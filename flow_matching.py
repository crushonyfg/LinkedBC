#!/usr/bin/env python3
"""
Route A: amortized graph-structured Flow-Matching Posterior Estimation (FMPE)
for the stochastic modular digital twin, plus an FM-proposal -> SMC-correction
hybrid.  See METHOD_REDESIGN.md sections 6 / 6'.

Idea (Dax et al. 2023, "Flow Matching for Scalable Simulation-Based Inference"):
train a conditional continuous normalizing flow q_phi(W | O) purely from
prior-predictive simulator draws (W, O) ~ p(W) p(O|W) -- NO simulator gradient,
NO likelihood evaluation at train time.  At inference, integrate the learned
ODE from N(0, I) conditioned on the real observation to draw posterior samples.

The novelty over a generic CNF is that the velocity field is *graph-structured*:
each module block W_j = (raw_theta_j, c_j, U_j) has its own velocity head that
sees only its neighbourhood in the faithful inverse graph (a chain b1-b2-b3,
from the moralized posterior of the cascade) and a *routed* observation
embedding (b1<-z1, b2<-z2,y, b3<-y).  This is the "information flow decided by
the probabilistic graph", not a monolithic MLP over the 101-D vector + all O.

Amortization scope (per user choice): FIXED design (fixed x grid + sensor
positions/counts); we amortize over data realizations, so q_phi targets the same
twin/dataset that Route B's SMC solves, giving a direct head-to-head plus an
SMC reference to correct against.

Hybrid correction (per user choice): FMPE particles seed a short graph-block
MCMC rejuvenation at lambda = 1 (reusing graph_transport.block_mh_sweep). No
flow density needed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import cascade_koh_flow_experiment as base
import gen_model as gm
import graph_transport as gt

FDTYPE = torch.float32   # the flow trains in float32; the model stays float64


# --------------------------------------------------------------------------- #
# W <-> unconstrained vector layout                                            #
# --------------------------------------------------------------------------- #
class Layout:
    """Flatten/unflatten W = (raw_theta, c1,c2,c3, U1,U2) and expose block slices."""

    def __init__(self, model: gm.StochasticCascade):
        n = model.n
        M1, M2, M3 = model.h1.M, model.h2.M, model.h3.M
        self.n, self.M1, self.M2, self.M3 = n, M1, M2, M3
        i = 0
        self.s_rawt = slice(i, i + 3); i += 3
        self.s_c1 = slice(i, i + M1); i += M1
        self.s_c2 = slice(i, i + M2); i += M2
        self.s_c3 = slice(i, i + M3); i += M3
        self.s_u1 = slice(i, i + n); i += n
        self.s_u2 = slice(i, i + n); i += n
        self.dim = i
        # block index vectors (into the flat W)
        r = np.arange(self.dim)
        self.b1 = np.concatenate([[0], r[self.s_c1], r[self.s_u1]])
        self.b2 = np.concatenate([[1], r[self.s_c2], r[self.s_u2]])
        self.b3 = np.concatenate([[2], r[self.s_c3]])
        self.nb = {"b1": [self.b1, self.b2],
                   "b2": [self.b1, self.b2, self.b3],
                   "b3": [self.b2, self.b3]}

    def to_flat(self, W: dict) -> torch.Tensor:
        th = torch.clamp(W["theta"], -0.999, 0.999)
        rawt = torch.atanh(th)
        return torch.cat([rawt, W["c1"], W["c2"], W["c3"], W["u1"], W["u2"]], dim=-1)

    def to_dict(self, flat: torch.Tensor) -> dict:
        return {
            "theta": torch.tanh(flat[:, self.s_rawt]).to(gm.DTYPE),
            "c1": flat[:, self.s_c1].to(gm.DTYPE),
            "c2": flat[:, self.s_c2].to(gm.DTYPE),
            "c3": flat[:, self.s_c3].to(gm.DTYPE),
            "u1": flat[:, self.s_u1].to(gm.DTYPE),
            "u2": flat[:, self.s_u2].to(gm.DTYPE),
        }


def obs_vector(model: gm.StochasticCascade, y, z1, z2) -> np.ndarray:
    return np.concatenate([y, z1, z2])


# --------------------------------------------------------------------------- #
# Training-pair generation: (W, O) ~ p(W) p(O|W)                               #
# --------------------------------------------------------------------------- #
def simulate_pairs(model: gm.StochasticCascade, layout: Layout, n_pairs: int,
                   chunk: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    Ws, Os = [], []
    ny, nz1, nz2 = model.n, len(model.z1_idx), len(model.z2_idx)
    done = 0
    while done < n_pairs:
        b = min(chunk, n_pairs - done)
        W = model.sample_prior(b, gen)
        # draw O ~ p(O | W)
        my, covy = model._y_mean_cov(W["u2"], W["theta"][:, 2], W["c3"])
        y = gm.mvn_sample(my, gm._chol(covy), gen)                       # (b, n)
        z1 = W["u1"][:, model.z1_idx] + model.sigma_z * torch.randn((b, nz1), dtype=gm.DTYPE, generator=gen)
        z2 = W["u2"][:, model.z2_idx] + model.sigma_z * torch.randn((b, nz2), dtype=gm.DTYPE, generator=gen)
        Ws.append(layout.to_flat(W).to(FDTYPE))
        Os.append(torch.cat([y, z1, z2], dim=-1).to(FDTYPE))
        done += b
    return torch.cat(Ws), torch.cat(Os)


# --------------------------------------------------------------------------- #
# Graph-structured velocity field                                             #
# --------------------------------------------------------------------------- #
def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.SiLU()]
    return torch.nn.Sequential(*layers[:-1])


class GraphVelocity(torch.nn.Module):
    """v_phi(W_t, O, t) with per-block heads masked by the faithful inverse graph."""

    def __init__(self, layout: Layout, nz1: int, nz2: int, ny: int, d_obs=48, hidden=192):
        super().__init__()
        self.L = layout
        # routed observation encoders  (b1<-z1, b2<-[z2,y], b3<-y)
        self.enc1 = mlp([nz1, hidden, d_obs])
        self.enc2 = mlp([nz2 + ny, hidden, d_obs])
        self.enc3 = mlp([ny, hidden, d_obs])
        self.d_obs = d_obs
        self.ny, self.nz1, self.nz2 = ny, nz1, nz2
        d_t = 8
        self.time = mlp([1, 32, d_t])
        # per-block velocity heads
        dim_b1, dim_b2, dim_b3 = len(layout.b1), len(layout.b2), len(layout.b3)
        in1 = dim_b1 + dim_b2 + d_obs + d_t
        in2 = dim_b1 + dim_b2 + dim_b3 + d_obs + d_t
        in3 = dim_b2 + dim_b3 + d_obs + d_t
        self.head1 = mlp([in1, hidden, hidden, dim_b1])
        self.head2 = mlp([in2, hidden, hidden, dim_b2])
        self.head3 = mlp([in3, hidden, hidden, dim_b3])
        self.bidx = {k: torch.tensor(getattr(layout, k), dtype=torch.long)
                     for k in ("b1", "b2", "b3")}

    def forward(self, Wt: torch.Tensor, O: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        y = O[:, :self.ny]
        z1 = O[:, self.ny:self.ny + self.nz1]
        z2 = O[:, self.ny + self.nz1:]
        e1 = self.enc1(z1)
        e2 = self.enc2(torch.cat([z2, y], -1))
        e3 = self.enc3(y)
        te = self.time(t)
        b1 = Wt[:, self.bidx["b1"]]
        b2 = Wt[:, self.bidx["b2"]]
        b3 = Wt[:, self.bidx["b3"]]
        v1 = self.head1(torch.cat([b1, b2, e1, te], -1))
        v2 = self.head2(torch.cat([b1, b2, b3, e2, te], -1))
        v3 = self.head3(torch.cat([b2, b3, e3, te], -1))
        v = torch.zeros_like(Wt)
        v[:, self.bidx["b1"]] += v1
        v[:, self.bidx["b2"]] += v2
        v[:, self.bidx["b3"]] += v3
        return v


# --------------------------------------------------------------------------- #
# Train / sample (rectified-flow conditional flow matching)                    #
# --------------------------------------------------------------------------- #
def train_fmpe(net, Wp, Op, Wm, Ws, Om, Os_, steps, batch, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    N = Wp.shape[0]
    Wn = (Wp - Wm) / Ws
    On = (Op - Om) / Os_
    hist = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (batch,), generator=g)
        W1 = Wn[idx]; O = On[idx]
        W0 = torch.randn(W1.shape, generator=g)
        t = torch.rand(batch, 1, generator=g)
        Wt = (1 - t) * W0 + t * W1
        target = W1 - W0
        v = net(Wt, O, t)
        loss = ((v - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if step == 1 or step % max(1, steps // 12) == 0:
            hist.append({"step": step, "loss": float(loss.detach())})
    return hist


@torch.no_grad()
def sample_fmpe(net, O_real_n, n_draw, n_steps, Wm, Ws, seed):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n_draw, Wm.shape[0], generator=g)
    O = O_real_n.unsqueeze(0).expand(n_draw, -1)
    dt = 1.0 / n_steps
    for k in range(n_steps):                          # Heun (2nd order) integrator
        t = torch.full((n_draw, 1), k * dt)
        v1 = net(W, O, t)
        Wm1 = W + dt * v1
        v2 = net(Wm1, O, torch.full((n_draw, 1), (k + 1) * dt))
        W = W + 0.5 * dt * (v1 + v2)
    return W * Ws + Wm                                # unstandardize -> flat W


# --------------------------------------------------------------------------- #
def theta_summary_np(theta, theta_true, w=None):
    if w is None:
        w = np.ones(len(theta)) / len(theta)
    mean = (theta * w[:, None]).sum(0)
    ci = np.zeros((3, 2))
    for j in range(3):
        o = np.argsort(theta[:, j]); cw = np.cumsum(w[o]); cw /= cw[-1]
        ci[j] = np.interp([0.025, 0.975], cw, theta[o, j])
    cover = [bool(ci[j, 0] <= theta_true[j] <= ci[j, 1]) for j in range(3)]
    return {"mean": mean.tolist(), "ci95": ci.tolist(),
            "rmse": float(np.sqrt(np.mean((mean - theta_true) ** 2))), "coverage95": cover}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=70)
    ap.add_argument("--n-field", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=120000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-draw", type=int, default=4000)
    ap.add_argument("--ode-steps", type=int, default=60)
    ap.add_argument("--correct-sweeps", type=int, default=40)
    ap.add_argument("--correct-particles", type=int, default=2000)
    ap.add_argument("--outdir", type=str, default="results/flow_matching")
    args = ap.parse_args()

    torch.set_num_threads(6)
    rng = np.random.default_rng(args.seed)
    theta_true = np.array([0.35, -0.55, 0.45])
    sigma_y, sigma_z = 0.06, 0.05

    # identical data/model construction to graph_transport (same seed => same twin)
    data = base.make_field_data(rng, args.n_field, theta_true, sigma_y, sigma_z)
    model = gm.build_cascade(rng, args.n_sim, data["x"], sigma_y, sigma_z)
    model._n_sim = args.n_sim
    gm.attach_data(model, data)
    layout = Layout(model)

    # ----- 1. simulate prior-predictive training pairs (gradient-free) ----- #
    t0 = time.time()
    Wp, Op = simulate_pairs(model, layout, args.n_train, 4096, args.seed + 3)
    sim_time = time.time() - t0
    Wm, Ws = Wp.mean(0), Wp.std(0) + 1e-6
    Om, Os_ = Op.mean(0), Op.std(0) + 1e-6

    # ----- 2. train graph-structured FMPE ----- #
    nz1, nz2, ny = len(model.z1_idx), len(model.z2_idx), model.n
    net = GraphVelocity(layout, nz1, nz2, ny)
    t0 = time.time()
    hist = train_fmpe(net, Wp, Op, Wm, Ws, Om, Os_, args.steps, args.batch, args.lr, args.seed + 4)
    train_time = time.time() - t0

    # ----- 3. amortized posterior at the REAL observation ----- #
    O_real = torch.tensor(obs_vector(model, data["y_obs"], data["z1_obs"], data["z2_obs"]), dtype=FDTYPE)
    O_real_n = (O_real - Om) / Os_
    flatW = sample_fmpe(net, O_real_n, args.n_draw, args.ode_steps, Wm, Ws, args.seed + 5)
    Wd = layout.to_dict(flatW)
    theta_fmpe = Wd["theta"].numpy()
    fmpe_summary = theta_summary_np(theta_fmpe, theta_true)

    # ----- 4. hybrid: FMPE particles -> graph-block MCMC correction (lambda=1) ----- #
    nc = min(args.correct_particles, args.n_draw)
    Wc = {k: v[:nc].clone() for k, v in Wd.items()}
    gen = torch.Generator().manual_seed(args.seed + 9)
    scales = {"p1_t": 0.05, "p1_c": 0.10, "u1_b": 0.12,
              "p2_t": 0.05, "p2_c": 0.10, "u2_b": 0.08,
              "p3_t": 0.05, "p3_c": 0.10}
    acc_last = {}
    for s in range(args.correct_sweeps):
        acc_last = gt.block_mh_sweep(model, Wc, 1.0, scales, gen)
        gt.adapt_scales(scales, acc_last)
    theta_corr = Wc["theta"].numpy()
    corr_summary = theta_summary_np(theta_corr, theta_true)

    # ----- SMC reference (from Route B run at the same seed, if available) ----- #
    smc_ref = None
    ref_path = Path("results/graph_transport/posterior.npz")
    if ref_path.exists():
        z = np.load(ref_path)
        if int(z["theta_true"].shape[0]) == 3 and np.allclose(z["x"], data["x"]):
            smc_ref = theta_summary_np(z["theta"], theta_true, z["weights"])

    out = {
        "setup": {"seed": args.seed, "n_field": args.n_field, "latent_dim": layout.dim,
                  "n_train": args.n_train, "steps": args.steps, "theta_true": theta_true.tolist(),
                  "sim_time_s": sim_time, "train_time_s": train_time},
        "fmpe": {"theta": fmpe_summary, "loss_hist": hist},
        "fmpe_smc_corrected": {"theta": corr_summary, "final_acc": acc_last, "sweeps": args.correct_sweeps},
        "smc_reference": smc_ref,
    }
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)
    np.savez(outdir / "posterior.npz", theta_fmpe=theta_fmpe, theta_corrected=theta_corr,
             theta_true=theta_true)

    print(f"\n===== Amortized graph-structured FMPE  |  latent dim = {layout.dim}  =====")
    print(f"true theta = {theta_true}   (sim {sim_time:.0f}s, train {train_time:.0f}s, "
          f"final loss {hist[-1]['loss']:.3f})")
    print(f"\n[FMPE q_phi]        theta mean = {np.round(fmpe_summary['mean'], 3)}  "
          f"RMSE={fmpe_summary['rmse']:.3f}  cover95={fmpe_summary['coverage95']}")
    print(f"[FMPE + SMC correct] theta mean = {np.round(corr_summary['mean'], 3)}  "
          f"RMSE={corr_summary['rmse']:.3f}  cover95={corr_summary['coverage95']}  "
          f"(acc p1/u1/p2/u2/p3 = {acc_last['p1']:.2f}/{acc_last['u1']:.2f}/{acc_last['p2']:.2f}/"
          f"{acc_last['u2']:.2f}/{acc_last['p3']:.2f})")
    if smc_ref is not None:
        print(f"[Route B SMC ref]   theta mean = {np.round(smc_ref['mean'], 3)}  "
              f"RMSE={smc_ref['rmse']:.3f}  cover95={smc_ref['coverage95']}")


if __name__ == "__main__":
    main()
