#!/usr/bin/env python3
"""
Graph-Structured Transport Calibration for Stochastic Modular Digital Twins.

Inference layer, Route B (see METHOD_REDESIGN.md, sections 4-6): a likelihood-
tempered Sequential Monte Carlo sampler whose transport / rejuvenation moves are
*module-block* Metropolis moves that exploit the posterior graph structure. No
simulator gradient, no linearization, no global Gaussian moment matching -- the
full non-Gaussian distributional uncertainty of the stochastic cascade is
propagated by the particles themselves.

Target:
    pi_lambda(W) propto p(W) * p(O | W)^lambda ,   lambda: 0 -> 1
with W = (theta, c_1, c_2, c_3, U_1, U_2). At lambda = 0 the target is the exact
generative prior, which we sample by ancestral simulation (gen_model). The
likelihood terms {z1, z2, y} are annealed in; the module transitions p(U_j|.)
stay at full strength throughout (they define the invariant prior).

Rejuvenation uses graph-local (Markov-blanket) block moves:
    W_1 = (theta_1, c_1, U_1) : touches  trans_1 + trans_2 + z1
    W_2 = (theta_2, c_2, U_2) : touches  trans_2 + y + z2
    W_3 = (theta_3, c_3)      : touches  y
Each block move recomputes ONLY the potentials incident on that block -- the
sparse structure of the faithful inverse graph is what makes this cheap and is
the methodological point, not an optimization afterthought.
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

DTYPE = gm.DTYPE


# --------------------------------------------------------------------------- #
# Resampling + ESS                                                             #
# --------------------------------------------------------------------------- #
def ess_of_logw(logw: torch.Tensor) -> float:
    w = torch.softmax(logw, dim=0)
    return float(1.0 / (w * w).sum())


def systematic_resample(logw: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    C = logw.shape[0]
    w = torch.softmax(logw, dim=0)
    positions = (torch.arange(C, dtype=DTYPE) + torch.rand(1, dtype=DTYPE, generator=gen)) / C
    cum = torch.cumsum(w, dim=0)
    return torch.searchsorted(cum, positions).clamp(max=C - 1)


def reindex(W: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx].clone() for k, v in W.items()}


# --------------------------------------------------------------------------- #
# One graph-structured block-MH sweep at temperature lambda                    #
# --------------------------------------------------------------------------- #
def perturb(x: torch.Tensor, scale: float, gen: torch.Generator) -> torch.Tensor:
    return x + scale * torch.randn(x.shape, dtype=DTYPE, generator=gen)


def _mh_accept(W: dict, prop: dict, logratio: torch.Tensor, keys, gen) -> float:
    a = torch.log(torch.rand(logratio.shape, dtype=DTYPE, generator=gen)) < logratio
    for k in keys:
        if prop[k].dim() == W[k].dim() and prop[k].dim() == 2:
            W[k] = torch.where(a.unsqueeze(-1), prop[k], W[k])
        else:
            W[k] = torch.where(a, prop[k], W[k])
    return float(a.double().mean())


def _pcn_propose(mean, L, u, beta, gen):
    xi = torch.randn(u.shape, dtype=DTYPE, generator=gen)
    step = torch.einsum("cij,cj->ci", L, xi)
    return mean + math.sqrt(1.0 - beta * beta) * (u - mean) + beta * step


def block_mh_sweep(model, W, lam, scales, gen) -> dict:
    """Metropolis-within-Gibbs sweep with graph-local moves.

    Parameter blocks (theta_j, c_j) use small random walks with the latent held
    fixed; latent states U_1, U_2 use pCN moves that leave their Gaussian module
    transition invariant, so each latent move's MH ratio contains ONLY the
    downstream / observation potentials incident on that node. Every move
    touches just its Markov blanket in the faithful inverse graph -- that
    locality is the graph-structured transport, and pCN is what lets the
    tightly-coupled GP cascade actually mix."""
    NEG = torch.full((W["theta"].shape[0],), -1e30, dtype=DTYPE)
    acc = {}

    def accept(prop, logratio, keys, name):
        acc[name] = _mh_accept(W, prop, logratio, keys, gen)

    # ---- param block 1: (theta_1, c_1), U_1 fixed -> touches trans_1 only --- #
    th = W["theta"].clone(); th[:, 0] = perturb(th[:, 0], scales["p1_t"], gen)
    c1p = perturb(W["c1"], scales["p1_c"], gen)
    prop = {**W, "theta": th, "c1": c1p}
    box = (th[:, 0] > -1.0) & (th[:, 0] < 1.0)
    cur = model.log_trans1(W) - 0.5 * (W["c1"] ** 2).sum(-1)
    new = model.log_trans1(prop) - 0.5 * (c1p ** 2).sum(-1)
    accept(prop, torch.where(box, new - cur, NEG), ("theta", "c1"), "p1")

    # ---- latent U_1 pCN (fixed parents) -> touches trans_2 + z1 ------------- #
    m1, cov1 = model._trans1_mean_cov(W["theta"][:, 0], W["c1"])
    u1p = _pcn_propose(m1, gm._chol(cov1), W["u1"], scales["u1_b"], gen)
    prop = {**W, "u1": u1p}
    cur = model.log_trans2(W) + lam * model.log_lik_z1(W)
    new = model.log_trans2(prop) + lam * model.log_lik_z1(prop)
    accept(prop, new - cur, ("u1",), "u1")

    # ---- param block 2: (theta_2, c_2), U_2 fixed -> touches trans_2 only --- #
    th = W["theta"].clone(); th[:, 1] = perturb(th[:, 1], scales["p2_t"], gen)
    c2p = perturb(W["c2"], scales["p2_c"], gen)
    prop = {**W, "theta": th, "c2": c2p}
    box = (th[:, 1] > -1.0) & (th[:, 1] < 1.0)
    cur = model.log_trans2(W) - 0.5 * (W["c2"] ** 2).sum(-1)
    new = model.log_trans2(prop) - 0.5 * (c2p ** 2).sum(-1)
    accept(prop, torch.where(box, new - cur, NEG), ("theta", "c2"), "p2")

    # ---- latent U_2 pCN (fixed parents) -> touches z2 + y ------------------- #
    m2, cov2 = model._trans2_mean_cov(W["u1"], W["theta"][:, 1], W["c2"])
    u2p = _pcn_propose(m2, gm._chol(cov2), W["u2"], scales["u2_b"], gen)
    prop = {**W, "u2": u2p}
    cur = lam * (model.log_lik_z2(W) + model.log_lik_y(W))
    new = lam * (model.log_lik_z2(prop) + model.log_lik_y(prop))
    accept(prop, new - cur, ("u2",), "u2")

    # ---- param block 3: (theta_3, c_3) -> touches y only ------------------- #
    th = W["theta"].clone(); th[:, 2] = perturb(th[:, 2], scales["p3_t"], gen)
    c3p = perturb(W["c3"], scales["p3_c"], gen)
    prop = {**W, "theta": th, "c3": c3p}
    box = (th[:, 2] > -1.0) & (th[:, 2] < 1.0)
    cur = -0.5 * (W["c3"] ** 2).sum(-1) + lam * model.log_lik_y(W)
    new = -0.5 * (c3p ** 2).sum(-1) + lam * model.log_lik_y(prop)
    accept(prop, torch.where(box, new - cur, NEG), ("theta", "c3"), "p3")
    return acc


_RW_KEYS = {"p1": ("p1_t", "p1_c"), "p2": ("p2_t", "p2_c"), "p3": ("p3_t", "p3_c")}


def adapt_scales(scales: dict, acc_hist: dict, target=0.3):
    """Robbins-Monro-ish nudge of RW scales and pCN betas toward target accept."""
    for name, keys in _RW_KEYS.items():
        fac = min(max(float(np.exp(0.5 * (acc_hist[name] - target))), 0.6), 1.5)
        for k in keys:
            scales[k] = float(np.clip(scales[k] * fac, 1e-3, 2.0))
    for name, k in (("u1", "u1_b"), ("u2", "u2_b")):
        fac = min(max(float(np.exp(0.5 * (acc_hist[name] - target))), 0.6), 1.5)
        scales[k] = float(np.clip(scales[k] * fac, 1e-3, 0.999))


# --------------------------------------------------------------------------- #
# Adaptive likelihood-tempered SMC with graph-block rejuvenation               #
# --------------------------------------------------------------------------- #
def run_graph_smc(model, n_particles, n_mh, ess_target_frac, resample_frac, seed):
    gen = torch.Generator().manual_seed(seed)
    W = model.sample_prior(n_particles, gen)   # lambda = 0 : exact prior draws
    logw = torch.zeros(n_particles, dtype=DTYPE)
    lam = 0.0
    scales = {
        "p1_t": 0.08, "p1_c": 0.15, "u1_b": 0.30,
        "p2_t": 0.08, "p2_c": 0.15, "u2_b": 0.12,
        "p3_t": 0.08, "p3_c": 0.15,
    }
    schedule, ess_trace, acc_trace = [], [], []
    loglik_cache = model.log_lik(W)            # incremental-weight base

    while lam < 1.0:
        # ----- adaptive temperature: pick d_lam so ESS drops to target ----- #
        base_logw = logw
        lo, hi = 0.0, 1.0 - lam
        target = ess_target_frac * n_particles
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            trial = base_logw + mid * loglik_cache
            if ess_of_logw(trial) < target:
                hi = mid
            else:
                lo = mid
        d_lam = max(hi, 1e-3)
        lam_new = min(1.0, lam + d_lam)
        d_lam = lam_new - lam

        # ----- reweight ----- #
        logw = logw + d_lam * loglik_cache
        lam = lam_new
        ess = ess_of_logw(logw)

        # ----- resample if degenerate ----- #
        if ess < resample_frac * n_particles:
            idx = systematic_resample(logw, gen)
            W = reindex(W, idx)
            logw = torch.zeros(n_particles, dtype=DTYPE)

        # ----- graph-block rejuvenation at pi_lambda ----- #
        sweep_acc = {"p1": 0.0, "u1": 0.0, "p2": 0.0, "u2": 0.0, "p3": 0.0}
        for _ in range(n_mh):
            a = block_mh_sweep(model, W, lam, scales, gen)
            for k in sweep_acc:
                sweep_acc[k] += a[k] / n_mh
        adapt_scales(scales, sweep_acc)

        loglik_cache = model.log_lik(W)        # refresh after moves
        schedule.append(lam)
        ess_trace.append(ess)
        acc_trace.append(sweep_acc)

    w = torch.softmax(logw, dim=0)
    return {
        "W": W, "weights": w, "logw": logw,
        "schedule": schedule, "ess_trace": ess_trace, "acc_trace": acc_trace,
        "n_temps": len(schedule), "final_scales": scales,
    }


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #
def weighted_theta_summary(theta: np.ndarray, w: np.ndarray, theta_true: np.ndarray) -> dict:
    mean = (theta * w[:, None]).sum(0)
    order = np.argsort(theta, axis=0)
    ci = np.zeros((3, 2))
    for j in range(3):
        o = order[:, j]
        cw = np.cumsum(w[o]); cw /= cw[-1]
        ci[j] = np.interp([0.025, 0.975], cw, theta[o, j])
    cover = [bool(ci[j, 0] <= theta_true[j] <= ci[j, 1]) for j in range(3)]
    return {
        "mean": mean.tolist(), "ci95": ci.tolist(),
        "abs_error": np.abs(mean - theta_true).tolist(),
        "rmse": float(np.sqrt(np.mean((mean - theta_true) ** 2))),
        "coverage95": cover,
    }


def resample_particles(W: dict, w: torch.Tensor, n: int, gen: torch.Generator) -> dict:
    idx = torch.multinomial(w, n, replacement=True, generator=gen)
    return {k: v[idx] for k, v in W.items()}


def predictive_uq(model_field, x_test, W, w, theta_true, gen, n_draw=4000):
    """Posterior-predictive at held-out x_test with FULL distributional draws.

    Returns RMSE for u1,u2,y (posterior-mean prediction) plus non-Gaussianity
    diagnostics (skew, excess kurtosis, bimodality) of p(y|O) at test points --
    the signature that distinguishes distributional propagation from linked-GaSP
    moment matching."""
    rng = np.random.default_rng(0)
    model_test = gm.build_cascade(rng, model_field._n_sim, x_test, model_field.sigma_y,
                                  model_field.sigma_z, model_field.sigma_xi)
    # copy the SAME fitted emulators so test == field twin
    model_test.gp1, model_test.gp2, model_test.gp3 = model_field.gp1, model_field.gp2, model_field.gp3
    idx = torch.multinomial(w, n_draw, replacement=True, generator=gen)
    # chunk the ancestral draws to keep the (chunk, n_test, n_test) chols bounded
    u1s, u2s, ys = [], [], []
    for s in range(0, n_draw, 500):
        sl = idx[s:s + 500]
        a, b, c = model_test.forward_sample(W["theta"][sl], W["c1"][sl], W["c2"][sl], W["c3"][sl], gen)
        u1s.append(a); u2s.append(b); ys.append(c)
    u1 = torch.cat(u1s).numpy(); u2 = torch.cat(u2s).numpy(); y = torch.cat(ys).numpy()
    u1t, u2t, yt = base.physical_forward_np(x_test, theta_true)

    def stats(col):
        m, s = col.mean(), col.std() + 1e-12
        z = (col - m) / s
        return float((z**3).mean()), float((z**4).mean() - 3.0)

    # pick a few interior test points for non-Gaussianity report
    pts = np.linspace(10, len(x_test) - 10, 5).astype(int)
    ng = [{"x": float(x_test[p]),
           "y_skew": stats(y[:, p])[0], "y_exkurt": stats(y[:, p])[1]} for p in pts]
    return {
        "u1_rmse": float(np.sqrt(np.mean((u1.mean(0) - u1t) ** 2))),
        "u2_rmse": float(np.sqrt(np.mean((u2.mean(0) - u2t) ** 2))),
        "y_rmse": float(np.sqrt(np.mean((y.mean(0) - yt) ** 2))),
        "y_nongaussianity": ng,
        "mean_abs_skew": float(np.mean([abs(s["y_skew"]) for s in ng])),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=70)
    ap.add_argument("--n-field", type=int, default=30)
    ap.add_argument("--particles", type=int, default=512)
    ap.add_argument("--n-mh", type=int, default=6)
    ap.add_argument("--ess-target", type=float, default=0.5,
                    help="ESS fraction to drop to per temperature; higher => more "
                         "bridging distributions (slower, better-mixed).")
    ap.add_argument("--resample-frac", type=float, default=0.5)
    ap.add_argument("--outdir", type=str, default="results/graph_transport")
    args = ap.parse_args()

    torch.set_num_threads(6)
    rng = np.random.default_rng(args.seed)
    theta_true = np.array([0.35, -0.55, 0.45])
    sigma_y, sigma_z = 0.06, 0.05

    data = base.make_field_data(rng, args.n_field, theta_true, sigma_y, sigma_z)
    model = gm.build_cascade(rng, args.n_sim, data["x"], sigma_y, sigma_z)
    model._n_sim = args.n_sim
    gm.attach_data(model, data)

    # ----- new method: graph-structured transport SMC ----- #
    t0 = time.time()
    smc = run_graph_smc(model, args.particles, args.n_mh, args.ess_target,
                        args.resample_frac, args.seed + 1)
    smc_time = time.time() - t0
    W, w = smc["W"], smc["weights"]
    theta_np = W["theta"].numpy(); w_np = w.numpy()
    theta_summary = weighted_theta_summary(theta_np, w_np, theta_true)

    gen = torch.Generator().manual_seed(args.seed + 5)
    x_test = np.linspace(0.02, 0.98, 60)
    pred = predictive_uq(model, x_test, W, w, theta_true, gen)

    # ----- baseline: final-output-only black-box KOH ----- #
    rng_bb = np.random.default_rng(args.seed)
    _ = base.make_field_data(rng_bb, args.n_field, theta_true, sigma_y, sigma_z)  # align rng
    gps_bb = base.make_simulation_design(rng_bb, args.n_sim)
    blackbox = base.run_black_box_koh(rng_bb, data, gps_bb, sigma_y, 16000, 3000)
    bb_theta = base.summarize_theta(blackbox["samples"], theta_true)
    bb_pred = base.evaluate_predictions("black_box", x_test, blackbox["samples"], gps_bb,
                                        theta_true, bb_data=data, bb_chol=blackbox["chol"])

    out = {
        "setup": {
            "seed": args.seed, "n_sim": args.n_sim, "n_field": args.n_field,
            "n_z1": int(len(data["z1_idx"])), "n_z2": int(len(data["z2_idx"])),
            "particles": args.particles, "n_mh": args.n_mh,
            "latent_dim": 3 + model.h1.M + model.h2.M + model.h3.M + 2 * args.n_field,
            "theta_true": theta_true.tolist(), "sigma_xi": list(model.sigma_xi),
        },
        "graph_smc": {
            "time_s": smc_time, "n_temps": smc["n_temps"],
            "ess_trace": smc["ess_trace"], "schedule": smc["schedule"],
            "final_acc": smc["acc_trace"][-1], "final_scales": smc["final_scales"],
            "theta": theta_summary, "prediction": pred,
        },
        "black_box_koh": {"theta": bb_theta, "prediction": bb_pred},
    }

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)
    np.savez(outdir / "posterior.npz", theta=theta_np, weights=w_np,
             u1=W["u1"].numpy(), u2=W["u2"].numpy(), theta_true=theta_true,
             x=data["x"], y_obs=data["y_obs"])

    # ----- console report ----- #
    print(f"\n===== Graph-Structured Transport SMC  |  latent dim = {out['setup']['latent_dim']}  =====")
    print(f"true theta = {theta_true}   (n_field={args.n_field}, n_z1={out['setup']['n_z1']}, n_z2={out['setup']['n_z2']})")
    fa = smc["acc_trace"][-1]
    print(f"\n[graph SMC] {smc['n_temps']} temperatures, {smc_time:.1f}s, final acc "
          f"p1/u1/p2/u2/p3 = {fa['p1']:.2f}/{fa['u1']:.2f}/{fa['p2']:.2f}/{fa['u2']:.2f}/{fa['p3']:.2f}")
    print(f"  theta mean = {np.round(theta_summary['mean'], 3)}  RMSE={theta_summary['rmse']:.3f}  "
          f"cover95={theta_summary['coverage95']}")
    print(f"  pred RMSE  u1={pred['u1_rmse']:.3f}  u2={pred['u2_rmse']:.3f}  y={pred['y_rmse']:.3f}")
    print(f"  p(y|O) mean |skew| over test pts = {pred['mean_abs_skew']:.3f}  "
          f"(0 => Gaussian; >0 => distributional propagation matters)")
    print(f"\n[black-box KOH] theta mean = {np.round(bb_theta['mean'], 3)}  RMSE={bb_theta['rmse']:.3f}  "
          f"cover95={bb_theta['coverage95']}")
    print(f"  pred RMSE  y={bb_pred['y_rmse']:.3f}")
    print(f"\n-> {'STRUCTURED WINS on theta' if theta_summary['rmse'] < bb_theta['rmse'] else 'check setup'}"
          f"  (theta RMSE {theta_summary['rmse']:.3f} vs {bb_theta['rmse']:.3f})")


if __name__ == "__main__":
    main()
