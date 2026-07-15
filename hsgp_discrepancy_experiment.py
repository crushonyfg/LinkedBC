#!/usr/bin/env python3
"""
GP-discrepancy version of the structured cascade KOH calibration.

The hand-picked 12-dim oracle basis of ``cascade_koh_flow_experiment`` is
replaced by a Hilbert-space (reduced-rank) GP discrepancy per module
(Solin & Sarkka 2020): a generic Laplacian-eigenfunction basis with a
squared-exponential spectral-density prior and *whitened* N(0,1) coefficients.

  delta_j(v) = sum_m sqrt(S_j(omega_m)) * phi_m(v) * c_{j,m},   c ~ N(0,I)

- module 1: delta_1(x)            -> 1-D HSGP over x
- module 2: delta_2(x, u1)        -> 2-D HSGP over (x, u1)
- module 3: delta_3(x, u2)        -> 2-D HSGP over (x, u2)

This is a legitimate (approximate) GP discrepancy on a *data-agnostic* basis, so
it removes the "oracle basis" criticism, propagates the upstream discrepancy
non-linearly through the downstream emulator (which, per the linked-GP
literature, cannot be marginalized -> the coefficients are sampled as latents,
i.e. the dgpsi "stochastic imputation" route), and raises the posterior to ~40D.

We then answer two questions empirically:
  (a) does the structured GP-discrepancy model still beat black-box KOH on theta?
  (b) is the ~40-D posterior solvable, and how do flow+IS / HMC / NeuTra compare?

Discrepancy GP hyperparameters (lengthscale, amplitude) are FIXED here to
isolate the theta-delta confounding; sampling them would add hierarchical
funnels (and further favor preconditioned/flow solvers).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import cascade_koh_flow_experiment as base
from flow_vs_mcmc_experiment import run_hmc, make_neutra_logp, psis_khat, theta_summary


# --------------------------------------------------------------------------- #
# Hilbert-space GP feature maps (scaled by sqrt spectral density)              #
# --------------------------------------------------------------------------- #
@dataclass
class HSGP1D:
    M: int
    L: float
    ell: float
    sigma: float

    def omega(self) -> np.ndarray:
        m = np.arange(1, self.M + 1, dtype=np.float64)
        return math.pi * m / (2 * self.L)

    def sqrt_S(self) -> np.ndarray:
        w = self.omega()
        S = self.sigma**2 * math.sqrt(2 * math.pi) * self.ell * np.exp(-0.5 * w**2 * self.ell**2)
        return np.sqrt(S)


@dataclass
class HSGP2D:
    Mx: int
    Mu: int
    Lx: float
    Lu: float
    ellx: float
    ellu: float
    sigma: float

    @property
    def M(self) -> int:
        return self.Mx * self.Mu

    def sqrt_S(self) -> np.ndarray:
        mx = np.arange(1, self.Mx + 1, dtype=np.float64)
        mu = np.arange(1, self.Mu + 1, dtype=np.float64)
        wx = math.pi * mx / (2 * self.Lx)
        wu = math.pi * mu / (2 * self.Lu)
        Sx = math.sqrt(2 * math.pi) * self.ellx * np.exp(-0.5 * wx**2 * self.ellx**2)
        Su = math.sqrt(2 * math.pi) * self.ellu * np.exp(-0.5 * wu**2 * self.ellu**2)
        S = self.sigma**2 * Sx[:, None] * Su[None, :]
        return np.sqrt(S).reshape(-1)   # (Mx*Mu,)


def phi_1d(t_std: torch.Tensor, M: int, L: float) -> torch.Tensor:
    # t_std: (...,)  in ~[-1,1];  returns (..., M) eigenfunctions on [-L, L]
    m = torch.arange(1, M + 1, dtype=t_std.dtype, device=t_std.device)
    w = math.pi * m / (2 * L)
    return math.sqrt(1.0 / L) * torch.sin(w * (t_std.unsqueeze(-1) + L))


def features_2d(x_std: torch.Tensor, u_std: torch.Tensor, Mx: int, Mu: int, Lx: float, Lu: float) -> torch.Tensor:
    # returns (..., Mx*Mu)
    px = phi_1d(x_std, Mx, Lx)               # (..., Mx)
    pu = phi_1d(u_std, Mu, Lu)               # (..., Mu)
    phi = px.unsqueeze(-1) * pu.unsqueeze(-2)  # (..., Mx, Mu)
    return phi.reshape(*phi.shape[:-2], Mx * Mu)


# standardization ranges (match emulator training design)
X_MID, X_HALF = 0.5, 0.5
U1_MID, U1_HALF = 0.25, 1.75
U2_MID, U2_HALF = 0.30, 2.10


@dataclass
class HSGPTarget:
    x: torch.Tensor
    y: torch.Tensor
    z1_idx: torch.Tensor
    z2_idx: torch.Tensor
    z1_obs: torch.Tensor
    z2_obs: torch.Tensor
    gps: tuple[dict, dict, dict]
    sigma_y: float
    sigma_z: float
    h1: HSGP1D
    h2: HSGP2D
    h3: HSGP2D
    sqrtS1: torch.Tensor
    sqrtS2: torch.Tensor
    sqrtS3: torch.Tensor

    @property
    def dim(self) -> int:
        return 3 + self.h1.M + self.h2.M + self.h3.M

    def _split(self, w: torch.Tensor):
        theta = torch.tanh(w[:, :3])
        i = 3
        c1 = w[:, i:i + self.h1.M]; i += self.h1.M
        c2 = w[:, i:i + self.h2.M]; i += self.h2.M
        c3 = w[:, i:i + self.h3.M]
        return theta, c1, c2, c3

    def forward_at(self, x: torch.Tensor, w: torch.Tensor):
        theta, c1, c2, c3 = self._split(w)
        b = w.shape[0]
        xb = x.unsqueeze(0).expand(b, -1)          # (C, n)
        x_std = (x - X_MID) / X_HALF               # (n,)

        # module 1
        t1 = theta[:, 0:1].expand(-1, xb.shape[1])
        u1 = base.gp_predict_torch(torch.stack([xb, t1], dim=-1), self.gps[0])
        phi1 = phi_1d(x_std, self.h1.M, self.h1.L) * self.sqrtS1   # (n, M1)
        u1 = u1 + torch.einsum("nm,cm->cn", phi1, c1)

        # module 2
        t2 = theta[:, 1:2].expand(-1, xb.shape[1])
        u2 = base.gp_predict_torch(torch.stack([xb, u1, t2], dim=-1), self.gps[1])
        u1_std = (u1 - U1_MID) / U1_HALF
        f2 = features_2d(x_std.unsqueeze(0).expand(b, -1), u1_std,
                         self.h2.Mx, self.h2.Mu, self.h2.Lx, self.h2.Lu) * self.sqrtS2
        u2 = u2 + torch.einsum("cnm,cm->cn", f2, c2)

        # module 3
        t3 = theta[:, 2:3].expand(-1, xb.shape[1])
        yhat = base.gp_predict_torch(torch.stack([xb, u2, t3], dim=-1), self.gps[2])
        u2_std = (u2 - U2_MID) / U2_HALF
        f3 = features_2d(x_std.unsqueeze(0).expand(b, -1), u2_std,
                         self.h3.Mx, self.h3.Mu, self.h3.Lx, self.h3.Lu) * self.sqrtS3
        yhat = yhat + torch.einsum("cnm,cm->cn", f3, c3)
        return theta, u1, u2, yhat

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        theta, u1, u2, yhat = self.forward_at(self.x, w)
        coefs = w[:, 3:]
        log_jac_theta = torch.sum(torch.log(torch.clamp(1.0 - theta * theta, min=1e-7)), dim=-1)
        log_prior_c = -0.5 * torch.sum(coefs * coefs, dim=-1)
        ry = (self.y.unsqueeze(0) - yhat) / self.sigma_y
        ll_y = -0.5 * torch.sum(ry * ry, dim=-1)
        rz1 = (self.z1_obs.unsqueeze(0) - u1[:, self.z1_idx]) / self.sigma_z
        ll_z1 = -0.5 * torch.sum(rz1 * rz1, dim=-1)
        rz2 = (self.z2_obs.unsqueeze(0) - u2[:, self.z2_idx]) / self.sigma_z
        ll_z2 = -0.5 * torch.sum(rz2 * rz2, dim=-1)
        return log_jac_theta + log_prior_c + ll_y + ll_z1 + ll_z2


def train_flow_generic(seed, target, dim, steps, batch, lr, device):
    torch.manual_seed(seed)
    flow = base.RealNVP(dim=dim, n_layers=10, hidden=128).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-5)
    for step in range(1, steps + 1):
        w, logq = flow.sample_and_logq(batch, device)
        logp = target.log_prob(w)
        loss = torch.mean(logq - logp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 25.0)
        opt.step()
    return flow


def structured_pred_rmse(target, x_test_t, theta_true, samples_w, weights):
    u1t, u2t, yt = base.physical_forward_np(x_test_t.numpy(), theta_true)
    with torch.no_grad():
        _, u1, u2, yh = target.forward_at(x_test_t, samples_w)
    u1 = u1.numpy(); u2 = u2.numpy(); yh = yh.numpy()
    pu1 = np.average(u1, axis=0, weights=weights)
    pu2 = np.average(u2, axis=0, weights=weights)
    py = np.average(yh, axis=0, weights=weights)
    return {
        "u1_rmse": float(np.sqrt(np.mean((pu1 - u1t) ** 2))),
        "u2_rmse": float(np.sqrt(np.mean((pu2 - u2t) ** 2))),
        "y_rmse": float(np.sqrt(np.mean((py - yt) ** 2))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=70)
    ap.add_argument("--n-field", type=int, default=40)
    ap.add_argument("--flow-steps", type=int, default=4000)
    ap.add_argument("--flow-batch", type=int, default=128)
    ap.add_argument("--hmc-warmup", type=int, default=600)
    ap.add_argument("--hmc-sample", type=int, default=1200)
    ap.add_argument("--hmc-chains", type=int, default=12)
    ap.add_argument("--hmc-leapfrog", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="results/hsgp_discrepancy")
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    theta_true = np.array([0.35, -0.55, 0.45])
    sigma_y, sigma_z = 0.06, 0.05

    gps = base.make_simulation_design(rng, args.n_sim)
    data = base.make_field_data(rng, args.n_field, theta_true, sigma_y, sigma_z)

    h1 = HSGP1D(M=8, L=1.5, ell=0.35, sigma=0.25)
    h2 = HSGP2D(Mx=5, Mu=3, Lx=1.5, Lu=1.5, ellx=0.35, ellu=0.60, sigma=0.22)
    h3 = HSGP2D(Mx=5, Mu=3, Lx=1.5, Lu=1.5, ellx=0.35, ellu=0.60, sigma=0.22)
    target = HSGPTarget(
        x=torch.tensor(data["x"], dtype=torch.float32),
        y=torch.tensor(data["y_obs"], dtype=torch.float32),
        z1_idx=torch.tensor(data["z1_idx"], dtype=torch.long),
        z2_idx=torch.tensor(data["z2_idx"], dtype=torch.long),
        z1_obs=torch.tensor(data["z1_obs"], dtype=torch.float32),
        z2_obs=torch.tensor(data["z2_obs"], dtype=torch.float32),
        gps=tuple(gp.torch_tensors(device) for gp in gps),  # type: ignore[arg-type]
        sigma_y=sigma_y, sigma_z=sigma_z,
        h1=h1, h2=h2, h3=h3,
        sqrtS1=torch.tensor(h1.sqrt_S(), dtype=torch.float32),
        sqrtS2=torch.tensor(h2.sqrt_S(), dtype=torch.float32),
        sqrtS3=torch.tensor(h3.sqrt_S(), dtype=torch.float32),
    )
    dim = target.dim
    x_test_t = torch.tensor(np.linspace(0.01, 0.99, 120), dtype=torch.float32)

    out = {"setup": {"seed": args.seed, "dim": dim, "theta_true": theta_true.tolist(),
                     "n_coef": dim - 3, "M": [h1.M, h2.M, h3.M]}}

    # --- black-box KOH baseline (unchanged) --------------------------------- #
    blackbox = base.run_black_box_koh(rng, data, gps, sigma_y, 16000, 3000)
    bb_samples = blackbox["samples"]
    out["black_box_koh"] = {
        "theta": base.summarize_theta(bb_samples, theta_true),
        "prediction": base.evaluate_predictions(
            "black_box", x_test_t.numpy(), bb_samples, gps, theta_true,
            bb_data=data, bb_chol=blackbox["chol"]),
    }

    # --- flow + IS ---------------------------------------------------------- #
    t0 = time.time()
    flow = train_flow_generic(args.seed + 100, target, dim, args.flow_steps, args.flow_batch, 1.5e-3, device)
    flow_time = time.time() - t0
    with torch.no_grad():
        w_f, logq = flow.sample_and_logq(12000, device)
        logp = target.log_prob(w_f)
    theta_f = torch.tanh(w_f[:, :3]).numpy()
    logw = (logp - logq).numpy(); logw -= logw.max()
    wts = np.exp(logw); wts /= wts.sum()
    ess = float(1.0 / np.sum(wts**2))
    khat = psis_khat(logp.numpy() - logq.numpy())
    theta_is_mean = (theta_f * wts[:, None]).sum(0)
    out["flow_is"] = {
        "train_time_s": flow_time, "ess": ess, "ess_frac": ess / len(wts), "psis_khat": khat,
        "theta_raw": theta_summary(theta_f, theta_true),
        "theta_is_mean": theta_is_mean.tolist(),
        "theta_is_rmse": float(np.sqrt(np.mean((theta_is_mean - theta_true) ** 2))),
        "prediction": structured_pred_rmse(target, x_test_t, theta_true, w_f, wts),
    }

    # --- HMC gold standard -------------------------------------------------- #
    t0 = time.time()
    hmc_w, hmc_diag = run_hmc(target.log_prob, dim, args.hmc_chains, args.hmc_warmup,
                              args.hmc_sample, args.hmc_leapfrog, args.seed + 1)
    hmc_time = time.time() - t0
    theta_hmc = np.tanh(hmc_w[:, :3])
    out["hmc"] = {"time_s": hmc_time, **hmc_diag, "theta": theta_summary(theta_hmc, theta_true),
                  "prediction": structured_pred_rmse(target, x_test_t, theta_true,
                                                     torch.tensor(hmc_w, dtype=torch.float32),
                                                     np.ones(len(hmc_w)) / len(hmc_w))}

    # --- NeuTra ------------------------------------------------------------- #
    logp_z, z_to_w = make_neutra_logp(flow, target)
    t0 = time.time()
    neutra_z, neutra_diag = run_hmc(logp_z, dim, args.hmc_chains, args.hmc_warmup,
                                    args.hmc_sample, args.hmc_leapfrog, args.seed + 2)
    with torch.no_grad():
        neutra_w = z_to_w(torch.tensor(neutra_z, dtype=torch.float32))
    neutra_time = time.time() - t0
    theta_neutra = np.tanh(neutra_w[:, :3].numpy())
    out["neutra"] = {"time_s": neutra_time, **neutra_diag, "theta": theta_summary(theta_neutra, theta_true),
                     "prediction": structured_pred_rmse(target, x_test_t, theta_true, neutra_w,
                                                        np.ones(len(neutra_w)) / len(neutra_w))}

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with (Path(args.outdir) / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)

    bb = out["black_box_koh"]
    print(f"\n===== GP(HSGP) discrepancy structured KOH  |  dim = {dim}  =====")
    print(f"true theta = {theta_true}")
    print(f"\n[black-box KOH]   theta RMSE={bb['theta']['rmse']:.3f}  y RMSE={bb['prediction']['y_rmse']:.3f}"
          f"  cover95={bb['theta']['coverage95']}")
    fi = out["flow_is"]
    print(f"\n[flow+IS]  ESS={fi['ess']:.0f}/12000 ({fi['ess_frac']*100:.1f}%)  PSIS k-hat={fi['psis_khat']:.3f}"
          f"  train={fi['train_time_s']:.1f}s")
    print(f"  theta RMSE={fi['theta_is_rmse']:.3f}  y RMSE={fi['prediction']['y_rmse']:.3f}  mean={np.round(fi['theta_is_mean'],3)}")
    hm = out["hmc"]
    print(f"\n[HMC gold] acc={hm['acceptance']:.2f}  Rhat_max={hm['rhat_max']:.3f}  time={hm['time_s']:.1f}s")
    print(f"  theta RMSE={hm['theta']['rmse']:.3f}  y RMSE={hm['prediction']['y_rmse']:.3f}"
          f"  cover95={hm['theta']['coverage95']}  mean={np.round(hm['theta']['mean'],3)}")
    nt = out["neutra"]
    print(f"\n[NeuTra]   acc={nt['acceptance']:.2f}  Rhat_max={nt['rhat_max']:.3f}  time={nt['time_s']:.1f}s")
    print(f"  theta RMSE={nt['theta']['rmse']:.3f}  y RMSE={nt['prediction']['y_rmse']:.3f}"
          f"  cover95={nt['theta']['coverage95']}  mean={np.round(nt['theta']['mean'],3)}")


if __name__ == "__main__":
    main()
