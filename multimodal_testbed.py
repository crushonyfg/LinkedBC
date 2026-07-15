#!/usr/bin/env python3
"""
E8 gate: a genuinely MULTIMODAL calibration testbed and the check that the
3-line comparison (E9) will have signal on it.

Mechanism (design locked 2026-07-15): put a SIGN SYMMETRY on the MIDDLE module.
Module 2's forward depends on theta_2 ONLY through an even function, and the
odd / non-even terms of the original eta_2 are removed:

    eta2_even(x, u1, theta2) = 0.65 u1 + cos(1.2 pi x)
                               + beta (theta2^2 - c) + gamma u1 (theta2^2 - c)

Because U_2 (hence the sensor z_2 AND the terminal y) is identical for +/-theta_2,
the sign of theta_2 is *structurally, observation-un-breakably* non-identifiable:
the posterior is a symmetric bimodal at theta_2 = +/- theta_2* (equal mass, exact
ground truth), while theta_1 (via z_1) and theta_3 (terminal) stay identified.

This script establishes the gate:
  (A) tempered graph SMC (Route B) is MODE-COVERING -> recovers BOTH modes ~50/50.
  (B) a reverse-KL VI flow (min KL(q||p)) is MODE-SEEKING -> COLLAPSES to one sign.
(Flow-matching FMPE, being a forward/distributional objective, is mode-covering
like SMC; the contrast that matters for E9 is reverse-KL VI vs the rest.)
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
from flow_matching import Layout

DTYPE = gm.DTYPE

# even-in-theta2 module-2 backbone -------------------------------------------- #
BETA, GAMMA, CENTER = 1.4, 0.14, 0.30


def eta2_even_np(x, u1, t2):
    even = t2 * t2 - CENTER
    return 0.65 * u1 + np.cos(1.2 * np.pi * x) + BETA * even + GAMMA * u1 * even


def physical_forward_even(x, theta):
    u1 = base.eta1_np(x, theta[0]) + base.delta1_true_np(x)
    u2 = eta2_even_np(x, u1, theta[1]) + base.delta2_true_np(x, u1)
    y = base.eta3_np(x, u2, theta[2]) + base.delta3_true_np(x, u2)
    return u1, u2, y


def make_field_data_even(rng, n_field, theta_true, sigma_y, sigma_z):
    x = np.sort(rng.uniform(0.02, 0.98, n_field))
    u1, u2, y_clean = physical_forward_even(x, theta_true)
    y_obs = y_clean + rng.normal(0, sigma_y, n_field)
    idx_pool = np.arange(2, n_field - 2)
    z1_idx = np.sort(rng.choice(idx_pool, size=max(5, n_field // 5), replace=False))
    z2_idx = np.sort(rng.choice(idx_pool, size=max(5, n_field // 5), replace=False))
    z1_obs = u1[z1_idx] + rng.normal(0, sigma_z, len(z1_idx))
    z2_obs = u2[z2_idx] + rng.normal(0, sigma_z, len(z2_idx))
    return {"x": x, "u1": u1, "u2": u2, "y_clean": y_clean, "y_obs": y_obs,
            "z1_idx": z1_idx, "z2_idx": z2_idx, "z1_obs": z1_obs, "z2_obs": z2_obs}


def build_emulators_even(rng, n_sim):
    x1 = rng.uniform(0, 1, n_sim); t1 = rng.uniform(-1, 1, n_sim)
    gp1 = gm.GPModule.fit(np.column_stack([x1, t1]), base.eta1_np(x1, t1), [0.22, 0.45])
    x2 = rng.uniform(0, 1, n_sim); u1 = rng.uniform(-1.5, 2.0, n_sim); t2 = rng.uniform(-1, 1, n_sim)
    gp2 = gm.GPModule.fit(np.column_stack([x2, u1, t2]), eta2_even_np(x2, u1, t2), [0.25, 0.65, 0.50])
    x3 = rng.uniform(0, 1, n_sim); u2 = rng.uniform(-1.8, 2.4, n_sim); t3 = rng.uniform(-1, 1, n_sim)
    gp3 = gm.GPModule.fit(np.column_stack([x3, u2, t3]), base.eta3_np(x3, u2, t3), [0.30, 0.75, 0.55])
    return gp1, gp2, gp3


def build_cascade_even(rng, n_sim, x, sigma_y, sigma_z, sigma_xi=(0.03, 0.03, 0.03)):
    gp1, gp2, gp3 = build_emulators_even(rng, n_sim)
    h1 = gm.HSGP1D(M=8, L=1.5, ell=0.35, sigma=0.25)
    h2 = gm.HSGP2D(Mx=5, Mu=3, Lx=1.5, Lu=1.5, ellx=0.35, ellu=0.60, sigma=0.22)
    h3 = gm.HSGP2D(Mx=5, Mu=3, Lx=1.5, Lu=1.5, ellx=0.35, ellu=0.60, sigma=0.22)
    return gm.StochasticCascade(
        x=torch.tensor(x, dtype=DTYPE), gp1=gp1, gp2=gp2, gp3=gp3, h1=h1, h2=h2, h3=h3,
        sqrtS1=torch.tensor(h1.sqrt_S(), dtype=DTYPE),
        sqrtS2=torch.tensor(h2.sqrt_S(), dtype=DTYPE),
        sqrtS3=torch.tensor(h3.sqrt_S(), dtype=DTYPE),
        sigma_xi=sigma_xi, sigma_y=sigma_y, sigma_z=sigma_z)


# --------------------------------------------------------------------------- #
# reverse-KL VI flow (mode-seeking baseline)                                   #
# --------------------------------------------------------------------------- #
def tanh_logjac(rawtheta: torch.Tensor) -> torch.Tensor:
    th = torch.tanh(rawtheta)
    return torch.sum(torch.log(torch.clamp(1.0 - th * th, min=1e-7)), dim=-1)


def train_reverse_kl(model, layout, dim, steps, batch, lr, seed):
    torch.manual_seed(seed)
    flow = base.RealNVP(dim=dim, n_layers=12, hidden=128)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    for step in range(1, steps + 1):
        w, logq = flow.sample_and_logq(batch, torch.device("cpu"))
        W = layout.to_dict(w)                       # tanh applied inside (theta in (-1,1))
        # box-free unconstrained target: tanh already enforces support, so use the
        # tanh log-Jacobian (clamped) instead of log_joint's hard -1e30 box penalty,
        # which would explode when float32 tanh saturates to +/-1.
        logp = (model.log_trans1(W) + model.log_trans2(W) + model.log_coef_prior(W)
                + model.log_lik(W)).to(torch.float32) + tanh_logjac(w[:, layout.s_rawt])
        loss = torch.mean(logq - logp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 25.0)
        opt.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            hist.append({"step": step, "loss": float(loss.detach())})
    return flow, hist


# --------------------------------------------------------------------------- #
def mode_report(theta2: np.ndarray, w: np.ndarray | None = None) -> dict:
    if w is None:
        w = np.ones(len(theta2)) / len(theta2)
    pos = theta2 > 0
    mass_pos = float(w[pos].sum())
    mass_neg = float(w[~pos].sum())
    mean_pos = float((theta2[pos] * w[pos]).sum() / max(mass_pos, 1e-12)) if mass_pos > 0 else float("nan")
    mean_neg = float((theta2[~pos] * w[~pos]).sum() / max(mass_neg, 1e-12)) if mass_neg > 0 else float("nan")
    minority = min(mass_pos, mass_neg)
    return {"mass_pos": mass_pos, "mass_neg": mass_neg, "mean_pos": mean_pos,
            "mean_neg": mean_neg, "minority_mass": minority,
            "bimodal": bool(minority > 0.1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=70)
    ap.add_argument("--n-field", type=int, default=30)
    ap.add_argument("--particles", type=int, default=1024)
    ap.add_argument("--n-mh", type=int, default=6)
    ap.add_argument("--ess-target", type=float, default=0.55)
    ap.add_argument("--vi-steps", type=int, default=4000)
    ap.add_argument("--vi-restarts", type=int, default=4)
    ap.add_argument("--reuse-smc", type=str, default="",
                    help="path to a saved smc_theta.npz to reuse instead of re-running SMC")
    ap.add_argument("--outdir", type=str, default="results/e8_gate")
    args = ap.parse_args()

    torch.set_num_threads(6)
    rng = np.random.default_rng(args.seed)
    theta_true = np.array([0.35, -0.55, 0.45])
    sigma_y, sigma_z = 0.06, 0.05

    data = make_field_data_even(rng, args.n_field, theta_true, sigma_y, sigma_z)
    model = build_cascade_even(rng, args.n_sim, data["x"], sigma_y, sigma_z)
    model._n_sim = args.n_sim
    gm.attach_data(model, data)
    layout = Layout(model)

    # (A) tempered graph SMC -> should be BIMODAL ---------------------------- #
    if args.reuse_smc:
        z = np.load(args.reuse_smc)
        th2 = z["theta"][:, 1]; w = z["weights"]
        smc_time, n_temps = 0.0, -1
        smc = None
    else:
        t0 = time.time()
        smc = gt.run_graph_smc(model, args.particles, args.n_mh, args.ess_target, 0.5, args.seed + 1)
        smc_time = time.time() - t0
        th2 = smc["W"]["theta"][:, 1].numpy()
        w = smc["weights"].numpy()
        n_temps = smc["n_temps"]
    smc_modes = mode_report(th2, w)

    # (B) reverse-KL VI flow x restarts -> should COLLAPSE ------------------- #
    vi_runs = []
    for r in range(args.vi_restarts):
        flow, hist = train_reverse_kl(model, layout, layout.dim, args.vi_steps, 256, 8e-4, args.seed + 100 + r)
        with torch.no_grad():
            wv, _ = flow.sample_and_logq(4000, torch.device("cpu"))
            th2v = torch.tanh(wv[:, 1]).numpy()
        vi_runs.append({"restart": r, "final_loss": hist[-1]["loss"], **mode_report(th2v)})

    n_collapsed = sum(1 for v in vi_runs if v["minority_mass"] < 0.05)

    out = {
        "setup": {"seed": args.seed, "n_field": args.n_field, "latent_dim": layout.dim,
                  "beta": BETA, "gamma": GAMMA, "theta_true": theta_true.tolist()},
        "smc": {"time_s": smc_time, "n_temps": n_temps, "modes": smc_modes},
        "vi_restarts": vi_runs,
        "gate": {"smc_bimodal": smc_modes["bimodal"],
                 "vi_collapsed_fraction": n_collapsed / max(1, args.vi_restarts)},
    }
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)
    if smc is not None:
        np.savez(outdir / "smc_theta.npz", theta=smc["W"]["theta"].numpy(), weights=w, theta_true=theta_true)

    print(f"\n===== E8 gate: sign-symmetric multimodal testbed (theta2 even)  |  dim {layout.dim} =====")
    print(f"true theta2 = {theta_true[1]:+.2f}   (modes expected at +/-{abs(theta_true[1]):.2f})")
    print(f"\n(A) tempered graph SMC  [{n_temps} temps, {smc_time:.0f}s]")
    print(f"    theta2 mode masses  neg={smc_modes['mass_neg']:.2f} (mean {smc_modes['mean_neg']:+.2f})  "
          f"pos={smc_modes['mass_pos']:.2f} (mean {smc_modes['mean_pos']:+.2f})")
    print(f"    -> BIMODAL: {smc_modes['bimodal']}  (minority mass {smc_modes['minority_mass']:.2f})")
    print(f"\n(B) reverse-KL VI flow  [{args.vi_restarts} restarts]")
    for v in vi_runs:
        print(f"    restart {v['restart']}: neg={v['mass_neg']:.2f} pos={v['mass_pos']:.2f}  "
              f"minority={v['minority_mass']:.2f}  loss={v['final_loss']:.1f}")
    print(f"    -> collapsed (minority<0.05) in {n_collapsed}/{args.vi_restarts} restarts")
    verdict = smc_modes["bimodal"] and n_collapsed >= 1
    print(f"\nGATE {'PASSED' if verdict else 'NOT PASSED'}: "
          f"SMC covers both modes AND reverse-KL VI collapses -> E9 3-line comparison has signal.")


if __name__ == "__main__":
    main()
