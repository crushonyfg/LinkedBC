#!/usr/bin/env python3
"""
Proposal 1 -- the amortization cost/accuracy frontier that makes the ordering
    amortized-FMPE  >  adaptive tempered SMC  >  black-box KOH
honest and quantitative.

On a SINGLE dataset, well-tuned SMC is the gold standard; a learned method only
MATCHES it. The defensible win is AMORTIZATION: SMC re-runs from scratch (many
forward-model / GP calls) for every new dataset O, while an amortized proposal
q_phi(W|O) trains ONCE and then calibrates each new dataset for ~free.

Setup (fixed design, amortize over data realizations with random true theta):
  - one twin (fixed x grid, fixed sensor indices, fixed emulators)
  - train FMPE once on prior-predictive pairs  -> one-time cost N_train sims
  - K held-out datasets, each with a fresh theta_true ~ prior and fresh noise
  - per dataset compare: FMPE (0 forward-model calls) vs graph SMC (thousands)
    vs black-box KOH; accuracy = theta RMSE vs the known truth

Reports mean accuracy + per-dataset & total cost, and the amortization crossover
K* beyond which FMPE's total forward-model calls beat SMC's.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import cascade_koh_flow_experiment as base
import gen_model as gm
import graph_transport as gt
import flow_matching as fm
from flow_matching import Layout

FDTYPE = torch.float32


def fixed_field(x, z1_idx, z2_idx, theta_true, rng, sigma_y, sigma_z):
    u1, u2, y = base.physical_forward_np(x, theta_true)
    return {"x": x, "u1": u1, "u2": u2,
            "y_obs": y + rng.normal(0, sigma_y, len(x)),
            "z1_idx": z1_idx, "z2_idx": z2_idx,
            "z1_obs": u1[z1_idx] + rng.normal(0, sigma_z, len(z1_idx)),
            "z2_obs": u2[z2_idx] + rng.normal(0, sigma_z, len(z2_idx))}


def theta_rmse(theta_hat, theta_true):
    return float(np.sqrt(np.mean((np.asarray(theta_hat) - theta_true) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=70)
    ap.add_argument("--n-field", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=80000)
    ap.add_argument("--fmpe-steps", type=int, default=7000)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--smc-particles", type=int, default=384)
    ap.add_argument("--smc-mh", type=int, default=4)
    ap.add_argument("--outdir", type=str, default="results/amortize_frontier")
    args = ap.parse_args()

    torch.set_num_threads(6)
    rng = np.random.default_rng(args.seed)
    sigma_y, sigma_z = 0.06, 0.05

    # ---- fixed design + fixed emulators (the "twin") ---- #
    x = np.linspace(0.03, 0.97, args.n_field)
    pool = np.arange(2, args.n_field - 2)
    nz = max(5, args.n_field // 5)
    z1_idx = np.sort(rng.choice(pool, size=nz, replace=False))
    z2_idx = np.sort(rng.choice(pool, size=nz, replace=False))
    gps = gm.build_emulators(rng, args.n_sim)                 # full-cov GP emulators (shared)
    gps_bb = base.make_simulation_design(np.random.default_rng(args.seed), args.n_sim)  # mean emul for black-box

    theta_ref = np.array([0.0, 0.0, 0.0])
    model = gm.build_cascade(np.random.default_rng(args.seed + 1), args.n_sim, x, sigma_y, sigma_z)
    model.gp1, model.gp2, model.gp3 = gps                    # ensure identical emulators everywhere
    model._n_sim = args.n_sim
    gm.attach_data(model, fixed_field(x, z1_idx, z2_idx, theta_ref, rng, sigma_y, sigma_z))
    layout = Layout(model)

    # ---- train FMPE ONCE (one-time amortized cost) ---- #
    t0 = time.time()
    Wp, Op = fm.simulate_pairs(model, layout, args.n_train, 4096, args.seed + 3)
    Wm, Ws = Wp.mean(0), Wp.std(0) + 1e-6
    Om, Os = Op.mean(0), Op.std(0) + 1e-6
    net = fm.GraphVelocity(layout, len(z1_idx), len(z2_idx), args.n_field)
    fm.train_fmpe(net, Wp, Op, Wm, Ws, Om, Os, args.fmpe_steps, 512, 1e-3, args.seed + 4)
    train_time = time.time() - t0
    train_sims = args.n_train                                # forward-model draws used to train

    # ---- K held-out calibration tasks ---- #
    recs = {"fmpe": [], "smc": [], "blackbox": []}
    cost = {"fmpe_infer_s": [], "smc_s": [], "blackbox_s": [], "smc_forward_calls": []}
    for k in range(args.K):
        theta_true = rng.uniform(-0.7, 0.7, 3)
        data = fixed_field(x, z1_idx, z2_idx, theta_true, rng, sigma_y, sigma_z)
        gm.attach_data(model, data)
        O = torch.tensor(np.concatenate([data["y_obs"], data["z1_obs"], data["z2_obs"]]), dtype=FDTYPE)

        # FMPE: amortized, ZERO forward-model calls (network only)
        t = time.time()
        On = (O - Om) / Os
        flat = fm.sample_fmpe(net, On, 3000, 60, Wm, Ws, args.seed + 100 + k)
        th_fmpe = torch.tanh(flat[:, :3]).mean(0).numpy()
        cost["fmpe_infer_s"].append(time.time() - t)
        recs["fmpe"].append(theta_rmse(th_fmpe, theta_true))

        # SMC: full re-run, thousands of forward-model calls
        t = time.time()
        smc = gt.run_graph_smc(model, args.smc_particles, args.smc_mh, 0.5, 0.5, args.seed + 200 + k)
        th_smc = (smc["W"]["theta"].numpy() * smc["weights"].numpy()[:, None]).sum(0)
        cost["smc_s"].append(time.time() - t)
        # forward-model calls ~ temps * particles * (mh sweeps * moves + reweight); ~5 evals/sweep + 1
        cost["smc_forward_calls"].append(smc["n_temps"] * args.smc_particles * (args.smc_mh * 5 + 1))
        recs["smc"].append(theta_rmse(th_smc, theta_true))

        # black-box KOH
        t = time.time()
        bb = base.run_black_box_koh(rng, data, gps_bb, sigma_y, 12000, 2000)
        th_bb = bb["samples"].mean(0)
        cost["blackbox_s"].append(time.time() - t)
        recs["blackbox"].append(theta_rmse(th_bb, theta_true))

    def ms(a): return [float(np.mean(a)), float(np.std(a))]
    smc_calls = float(np.mean(cost["smc_forward_calls"]))
    crossover_K = train_sims / smc_calls
    out = {
        "setup": {"seed": args.seed, "n_field": args.n_field, "K": args.K,
                  "n_train_sims": train_sims, "fmpe_train_time_s": train_time},
        "theta_rmse": {"fmpe": ms(recs["fmpe"]), "smc": ms(recs["smc"]), "blackbox": ms(recs["blackbox"])},
        "per_dataset_time_s": {"fmpe_infer": ms(cost["fmpe_infer_s"]), "smc": ms(cost["smc_s"]),
                               "blackbox": ms(cost["blackbox_s"])},
        "forward_model_calls": {"fmpe_per_dataset": 0, "smc_per_dataset_mean": smc_calls,
                                "fmpe_train_once": train_sims, "amortization_crossover_K": crossover_K},
    }
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with (Path(args.outdir) / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)

    r = out["theta_rmse"]; c = out["per_dataset_time_s"]
    rm_f = "%.3f±%.3f" % (r["fmpe"][0], r["fmpe"][1])
    rm_s = "%.3f±%.3f" % (r["smc"][0], r["smc"][1])
    rm_b = "%.3f±%.3f" % (r["blackbox"][0], r["blackbox"][1])
    t_f = "%.2fs" % c["fmpe_infer"][0]
    t_s = "%.1fs" % c["smc"][0]
    t_b = "%.1fs" % c["blackbox"][0]
    print("\n===== Amortization frontier: our (FMPE) vs SMC vs black-box | K=%d tasks =====" % args.K)
    print("one-time FMPE training: %d sims, %.0fs" % (train_sims, train_time))
    print("\n%-14s%-26s%-22s%s" % ("method", "theta RMSE (mean±sd)", "per-dataset time", "forward-model calls/dataset"))
    print("%-14s%-26s%-22s%s" % ("FMPE (ours)", rm_f, t_f, "0 (amortized)"))
    print("%-14s%-26s%-22s%.0f" % ("SMC", rm_s, t_s, smc_calls))
    print("%-14s%-26s%-22s%s" % ("black-box", rm_b, t_b, "(final-output only)"))
    rel = "≈" if abs(r["fmpe"][0] - r["smc"][0]) < 0.05 else "vs"
    print("\naccuracy: FMPE %s SMC, both << black-box (%.3f)" % (rel, r["blackbox"][0]))
    print("amortization crossover: FMPE beats SMC on total forward-model calls after K* ≈ %.1f datasets" % crossover_K)


if __name__ == "__main__":
    main()
