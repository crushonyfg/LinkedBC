#!/usr/bin/env python3
"""
E9 main comparison: three training objectives on the SAME graph-structured
autoregressive proposal (seq_proposal.SeqProposal), evaluated on the E8
sign-symmetric bimodal testbed whose posterior is EXACTLY 50/50 in +/-theta2.

Lines (fairness rule: identical architecture, only the objective changes):
  1. VI          -- reverse-KL, pathwise         -> mode-seeking (collapses)
  2. GFlowNet    -- VarGrad / log-Z-free TB,
                    OFF-POLICY (temperature explore) -> mode-covering
  3. SMC         -- tempered graph-block SMC from the prior (Route B, reused
                    from the E8 gate run) -> correctness-guaranteed, covers both

Plus the honest punchline about importance/SMC correction:
  - IS-correcting a mode-SEEKING proposal (VI) CANNOT recover the missing mode
    (the proposal never proposes it) -- and PSIS k-hat can look deceptively fine
    because it only diagnoses the tail of the *sampled* weights, not a missing mode.
  - IS-correcting a mode-COVERING proposal (GFlowNet) works.

Ground truth for mode mass is the analytic symmetry (exactly 0.5 / 0.5), so
mode recall = min(mass_neg, mass_pos) / 0.5.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import gen_model as gm
import multimodal_testbed as mm
import seq_proposal as sp
from flow_matching import Layout
from flow_vs_mcmc_experiment import psis_khat

FDTYPE = torch.float32


# --------------------------------------------------------------------------- #
# Line 2 -- GFlowNet, VarGrad (log-Z-free trajectory balance), OFF-POLICY      #
# --------------------------------------------------------------------------- #
def train_gflownet_vargrad(model, layout, prop, O, steps, batch, lr, seed,
                           temps=(1.0, 2.0)):
    """VarGrad objective:  loss = Var_batch( target_logp(W) - log q_phi(W) ).
    At the optimum log q_phi = target_logp - const => q_phi propto posterior.
    W is drawn OFF-POLICY and DETACHED. The behavior distribution mixes (a)
    gradient-free PRIOR ancestral samples (theta2 uniform => BOTH modes always
    covered, independent of the current policy) with (b) temperature-scaled
    policy samples. Exploring via the prior is what stops the mode collapse that
    policy-only tempered exploration suffers once the policy sharpens."""
    torch.manual_seed(seed)
    gen64 = torch.Generator().manual_seed(seed + 777)
    opt = torch.optim.AdamW(prop.parameters(), lr=lr, weight_decay=1e-5)
    per = max(1, batch // (len(temps) + 1))
    hist = []
    for step in range(1, steps + 1):
        oc = prop.obs_context(O)                      # grad path (for log_prob)
        with torch.no_grad():
            oc_b = prop.obs_context(O)
            Wp = model.sample_prior(per, gen64)       # prior: both modes, no grad
            flat_prior = layout.to_flat(Wp).to(FDTYPE)
            flats = [flat_prior] + [prop.behavior_sample(per, oc_b, T) for T in temps]
            flat = torch.cat(flats, 0)                # off-policy, detached
        logqW = prop.log_prob(flat, oc)               # differentiable in phi
        logpW = sp.target_logp(model, layout, flat)   # no phi-grad
        delta = logpW - logqW
        loss = delta.var()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 25.0)
        opt.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            hist.append({"step": step, "vargrad": float(loss.detach())})
    return hist


# --------------------------------------------------------------------------- #
# Evaluation: proposal draws + importance correction                          #
# --------------------------------------------------------------------------- #
def eval_proposal(prop, model, layout, O, n=6000):
    with torch.no_grad():
        oc = prop.obs_context(O)
        flat, logq = prop.sample(n, oc)
        logp = sp.target_logp(model, layout, flat)
    th2 = torch.tanh(flat[:, 1]).numpy()
    raw = mm.mode_report(th2)                                    # unweighted (proposal)
    logw = (logp - logq).numpy()
    lw = logw - logw.max()
    w = np.exp(lw); w = w / w.sum()
    is_modes = mode_report_w(th2, w)                            # IS-corrected
    ess = float(1.0 / np.sum(w * w))
    khat = psis_khat(logw)
    return {"proposal": raw, "is_corrected": is_modes,
            "ess": ess, "ess_frac": ess / n, "psis_khat": khat}


def mode_report_w(th2, w):
    pos = th2 > 0
    mp, mn = float(w[pos].sum()), float(w[~pos].sum())
    return {"mass_pos": mp, "mass_neg": mn, "minority_mass": float(min(mp, mn)),
            "recall": float(min(mp, mn) / 0.5)}


def recall_of(modes):
    return float(min(modes["mass_pos"], modes["mass_neg"]) / 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=70)
    ap.add_argument("--n-field", type=int, default=30)
    ap.add_argument("--vi-steps", type=int, default=1800)
    ap.add_argument("--gf-steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--smc-npz", type=str, default="results/e8_gate/smc_theta.npz")
    ap.add_argument("--outdir", type=str, default="results/e9_compare")
    args = ap.parse_args()

    torch.set_num_threads(6)
    rng = np.random.default_rng(args.seed)
    theta_true = np.array([0.35, -0.55, 0.45])
    sigma_y, sigma_z = 0.06, 0.05

    data = mm.make_field_data_even(rng, args.n_field, theta_true, sigma_y, sigma_z)
    model = mm.build_cascade_even(rng, args.n_sim, data["x"], sigma_y, sigma_z)
    model._n_sim = args.n_sim
    gm.attach_data(model, data)
    layout = Layout(model)
    O = torch.tensor(np.concatenate([data["y_obs"], data["z1_obs"], data["z2_obs"]]), dtype=FDTYPE)
    nz1, nz2 = len(model.z1_idx), len(model.z2_idx)

    def fresh():
        return sp.SeqProposal(layout, model.n, nz1, nz2)

    # ---- Line 1: reverse-KL VI ---- #
    vi_runs = []
    for r in range(args.restarts):
        prop = fresh(); t0 = time.time()
        sp.train_vi(model, layout, prop, O, args.vi_steps, args.batch, 1e-3, args.seed + 10 + r)
        ev = eval_proposal(prop, model, layout, O)
        vi_runs.append({"restart": r, "train_s": time.time() - t0, **ev})

    # ---- Line 2: GFlowNet VarGrad (off-policy) ---- #
    gf_runs = []
    for r in range(args.restarts):
        prop = fresh(); t0 = time.time()
        train_gflownet_vargrad(model, layout, prop, O, args.gf_steps, args.batch, 1e-3, args.seed + 30 + r)
        ev = eval_proposal(prop, model, layout, O)
        gf_runs.append({"restart": r, "train_s": time.time() - t0, **ev})

    # ---- Line 3: tempered graph-block SMC (reuse E8 gate result) ---- #
    smc = None
    if Path(args.smc_npz).exists():
        z = np.load(args.smc_npz)
        smc = mode_report_w(z["theta"][:, 1], z["weights"])

    def avg(runs, path):
        vals = []
        for rr in runs:
            d = rr
            for k in path:
                d = d[k]
            vals.append(d)
        return float(np.mean(vals)), float(np.std(vals))

    out = {
        "setup": {"seed": args.seed, "latent_dim": layout.dim,
                  "testbed": "bimodal_even_theta2", "gt_mode_mass": [0.5, 0.5]},
        "line1_vi": {"runs": vi_runs,
                     "proposal_minority_mean": avg(vi_runs, ["proposal", "minority_mass"]),
                     "is_recall_mean": avg(vi_runs, ["is_corrected", "recall"]),
                     "khat_mean": avg(vi_runs, ["psis_khat"]),
                     "ess_frac_mean": avg(vi_runs, ["ess_frac"])},
        "line2_gflownet": {"runs": gf_runs,
                           "proposal_minority_mean": avg(gf_runs, ["proposal", "minority_mass"]),
                           "is_recall_mean": avg(gf_runs, ["is_corrected", "recall"]),
                           "khat_mean": avg(gf_runs, ["psis_khat"]),
                           "ess_frac_mean": avg(gf_runs, ["ess_frac"])},
        "line3_smc": {"modes": smc, "recall": recall_of(smc) if smc else None},
    }
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)

    def fmt(m):
        return f"{m[0]:.2f}±{m[1]:.2f}"
    print("\n===== E9 main table: 3 objectives on ONE graph proposal | bimodal (GT 0.5/0.5) =====")
    print(f"{'line':<26}{'proposal minority':<20}{'IS-corrected recall':<22}{'PSIS k-hat':<14}{'ESS frac'}")
    v, g = out["line1_vi"], out["line2_gflownet"]
    print(f"{'1. reverse-KL VI':<26}{fmt(v['proposal_minority_mean']):<20}{fmt(v['is_recall_mean']):<22}"
          f"{fmt(v['khat_mean']):<14}{fmt(v['ess_frac_mean'])}")
    print(f"{'2. GFlowNet (VarGrad)':<26}{fmt(g['proposal_minority_mean']):<20}{fmt(g['is_recall_mean']):<22}"
          f"{fmt(g['khat_mean']):<14}{fmt(g['ess_frac_mean'])}")
    if smc:
        print(f"{'3. tempered graph SMC':<26}{smc['minority_mass']:<20.2f}{'(is the reference)':<22}{'-':<14}-")
    print("\nreading: minority mass -> 0=collapsed, 0.5=balanced; recall = min-mass/0.5 (1=both modes).")
    print("punchline: VI collapses; IS on a collapsed proposal can't restore the missing mode")
    print("           (note k-hat may look OK yet the posterior is half-missing); GFlowNet & SMC cover both.")


if __name__ == "__main__":
    main()
