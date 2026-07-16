#!/usr/bin/env python3
"""
Bidirectional Bayesian calibration of a FEEDBACK-COUPLED two-module emulator
network (prototype for `bidirectional_bayesian_calibration.md`).

Unlike the feed-forward cascade (gen_model.py), the two modules are coupled both
ways and the state is defined IMPLICITLY as a fixed point (equilibrium):

    u1 = f1(x, u2, theta1) + delta1(x, u2)
    u2 = f2(x, u1, theta2) + delta2(x, u1)        =>   U* = F(U*)  (solve R(U*)=0)

We solve U* with a damped Gauss-Seidel co-simulation iteration (no simulator
gradient, DEQ-style implicit output). The generative model is SOLVER-DEFINED:
sample theta, discrepancy coefs, a branch B, run the coupled solver, observe
Y = u2* (+noise) and sparse Z1 on u1*. Emulator (GP) predictive uncertainty is
propagated into the likelihood.

Two regimes:
  * UNIQUE  (weak coupling, loop gain < 1): a single equilibrium. We show that a
    baseline which IGNORES the feedback edge (feed-forward KOH) mis-calibrates,
    while the feedback-aware network KOH recovers theta.
  * BISTABLE (strong coupling, loop gain > 1): two stable equilibria (+/- branch).
    With a branch-symmetric observable the posterior is genuinely BIMODAL over
    the branch B; a branch-aware tempered SMC (with branch-flip moves) covers
    both equilibria, a single-branch sampler misses one.

Scope notes (prototype): scalar module states per field condition; low-rank
(basis) discrepancy; emulator uncertainty entered as marginal (diagonal)
variance in the likelihood. Full-covariance GP UQ + HSGP discrepancy were
already demonstrated in the feed-forward gen_model.py and slot in directly.
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

import gen_model as gm

DTYPE = gm.DTYPE


# --------------------------------------------------------------------------- #
# Coupled backbone (ground truth) + regimes                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Regime:
    name: str
    G: float; k: float; s: float
    A1: float; A2: float; b1: float; b2: float
    init_mag: float
    bistable: bool
    even_obs: bool          # observe |u2*| (branch-symmetric) -> bimodal branch
    odd_sym: bool = False    # gain-modulated ODD coupling (mirror-symmetric branches)
    c: float = 0.0           # theta modulates the coupling gain


# strong but still-contracting feedback (G*k=0.8<1 -> unique) so the feedback
# edge genuinely carries information; small additive theta term, identifiable.
UNIQUE = Regime("unique", G=0.8, k=1.0, s=0.0, A1=0.8, A2=0.6, b1=0.6, b2=0.5,
                init_mag=0.0, bistable=False, even_obs=False)
# gain-modulated odd coupling: f_j=(G+c*theta_j)*tanh(k*u).  Pure odd in (u1,u2)
# => two EXACTLY mirror-symmetric equilibria; |u2*| is branch-invariant, theta
# enters via the branch magnitude.
BISTABLE = Regime("bistable", G=1.6, k=1.4, s=0.0, A1=0.0, A2=0.0, b1=0.0, b2=0.0,
                  init_mag=2.0, bistable=True, even_obs=True, odd_sym=True, c=0.4)


def f1_true(x, u2, t1, R: Regime):
    if R.odd_sym:
        return (R.G + R.c * t1) * np.tanh(R.k * u2)
    return R.A1 * np.sin(1.6 * np.pi * x) + R.b1 * t1 + R.G * np.tanh(R.k * u2 + R.s * t1)


def f2_true(x, u1, t2, R: Regime):
    if R.odd_sym:
        return (R.G + R.c * t2) * np.tanh(R.k * u1)
    return R.A2 * np.cos(1.2 * np.pi * x) + R.b2 * t2 + R.G * np.tanh(R.k * u1 + R.s * t2)


def delta1_true(x, u2, R: Regime):
    return 0.03 * u2 if R.odd_sym else 0.10 * np.cos(2.2 * np.pi * x) + 0.03 * u2


def delta2_true(x, u1, R: Regime):
    return 0.03 * u1 if R.odd_sym else 0.09 * np.sin(1.7 * np.pi * x) + 0.03 * u1


def solve_true(x, theta, R: Regime, init_sign=1.0, iters=200, damp=0.6):
    u1 = init_sign * R.init_mag * np.ones_like(x)
    u2 = init_sign * R.init_mag * np.ones_like(x)
    for _ in range(iters):
        u1 = (1 - damp) * u1 + damp * (f1_true(x, u2, theta[0], R) + delta1_true(x, u2, R))
        u2 = (1 - damp) * u2 + damp * (f2_true(x, u1, theta[1], R) + delta2_true(x, u1, R))
    return u1, u2


# --------------------------------------------------------------------------- #
# Emulators (biased simulator = f_true without discrepancy)                     #
# --------------------------------------------------------------------------- #
def build_emulators(rng, n_sim, R: Regime):
    x1 = rng.uniform(0, 1, n_sim); u2 = rng.uniform(-2.6, 2.6, n_sim); t1 = rng.uniform(-1, 1, n_sim)
    gp1 = gm.GPModule.fit(np.column_stack([x1, u2, t1]), f1_true(x1, u2, t1, R), [0.25, 0.7, 0.55])
    x2 = rng.uniform(0, 1, n_sim); u1 = rng.uniform(-2.6, 2.6, n_sim); t2 = rng.uniform(-1, 1, n_sim)
    gp2 = gm.GPModule.fit(np.column_stack([x2, u1, t2]), f2_true(x2, u1, t2, R), [0.25, 0.7, 0.55])
    return gp1, gp2


# --------------------------------------------------------------------------- #
# Discrepancy basis (low-rank), torch                                          #
# --------------------------------------------------------------------------- #
SCALE_D = 0.10


def basis1(x, u2):                       # x (n,), u2 (C,n) -> (C,n,4)
    xb = x.unsqueeze(0).expand_as(u2)
    return torch.stack([torch.ones_like(u2), torch.cos(2 * math.pi * xb),
                        torch.sin(1.4 * math.pi * xb), torch.tanh(u2)], -1)


def basis2(x, u1):
    xb = x.unsqueeze(0).expand_as(u1)
    return torch.stack([torch.ones_like(u1), torch.sin(2 * math.pi * xb),
                        torch.cos(1.4 * math.pi * xb), torch.tanh(u1)], -1)


# --------------------------------------------------------------------------- #
# Coupled equilibrium solver (torch, vectorized over C particles x n fields)    #
# --------------------------------------------------------------------------- #
class Equil:
    def __init__(self, gp1, gp2, x, R: Regime, feedback=True, iters=35, damp=0.6):
        self.gp1, self.gp2 = gp1, gp2
        self.x = x; self.R = R; self.feedback = feedback
        self.iters, self.damp = iters, damp
        self.n = x.shape[0]

    def _d1(self, u2, a1):
        return SCALE_D * torch.einsum("cnk,ck->cn", basis1(self.x, u2), a1)

    def _d2(self, u1, a2):
        return SCALE_D * torch.einsum("cnk,ck->cn", basis2(self.x, u1), a2)

    def solve(self, theta, a1, a2, branch):
        C = theta.shape[0]
        xb = self.x.unsqueeze(0).expand(C, -1)
        sign = torch.where(branch > 0, 1.0, -1.0).unsqueeze(-1).to(DTYPE)
        u1 = sign * self.R.init_mag * torch.ones(C, self.n, dtype=DTYPE)
        u2 = u1.clone()
        for _ in range(self.iters):
            u2_in = u2 if self.feedback else torch.zeros_like(u2)   # drop feedback edge if baseline
            V1 = torch.stack([xb, u2_in, theta[:, 0:1].expand(-1, self.n)], -1)
            m1 = self.gp1.predict_mean(V1) + self._d1(u2_in, a1)
            u1 = (1 - self.damp) * u1 + self.damp * m1
            V2 = torch.stack([xb, u1, theta[:, 1:2].expand(-1, self.n)], -1)
            m2 = self.gp2.predict_mean(V2) + self._d2(u1, a2)
            u2 = (1 - self.damp) * u2 + self.damp * m2
        # emulator marginal variance at the converged equilibrium inputs
        u2_in = u2 if self.feedback else torch.zeros_like(u2)
        _, var1 = self.gp1.predict_mean_var(torch.stack([xb, u2_in, theta[:, 0:1].expand(-1, self.n)], -1))
        _, var2 = self.gp2.predict_mean_var(torch.stack([xb, u1, theta[:, 1:2].expand(-1, self.n)], -1))
        return u1, u2, var1, var2


# --------------------------------------------------------------------------- #
# Target log-density (theta uniform, a ~ N(0,I), branch uniform)               #
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    eq: Equil
    y_obs: torch.Tensor
    z1_idx: torch.Tensor
    z1_obs: torch.Tensor
    sigma_y: float
    sigma_z: float
    sigma_xi: float
    even_obs: bool

    def loglik(self, W):
        u1, u2, var1, var2 = self.eq.solve(W["theta"], W["a1"], W["a2"], W["branch"])
        pred_y = torch.abs(u2) if self.even_obs else u2
        vy = self.sigma_y**2 + self.sigma_xi**2 + var2
        ry = (self.y_obs.unsqueeze(0) - pred_y)
        ll_y = torch.sum(-0.5 * ry * ry / vy - 0.5 * torch.log(2 * math.pi * vy), -1)
        vz = self.sigma_z**2 + self.sigma_xi**2 + var1[:, self.z1_idx]
        rz = (self.z1_obs.unsqueeze(0) - u1[:, self.z1_idx])
        ll_z = torch.sum(-0.5 * rz * rz / vz - 0.5 * torch.log(2 * math.pi * vz), -1)
        return ll_y + ll_z

    def logprior(self, W):
        box = ((W["theta"] > -1) & (W["theta"] < 1)).all(-1)
        lp = -0.5 * (W["a1"] ** 2).sum(-1) - 0.5 * (W["a2"] ** 2).sum(-1)
        return torch.where(box, lp, torch.full_like(lp, -1e30))


# --------------------------------------------------------------------------- #
# Adaptive tempered SMC with RW rejuvenation + branch-flip moves               #
# --------------------------------------------------------------------------- #
def ess(logw):
    w = torch.softmax(logw, 0); return float(1.0 / (w * w).sum())


def systematic(logw, gen):
    C = logw.shape[0]; w = torch.softmax(logw, 0)
    pos = (torch.arange(C, dtype=DTYPE) + torch.rand(1, dtype=DTYPE, generator=gen)) / C
    return torch.searchsorted(torch.cumsum(w, 0), pos).clamp(max=C - 1)


def run_smc(target: Target, n, use_branch, n_mh, ess_target, seed, iters=60):
    gen = torch.Generator().manual_seed(seed)
    theta = 2 * torch.rand(n, 2, dtype=DTYPE, generator=gen) - 1
    a1 = torch.randn(n, 4, dtype=DTYPE, generator=gen)
    a2 = torch.randn(n, 4, dtype=DTYPE, generator=gen)
    branch = (torch.rand(n, generator=gen) > 0.5).long() if use_branch else torch.ones(n).long()
    W = {"theta": theta, "a1": a1, "a2": a2, "branch": branch}
    logw = torch.zeros(n, dtype=DTYPE)
    lam = 0.0
    ll = target.loglik(W)
    sc = {"t": 0.06, "a": 0.15}
    n_temps = 0
    while lam < 1.0:
        lo, hi = 0.0, 1.0 - lam
        tgt = ess_target * n
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if ess(logw + mid * ll) < tgt: hi = mid
            else: lo = mid
        dl = min(max(hi, 1e-3), 1.0 - lam); lam = lam + dl
        logw = logw + dl * ll
        n_temps += 1
        if ess(logw) < 0.5 * n:
            idx = systematic(logw, gen); W = {k: v[idx].clone() for k, v in W.items()}
            logw = torch.zeros(n, dtype=DTYPE)
        # rejuvenation
        for _ in range(n_mh):
            cur = target.logprior(W) + lam * target.loglik(W)
            prop = {k: v.clone() for k, v in W.items()}
            prop["theta"] = W["theta"] + sc["t"] * torch.randn(n, 2, dtype=DTYPE, generator=gen)
            prop["a1"] = W["a1"] + sc["a"] * torch.randn(n, 4, dtype=DTYPE, generator=gen)
            prop["a2"] = W["a2"] + sc["a"] * torch.randn(n, 4, dtype=DTYPE, generator=gen)
            newp = target.logprior(prop) + lam * target.loglik(prop)
            acc = torch.log(torch.rand(n, dtype=DTYPE, generator=gen)) < (newp - cur)
            for key in ("theta", "a1", "a2"):
                W[key] = torch.where(acc.unsqueeze(-1), prop[key], W[key])
            a_acc = float(acc.double().mean())
            sc["t"] = float(np.clip(sc["t"] * math.exp(0.4 * (a_acc - 0.3)), 1e-3, 1.0))
            sc["a"] = float(np.clip(sc["a"] * math.exp(0.4 * (a_acc - 0.3)), 1e-3, 1.0))
            # branch-flip move (mode jump across equilibria)
            if use_branch:
                cur2 = target.logprior(W) + lam * target.loglik(W)
                pf = {k: v.clone() for k, v in W.items()}
                pf["branch"] = 1 - W["branch"]
                new2 = target.logprior(pf) + lam * target.loglik(pf)
                accb = torch.log(torch.rand(n, dtype=DTYPE, generator=gen)) < (new2 - cur2)
                W["branch"] = torch.where(accb, pf["branch"], W["branch"])
        ll = target.loglik(W)
    return W, torch.softmax(logw, 0), n_temps


# --------------------------------------------------------------------------- #
def wmean(x, w):
    return float((x * w).sum())


def theta_report(W, w, theta_true):
    th = W["theta"].numpy(); wn = w.numpy()
    mean = (th * wn[:, None]).sum(0)
    ci = np.zeros((2, 2)); cov = []
    for j in range(2):
        o = np.argsort(th[:, j]); cw = np.cumsum(wn[o]); cw /= cw[-1]
        ci[j] = np.interp([0.025, 0.975], cw, th[o, j])
        cov.append(bool(ci[j, 0] <= theta_true[j] <= ci[j, 1]))
    return {"mean": mean.tolist(), "rmse": float(np.sqrt(np.mean((mean - theta_true) ** 2))),
            "coverage95": cov}


def make_data(rng, R: Regime, n_field, theta_true, sigma_y, sigma_z, true_sign=1.0):
    x = np.sort(rng.uniform(0.02, 0.98, n_field))
    u1s, u2s = solve_true(x, theta_true, R, init_sign=true_sign)
    obs_y = np.abs(u2s) if R.even_obs else u2s
    y_obs = obs_y + rng.normal(0, sigma_y, n_field)
    z1_idx = np.sort(rng.choice(np.arange(2, n_field - 2), size=max(6, n_field // 4), replace=False))
    z1_obs = u1s[z1_idx] + rng.normal(0, sigma_z, len(z1_idx))
    return {"x": x, "u1": u1s, "u2": u2s, "y_obs": y_obs, "z1_idx": z1_idx, "z1_obs": z1_obs}


def build_target(gp1, gp2, data, R, feedback, sigma_y, sigma_z, sigma_xi, iters=35):
    eq = Equil(gp1, gp2, torch.tensor(data["x"], dtype=DTYPE), R, feedback=feedback, iters=iters)
    return Target(eq, torch.tensor(data["y_obs"], dtype=DTYPE),
                  torch.tensor(data["z1_idx"], dtype=torch.long),
                  torch.tensor(data["z1_obs"], dtype=DTYPE),
                  sigma_y, sigma_z, sigma_xi, R.even_obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=90)
    ap.add_argument("--n-field", type=int, default=24)
    ap.add_argument("--particles", type=int, default=512)
    ap.add_argument("--n-mh", type=int, default=4)
    ap.add_argument("--ess-target", type=float, default=0.55)
    ap.add_argument("--outdir", type=str, default="results/equilibrium")
    args = ap.parse_args()

    torch.set_num_threads(6)
    theta_true = np.array([0.35, -0.55])
    sy, sz, sxi = 0.06, 0.05, 0.03
    out = {"setup": {"seed": args.seed, "theta_true": theta_true.tolist()}}

    # ============ Experiment 1: UNIQUE equilibrium, feedback matters ============ #
    rng = np.random.default_rng(args.seed)
    R = UNIQUE
    data = make_data(rng, R, args.n_field, theta_true, sy, sz)
    gp1, gp2 = build_emulators(rng, args.n_sim, R)
    tgt_net = build_target(gp1, gp2, data, R, True, sy, sz, sxi)
    tgt_ff = build_target(gp1, gp2, data, R, False, sy, sz, sxi)   # feedback-ignoring baseline
    t0 = time.time()
    Wn, wn, tn = run_smc(tgt_net, args.particles, False, args.n_mh, args.ess_target, args.seed + 1)
    Wf, wf, tf = run_smc(tgt_ff, args.particles, False, args.n_mh, args.ess_target, args.seed + 2)
    out["exp1_unique"] = {
        "network_KOH": theta_report(Wn, wn, theta_true),
        "feedback_ignoring": theta_report(Wf, wf, theta_true),
        "time_s": time.time() - t0,
    }

    # ============ Experiment 2: BISTABLE, branch-symmetric obs -> bimodal ======= #
    rng = np.random.default_rng(args.seed + 100)
    R = BISTABLE
    data = make_data(rng, R, args.n_field, theta_true, sy, sz, true_sign=1.0)
    gp1, gp2 = build_emulators(rng, args.n_sim, R)
    # verify solver finds two distinct equilibria from +/- init
    xb = torch.tensor(data["x"], dtype=DTYPE)
    eqv = Equil(gp1, gp2, xb, R, feedback=True)
    thn = torch.tensor(theta_true, dtype=DTYPE).unsqueeze(0)
    up = eqv.solve(thn, torch.zeros(1, 4, dtype=DTYPE), torch.zeros(1, 4, dtype=DTYPE), torch.ones(1).long())
    um = eqv.solve(thn, torch.zeros(1, 4, dtype=DTYPE), torch.zeros(1, 4, dtype=DTYPE), torch.zeros(1).long())
    branch_gap = float((up[1] - um[1]).abs().mean())
    tgt = build_target(gp1, gp2, data, R, True, sy, sz, sxi)
    t0 = time.time()
    Wb, wb, tb = run_smc(tgt, args.particles, True, args.n_mh, args.ess_target, args.seed + 3)
    Ws, ws, ts = run_smc(tgt, args.particles, False, args.n_mh, args.ess_target, args.seed + 4)  # single-branch
    bmass_b = wmean((Wb["branch"] == 1).to(DTYPE), wb)
    bmass_s = wmean((Ws["branch"] == 1).to(DTYPE), ws)
    out["exp2_bistable"] = {
        "solver_branch_gap": branch_gap,
        "branch_aware": {"branch_pos_mass": bmass_b, "minority_mass": float(min(bmass_b, 1 - bmass_b)),
                         "theta": theta_report(Wb, wb, theta_true), "n_temps": tb},
        "single_branch": {"branch_pos_mass": bmass_s, "minority_mass": float(min(bmass_s, 1 - bmass_s))},
        "time_s": time.time() - t0,
    }

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with (Path(args.outdir) / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)

    e1 = out["exp1_unique"]
    print("\n===== Bidirectional equilibrium calibration =====")
    print(f"true theta = {theta_true}")
    print(f"\n[Exp1 UNIQUE equilibrium]  ({e1['time_s']:.0f}s)")
    print(f"  network KOH (feedback-aware): theta={np.round(e1['network_KOH']['mean'],3)}  "
          f"RMSE={e1['network_KOH']['rmse']:.3f}  cover={e1['network_KOH']['coverage95']}")
    print(f"  feedback-IGNORING baseline:   theta={np.round(e1['feedback_ignoring']['mean'],3)}  "
          f"RMSE={e1['feedback_ignoring']['rmse']:.3f}  cover={e1['feedback_ignoring']['coverage95']}")
    print(f"  -> feedback matters: {'YES' if e1['network_KOH']['rmse'] < e1['feedback_ignoring']['rmse'] else 'check'}")
    e2 = out["exp2_bistable"]
    print(f"\n[Exp2 BISTABLE, branch-symmetric observable]  ({e2['time_s']:.0f}s)")
    print(f"  solver finds 2 equilibria: mean |u2+ - u2-| = {e2['solver_branch_gap']:.2f}")
    print(f"  branch-aware SMC: pos-branch mass={e2['branch_aware']['branch_pos_mass']:.2f}  "
          f"minority={e2['branch_aware']['minority_mass']:.2f}  (bimodal if ~0.5)")
    print(f"  single-branch SMC: pos-branch mass={e2['single_branch']['branch_pos_mass']:.2f}  "
          f"minority={e2['single_branch']['minority_mass']:.2f}  (misses a branch if ~0)")


if __name__ == "__main__":
    main()
