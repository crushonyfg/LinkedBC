#!/usr/bin/env python3
"""
Point-3 probe: is the RealNVP reverse-KL flow a good/elegant solver for the
15-D structured cascade-KOH posterior, and how does it compare with exact
gradient-based MCMC?

We reuse the exact target from ``cascade_koh_flow_experiment`` and compare three
posteriors over w = (raw_theta[3], beta[12]):

1. flow + self-normalized importance sampling (current method), with PSIS k-hat.
2. HMC gold standard: batched leapfrog HMC with dual-averaging step size and a
   diagonal mass matrix, run directly on the exact log-density. No package needed.
3. NeuTra: the trained flow is used as a reparameterization preconditioner and
   HMC is run in the flow's latent z-space (asymptotically exact for the true
   target, but mixes in a near-isotropic geometry).

The point is not speed at 15-D (HMC wins there); it is to (a) check the flow
posterior is correct, and (b) demonstrate a strictly better solving idea
(NeuTra) that keeps the flow's learned geometry while being exact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import cascade_koh_flow_experiment as base


# --------------------------------------------------------------------------- #
# PSIS k-hat (Vehtari et al., ported compactly from the arviz implementation)  #
# --------------------------------------------------------------------------- #
def _gpdfit(ary: np.ndarray) -> tuple[float, float]:
    prior_bs = 3.0
    prior_k = 10.0
    n = len(ary)
    m_est = 30 + int(n**0.5)
    b_ary = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1.0 / ary[-1]
    k_ary = np.log1p(-b_ary[:, None] * ary[None, :]).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1.0)
    weights = 1.0 / np.exp(len_scale[:, None] - len_scale).sum(axis=1)
    real = weights >= 10 * np.finfo(float).eps
    weights = weights[real]
    b_ary = b_ary[real]
    weights /= weights.sum()
    b_post = float(np.sum(b_ary * weights))
    k_post = float(np.log1p(-b_post * ary).mean())
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return k_post, -k_post / b_post


def psis_khat(log_w: np.ndarray) -> float:
    log_w = np.asarray(log_w, dtype=float)
    n = len(log_w)
    lw = log_w - log_w.max()
    n_tail = int(min(0.2 * n, 3.0 * np.sqrt(n)))
    order = np.argsort(lw)
    cutoff = lw[order[-n_tail - 1]]
    tail = np.exp(lw[order[-n_tail:]]) - np.exp(cutoff)
    tail = np.sort(tail)
    if tail[0] <= 0 or n_tail < 5:
        return float("nan")
    k, _ = _gpdfit(tail)
    return float(k)


# --------------------------------------------------------------------------- #
# Batched HMC with dual-averaging step size + diagonal mass matrix             #
# --------------------------------------------------------------------------- #
def _potential_and_grad(logp_fn, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.detach().requires_grad_(True)
    lp = logp_fn(w)                       # (C,)
    grad = torch.autograd.grad(lp.sum(), w)[0]
    return (-lp).detach(), (-grad).detach()   # U, dU/dw   (chains independent)


def run_hmc(
    logp_fn,
    dim: int,
    n_chains: int,
    warmup: int,
    n_sample: int,
    n_leapfrog: int,
    seed: int,
    init: torch.Tensor | None = None,
    target_accept: float = 0.8,
) -> tuple[np.ndarray, dict]:
    g = torch.Generator().manual_seed(seed)
    if init is None:
        w = 0.1 * torch.randn(n_chains, dim, generator=g)
    else:
        w = init.clone()
    inv_mass = torch.ones(dim)            # metric ~ posterior covariance (diag)

    # dual averaging state
    eps = 0.1
    mu = float(np.log(10 * eps))
    log_eps_bar, H_bar = 0.0, 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75

    warm_samples = []
    samples = []
    n_accept_total = 0
    total_steps = warmup + n_sample
    U, gradU = _potential_and_grad(logp_fn, w)

    for step in range(1, total_steps + 1):
        p = torch.randn(n_chains, dim, generator=g) / torch.sqrt(inv_mass)
        current_K = 0.5 * torch.sum(p * p * inv_mass, dim=-1)
        w_new = w.clone()
        p_new = p.clone()
        gU = gradU.clone()
        # leapfrog
        p_new = p_new - 0.5 * eps * gU
        for _ in range(n_leapfrog):
            w_new = w_new + eps * inv_mass * p_new
            U_new, gU = _potential_and_grad(logp_fn, w_new)
            p_new = p_new - eps * gU
        p_new = p_new + 0.5 * eps * gU     # undo half over-step
        U_new, gU_final = _potential_and_grad(logp_fn, w_new)
        proposed_K = 0.5 * torch.sum(p_new * p_new * inv_mass, dim=-1)

        log_accept = (U - U_new) + (current_K - proposed_K)
        log_accept = torch.nan_to_num(log_accept, nan=-float("inf"))
        accept = torch.log(torch.rand(n_chains, generator=g)) < log_accept
        acc_prob = float(torch.clamp(torch.exp(log_accept), max=1.0).mean())

        w = torch.where(accept[:, None], w_new, w)
        U = torch.where(accept, U_new, U)
        gradU = torch.where(accept[:, None], gU_final, gradU)
        n_accept_total += int(accept.sum())

        if step <= warmup:
            eta = 1.0 / (step + t0)
            H_bar = (1 - eta) * H_bar + eta * (target_accept - acc_prob)
            log_eps = mu - (step**0.5) / gamma * H_bar
            x_eta = step ** (-kappa)
            log_eps_bar = x_eta * log_eps + (1 - x_eta) * log_eps_bar
            eps = float(np.exp(log_eps))
            warm_samples.append(w.clone())
            # estimate diagonal mass once, midway through warmup
            if step == warmup // 2 and len(warm_samples) > 10:
                ws = torch.stack(warm_samples[len(warm_samples) // 2:], dim=0)
                inv_mass = ws.reshape(-1, dim).var(dim=0) + 1e-6
        else:
            if step == warmup + 1:
                eps = float(np.exp(log_eps_bar))
            samples.append(w.clone())

    chain = torch.stack(samples, dim=0).numpy()   # (n_sample, n_chains, dim)
    diag = {
        "acceptance": n_accept_total / (total_steps * n_chains),
        "final_eps": eps,
        "n_leapfrog": n_leapfrog,
        "rhat_max": float(_max_rhat(chain)),
    }
    return chain.reshape(-1, dim), diag


def _max_rhat(chain: np.ndarray) -> float:
    # chain: (n_draws, n_chains, dim) -> split-Rhat max over dims
    n_draws, n_chains, dim = chain.shape
    if n_chains < 2 or n_draws < 4:
        return float("nan")
    half = n_draws // 2
    a = chain[:half]
    b = chain[half:2 * half]
    split = np.concatenate([a, b], axis=1)   # (half, 2*n_chains, dim)
    m = split.shape[1]
    means = split.mean(axis=0)               # (m, dim)
    vars = split.var(axis=0, ddof=1)         # (m, dim)
    B = half * means.var(axis=0, ddof=1)
    W = vars.mean(axis=0)
    var_hat = (half - 1) / half * W + B / half
    rhat = np.sqrt(var_hat / np.maximum(W, 1e-12))
    return float(np.max(rhat))


# --------------------------------------------------------------------------- #
# NeuTra: HMC in the flow's latent z-space                                     #
# --------------------------------------------------------------------------- #
def make_neutra_logp(flow: base.RealNVP, target: base.TorchTarget):
    def logp_z(z: torch.Tensor) -> torch.Tensor:
        x = z
        logdet = torch.zeros(z.shape[0])
        for layer in flow.layers:
            x, ld = layer(x)
            logdet = logdet + ld            # log|det dT/dz|
        return target.log_prob(x) + logdet  # pushforward density in z-space

    def z_to_w(z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in flow.layers:
            x, _ = layer(x)
        return x

    return logp_z, z_to_w


# --------------------------------------------------------------------------- #
def theta_summary(theta_samples: np.ndarray, theta_true: np.ndarray) -> dict:
    mean = theta_samples.mean(axis=0)
    std = theta_samples.std(axis=0)
    q = np.quantile(theta_samples, [0.025, 0.975], axis=0)
    cover = [bool(q[0, j] <= theta_true[j] <= q[1, j]) for j in range(3)]
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "rmse": float(np.sqrt(np.mean((mean - theta_true) ** 2))),
        "coverage95": cover,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=70)
    ap.add_argument("--n-field", type=int, default=40)
    ap.add_argument("--flow-steps", type=int, default=3000)
    ap.add_argument("--flow-batch", type=int, default=128)
    ap.add_argument("--hmc-warmup", type=int, default=600)
    ap.add_argument("--hmc-sample", type=int, default=1500)
    ap.add_argument("--hmc-chains", type=int, default=16)
    ap.add_argument("--hmc-leapfrog", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="results/flow_vs_mcmc")
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    theta_true = np.array([0.35, -0.55, 0.45])
    sigma_y, sigma_z, alpha_scale = 0.06, 0.05, 0.18

    gps = base.make_simulation_design(rng, args.n_sim)
    data = base.make_field_data(rng, args.n_field, theta_true, sigma_y, sigma_z)
    target = base.TorchTarget(
        x=torch.tensor(data["x"], dtype=torch.float32),
        y=torch.tensor(data["y_obs"], dtype=torch.float32),
        z1_idx=torch.tensor(data["z1_idx"], dtype=torch.long),
        z2_idx=torch.tensor(data["z2_idx"], dtype=torch.long),
        z1_obs=torch.tensor(data["z1_obs"], dtype=torch.float32),
        z2_obs=torch.tensor(data["z2_obs"], dtype=torch.float32),
        gps=tuple(gp.torch_tensors(device) for gp in gps),  # type: ignore[arg-type]
        sigma_y=sigma_y, sigma_z=sigma_z, alpha_scale=alpha_scale,
    )

    def theta_of(w: torch.Tensor) -> np.ndarray:
        return torch.tanh(w[:, :3]).numpy()

    out = {"setup": {"seed": args.seed, "theta_true": theta_true.tolist()}}

    # --- 1. flow + IS -------------------------------------------------------- #
    t0 = time.time()
    flow, _ = base.train_flow(args.seed + 100, target, args.flow_steps, args.flow_batch, 2e-3, device)
    flow_train_time = time.time() - t0
    with torch.no_grad():
        w_f, logq = flow.sample_and_logq(12000, device)
        logp = target.log_prob(w_f)
    theta_f = theta_of(w_f)
    logw = (logp - logq).numpy()
    logw -= logw.max()
    wts = np.exp(logw); wts /= wts.sum()
    ess = float(1.0 / np.sum(wts**2))
    khat = psis_khat(logp.numpy() - logq.numpy())
    theta_f_is_mean = (theta_f * wts[:, None]).sum(0)
    out["flow_is"] = {
        "train_time_s": flow_train_time,
        "ess": ess, "ess_frac": ess / len(wts), "psis_khat": khat,
        "theta_raw": theta_summary(theta_f, theta_true),
        "theta_is_mean": theta_f_is_mean.tolist(),
        "theta_is_rmse": float(np.sqrt(np.mean((theta_f_is_mean - theta_true) ** 2))),
    }

    # --- 2. HMC gold standard ----------------------------------------------- #
    t0 = time.time()
    hmc_w, hmc_diag = run_hmc(
        target.log_prob, 15, args.hmc_chains, args.hmc_warmup, args.hmc_sample,
        args.hmc_leapfrog, args.seed + 1,
    )
    hmc_time = time.time() - t0
    theta_hmc = np.tanh(hmc_w[:, :3])
    out["hmc"] = {"time_s": hmc_time, **hmc_diag, "theta": theta_summary(theta_hmc, theta_true)}

    # --- 3. NeuTra: HMC in flow latent space -------------------------------- #
    logp_z, z_to_w = make_neutra_logp(flow, target)
    t0 = time.time()
    neutra_z, neutra_diag = run_hmc(
        logp_z, 15, args.hmc_chains, args.hmc_warmup, args.hmc_sample,
        args.hmc_leapfrog, args.seed + 2,
    )
    with torch.no_grad():
        neutra_w = z_to_w(torch.tensor(neutra_z, dtype=torch.float32)).numpy()
    neutra_time = time.time() - t0
    theta_neutra = np.tanh(neutra_w[:, :3])
    out["neutra"] = {"time_s": neutra_time, **neutra_diag, "theta": theta_summary(theta_neutra, theta_true)}

    # --- cross-method agreement (HMC as reference) -------------------------- #
    ref_mean = theta_hmc.mean(0)
    ref_std = theta_hmc.std(0)
    def agree(name, mean, std):
        return {
            "theta_mean_absdiff_vs_hmc": np.abs(np.array(mean) - ref_mean).tolist(),
            "theta_std_ratio_vs_hmc": (np.array(std) / ref_std).tolist(),
        }
    out["agreement_vs_hmc"] = {
        "flow_is": agree("flow_is", theta_f_is_mean, np.sqrt(np.average((theta_f - theta_f_is_mean)**2, axis=0, weights=wts))),
        "neutra": agree("neutra", theta_neutra.mean(0), theta_neutra.std(0)),
    }

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with (Path(args.outdir) / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)
    np.savez(
        Path(args.outdir) / "samples.npz",
        theta_flow=theta_f, flow_weights=wts,
        theta_hmc=theta_hmc, theta_neutra=theta_neutra, theta_true=theta_true,
    )

    # --- console report ----------------------------------------------------- #
    print("\n===== structured 15-D posterior: solver comparison =====")
    print(f"true theta = {theta_true}")
    print(f"\n[flow+IS]  ESS={ess:.0f}/12000 ({ess/len(wts)*100:.1f}%)  PSIS k-hat={khat:.3f}"
          f"  train={flow_train_time:.1f}s")
    print(f"  theta IS-mean = {np.round(theta_f_is_mean,3)}  RMSE={out['flow_is']['theta_is_rmse']:.3f}")
    print(f"\n[HMC gold] acc={hmc_diag['acceptance']:.2f}  Rhat_max={hmc_diag['rhat_max']:.3f}"
          f"  time={hmc_time:.1f}s")
    print(f"  theta mean = {np.round(out['hmc']['theta']['mean'],3)}  RMSE={out['hmc']['theta']['rmse']:.3f}"
          f"  cover95={out['hmc']['theta']['coverage95']}")
    print(f"\n[NeuTra]   acc={neutra_diag['acceptance']:.2f}  Rhat_max={neutra_diag['rhat_max']:.3f}"
          f"  time={neutra_time:.1f}s")
    print(f"  theta mean = {np.round(out['neutra']['theta']['mean'],3)}  RMSE={out['neutra']['theta']['rmse']:.3f}"
          f"  cover95={out['neutra']['theta']['coverage95']}")
    print(f"\nflow-IS theta mean |diff| vs HMC = {np.round(out['agreement_vs_hmc']['flow_is']['theta_mean_absdiff_vs_hmc'],3)}")
    print(f"flow-IS std ratio vs HMC        = {np.round(out['agreement_vs_hmc']['flow_is']['theta_std_ratio_vs_hmc'],3)}")
    print(f"NeuTra  std ratio vs HMC        = {np.round(out['agreement_vs_hmc']['neutra']['theta_std_ratio_vs_hmc'],3)}")


if __name__ == "__main__":
    main()
