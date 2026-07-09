#!/usr/bin/env python3
"""
Partially observed linked digital twin calibration experiment.

The experiment builds a three-module feed-forward digital twin. Field data are
generated from known ground-truth module functions, while the digital twin uses
pre-trained biased GP emulators for each module. Two calibration methods are
compared:

1. Black-box KOH: final-output-only Bayesian calibration with a final GP
   discrepancy integrated out, sampled by random-walk Metropolis.
2. Structured cascade KOH: module-level low-rank discrepancies plus sparse
   intermediate observations, approximated by a RealNVP normalizing flow and
   optionally corrected by self-normalized importance weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve


Array = np.ndarray


def eta1_np(x: Array, theta1: Array) -> Array:
    return np.sin(1.6 * np.pi * x) + 0.75 * theta1 + 0.15 * x * theta1


def eta2_np(x: Array, u1: Array, theta2: Array) -> Array:
    return 0.65 * u1 + np.cos(1.2 * np.pi * x + 0.35 * theta2) + 0.45 * theta2 + 0.08 * u1 * theta2


def eta3_np(x: Array, u2: Array, theta3: Array) -> Array:
    return 0.80 * u2 + np.sin(0.75 * u2 + theta3) + 0.25 * theta3 + 0.10 * x


def delta1_true_np(x: Array) -> Array:
    return 0.22 * np.cos(2.2 * np.pi * x) + 0.08 * (x - 0.5)


def delta2_true_np(x: Array, u1: Array) -> Array:
    return 0.16 * np.sin(1.7 * np.pi * x + 0.6 * u1) + 0.045 * (u1**2 - 0.7)


def delta3_true_np(x: Array, u2: Array) -> Array:
    return -0.18 * np.cos(1.4 * np.pi * x - 0.35 * u2) + 0.06 * np.sin(3.0 * np.pi * x)


def physical_forward_np(x: Array, theta: Array) -> tuple[Array, Array, Array]:
    u1 = eta1_np(x, theta[0]) + delta1_true_np(x)
    u2 = eta2_np(x, u1, theta[1]) + delta2_true_np(x, u1)
    y = eta3_np(x, u2, theta[2]) + delta3_true_np(x, u2)
    return u1, u2, y


def rbf_kernel_np(X1: Array, X2: Array, lengthscales: Array, signal: float = 1.0) -> Array:
    X1s = X1 / lengthscales
    X2s = X2 / lengthscales
    sq = np.sum(X1s**2, axis=1, keepdims=True) + np.sum(X2s**2, axis=1) - 2.0 * X1s @ X2s.T
    return signal**2 * np.exp(-0.5 * np.maximum(sq, 0.0))


@dataclass
class RBFEmulator:
    X: Array
    alpha: Array
    lengthscales: Array
    y_mean: float
    y_std: float
    signal: float = 1.0

    @classmethod
    def fit(cls, X: Array, y: Array, lengthscales: list[float], noise: float = 1e-5) -> "RBFEmulator":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y_mean = float(y.mean())
        y_std = float(y.std() + 1e-8)
        yz = (y - y_mean) / y_std
        ls = np.asarray(lengthscales, dtype=np.float64)
        K = rbf_kernel_np(X, X, ls) + (noise + 1e-8) * np.eye(len(X))
        cf = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve(cf, yz, check_finite=False)
        return cls(X=X, alpha=alpha, lengthscales=ls, y_mean=y_mean, y_std=y_std)

    def predict_np(self, Xnew: Array) -> Array:
        K = rbf_kernel_np(np.asarray(Xnew, dtype=np.float64), self.X, self.lengthscales, self.signal)
        return self.y_mean + self.y_std * (K @ self.alpha)

    def torch_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "X": torch.tensor(self.X, dtype=torch.float32, device=device),
            "alpha": torch.tensor(self.alpha, dtype=torch.float32, device=device),
            "lengthscales": torch.tensor(self.lengthscales, dtype=torch.float32, device=device),
            "y_mean": torch.tensor(self.y_mean, dtype=torch.float32, device=device),
            "y_std": torch.tensor(self.y_std, dtype=torch.float32, device=device),
        }


def gp_predict_torch(Xnew: torch.Tensor, gp: dict[str, torch.Tensor]) -> torch.Tensor:
    # Xnew shape: (..., d). Return shape: (...).
    diff = (Xnew.unsqueeze(-2) - gp["X"]) / gp["lengthscales"]
    K = torch.exp(-0.5 * torch.sum(diff * diff, dim=-1))
    return gp["y_mean"] + gp["y_std"] * torch.matmul(K, gp["alpha"])


def basis1_np(x: Array) -> Array:
    return np.stack(
        [np.ones_like(x), np.sin(2 * np.pi * x), np.cos(2 * np.pi * x), x - 0.5],
        axis=-1,
    )


def basis2_np(x: Array, u1: Array) -> Array:
    return np.stack(
        [np.ones_like(x), np.sin(np.pi * x), u1, u1**2 - 0.7],
        axis=-1,
    )


def basis3_np(x: Array, u2: Array) -> Array:
    return np.stack(
        [np.ones_like(x), np.cos(np.pi * x), np.sin(u2), x * u2],
        axis=-1,
    )


def basis1_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.ones_like(x), torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x), x - 0.5],
        dim=-1,
    )


def basis2_torch(x: torch.Tensor, u1: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.ones_like(x), torch.sin(math.pi * x), u1, u1 * u1 - 0.7],
        dim=-1,
    )


def basis3_torch(x: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.ones_like(x), torch.cos(math.pi * x), torch.sin(u2), x * u2],
        dim=-1,
    )


def make_simulation_design(rng: np.random.Generator, n_sim: int) -> tuple[RBFEmulator, RBFEmulator, RBFEmulator]:
    x1 = rng.uniform(0, 1, n_sim)
    t1 = rng.uniform(-1, 1, n_sim)
    X1 = np.column_stack([x1, t1])
    y1 = eta1_np(x1, t1)

    x2 = rng.uniform(0, 1, n_sim)
    u1 = rng.uniform(-1.5, 2.0, n_sim)
    t2 = rng.uniform(-1, 1, n_sim)
    X2 = np.column_stack([x2, u1, t2])
    y2 = eta2_np(x2, u1, t2)

    x3 = rng.uniform(0, 1, n_sim)
    u2 = rng.uniform(-1.8, 2.4, n_sim)
    t3 = rng.uniform(-1, 1, n_sim)
    X3 = np.column_stack([x3, u2, t3])
    y3 = eta3_np(x3, u2, t3)

    return (
        RBFEmulator.fit(X1, y1, [0.22, 0.45]),
        RBFEmulator.fit(X2, y2, [0.25, 0.65, 0.50]),
        RBFEmulator.fit(X3, y3, [0.30, 0.75, 0.55]),
    )


def linked_emulator_np(x: Array, theta: Array, gps: tuple[RBFEmulator, RBFEmulator, RBFEmulator]) -> tuple[Array, Array, Array]:
    t1 = np.full_like(x, theta[0], dtype=np.float64)
    u1 = gps[0].predict_np(np.column_stack([x, t1]))
    t2 = np.full_like(x, theta[1], dtype=np.float64)
    u2 = gps[1].predict_np(np.column_stack([x, u1, t2]))
    t3 = np.full_like(x, theta[2], dtype=np.float64)
    y = gps[2].predict_np(np.column_stack([x, u2, t3]))
    return u1, u2, y


def structured_forward_np(
    x: Array,
    theta: Array,
    beta: Array,
    gps: tuple[RBFEmulator, RBFEmulator, RBFEmulator],
    alpha_scale: float,
) -> tuple[Array, Array, Array]:
    a1, a2, a3 = alpha_scale * beta[:4], alpha_scale * beta[4:8], alpha_scale * beta[8:12]
    t1 = np.full_like(x, theta[0], dtype=np.float64)
    u1 = gps[0].predict_np(np.column_stack([x, t1])) + basis1_np(x) @ a1
    t2 = np.full_like(x, theta[1], dtype=np.float64)
    u2 = gps[1].predict_np(np.column_stack([x, u1, t2])) + np.sum(basis2_np(x, u1) * a2, axis=-1)
    t3 = np.full_like(x, theta[2], dtype=np.float64)
    y = gps[2].predict_np(np.column_stack([x, u2, t3])) + np.sum(basis3_np(x, u2) * a3, axis=-1)
    return u1, u2, y


def make_field_data(rng: np.random.Generator, n_field: int, theta_true: Array, sigma_y: float, sigma_z: float) -> dict[str, Array]:
    x = np.sort(rng.uniform(0.02, 0.98, n_field))
    u1, u2, y_clean = physical_forward_np(x, theta_true)
    y_obs = y_clean + rng.normal(0, sigma_y, n_field)
    idx_pool = np.arange(2, n_field - 2)
    z1_idx = np.sort(rng.choice(idx_pool, size=max(5, n_field // 5), replace=False))
    z2_idx = np.sort(rng.choice(idx_pool, size=max(5, n_field // 5), replace=False))
    z1_obs = u1[z1_idx] + rng.normal(0, sigma_z, len(z1_idx))
    z2_obs = u2[z2_idx] + rng.normal(0, sigma_z, len(z2_idx))
    return {
        "x": x,
        "u1": u1,
        "u2": u2,
        "y_clean": y_clean,
        "y_obs": y_obs,
        "z1_idx": z1_idx,
        "z2_idx": z2_idx,
        "z1_obs": z1_obs,
        "z2_obs": z2_obs,
    }


def koh_kernel(x1: Array, x2: Array, amp: float = 0.35, length: float = 0.32) -> Array:
    d = (x1[:, None] - x2[None, :]) / length
    return amp**2 * np.exp(-0.5 * d * d)


def run_black_box_koh(
    rng: np.random.Generator,
    data: dict[str, Array],
    gps: tuple[RBFEmulator, RBFEmulator, RBFEmulator],
    sigma_y: float,
    n_steps: int,
    burn: int,
) -> dict[str, Array | float]:
    x, y = data["x"], data["y_obs"]
    Ky = koh_kernel(x, x) + sigma_y**2 * np.eye(len(x)) + 1e-8 * np.eye(len(x))
    cf = cho_factor(Ky, lower=True, check_finite=False)

    def logpost(theta: Array) -> float:
        if np.any(theta < -1.0) or np.any(theta > 1.0):
            return -np.inf
        _, _, mu = linked_emulator_np(x, theta, gps)
        r = y - mu
        return float(-0.5 * r @ cho_solve(cf, r, check_finite=False))

    theta = np.array([0.0, 0.0, 0.0])
    lp = logpost(theta)
    prop_sd = np.array([0.055, 0.055, 0.055])
    samples = []
    accepted = 0
    for step in range(n_steps):
        proposal = theta + rng.normal(0, prop_sd)
        lp_prop = logpost(proposal)
        if math.log(rng.uniform()) < lp_prop - lp:
            theta, lp = proposal, lp_prop
            accepted += 1
        if step >= burn and (step - burn) % 4 == 0:
            samples.append(theta.copy())
    return {"samples": np.asarray(samples), "acceptance": accepted / n_steps, "chol": cf}


class CouplingLayer(torch.nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 96):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2 * dim),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xm = x * self.mask
        s, t = self.net(xm).chunk(2, dim=-1)
        s = 1.4 * torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        y = xm + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = torch.sum(s, dim=-1)
        return y, logdet


class RealNVP(torch.nn.Module):
    def __init__(self, dim: int, n_layers: int = 8, hidden: int = 96):
        super().__init__()
        layers = []
        for i in range(n_layers):
            mask_vals = [(j + i) % 2 for j in range(dim)]
            layers.append(CouplingLayer(dim, torch.tensor(mask_vals, dtype=torch.float32), hidden))
        self.layers = torch.nn.ModuleList(layers)
        self.dim = dim

    def sample_and_logq(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(n, self.dim, device=device)
        x = z
        logdet = torch.zeros(n, device=device)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        base_logp = -0.5 * torch.sum(z * z, dim=-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return x, base_logp - logdet


@dataclass
class TorchTarget:
    x: torch.Tensor
    y: torch.Tensor
    z1_idx: torch.Tensor
    z2_idx: torch.Tensor
    z1_obs: torch.Tensor
    z2_obs: torch.Tensor
    gps: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]
    sigma_y: float
    sigma_z: float
    alpha_scale: float

    def forward_model(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_theta = w[:, :3]
        theta = torch.tanh(raw_theta)
        beta = w[:, 3:]
        a1 = self.alpha_scale * beta[:, 0:4]
        a2 = self.alpha_scale * beta[:, 4:8]
        a3 = self.alpha_scale * beta[:, 8:12]
        b = w.shape[0]
        x = self.x.unsqueeze(0).expand(b, -1)

        t1 = theta[:, 0:1].expand(-1, x.shape[1])
        u1 = gp_predict_torch(torch.stack([x, t1], dim=-1), self.gps[0])
        u1 = u1 + torch.sum(basis1_torch(x) * a1[:, None, :], dim=-1)

        t2 = theta[:, 1:2].expand(-1, x.shape[1])
        u2 = gp_predict_torch(torch.stack([x, u1, t2], dim=-1), self.gps[1])
        u2 = u2 + torch.sum(basis2_torch(x, u1) * a2[:, None, :], dim=-1)

        t3 = theta[:, 2:3].expand(-1, x.shape[1])
        yhat = gp_predict_torch(torch.stack([x, u2, t3], dim=-1), self.gps[2])
        yhat = yhat + torch.sum(basis3_torch(x, u2) * a3[:, None, :], dim=-1)
        return theta, beta, u1, u2, yhat

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        theta, beta, u1, u2, yhat = self.forward_model(w)
        log_jac_theta = torch.sum(torch.log(torch.clamp(1.0 - theta * theta, min=1e-7)), dim=-1)
        log_prior_beta = -0.5 * torch.sum(beta * beta, dim=-1) - 0.5 * beta.shape[1] * math.log(2 * math.pi)
        ry = (self.y.unsqueeze(0) - yhat) / self.sigma_y
        log_like_y = -0.5 * torch.sum(ry * ry, dim=-1) - self.y.numel() * math.log(self.sigma_y * math.sqrt(2 * math.pi))

        z1_pred = u1[:, self.z1_idx]
        rz1 = (self.z1_obs.unsqueeze(0) - z1_pred) / self.sigma_z
        log_like_z1 = -0.5 * torch.sum(rz1 * rz1, dim=-1) - self.z1_obs.numel() * math.log(self.sigma_z * math.sqrt(2 * math.pi))

        z2_pred = u2[:, self.z2_idx]
        rz2 = (self.z2_obs.unsqueeze(0) - z2_pred) / self.sigma_z
        log_like_z2 = -0.5 * torch.sum(rz2 * rz2, dim=-1) - self.z2_obs.numel() * math.log(self.sigma_z * math.sqrt(2 * math.pi))
        return log_jac_theta + log_prior_beta + log_like_y + log_like_z1 + log_like_z2


def train_flow(
    seed: int,
    target: TorchTarget,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[RealNVP, list[dict[str, float]]]:
    torch.manual_seed(seed)
    flow = RealNVP(dim=15, n_layers=8, hidden=96).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-5)
    history = []
    for step in range(1, steps + 1):
        w, logq = flow.sample_and_logq(batch_size, device)
        logp = target.log_prob(w)
        loss = torch.mean(logq - logp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 25.0)
        opt.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            with torch.no_grad():
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "mean_logp": float(logp.mean().detach().cpu()),
                        "mean_logq": float(logq.mean().detach().cpu()),
                    }
                )
    return flow, history


def weighted_quantile(values: Array, weights: Array, quantiles: list[float]) -> Array:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(quantiles, cw, v)


def summarize_theta(samples: Array, theta_true: Array, weights: Array | None = None) -> dict[str, object]:
    if weights is None:
        weights = np.ones(len(samples)) / len(samples)
    mean = np.sum(samples * weights[:, None], axis=0)
    ci = np.stack([weighted_quantile(samples[:, j], weights, [0.025, 0.975]) for j in range(samples.shape[1])])
    coverage = [bool(ci[j, 0] <= theta_true[j] <= ci[j, 1]) for j in range(samples.shape[1])]
    return {
        "mean": mean.tolist(),
        "ci95": ci.tolist(),
        "abs_error": np.abs(mean - theta_true).tolist(),
        "rmse": float(np.sqrt(np.mean((mean - theta_true) ** 2))),
        "coverage95": coverage,
    }


def evaluate_predictions(
    method: str,
    x_test: Array,
    theta_samples: Array,
    gps: tuple[RBFEmulator, RBFEmulator, RBFEmulator],
    theta_true: Array,
    weights: Array | None = None,
    beta_samples: Array | None = None,
    alpha_scale: float = 0.18,
    bb_data: dict[str, Array] | None = None,
    bb_chol: tuple[Array, bool] | None = None,
) -> dict[str, float]:
    if weights is None:
        weights = np.ones(len(theta_samples)) / len(theta_samples)
    max_eval = min(2500, len(theta_samples))
    idx = np.linspace(0, len(theta_samples) - 1, max_eval).astype(int)
    weights_eval = weights[idx]
    weights_eval = weights_eval / weights_eval.sum()
    u1_true, u2_true, y_true = physical_forward_np(x_test, theta_true)
    pred_u1 = np.zeros_like(x_test)
    pred_u2 = np.zeros_like(x_test)
    pred_y = np.zeros_like(x_test)

    if method == "black_box":
        assert bb_data is not None and bb_chol is not None
        x_train = bb_data["x"]
        y_train = bb_data["y_obs"]
        Kxt = koh_kernel(x_test, x_train)
        for wi, s in zip(weights_eval, idx):
            theta = theta_samples[s]
            u1, u2, y = linked_emulator_np(x_test, theta, gps)
            _, _, y_train_mu = linked_emulator_np(x_train, theta, gps)
            delta_mean = Kxt @ cho_solve(bb_chol, y_train - y_train_mu, check_finite=False)
            pred_u1 += wi * u1
            pred_u2 += wi * u2
            pred_y += wi * (y + delta_mean)
    else:
        assert beta_samples is not None
        for wi, s in zip(weights_eval, idx):
            u1, u2, y = structured_forward_np(x_test, theta_samples[s], beta_samples[s], gps, alpha_scale)
            pred_u1 += wi * u1
            pred_u2 += wi * u2
            pred_y += wi * y

    return {
        "u1_rmse": float(np.sqrt(np.mean((pred_u1 - u1_true) ** 2))),
        "u2_rmse": float(np.sqrt(np.mean((pred_u2 - u2_true) ** 2))),
        "y_rmse": float(np.sqrt(np.mean((pred_y - y_true) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-sim", type=int, default=70)
    parser.add_argument("--n-field", type=int, default=40)
    parser.add_argument("--blackbox-steps", type=int, default=16000)
    parser.add_argument("--blackbox-burn", type=int, default=3000)
    parser.add_argument("--flow-steps", type=int, default=3000)
    parser.add_argument("--flow-batch", type=int, default=128)
    parser.add_argument("--outdir", type=str, default="results/cascade_koh_flow")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)
    theta_true = np.array([0.35, -0.55, 0.45])
    sigma_y = 0.06
    sigma_z = 0.05
    alpha_scale = 0.18

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gps = make_simulation_design(rng, args.n_sim)
    data = make_field_data(rng, args.n_field, theta_true, sigma_y, sigma_z)

    blackbox = run_black_box_koh(rng, data, gps, sigma_y, args.blackbox_steps, args.blackbox_burn)
    bb_samples = blackbox["samples"]

    device = torch.device("cpu")
    target = TorchTarget(
        x=torch.tensor(data["x"], dtype=torch.float32, device=device),
        y=torch.tensor(data["y_obs"], dtype=torch.float32, device=device),
        z1_idx=torch.tensor(data["z1_idx"], dtype=torch.long, device=device),
        z2_idx=torch.tensor(data["z2_idx"], dtype=torch.long, device=device),
        z1_obs=torch.tensor(data["z1_obs"], dtype=torch.float32, device=device),
        z2_obs=torch.tensor(data["z2_obs"], dtype=torch.float32, device=device),
        gps=tuple(gp.torch_tensors(device) for gp in gps),  # type: ignore[arg-type]
        sigma_y=sigma_y,
        sigma_z=sigma_z,
        alpha_scale=alpha_scale,
    )
    flow, history = train_flow(args.seed + 100, target, args.flow_steps, args.flow_batch, 2e-3, device)

    with torch.no_grad():
        w, logq = flow.sample_and_logq(12000, device)
        logp = target.log_prob(w)
        theta_t, beta_t, _, _, _ = target.forward_model(w)
    theta_flow = theta_t.cpu().numpy()
    beta_flow = beta_t.cpu().numpy()
    logw = (logp - logq).cpu().numpy()
    logw = logw - np.max(logw)
    weights = np.exp(logw)
    weights = weights / weights.sum()
    ess = float(1.0 / np.sum(weights * weights))

    x_test = np.linspace(0.01, 0.99, 120)
    results = {
        "setup": {
            "seed": args.seed,
            "n_sim_per_module": args.n_sim,
            "n_field": args.n_field,
            "n_z1": int(len(data["z1_idx"])),
            "n_z2": int(len(data["z2_idx"])),
            "theta_true": theta_true.tolist(),
            "sigma_y": sigma_y,
            "sigma_z": sigma_z,
            "alpha_scale": alpha_scale,
        },
        "black_box_koh": {
            "acceptance": float(blackbox["acceptance"]),
            "theta": summarize_theta(bb_samples, theta_true),
            "prediction": evaluate_predictions(
                "black_box",
                x_test,
                bb_samples,
                gps,
                theta_true,
                bb_data=data,
                bb_chol=blackbox["chol"],  # type: ignore[arg-type]
            ),
        },
        "structured_flow_raw": {
            "theta": summarize_theta(theta_flow, theta_true),
            "prediction": evaluate_predictions(
                "structured",
                x_test,
                theta_flow,
                gps,
                theta_true,
                beta_samples=beta_flow,
                alpha_scale=alpha_scale,
            ),
        },
        "structured_flow_importance_corrected": {
            "ess": ess,
            "ess_fraction": ess / len(weights),
            "theta": summarize_theta(theta_flow, theta_true, weights),
            "prediction": evaluate_predictions(
                "structured",
                x_test,
                theta_flow,
                gps,
                theta_true,
                weights=weights,
                beta_samples=beta_flow,
                alpha_scale=alpha_scale,
            ),
        },
        "flow_training_history": history,
    }

    np.savez(
        outdir / "posterior_samples.npz",
        theta_blackbox=bb_samples,
        theta_flow=theta_flow,
        beta_flow=beta_flow,
        flow_importance_weights=weights,
        x=data["x"],
        y_obs=data["y_obs"],
        z1_idx=data["z1_idx"],
        z1_obs=data["z1_obs"],
        z2_idx=data["z2_idx"],
        z2_obs=data["z2_obs"],
        theta_true=theta_true,
    )
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
