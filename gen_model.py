#!/usr/bin/env python3
"""
Stochastic function graph model for modular digital-twin calibration.

This is the model layer of the redesign (see METHOD_REDESIGN.md, sections 2-3).
It replaces the deterministic ``linked emulator mean`` of the earlier scripts
with a genuine *stochastic* cascade:

  U_j | U_pa(j), theta_j, c_j
      ~ N( m_j(V_j) + Phi_j(V_j) (sqrtS_j . c_j),   K_j(V_j, V_j) + sigma_xi_j^2 I )

where
  * (m_j, K_j) are the FULL Gaussian-process emulator posterior mean and
    covariance of module j, trained on the biased simulator runs D_j^sim.
    K_j is the whole n x n cross-field-point covariance, not a per-point
    marginal variance -- so emulator uncertainty is propagated with its
    field-point correlations intact (redesign section 1 / 7).
  * Phi_j(V_j) (sqrtS_j . c_j) is a Hilbert-space GP discrepancy Delta_j with
    whitened coefficients c_j ~ N(0, I) (redesign section 1).
  * sigma_xi_j is a small module process-noise floor.
  * V_j = [x, U_pa(j), theta_j] is the (random, because U_pa is random) module
    input, so K_j itself is a random matrix that differs per particle.

The intermediate states U_1, U_2 are EXPLICIT latent vectors (n-dim each): they
feed the downstream kernels non-linearly and cannot be marginalized (dgpsi
"stochastic imputation"). The terminal state U_3 IS marginalized analytically
into the y likelihood (still keeping emulator-3 covariance), because nothing
downstream reads it.

Latent vector of the posterior:
    W = ( theta_1, theta_2, theta_3,  c_1, c_2, c_3,  U_1, U_2 )

Everything is vectorized over C particles and runs in float64 on CPU. No
simulator gradient, no local linearization, no global Gaussian moment matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

import cascade_koh_flow_experiment as base
from hsgp_discrepancy_experiment import (
    HSGP1D,
    HSGP2D,
    phi_1d,
    features_2d,
    X_MID,
    X_HALF,
    U1_MID,
    U1_HALF,
    U2_MID,
    U2_HALF,
)

DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# Full-covariance GP emulator (mean AND joint predictive covariance)           #
# --------------------------------------------------------------------------- #
@dataclass
class GPModule:
    """Zero-mean-in-standardized-space RBF GP with a full predictive covariance.

    Fit on standardized targets; ``predict_mean_cov`` returns the joint Gaussian
    over an arbitrary set of (per-particle) query points.
    """

    Xtrain: torch.Tensor      # (m, d)
    alpha: torch.Tensor       # (m,)   = Kinv @ y_std_targets
    Kinv: torch.Tensor        # (m, m)
    lengthscales: torch.Tensor  # (d,)
    y_mean: float
    y_std: float
    signal: float = 1.0

    @classmethod
    def fit(cls, X, y, lengthscales, signal=1.0, noise=1e-6):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y_mean = float(y.mean())
        y_std = float(y.std() + 1e-8)
        yz = (y - y_mean) / y_std
        ls = np.asarray(lengthscales, dtype=np.float64)
        K = base.rbf_kernel_np(X, X, ls, signal) + (noise + 1e-8) * np.eye(len(X))
        Kinv = np.linalg.inv(K)
        alpha = Kinv @ yz
        return cls(
            Xtrain=torch.tensor(X, dtype=DTYPE),
            alpha=torch.tensor(alpha, dtype=DTYPE),
            Kinv=torch.tensor(Kinv, dtype=DTYPE),
            lengthscales=torch.tensor(ls, dtype=DTYPE),
            y_mean=y_mean,
            y_std=y_std,
            signal=signal,
        )

    def _rbf(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: (..., n, d), B: (..., p, d)  ->  (..., n, p)
        As = A / self.lengthscales
        Bs = B / self.lengthscales
        a2 = (As * As).sum(-1, keepdim=True)                 # (..., n, 1)
        b2 = (Bs * Bs).sum(-1).unsqueeze(-2)                 # (..., 1, p)
        cross = As @ Bs.transpose(-1, -2)                    # (..., n, p)
        d2 = torch.clamp(a2 + b2 - 2.0 * cross, min=0.0)
        return self.signal**2 * torch.exp(-0.5 * d2)

    def predict_mean_cov(self, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """V: (C, n, d) -> mean (C, n), cov (C, n, n)."""
        C = V.shape[0]
        Xb = self.Xtrain.unsqueeze(0).expand(C, -1, -1)      # (C, m, d)
        kVX = self._rbf(V, Xb)                               # (C, n, m)
        kVV = self._rbf(V, V)                                # (C, n, n)
        mean = self.y_mean + self.y_std * (kVX @ self.alpha) # (C, n)
        kVX_Kinv = kVX @ self.Kinv                           # (C, n, m)
        cov = kVV - kVX_Kinv @ kVX.transpose(-1, -2)         # (C, n, n)
        cov = self.y_std**2 * cov
        return mean, cov


def _chol(cov: torch.Tensor, jitter: float = 1e-9) -> torch.Tensor:
    """Batched Cholesky with escalating jitter (cov: (..., n, n))."""
    n = cov.shape[-1]
    eye = torch.eye(n, dtype=cov.dtype, device=cov.device)
    for k in range(8):
        try:
            return torch.linalg.cholesky(cov + (jitter * 10.0**k) * eye)
        except RuntimeError:
            continue
    return torch.linalg.cholesky(cov + 1e-2 * eye)


def mvn_logpdf(x: torch.Tensor, mean: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """x, mean: (C, n); L: (C, n, n) lower Cholesky -> (C,)."""
    d = (x - mean).unsqueeze(-1)                             # (C, n, 1)
    sol = torch.linalg.solve_triangular(L, d, upper=False)  # (C, n, 1)
    quad = (sol.squeeze(-1) ** 2).sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    n = x.shape[-1]
    return -0.5 * (quad + logdet + n * math.log(2 * math.pi))


def mvn_sample(mean: torch.Tensor, L: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    eps = torch.randn(mean.shape, dtype=mean.dtype, generator=gen)
    return mean + torch.einsum("cij,cj->ci", L, eps)


# --------------------------------------------------------------------------- #
# The stochastic cascade                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class StochasticCascade:
    """Full generative graph model + factorized potentials for graph transport."""

    x: torch.Tensor                       # (n,)  field inputs
    gp1: GPModule
    gp2: GPModule
    gp3: GPModule
    h1: HSGP1D
    h2: HSGP2D
    h3: HSGP2D
    sqrtS1: torch.Tensor
    sqrtS2: torch.Tensor
    sqrtS3: torch.Tensor
    sigma_xi: tuple[float, float, float]
    sigma_y: float
    sigma_z: float
    # observations (filled by attach_data)
    y_obs: torch.Tensor | None = None
    z1_idx: torch.Tensor | None = None
    z2_idx: torch.Tensor | None = None
    z1_obs: torch.Tensor | None = None
    z2_obs: torch.Tensor | None = None

    n: int = field(init=False)

    def __post_init__(self):
        self.n = self.x.shape[0]
        self._x_std = (self.x - X_MID) / X_HALF

    # ----- discrepancy feature maps (return Phi.sqrtS, shape (C,n,M)) ------- #
    def _disc1(self, C: int) -> torch.Tensor:
        phi = phi_1d(self._x_std, self.h1.M, self.h1.L) * self.sqrtS1   # (n, M1)
        return phi.unsqueeze(0).expand(C, -1, -1)

    def _disc2(self, u1: torch.Tensor) -> torch.Tensor:
        C = u1.shape[0]
        x_std = self._x_std.unsqueeze(0).expand(C, -1)
        u1_std = (u1 - U1_MID) / U1_HALF
        return features_2d(x_std, u1_std, self.h2.Mx, self.h2.Mu, self.h2.Lx, self.h2.Lu) * self.sqrtS2

    def _disc3(self, u2: torch.Tensor) -> torch.Tensor:
        C = u2.shape[0]
        x_std = self._x_std.unsqueeze(0).expand(C, -1)
        u2_std = (u2 - U2_MID) / U2_HALF
        return features_2d(x_std, u2_std, self.h3.Mx, self.h3.Mu, self.h3.Lx, self.h3.Lu) * self.sqrtS3

    # ----- module transition moments -------------------------------------- #
    def _trans1_mean_cov(self, theta1: torch.Tensor, c1: torch.Tensor):
        C = theta1.shape[0]
        xb = self.x.unsqueeze(0).expand(C, -1)
        V1 = torch.stack([xb, theta1.unsqueeze(-1).expand(-1, self.n)], dim=-1)  # (C,n,2)
        m1, K1 = self.gp1.predict_mean_cov(V1)
        mean = m1 + torch.einsum("cnm,cm->cn", self._disc1(C), c1)
        cov = K1 + (self.sigma_xi[0] ** 2) * torch.eye(self.n, dtype=DTYPE)
        return mean, cov

    def _trans2_mean_cov(self, u1: torch.Tensor, theta2: torch.Tensor, c2: torch.Tensor):
        C = u1.shape[0]
        xb = self.x.unsqueeze(0).expand(C, -1)
        V2 = torch.stack([xb, u1, theta2.unsqueeze(-1).expand(-1, self.n)], dim=-1)
        m2, K2 = self.gp2.predict_mean_cov(V2)
        mean = m2 + torch.einsum("cnm,cm->cn", self._disc2(u1), c2)
        cov = K2 + (self.sigma_xi[1] ** 2) * torch.eye(self.n, dtype=DTYPE)
        return mean, cov

    def _y_mean_cov(self, u2: torch.Tensor, theta3: torch.Tensor, c3: torch.Tensor):
        """Terminal: U_3 marginalized -> y ~ N(mean3, K3 + (xi3^2 + sigma_y^2) I)."""
        C = u2.shape[0]
        xb = self.x.unsqueeze(0).expand(C, -1)
        V3 = torch.stack([xb, u2, theta3.unsqueeze(-1).expand(-1, self.n)], dim=-1)
        m3, K3 = self.gp3.predict_mean_cov(V3)
        mean = m3 + torch.einsum("cnm,cm->cn", self._disc3(u2), c3)
        cov = K3 + (self.sigma_xi[2] ** 2 + self.sigma_y ** 2) * torch.eye(self.n, dtype=DTYPE)
        return mean, cov

    # ----- ancestral prior sampling (this IS the SMC lambda=0 target) ------ #
    def sample_prior(self, C: int, gen: torch.Generator) -> dict:
        theta = (2.0 * torch.rand(C, 3, dtype=DTYPE, generator=gen) - 1.0)
        c1 = torch.randn(C, self.h1.M, dtype=DTYPE, generator=gen)
        c2 = torch.randn(C, self.h2.M, dtype=DTYPE, generator=gen)
        c3 = torch.randn(C, self.h3.M, dtype=DTYPE, generator=gen)
        m1, cov1 = self._trans1_mean_cov(theta[:, 0], c1)
        u1 = mvn_sample(m1, _chol(cov1), gen)
        m2, cov2 = self._trans2_mean_cov(u1, theta[:, 1], c2)
        u2 = mvn_sample(m2, _chol(cov2), gen)
        return {"theta": theta, "c1": c1, "c2": c2, "c3": c3, "u1": u1, "u2": u2}

    def sample_forward_y(self, W: dict, gen: torch.Generator) -> torch.Tensor:
        """Posterior-predictive terminal draw y ~ p(y | W) (for UQ diagnostics)."""
        mean, cov = self._y_mean_cov(W["u2"], W["theta"][:, 2], W["c3"])
        return mvn_sample(mean, _chol(cov), gen)

    def forward_sample(self, theta, c1, c2, c3, gen: torch.Generator):
        """Ancestral forward pass at this cascade's own x, given (theta, c).

        Draws fresh latent U_1, U_2 and a terminal U_3(=y) so it can be used for
        posterior-predictive UQ at held-out inputs. Returns (u1, u2, y)."""
        m1, cov1 = self._trans1_mean_cov(theta[:, 0], c1)
        u1 = mvn_sample(m1, _chol(cov1), gen)
        m2, cov2 = self._trans2_mean_cov(u1, theta[:, 1], c2)
        u2 = mvn_sample(m2, _chol(cov2), gen)
        my, covy = self._y_mean_cov(u2, theta[:, 2], c3)
        y = mvn_sample(my, _chol(covy), gen)
        return u1, u2, y

    # ----- factorized potentials (log-densities), used by graph transport -- #
    # transitions (always at full strength under tempering)
    def log_trans1(self, W: dict) -> torch.Tensor:
        m, cov = self._trans1_mean_cov(W["theta"][:, 0], W["c1"])
        return mvn_logpdf(W["u1"], m, _chol(cov))

    def log_trans2(self, W: dict) -> torch.Tensor:
        m, cov = self._trans2_mean_cov(W["u1"], W["theta"][:, 1], W["c2"])
        return mvn_logpdf(W["u2"], m, _chol(cov))

    def log_coef_prior(self, W: dict) -> torch.Tensor:
        return (
            -0.5 * (W["c1"] ** 2).sum(-1)
            - 0.5 * (W["c2"] ** 2).sum(-1)
            - 0.5 * (W["c3"] ** 2).sum(-1)
        )

    def theta_in_box(self, W: dict) -> torch.Tensor:
        return ((W["theta"] > -1.0) & (W["theta"] < 1.0)).all(-1)

    # likelihood potentials (tempered by lambda)
    def log_lik_z1(self, W: dict) -> torch.Tensor:
        if self.z1_obs is None:
            return torch.zeros(W["u1"].shape[0], dtype=DTYPE)
        r = (self.z1_obs.unsqueeze(0) - W["u1"][:, self.z1_idx]) / self.sigma_z
        k = self.z1_obs.numel()
        return -0.5 * (r * r).sum(-1) - k * math.log(self.sigma_z * math.sqrt(2 * math.pi))

    def log_lik_z2(self, W: dict) -> torch.Tensor:
        if self.z2_obs is None:
            return torch.zeros(W["u2"].shape[0], dtype=DTYPE)
        r = (self.z2_obs.unsqueeze(0) - W["u2"][:, self.z2_idx]) / self.sigma_z
        k = self.z2_obs.numel()
        return -0.5 * (r * r).sum(-1) - k * math.log(self.sigma_z * math.sqrt(2 * math.pi))

    def log_lik_y(self, W: dict) -> torch.Tensor:
        mean, cov = self._y_mean_cov(W["u2"], W["theta"][:, 2], W["c3"])
        return mvn_logpdf(self.y_obs, mean, _chol(cov))

    # ----- assembled densities -------------------------------------------- #
    def log_prior(self, W: dict) -> torch.Tensor:
        lp = self.log_trans1(W) + self.log_trans2(W) + self.log_coef_prior(W)
        lp = torch.where(self.theta_in_box(W), lp, torch.full_like(lp, -1e30))
        return lp

    def log_lik(self, W: dict) -> torch.Tensor:
        return self.log_lik_z1(W) + self.log_lik_z2(W) + self.log_lik_y(W)

    def log_joint(self, W: dict, lam: float = 1.0) -> torch.Tensor:
        return self.log_prior(W) + lam * self.log_lik(W)


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #
def build_emulators(rng: np.random.Generator, n_sim: int) -> tuple[GPModule, GPModule, GPModule]:
    x1 = rng.uniform(0, 1, n_sim); t1 = rng.uniform(-1, 1, n_sim)
    gp1 = GPModule.fit(np.column_stack([x1, t1]), base.eta1_np(x1, t1), [0.22, 0.45])

    x2 = rng.uniform(0, 1, n_sim); u1 = rng.uniform(-1.5, 2.0, n_sim); t2 = rng.uniform(-1, 1, n_sim)
    gp2 = GPModule.fit(np.column_stack([x2, u1, t2]), base.eta2_np(x2, u1, t2), [0.25, 0.65, 0.50])

    x3 = rng.uniform(0, 1, n_sim); u2 = rng.uniform(-1.8, 2.4, n_sim); t3 = rng.uniform(-1, 1, n_sim)
    gp3 = GPModule.fit(np.column_stack([x3, u2, t3]), base.eta3_np(x3, u2, t3), [0.30, 0.75, 0.55])
    return gp1, gp2, gp3


def build_cascade(rng, n_sim, x, sigma_y, sigma_z, sigma_xi=(0.03, 0.03, 0.03)) -> StochasticCascade:
    gp1, gp2, gp3 = build_emulators(rng, n_sim)
    h1 = HSGP1D(M=8, L=1.5, ell=0.35, sigma=0.25)
    h2 = HSGP2D(Mx=5, Mu=3, Lx=1.5, Lu=1.5, ellx=0.35, ellu=0.60, sigma=0.22)
    h3 = HSGP2D(Mx=5, Mu=3, Lx=1.5, Lu=1.5, ellx=0.35, ellu=0.60, sigma=0.22)
    return StochasticCascade(
        x=torch.tensor(x, dtype=DTYPE),
        gp1=gp1, gp2=gp2, gp3=gp3,
        h1=h1, h2=h2, h3=h3,
        sqrtS1=torch.tensor(h1.sqrt_S(), dtype=DTYPE),
        sqrtS2=torch.tensor(h2.sqrt_S(), dtype=DTYPE),
        sqrtS3=torch.tensor(h3.sqrt_S(), dtype=DTYPE),
        sigma_xi=sigma_xi, sigma_y=sigma_y, sigma_z=sigma_z,
    )


def attach_data(model: StochasticCascade, data: dict) -> None:
    model.y_obs = torch.tensor(data["y_obs"], dtype=DTYPE)
    model.z1_idx = torch.tensor(data["z1_idx"], dtype=torch.long)
    model.z2_idx = torch.tensor(data["z2_idx"], dtype=torch.long)
    model.z1_obs = torch.tensor(data["z1_obs"], dtype=DTYPE)
    model.z2_obs = torch.tensor(data["z2_obs"], dtype=DTYPE)


if __name__ == "__main__":
    # smoke test: build, sample prior, evaluate densities
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    theta_true = np.array([0.35, -0.55, 0.45])
    data = base.make_field_data(rng, 30, theta_true, 0.06, 0.05)
    model = build_cascade(rng, 70, data["x"], 0.06, 0.05)
    attach_data(model, data)
    gen = torch.Generator().manual_seed(0)
    W = model.sample_prior(256, gen)
    print("prior sample shapes:", {k: tuple(v.shape) for k, v in W.items()})
    print("log_prior  mean/std:", float(model.log_prior(W).mean()), float(model.log_prior(W).std()))
    print("log_lik    mean/std:", float(model.log_lik(W).mean()), float(model.log_lik(W).std()))
    ypp = model.sample_forward_y(W, gen)
    print("post-pred y shape:", tuple(ypp.shape))
