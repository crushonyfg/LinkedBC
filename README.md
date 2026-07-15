# Linked Cascade KOH Calibration Experiment

This repository contains a reproducible study of partially observed cascade
calibration of a linked digital twin.

## Current main line: Graph-Structured Transport Calibration

The method has been redesigned around a **stochastic function-graph model** with
**graph-structured, gradient-free transport inference**. See
[`METHOD_REDESIGN.md`](METHOD_REDESIGN.md) for the full specification. Core files:

- `gen_model.py` — model layer: each module is a *full-covariance* GP posterior
  transition plus a GP discrepancy; intermediate states `U_1, U_2` are explicit
  latents drawn by ancestral sampling (`U_j = m_j + L_j eps`, full `K_j`). Five
  uncertainty sources are propagated distributionally — no mean-only surrogate,
  no moment matching.
- `graph_transport.py` — inference layer (Route B): a likelihood-tempered SMC
  whose rejuvenation uses **module-block** moves (small-step RW on parameters,
  pCN on latents) that touch only each block's Markov blanket in the faithful
  inverse graph. No simulator gradient, no linearization.
- `flow_matching.py` — inference layer (Route A): **amortized graph-structured
  FMPE**. A conditional flow-matching CNF `q_phi(W|O)` trained purely on
  prior-predictive simulator draws (no gradient, no likelihood at train time);
  the velocity field is graph-masked (per-module heads over a faithful-inverse
  chain + routed observation embeddings). Includes the hybrid: FMPE particles
  seed a short graph-block MCMC correction at `lambda=1` (reusing Route B moves).

```bash
python3 gen_model.py                    # generative-model smoke test
python3 graph_transport.py              # graph transport SMC vs black-box KOH
# -> results/graph_transport/{metrics.json, posterior.npz}
# --ess-target trades speed vs mixing; --particles / --n-mh scale effort.
```

Reference behavior (seeds 11/23, n_field=30, 6 intermediate sensors each):
structured θ RMSE ≈ 0.08 with correct 95% coverage vs black-box 0.33–0.40; the
posterior-predictive `p(y|O)` is visibly non-Gaussian (mean |skew| ≈ 0.3–0.4),
which is the signature that full distributional propagation matters.

## Legacy proof-of-concept (baselines / ablations)

The script `cascade_koh_flow_experiment.py` builds a three-module feed-forward
system:

```text
x, theta_1 -> u_1
x, u_1, theta_2 -> u_2
x, u_2, theta_3 -> y
```

Field data are generated from known ground-truth module functions. The digital
twin is deliberately biased: each module emulator is an exact RBF-GP surrogate
trained only on simulator runs from the biased module simulator. The field data
include all final outputs and sparse noisy observations of `u_1` and `u_2`.

Two calibration methods are compared:

- `black_box_koh`: final-output-only KOH calibration. It composes the linked
  module GP means into a black-box final-output emulator and integrates a final
  GP discrepancy by Gaussian conditioning. The posterior over `theta` is sampled
  with random-walk Metropolis.
- `structured_flow`: cascade KOH calibration. It keeps module structure, adds
  low-rank module discrepancy bases, uses sparse intermediate observations, and
  approximates the posterior over `theta` and discrepancy coefficients with a
  RealNVP normalizing flow. Self-normalized importance weights are reported as a
  correction diagnostic.

Run:

```bash
python3 cascade_koh_flow_experiment.py
```

Outputs are written to `results/cascade_koh_flow/`:

- `metrics.json`: setup, posterior summaries, prediction RMSEs, flow training
  history.
- `posterior_samples.npz`: posterior samples and generated observations.

## Reference Run

With seed `7`, `n_field=40`, `n_sim=70`, and `flow_steps=3000`, the black-box
calibration fits final output well but misidentifies downstream parameters:

```text
black-box theta RMSE: 0.850
black-box y RMSE:     0.051
```

The structured flow posterior is much closer to the true module parameters:

```text
flow + importance theta RMSE: 0.076
flow + importance y RMSE:     0.097
flow importance ESS:          1264 / 12000
```

This is the expected behavior in the designed example: a flexible final
discrepancy can absorb module-level errors and preserve output prediction while
losing physical parameter identifiability. Sparse intermediate observations and
module-level discrepancy structure improve calibration of the linked system.
