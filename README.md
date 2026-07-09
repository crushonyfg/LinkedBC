# Linked Cascade KOH Calibration Experiment

This repository contains a small reproducible experiment for partially observed
cascade calibration of a linked digital twin.

## Experiment

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
