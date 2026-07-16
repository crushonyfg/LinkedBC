# Project Status & TODO (linkedC) — snapshot 2026-07-16

Two research threads, both "Bayesian calibration of modular digital twins with
per-module GP emulators + KOH discrepancy + partial observation, gradient-free":
**(1) feed-forward cascade** (done, strong results) and **(2) bidirectional /
feedback-coupled equilibrium networks** (prototype, in progress).

See `METHOD_REDESIGN.md` for the full method. This file = current state + TODOs.

---

## 1. What works (committed, reproducible)

### Cascade (feed-forward) — the mature thread
- `gen_model.py` — stochastic function-graph model: full-covariance GP module
  transitions (keep n×n K_j), HSGP GP discrepancy, explicit latent U_1,U_2 by
  ancestral sampling (U=m+Lε), terminal U_3 marginalized. Five UQ sources
  propagated distributionally (no moment matching).
- `graph_transport.py` — **Route B**: likelihood-tempered adaptive SMC, graph-local
  Metropolis-within-Gibbs (RW on params, **pCN on latents** — key to mixing the
  tight GP cascade), each move touches only its Markov blanket.
- `flow_matching.py` — **Route A**: amortized graph-structured FMPE (rectified-flow
  CFM, graph-masked per-block velocity + routed obs) + FMPE→SMC hybrid.
- `multimodal_testbed.py` — **E8 gate (PASSED)**: θ₂ sign-symmetric (even-function)
  bimodal testbed. Tempered SMC covers both ±modes (mean −0.50/+0.49, mass
  0.44/0.56); reverse-KL VI collapses (3/4 restarts). GOTCHA: VI target must be
  box-free (float32 tanh saturation → −1e30 penalty → gradient blowup).
- `seq_proposal.py` — **E9 shared substrate + Line 1**: autoregressive per-block
  proposal q_φ(W|O)=∏_j q_j(W_j|W_π(<j),O); exact factorized log q VERIFIED
  (self-consistency 0.00). VI line collapses 3/3 (mode-seeking, expected).
- `e9_compare.py` — **E9 3-line** (VI / GFlowNet-VarGrad / tempered-SMC on ONE
  proposal). VI collapses, GFlowNet (off-policy, prior-mixed behavior) covers both
  modes but noisily (minority 0.19±0.17), SMC balanced (0.44). Unifying identity:
  per-module local reward = incremental SMC log-weight.
- `amortize_frontier.py` — **★ the "our > SMC > black-box" result ★**: train FMPE
  once (80k sims, 83s), calibrate K=12 held-out datasets. **FMPE θ RMSE
  0.143±0.057 at 0 forward-calls/0.79s ≈ strong SMC 0.164 at 191,787 calls/7.6s,
  both ≫ black-box 0.603.** Amortization crossover K*≈0.4 (training < one SMC run).
  HONEST FRAME: on single-dataset accuracy SMC is gold and we only MATCH it — the
  win is COST / AMORTIZATION / EFFICIENCY / MODE-COVERAGE, not single-shot accuracy.

### Bidirectional (equilibrium) — prototype
- `equilibrium_model.py` — 2-module feedback fixed-point (damped Gauss-Seidel,
  DEQ-style implicit output), solver-defined generative model, tempered SMC over
  (θ, δ-coefs, branch B) with branch-flip moves, feedback-ignoring baseline.
  - **Exp1 UNIQUE (feedback matters): WORKS** — network KOH θ RMSE 0.128 vs
    feedback-ignoring 0.534 (θ₁ wrong, coverage False). Strong coupling (G·k=0.8)
    makes the dropped feedback term ~2.7× the θ signal → ignoring it mis-calibrates.
  - **Exp2 BISTABLE (bimodal branch): STILL FAILS** — solver finds 2 equilibria
    (gap 2.6), true |u2+|=|u2-| exactly (mirror), but branch-aware SMC still puts
    100% mass on one branch. See TODO.

### Results snapshot (θ RMSE, key numbers)
| setting | our method | SMC | black-box |
|---|---|---|---|
| cascade amortized (K=12) | **0.143** (0 fwd-calls) | 0.164 (191k calls) | 0.603 |
| equilibrium Exp1 unique | **0.128** (feedback-aware) | — | feedback-ignoring 0.534 |

---

## 2. Literature / positioning (verified by subagents, with URLs in git history / METHOD_REDESIGN)

Novelty CONFIRMED (multiple targeted searches) on the *assembled* contribution:
- **Feedback-coupled network KOH calibration** — closest: Abboud/fuel-bow 2026
  (arXiv:2601.18480, coupled GP surrogates iterated to equilibrium, but FORWARD-only,
  no calibration/discrepancy); Baldé 2023 (arXiv:2307.01111, modular chained KOH but
  feed-forward + CUT, not projection). Ming-Guillas explicitly "convert feedback to
  feed-forward by decoupling". DEQ (Bai 2019) = implicit output via IFT, not Bayesian.
- **Module-wise / graph PROJECTED calibration** (δ_j ⊥ ∂η_j/∂θ_j per node) — UNCLAIMED.
  Single-model only: Tuo-Wu 2015/2016 (L2 projection, Annals/JUQ), Plumlee 2017 (JASA,
  orthogonal GP discrepancy — uses gradient), Xie-Xu 2021 (JASA, Bayesian projected),
  Tuo 2019 (projected kernel). iPro-NC (2505.18176) leaves θ–δ non-id UNRESOLVED.
- **3-objective (VI/GFlowNet/SMC) on one graph proposal** for modular calibration — unclaimed.
- **Equilibrium branch as first-class Bayesian variable in calibration** — under-explored.
- Concurrent to watch: DAG-DGP (arXiv:2607.09645, Perlino et al., VERIFIED real) —
  DAG-of-GPs + structured VI; differentiate on inference (our transport/SMC/amortized)
  + identifiability/multimodality focus.
- Frontier hooks (verified): flow-matching SBI = Wildberger et al. 2023 NeurIPS (Dax
  co-author); GFlowNet TB / off-policy mode coverage (Malkin 2022/2023); VarGrad
  (Richter 2020); AFT/CRAFT/SNF (transport+SMC correctness); FIVO/NASMC (learned-proposal
  SMC); free-energy-barrier failure of local MCMC (2209.02001); Wehenkel-Louppe /
  Weilbach / Simformer (structured amortized inference).

---

## 3. TODO / open decisions (priority order)

### ★ PENDING USER DECISION — projected calibration route (derivative-free)
Implement projected-KOH and compare `plain-KOH (confounded ridge) vs projected-KOH
(identifiable θ)`. Options (user to pick; recommendation = B):
- **Route B (recommended):** keep joint graph model; make δ_j orthogonal to the
  EMPIRICAL parameter-response subspace V_j (sample θ_j, EVALUATE η_j, PCA → V_j;
  Φ̃_j=(I−V_jV_jᵀ)Φ_j). Fully derivative-free (Plumlee's orthogonality without the
  gradient — a small independent novelty). Sampler unchanged.
- **Route A:** strict two-stage L2 (Tuo-Wu form). Stage-1 nonparametric reconstruct
  ζ̂ from O (GP smoother, no simulator); Stage-2 per-module θ_j=argmin‖û_j−η_j(·,θ_j)‖²
  (evaluation-only). Bayesian/cut posterior.
- Both, as a 3-way comparison.
NOTE: estimand becomes the L2-projection θ* (best-fit under imperfect model), not
necessarily the "true physical" θ — state clearly in the paper.
(User paused here to clarify before choosing; machine may be shut down.)

### Identifiability theme (the user's central concern)
- Core figure: θ_j contraction / RMSE / coverage vs #intermediate sensors n_z
  (structured vs black-box). Reduction lever = observations, not sampler.
- ★ Amortized identifiability diagnosis + optimal sensor placement (use cheap
  q_φ(W|O) to map posterior contraction / EIG across the DESIGN space → which sensor
  most contracts θ; SMC can't — re-run per design). Strongest novel angle.
- SBC rank uniformity; coverage; θ–δ confounding-direction visualization.

### Bidirectional
- FIX Exp2 bimodal branch: the GP emulator of the odd system isn't exactly odd
  (breaks mirror symmetry) and the branch-flip keeps (θ,a) fixed (flip rejected).
  Fixes to try: (i) symmetrize the emulator or train on symmetrized data;
  (ii) joint (branch, a) proposal / re-fit a on flip; (iii) a truly branch-symmetric
  observable robust to emulator asymmetry. Then show branch-aware SMC recovers ~50/50
  vs single-branch stuck.
- Speed: solver-in-loop SMC is slow at 512 particles (>10 min, killed). Use ≤256
  particles / warm-start the solver across particles (parameter continuation).
- Full GP covariance UQ through the loop (currently diagonal var); HSGP discrepancy.

### Cascade / method polish
- E9: GFlowNet coverage noisy (0.19) + IS degenerate (k̂ nan) — more training / better
  off-policy exploration; strict IS/PSIS hybrid (now unblocked by exact autoregressive
  log q).
- Transport-SMC efficiency experiment (learned graph-transport between anneal steps →
  fewer temperatures / forward-calls than plain SMC at equal accuracy; AFT/CRAFT-style).
- Variable-design FMPE (set-encoder over sensors) for amortized optimal design.

### Writing
- Final report / paper draft: two threads (cascade + bidirectional), the
  our>SMC>black-box framing (cost/amortization/coverage, NOT single-shot accuracy),
  projected calibration as the identifiability resolution, positioning vs the
  verified related work.

---

## 4. Reproduce

```bash
python3 gen_model.py                  # generative-model smoke test
python3 graph_transport.py            # Route B graph SMC vs black-box
python3 flow_matching.py              # Route A amortized FMPE + hybrid
python3 multimodal_testbed.py         # E8 bimodal gate
python3 e9_compare.py                 # E9 3-line (VI/GFlowNet/SMC)
python3 amortize_frontier.py          # ★ our > SMC > black-box (amortization)
python3 equilibrium_model.py --particles 256 --n-mh 3 --ess-target 0.45   # bidirectional
```
Results (metrics.json) under `results/<experiment>/`. Legacy proof-of-concept:
`cascade_koh_flow_experiment.py`, `flow_vs_mcmc_experiment.py`,
`hsgp_discrepancy_experiment.py` (baselines/ablations).
