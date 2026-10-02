# Project State

Last updated: 2026-10-01

## Research question

Can next-token prediction on language alone recover both latent physical **states** z and the **operators** f: z → z' that govern them? Hold the world W fixed, vary the verbalization L_i = g_i(W), and measure which variables and dynamics become identifiable, represented, causally functional and compositionally reusable.

Working title: *From Language to Dynamics: Identifying Latent Physical States and Operators Learned from Linguistic Observations.*

## Files

| File | Purpose | Status |
|---|---|---|
| `project_idea.md` | Original idea and novelty analysis (LLM-generated citations, unverified) | reference |
| `Simulator_Verbalizer_Spec.md` | Spec v0: simulator, channels, information map, splits, episode record | current |
| `sim.py` | Phase-1 simulator, `num` + `qual` verbalizers and parsers, counterfactual twins, self-checks | working, all checks pass |
| `train.py` | Char-level GPT (6 layers, d=256, 4.9M params), one episode per sequence, writes `ckpt/<channel>.pt` | working |
| `probe.py` | Ridge and MLP probes on the residual stream at span-end tokens. Baselines: random-init model, and observables (ridge and MLP on the ground-truth stated values, most recent span first). Hidden params split by per-param identifiability | working |
| `PROJECT_STATE.md` | This file | |
| `HANDOFF.md` | How to resume in a new chat | |

## Decisions made

1. **Operator recovery is defined through interchange interventions.** Patch the internal representation of m (or F, or mu) from episode B into episode A. The model's continuation should match the simulator's counterfactual A-with-B's-value. Score this on held-out (m, F) regions so lookup is separated from computation. Use the causal abstraction / DAS framework, with the simulator as the high-level model.
2. **Drop direct estimation of I(W;L_i).** The verbalizers are designed by us, so each channel's information content is known by construction (the information map in the spec). Optional later: an InfoNCE dual encoder as a lower-bound "capacity meter", or a supervised text-to-z regressor as a decodability ceiling.
3. **A dual encoder is not the main model.** It would see z (which breaks the language-only condition), it has no dynamics, and it tests a different objective. Keep it only as a measuring instrument.
4. **Train small GPTs from scratch as the primary setting** (6–8 layers, 20–50M params). Pretrained LMs are a secondary comparison only, because of their physics priors.
5. **The headline probes target inferable-but-unstated variables** (the I cells in the information map). Probes on stated variables measure token copying and serve as baselines.
6. **Phase 1 is one cart with piecewise force, kinetic friction and stiction.** Collisions are phase 2.
7. **mu was narrowed from 0–0.5 to 0–0.3.** With 0–0.5, static friction (up to 24.5 N) swallowed many force segments: 33% of steps stuck, 71% of episodes identifiable. With 0–0.3: 19% stuck, 84% identifiable (2000 seeds). About 10% of episodes are deliberate single-segment controls, so the ceiling is roughly 90%.

## sim.py facts

- Recording: 50 steps at 0.1 s, integrated with 10 substeps of 0.01 s. Constant acceleration is integrated exactly; a zero-velocity crossing inside a substep is solved exactly and friction is re-evaluated there.
- Force segments start on 0.1 s boundaries, and F is rounded to 0.1 N.
- Emission: every 5 recorded steps, plus every step that has an event (`force_change`, `start`, `stop`).
- `identifiable(tr)` returns `{"m": bool, "mu": bool}`. In the kinetic regime a = F·(1/m) − g·s·mu, so both are identifiable iff two observed (F, s) rows are linearly independent. mu alone is also identifiable from any coast row (F = 0). Over 500 seeds: m is identifiable in 414 episodes, mu in 416.
- `make_episode(seed, **changes)` builds a counterfactual twin (for example `m=4.0`), with `cf_of` pointing to the base id.
- The surface template RNG is seeded with `f"{seed}-surface"`.
- Commands: `python sim.py` runs the self-checks. `python sim.py N out.jsonl` writes N episodes.

## Results

### Run 1: `num` channel, 2026-10-01

Setup: 100k train episodes (seeds 0..99999), 5000 steps at batch 64 (about 3.2 epochs), AdamW lr 1e-3 cosine, bf16, RTX 3070 Ti Laptop GPU, 873 s. Validation loss 3.29 → 0.294, still falling at the end. Probes: 3000 episodes (seeds 20M+), split 75/25 by episode, RidgeCV on standardized features, read at the last token of each span.

Best-layer R² (trained vs random-init):

| | x | v | a | F | log_m | mu |
|---|---|---|---|---|---|---|
| trained | 0.77 (L3) | 0.70 (L3) | 0.57 (L4) | 0.86 (L5) | 0.44 (L4) | 0.16 (L6) |
| random-init | 0.50 | 0.46 | 0.09 | 0.24 | 0.15 | 0.05 |

Late spans (step ≥ 25), trained model: log_m R² is 0.50 on identifiable episodes vs 0.34 on non-identifiable. mu R² is 0.11 on identifiable vs 0.32 on non-identifiable (reversed).

Reading:
- Positive signal: the unstated variables a and m decode far above the random-init baseline.
- The probe readout is weak even for stated variables (x and v reach only about 0.75), so the linear probe at the "." token is not near its ceiling.
- Random-init already reaches R² 0.15 for log_m. Observable correlates leak mass information (light carts reach larger |v| and |x|).
- The identifiability control is confounded. The flag is joint, but a coast-only episode identifies mu without m, which likely explains why mu is reversed. The non-identifiable subset is also small (about 80 episodes).

### Run 1, re-probed with the new `probe.py` (2026-10-02, partial)

The run was stopped before the observables baselines finished. Best layer per target:

| | x | v | a | F | log_m | mu |
|---|---|---|---|---|---|---|
| trained ridge | 0.77 | 0.70 | 0.57 | 0.86 | 0.44 | 0.16 |
| trained MLP | 0.92 | 0.87 | 0.80 | 0.97 | 0.51 | 0.14 |
| random-init ridge | 0.50 | 0.46 | 0.09 | 0.24 | 0.15 | 0.05 |

Reading: much of the weak linear readout was a limit of the linear probe. With the MLP, stated values reach about 0.9 and a reaches 0.80. mu stays weak under every probe. Still missing: the observables baselines, which are the key comparison.

### Run 2: `num` channel, 15k steps on RunPod (2026-10-02)

Setup: same as run 1 but 15k steps, on an RTX 4090 (283 s training, about 30× faster than the laptop). Validation loss 0.2309 (run 1: 0.294). Checkpoint `ckpt/num_run2.pt`, full log `train_probe_run2.log` (both gitignored). A first local attempt was aborted at step 7000.

Best layer per target:

| | x | v | a | F | log_m | mu |
|---|---|---|---|---|---|---|
| trained ridge | 0.79 | 0.68 | 0.65 | 0.87 | **0.61** | 0.22 |
| trained MLP | 0.92 | 0.90 | 0.85 | 0.96 | 0.62 | 0.20 |
| random-init ridge | 0.50 | 0.46 | 0.09 | 0.24 | 0.15 | 0.05 |
| observables ridge | 1.00 | 1.00 | 0.75 | 1.00 | **0.02** | 0.03 |
| observables MLP (ceiling) | 1.00 | 1.00 | 0.88 | 1.00 | 0.64 | 0.18 |

Late spans (step ≥ 25), split by each parameter's own flag:

| | log_m id | log_m non-id | mu id | mu non-id |
|---|---|---|---|---|
| trained ridge | 0.67 | 0.39 | 0.21 | 0.31 |
| observables MLP | 0.75 | 0.51 | 0.20 | 0.29 |

Reading:
- **Mass is the headline result.** A linear probe on the trained model gets log_m R² 0.61. A linear probe on the stated numbers gets 0.02. So the model has computed a nonlinear function of the text (mass from dynamics) and stores it linearly, close to the nonlinear ceiling of 0.64.
- **Acceleration is not evidence of computation.** a is nearly linear in the stated values (Δv/Δt with a mostly fixed Δt), so the observables ridge already reaches 0.75. The trained MLP (0.85) is near the ceiling (0.88).
- **mu is limited by the data, not the model.** Even the observables MLP reaches only 0.18. The trained model matches that ceiling.
- **The identifiability gaps are a data property.** The log_m id > non-id gap and the reversed mu gap both appear in the observables ceiling too, so they come from the data, not from the model. Non-identifiable episodes still carry partial information (stiction bounds, the parameter range). The non-id subset is small (about 70 episodes).
- **Decision gate passed for mass:** the trained model clearly beats the linear observables baseline. Move on to more channels.

## Known issues and open questions

- mu has a low decodability ceiling, even from stated values with an MLP (0.18). Possible causes: mu's small effect on the dynamics (mu·g ≤ 2.94 m/s²), the confound with m, or MLP baseline capacity and data size. Worth checking with a closed-form estimator from the trajectory as a true ceiling.
- Why non-identifiable episodes show higher mu R² than identifiable ones is unexplained (seen in the ceiling too).
- `qual` texts exceed the 512-token block in `train.py` (assertion in `encode`). Raise the block size or shorten `qual` (for example, collapse repeated "It stays still.") before training on it.

- `qual` repeats "It stays still." during long stuck stretches. This inflates token counts (the length confound). It could be collapsed into one sentence.
- `qual` reports "a gentle push starts" when the force changes within the same bin. This is intended information loss, but worth noting in the paper.
- The citations in `project_idea.md` must be verified, especially arXiv 2607.27017, Paperlayer 2607.20058, and the PhysLang characterization.
- Related work missing from `project_idea.md`: Vafa et al. 2025 ("What has a foundation model found?", inductive-bias probe on orbital mechanics; the closest prior work), Vafa et al. 2024 (world-model evaluation metrics), Li, Nye, Andreas 2021 (implicit entity state in text), and Othello-GPT.

## Not built yet

- Verbalizers: `nat`, `rel`, `sym`, and the ablations `nat-noterm`, `nat-notime`, `nat-nocause`, `nat-short`.
- Dataset splits: `iid`, `ood-combo`, `ood-extrap`, `compose`, `cross-channel`. The `ood-combo` and `ood-extrap` splits need param-region filtering in `sample_params` or at dataset-build time.
- BPE tokenizer (needed once `nat` exists; `num` uses char-level) and the interchange intervention code.
- Phase 2: collisions.
