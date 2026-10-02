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
| `sim.py` | Phase-1 simulator, `num` + `qual` + `nat` + `rel` + `sym` verbalizers and parsers, counterfactual twins, self-checks | working, all checks pass |
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
8. **One char-level tokenizer for every channel, no BPE.** A shared tokenizer removes the tokenizer as a confound between channels. `train.py` sets the context length from the data (1.25 × the longest training text, rounded up to 64): about 576 tokens for `num`, about 1472 for `nat`.

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

### Run 3: `nat` channel, 15k steps on RunPod (2026-10-02)

Setup: same as run 2 (100k episodes, 15k steps, batch 64, RTX 4090) on `nat` text, context length 1472. Training took 783 s. Final validation loss 0.0869 per character, which is not comparable to `num`, because most `nat` characters are predictable template words. Checkpoint `ckpt/nat_run1.pt`, log `train_probe_nat_run1.log` (both gitignored). The observables baselines use `nat`'s precision (x and v to 0.1).

Best layer per target, with run 2 (`num`) in brackets:

| | x | v | a | F | log_m | mu |
|---|---|---|---|---|---|---|
| trained ridge | 0.37 (0.79) | 0.43 (0.68) | 0.15 (0.65) | 0.37 (0.87) | **0.39 (0.61)** | 0.13 (0.22) |
| trained MLP | 0.79 (0.92) | 0.84 (0.90) | 0.47 (0.85) | 0.60 (0.96) | 0.47 (0.62) | 0.09 (0.20) |
| random-init ridge | 0.00 | 0.01 | 0.01 | 0.03 | 0.00 | 0.00 |
| observables ridge | 1.00 | 1.00 | 0.75 | 1.00 | 0.02 | 0.03 |
| observables MLP (ceiling) | 1.00 | 1.00 | 0.88 | 1.00 | 0.63 (0.64) | 0.18 |

Late spans, log_m: trained ridge 0.43 on identifiable vs 0.24 on non-identifiable episodes (ceiling 0.74 vs 0.52).

Reading:
- **First cross-channel result: the information is still there, but the model extracts less of it.** The `nat` ceiling for log_m (0.63) matches `num` (0.64), so rounding to 0.1 removed almost no mass information. Yet the trained model's linear log_m drops from 0.61 to 0.39. This gap between information present and information extracted is exactly what the project measures.
- Mass is still computed: a linear probe gets 0.39 on the trained model vs 0.02 on the stated values.
- Every variable decodes worse at the span-end token, including stated ones (x 0.37). Possible causes:
  - (a) Language makes extraction harder.
  - (b) The probe position: x and v sit at varying positions inside templated sentences, while in `num` they sit at fixed offsets.
  - (c) The compute budget: same steps and model size, but the model must also learn the templates, and loss was still falling slowly.
  - (b) and (c) must be ruled out before claiming (a).
- Random-init is about 0 everywhere, unlike `num` (about 0.5 for x and v). In `num`, a fixed format makes position a proxy for time; in `nat`, variable-length sentences break that.

### Probe-position check: mean pooling over each span (2026-10-02, local)

Same checkpoints, with features averaged over the span's tokens instead of read at its last token. Logs: `probe_nat_run1_mean.log`, `probe_num_run2_mean.log`.

| trained ridge, log_m | end of span | mean over span |
|---|---|---|
| `num` (run 2) | 0.61 | 0.67 |
| `nat` (run 3) | 0.39 | 0.48 |

Reading:
- **Probe position does not explain the `nat` vs `num` gap.** Pooling raises both channels by a similar amount, and the gap stays at about 0.2.
- **The observables-MLP "ceiling" is not a ceiling.** With mean pooling, the `num` model beats it on log_m (0.67 vs 0.64) and on mu (0.34 vs 0.18). The sklearn MLP on about 26k rows is underfit. This corrects two earlier readings: "close to the nonlinear ceiling" (run 2) and "mu is limited by the data, not the model". A stronger reference is needed before any ceiling claim.
- Mean pooling also lifts random-init (log_m 0.15 → 0.26 for `num`). Averaged random features act like a bag of characters, so the trained-vs-random comparison must use the same pooling.

### Stronger ceiling (2026-10-02, RunPod)

`probe.ceiling()`: a 2-layer GPU MLP (512 wide, early stopping) on the observables from fresh episodes (seeds 30M+), scored on the same test split. `probe.py` prints rows for 5k and 50k episodes. The 200k result is in `ceiling_scaling_200k.log`.

| ceiling trained on | log_m | mu | a |
|---|---|---|---|
| 5k episodes | 0.71 | 0.28 | 0.92 |
| 50k episodes | 0.77 | 0.57 | 0.94 |
| 200k episodes | 0.80 | 0.67 | 0.95 |

- `num` and `nat` give the same ceiling to 3 decimals, so rounding to 0.1 loses essentially nothing.
- log_m is levelling off at about 0.8. mu is still rising, so its true ceiling is above 0.67.
- With the 50k ceiling, mu decodes higher on identifiable than on non-identifiable episodes (0.72 vs 0.57). The earlier "reversed mu" pattern was an artifact of the weak sklearn MLP.

### Run 4: `nat`, 45k steps on RunPod (2026-10-02)

Fresh run, cosine schedule over 45k steps, 2349 s on an RTX 4090. Validation loss 0.0833 (15k run: 0.0869). Checkpoint `ckpt/nat_45k.pt`, logs `train_probe_nat_45k.log` (end pooling) and `probe_nat_45k_mean.log` (mean pooling).

| trained ridge | log_m end | log_m mean | mu end | mu mean |
|---|---|---|---|---|
| `num` 15k (run 2) | 0.61 | 0.67 | 0.22 | 0.34 |
| `nat` 15k (run 3) | 0.39 | 0.48 | 0.13 | 0.16 |
| `nat` 45k (run 4) | 0.55 | 0.60 | 0.18 | 0.32 |
| ceiling (200k episodes) | 0.80 | 0.80 | ≥0.67 | ≥0.67 |

Reading:
- Tripling the steps narrows the gap to `num` 15k from 0.19 to 0.07 (mean pooling). **Superseded by run 5:** at equal steps, the gap is about 0.1.
- **Both channels sit well below the ceiling,** for mass (about 0.6 vs 0.8) and especially for mu (about 0.3 vs at least 0.67). The information is in the text, but the model doesn't extract all of it. This gap is a headline quantity for the project.
- Stated variables still decode worse in `nat` at the end token (x 0.40 vs 0.79 for `num`). With mean pooling they recover (x 0.61, v 0.60).

### Run 5: `num`, 45k steps on RunPod (2026-10-02) — channels compared at equal training

Fresh run, 1074 s on an RTX 4090, validation loss 0.2133 (15k: 0.2309). Checkpoint `ckpt/num_45k.pt`, logs `train_num_45k.log`, `probe_num_45k_end.log`, `probe_num_45k_mean.log`.

| trained ridge | log_m end | log_m mean | mu end | mu mean |
|---|---|---|---|---|
| `num` 45k | 0.64 | **0.72** | 0.31 | **0.41** |
| `nat` 45k | 0.55 | **0.60** | 0.18 | **0.32** |
| gap | 0.09 | 0.12 | 0.13 | 0.09 |
| ceiling (200k episodes) | 0.80 | 0.80 | ≥0.67 | ≥0.67 |

Reading:
- **Correction to run 4's reading.** "70% of the gap was undertraining" compared `nat` 45k with `num` 15k. `num` also improves with training, so at equal steps a real gap of about 0.1 remains on both hidden parameters. Language costs extraction even though the information ceiling is identical.
- `num` 45k reaches about 0.72 of the 0.80 mass ceiling (mean pooling). `nat` reaches 0.60.
- mu stays far below its ceiling in both channels (about 0.3–0.4 vs at least 0.67).

### Run 6: `sym`, 45k steps on RunPod (2026-10-02): sanity check and noise floor

`num` text under a fixed character substitution, same settings as run 5. Validation loss 0.2079 (`num` 45k: 0.2133). Checkpoint `ckpt/sym_45k.pt`, logs `train_sym_45k.log`, `probe_sym_45k_end.log`, `probe_sym_45k_mean.log`. Ceiling skipped (`CEIL=`), since it equals `num`'s.

| trained ridge | log_m end | log_m mean | mu end | mu mean |
|---|---|---|---|---|
| `num` 45k | 0.64 | 0.72 | 0.31 | 0.41 |
| `sym` 45k | 0.62 | 0.70 | 0.25 | 0.40 |
| difference | 0.02 | 0.02 | 0.07 | 0.01 |

Reading:
- **Sanity check passed.** A char-level model from scratch does as well on ciphered `num` as on `num`.
- **Run-to-run noise floor.** `sym` and `num` carry identical information, so their difference estimates seed-level variation: about 0.02 for log_m and up to 0.07 for mu (end pooling).
- So the `nat` vs `num` gap on log_m (0.09–0.12) is well above noise. The mu gap (0.09–0.13) is only marginally above it and needs several seeds per channel before any claim.

## Channel facts (`rel`, `sym`, added 2026-10-02)

- `rel` states only comparisons with the previous sentence: faster/slower/about as fast, reversed direction, further left/right/about where it was, and push starts/stops/reverses/stronger/weaker. A push's direction appears only relative to the motion ("with/against its motion"). There are no numbers and **no times**. m and mu are not identifiable, but `rel` still carries partial information about them: a ridge on its stated codes gets log_m about 0.2 (400 episodes), probably because heavy carts get stuck more often. So the negative control is "trained probe ≤ `rel` ceiling", not "probe ≈ 0". Thresholds: `SAME_V = SAME_X = 0.05`.
- `sym` is `num` under a fixed random character substitution, spaces included. It has the same spans and lengths as `num`. For a from-scratch char model it should match `num`, so it's a sanity check. It matters only for pretrained models.
- Observables for `qual` and `rel` (`probe.code_observables`): each sentence is read with the channel's own parser into one-hot "key=value" codes (qual 23 codes, rel 24). Each row holds the latest value of every key, plus the per-sentence history, most recent first. A ridge on 400 episodes gives `qual` log_m 0.88 (the mass bin is stated) and F 0.96. `CEIL=""` skips the ceiling for smoke tests; `CEIL=5000,50000,200000` sets the sizes.
- The `qual` and `rel` parsers now also return the motion direction, change word and start/stop event (`qual`), and the push-vs-motion relation plus the first sentence (`rel`, via `parse_rel_first`). All are checked in `_checks()`.

## Known issues and open questions

- Single seed per channel. Seed noise is about 0.02 on log_m and up to 0.07 on mu (run 6). Any mu comparison needs 2–3 seeds per channel.

- The observables MLP (sklearn) is underfit. Use the `ceiling` rows instead. The mu ceiling has not saturated even at 200k episodes.
- `qual` still needs a binned observables baseline in `probe.py` (`STATED_DIGITS` covers only `num` and `nat`). The block-size overflow is fixed.

- `qual` repeats "It stays still." during long stuck stretches. This inflates token counts (the length confound). It could be collapsed into one sentence.
- `qual` reports "a gentle push starts" when the force changes within the same bin. This is intended information loss, but worth noting in the paper.
- The citations in `project_idea.md` must be verified, especially arXiv 2607.27017, Paperlayer 2607.20058, and the PhysLang characterization.
- Related work missing from `project_idea.md`: Vafa et al. 2025 ("What has a foundation model found?", inductive-bias probe on orbital mechanics; the closest prior work), Vafa et al. 2024 (world-model evaluation metrics), Li, Nye, Andreas 2021 (implicit entity state in text), and Othello-GPT.

## Not built yet

- Verbalizers: the ablations `nat-noterm`, `nat-notime`, `nat-nocause`, `nat-short`.
- Dataset splits: `iid`, `ood-combo`, `ood-extrap`, `compose`, `cross-channel`. The `ood-combo` and `ood-extrap` splits need param-region filtering in `sample_params` or at dataset-build time.
- The interchange intervention code.
- Phase 2: collisions.
