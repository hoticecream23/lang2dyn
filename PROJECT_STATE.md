# Project State

Last updated: 2026-10-04

## Research question

Can next-token prediction on language alone recover both latent physical **states** z and the **operators** f: z → z' that govern them? Hold the world W fixed, vary the verbalization L_i = g_i(W), and measure which variables and dynamics become identifiable, represented, causally functional and compositionally reusable.

Working title: *From Language to Dynamics: Identifying Latent Physical States and Operators Learned from Linguistic Observations.*

## Headline findings so far

All numbers are best-layer linear-probe scores on held-out episodes. The hidden parameters are log_m (mass) and mu (friction), which `num`, `nat` and `sym` never state.

1. **Models compute hidden physical parameters from text.** `num` (3 seeds, 45k steps): log_m 0.707 ± 0.011 (mean pooling), while a linear probe on the stated numbers gets 0.02. Mass is computed from the dynamics and stored as a linear feature.
2. **Language costs extraction, not information.** `nat` (the same trajectories in words) has the same information ceiling as `num` (log_m about 0.80), but the model extracts less: log_m 0.613 ± 0.017 vs 0.707 ± 0.011, and mu at the end token 0.189 ± 0.007 vs 0.296 ± 0.015. All gaps are many SEs from 0, except mean-pooled mu (t 2.6).
3. **Models stay below the ceiling, mu most of all.** Ceiling (GPU MLP on the stated values, 200k episodes): log_m 0.80, mu ≥ 0.67, not saturated. Models reach mu of about 0.3–0.4.
4. **Controls pass.** `sym` (ciphered `num`) matches `num` within seed noise. `rel` (comparisons only) stays below its own ceiling (log_m 0.21 vs 0.27): no information is invented.
5. **Generalization (first look).** On unseen masses (5–8), next-token loss on moving spans rises 10–14% (`qual`: 36%). Probes fit on m ≤ 5 fail to extrapolate for every decoder, so probes are the wrong tool there.
6. **Interventions (`num` and `nat`; mass and friction; 3 seeds; CIs; random-init control).**
   - The log_m probe direction is causally inert, and no 1-D mass variable exists.
   - **Write:** writing the true value into a 32-D subspace at L4–L5 makes the model produce the counterfactual, F-dependent velocities.
     - Mass in `num`: robust slope 0.67–0.91 on 3 seeds, with tight CIs.
     - Mass in `nat`: 0.92 at L5.
     - Friction in `num`: 0.84 at L5.
     - The CE gap is 0.03–0.29 against 0.001 on a random-init model.
   - **Read:** swapping in another episode's own activations transfers little.
     - Mass in `num`: CE gap +0.007 to +0.018, CIs excluding 0 and the shuffled control on every seed. Slope 0.14 on seed 0, 0.01 on seeds 1–2.
     - `nat` and mu: about 0.
     - So the variable is held distributed and redundantly, and a 32-D swap moves little of it.
   - **Computation, not lookup (2 seeds):** `ood-combo`-trained models produce the right velocity for an unseen mass × force combination when the mass is written in. Held-out effect 0.86 / 0.87 vs seen 0.78 / 0.72. Their moving-span loss on real held-out-combination episodes is about 4% above the iid model's.
   - **No composition:** a model trained on frictionless-with-force and friction-while-coasting episodes (`compose`) fails on friction and force together. Moving-span loss is 3.7× its training-part loss; stuck-span loss rises from 0.006 to 1.08, since it never saw stiction under a force. Its mass pathway (F/m) still responds correctly to written mass, but it doesn't combine friction with force.
7. **Not yet tested:** a second `compose` seed, `nat` for the `ood-combo` / `compose` designs, and the `nat` ablations. F is not an intervention target, because it is stated in every span.

## Files

| File | Purpose | Status |
|---|---|---|
| `project_idea.md` | Original idea and novelty analysis (LLM-generated citations, unverified) | reference |
| `Simulator_Verbalizer_Spec.md` | Spec: simulator, channels, information map, splits, episode record | matches code (v1) |
| `sim.py` | Simulator; `num`, `nat`, `qual`, `rel`, `sym` verbalizers and parsers; identifiability; counterfactual twins; dataset splits; self-checks | all checks pass |
| `train.py` | Char-level GPT from scratch (6 layers, d=256, 8 heads, about 5M params), one episode per sequence, with resume, seed and split arguments | working |
| `probe.py` | Ridge/MLP probes on the residual stream (end or mean pooling), random-init and observables baselines, GPU-MLP ceiling, OOD scoring, next-token loss split by moving/stuck spans | working |
| `intervene.py` | Interchange interventions on mass or friction (`num` and `nat`): counterfactual pairs (iid / `ood-combo` / `compose` designs), DAS / steering / probe / random directions, greedy-decode and CE scoring with CIs | working |
| `requirements.txt` | torch ≥ 2.6, numpy, scikit-learn, psutil | |
| `HANDOFF.md` | How to resume in a new chat | |
| `ckpt/*.pt`, `*.log` | Checkpoints and logs (gitignored, local only). Each results section names its files | |

## Decisions made

1. **Operator recovery is defined through interchange interventions.** Patch the internal representation of m (or F, or mu) from episode B into episode A. The model's continuation should match the simulator's counterfactual A-with-B's-value. Score this on held-out (m, F) regions so lookup is separated from computation. Use the causal abstraction / DAS framework, with the simulator as the high-level model.
2. **Drop direct estimation of I(W;L_i).** The verbalizers are designed by us, so each channel's information content is known by construction (the information map in the spec). The decodability ceiling is a GPU MLP trained on the stated values (see 9).
3. **A dual encoder is not the main model.** It would see z (which breaks the language-only condition), it has no dynamics, and it tests a different objective. Keep it only as a possible measuring instrument.
4. **Train small GPTs from scratch as the primary setting** (so far 6 layers, d=256, about 5M params). Pretrained LMs are a secondary comparison only, because of their physics priors.
5. **The headline probes target inferable-but-unstated variables** (the I cells in the information map). Probes on stated variables measure token copying and serve as baselines.
6. **Phase 1 is one cart with piecewise force, kinetic friction and stiction.** Collisions are phase 2.
7. **mu was narrowed from 0–0.5 to 0–0.3.** With 0–0.5, static friction (up to 24.5 N) swallowed many force segments: 33% of steps stuck, 71% of episodes identifiable. With 0–0.3: 19% stuck, 84% identifiable (2000 seeds). About 10% of episodes are deliberate single-segment controls.
8. **One char-level tokenizer for every channel, no BPE.** A shared tokenizer removes the tokenizer as a confound between channels. `train.py` sets the context length from the data (1.25 × the longest training text, rounded up to 64): 576 tokens for `num`/`sym`, 1472 for `nat`.
9. **Ceiling = GPU MLP on the observables** (`probe.ceiling`, two 512-wide hidden layers, early stopping), trained on fresh episodes. The sklearn "observables MLP" row is underfit and is not a ceiling. Ceilings are lower bounds on the true information content.
10. **Standard comparison setting:** 100k training episodes, 45k steps, batch 64, both poolings reported. Channel gaps need 2–3 seeds or more (seed noise: log_m about 0.02, mu up to 0.05).
11. **OOD evaluation:** probes are fit on a split's train part and scored on its test part as 1 − MSE/Var_train (plain R² explodes on narrow test regions). Model behavior is judged by next-token loss on **moving** spans only, because stuck text is about 9× easier and its share varies between parts.
12. **Interventions are scored by the source-difference metrics** in `intervene.py` (CE gap, and the Theil–Sen `src slope`), not by IIA or effect alone. Erasing A's mass already moves predictions toward a random B's counterfactual, so IIA rises even with a source-independent patch. Pairs use force-change prompts, because elsewhere the model can extrapolate Δv without mass. `das-steer` (writing the true value) is reported as an upper bound, not as an interchange.

## Code facts

### `sim.py`

- Recording: 50 steps at 0.1 s, integrated with 10 substeps of 0.01 s. Constant acceleration is integrated exactly; a zero-velocity crossing inside a substep is solved exactly and friction is re-evaluated there.
- Sampling: 10% of episodes have one force segment (mostly non-identifiable), the rest have 2–4. 30% of multi-segment episodes get one coast segment (F = 0). Segments start on 0.1 s boundaries, and F is rounded to 0.1 N.
- Emission: every 5 recorded steps, plus every step with an event (`force_change`, `start`, `stop`). Each emitted step is one span `[char_start, char_end, step]`.
- `identifiable(tr)` returns `{"m": bool, "mu": bool}`. In the kinetic regime a = F·(1/m) − g·s·mu, so both are identifiable iff two observed (F, s) rows are linearly independent. mu alone is also identifiable from any coast row. Over 500 seeds: m is identifiable in 414 episodes, mu in 416.
- `make_episode(seed, **changes)`: changes make a counterfactual twin (for example `m=4.0`), with `cf_of` pointing to the base id.
- `make_episode(seed, m_switch=[r0, m])` / `mu_switch=[r0, mu]`: the mass (friction) changes at recorded step r0 (the state at r0 is unchanged). The text is identical to the original up to step r0. Used as the intervention target.
- Template RNG seeds: `qual` uses `f"{seed}-surface"`, `nat` `f"{seed}-surface-nat"`, `rel` `f"{seed}-surface-rel"`. Each sentence type has 2–3 phrasings.
- Commands: `python sim.py` runs the self-checks. `python sim.py N out.jsonl` writes N episodes.

### Channels

- `num`: `t=… x=… v=… F=… .` per span; x and v to 0.01, F exact.
- `nat`: words, with x and v to 0.1, F exact when it changes, and time only as gaps ("0.5 s later"). Never states a, m or mu.
- `qual`: bins only.
  - mass light/medium/heavy (< 1, 1–2.5, > 2.5 kg)
  - push gentle/firm/hard (< 3, 3–7, > 7 N), with direction
  - speed still/slow/steady/fast (< 0.05, < 1, < 3, ≥ 3 m/s)
  - change words: speeds up slowly/quickly, slows down gently/sharply, keeps its pace
  - events: starts moving, halts / comes to a stop, stays still

  **It states a mass bin.** It repeats "It stays still." while stuck (a length confound).
- `rel`: only comparisons with the previous sentence: faster/slower/about as fast, reversed direction, further left/right/about where it was, push starts/stops/reverses/stronger/weaker, push with/against the motion. No numbers, no times. Thresholds `SAME_V = SAME_X = 0.05`. m and mu are not identifiable, but `rel` still carries partial information (ceiling log_m 0.27), probably because heavy carts get stuck more often. So the negative control is "probe ≤ `rel` ceiling", not "probe ≈ 0".
- `sym`: `num` under a fixed random character substitution (spaces included), with the same spans and lengths. For a from-scratch char model it should match `num`; it matters only for pretrained models.

### `train.py`

- `python train.py [channel=num] [n=100000] [steps=5000] [seed=0] [split=iid]` writes `ckpt/<channel>[_<split>][_s<seed>].pt`. The default is 5000 steps, but the standard setting is 45000.
- AdamW lr 1e-3, betas (0.9, 0.95), weight decay 0.1, 200-step warmup, cosine decay to 10%, batch 64, bf16 autocast, grad clip 1.0. Validation: 2000 episodes from seed 10M (from that split's train part).
- Resume: atomic save to `ckpt/<name>_state.pt` every 500 steps. Rerun the same command to resume; the state file is deleted at the end.
- The seed changes weight init and batch order only; the training episodes are seeds 0..n−1 for every seed.

### `probe.py`

- `python probe.py [ckpt=ckpt/num.pt] [n=3000] [pool=end|mean] [split=iid]`. The `CEIL` env variable sets the ceiling sizes (default `5000,50000`). `CEIL=` skips the ceiling, for smoke tests or when it's already known.
- Probe episodes: seeds 20M+. iid: fit on 75%, score on 25%, split by episode. OOD splits: fit on n train-part episodes, score on n/3 test-part episodes.
- Rows: trained ridge/MLP (every layer), random-init ridge, observables ridge/MLP (sklearn), and the ceiling (GPU MLP, seeds 30M+). Then the late-span (step ≥ 25) split by each parameter's identifiability flag.
- Observables:
  - `num`/`nat`/`sym`: the stated values (t, x, v, F) at the channel's precision, history most recent first.
  - `qual`/`rel`: one-hot parser codes (`code_observables`; 23 `qual` codes, 24 `rel` codes). Each row holds the latest value of every key plus the per-sentence history. The vocab comes from seeds 40M+.
- `lm_loss_on(model, eps, stoi, channel)` returns per-token loss over all / moving / stuck spans. It is printed for OOD splits.

### `intervene.py`

- `python intervene.py [ckpt=ckpt/num_45k.pt] [n_train=2000] [n_test=400] [n_probe=2000] [k=1] [pos=all|last|span]`. The channel (`num` or `nat`) comes from the checkpoint. Env options:
  - `TARGET=m|mu` (default m);
  - `SPLIT=iid|combo|compose` (`COMBO=1` is an alias for combo; see below);
  - `LAYERS=3,4,5` (default 1–6);
  - `SRC=mean|end` (B's source activation), `BSRC=last|fc` (B's source span), `BID=1` (B's prefix must identify B's value);
  - `STEER=logm|invm|lin` (default log m for mass, linear for mu);
  - `RANDOM=1` (random-init model control; it can't decode, so only the CE gap is defined);
  - `OUT=path.npz` (save per-pair decoded v and target CE for every condition).
- Pairs (seeds 50M+, A = even seed, B = the next seed): A's prompt ends at a span where a new nonzero force starts (step r0 ≥ 10). Target: `<TARGET>_switch=[r0, B's value]`. Pairs are kept when:
  - the counterfactual changes v at the next emitted step r1 by at least 0.05 (`num`) or 0.15 (`nat`, which states v to 0.1);
  - it emits the same step;
  - its text up to r0 is identical (`nat` states events at r0, e.g. "starts moving", which can depend on the switched value).
- Held-out pairs (never used to train DAS):
  - `iid`: B's value in [2, 3] kg (mu: [0.1, 0.15]).
  - `combo` (for an `ood-combo` model): A and B are train-part episodes, and held out = an unseen combination (m_B in [3, 5] and |F| ≥ 7 at r0).
  - `compose` (for a `compose` model): held out = A is a test-part episode (friction and force together); seen = A is a frictionless train-part episode. B is a frictionless train-part episode.
  - For `combo` and `compose`, the test set is half held-out pairs.
- Intervention: at layer L's output, overwrite A's coordinates in a k-dim orthonormal subspace U with c, at positions given by `pos`: `all`; `last` = from A's last prompt span on, generated tokens included; `span` = that span only.
- Conditions:
  - `das`: U learned with the model frozen, teacher-forced CE on the counterfactual span; c = B's activations (mean over its source span) projected on U.
  - `das-shuf`: as `das`, but trained with another pair's B as source.
  - `ablate`: c = the mean c.
  - `das-steer`: c = α·f(z_B) + β, learned (f = log m, or mu). Not an interchange: it writes the true value.
  - `probe` / `random` directions (k = 1).
- Metrics on held-out pairs:
  - greedy-decoded v at r1: `effect` = (v_patched − v_clean)/(v_cf − v_A), median; `iia` = share closer to v_cf than to v_A.
  - **`src slope` / `r`** = Theil–Sen slope (since 2026-10-03 round 2; earlier logs use a slope through 0) and correlation of v_patched(B) − v_patched(B′) on v_cf(B) − v_cf(B′), with B′ a second source for the same A. `r` is outlier-sensitive. This is the key number: erasing A's mass already moves v toward a random B's counterfactual (regression to the mean), and `src` is 0 for any source-independent change.
  - Teacher-forced target CE with the matched source and with B′. **CE gap** = the difference: mean with a bootstrap 95% CI (1000 resamples of pairs), also split held-out / seen. The src slope has a Theil–Sen 95% CI.
- `nat` decoding stops at the state sentence ("… mark." / "… m/s."), and r1 is checked through the stated time gap.
- Timing on a 4090: about 3–4 min per layer at 2000/400 pairs (`nat` about 2× longer). Several runs can share the GPU.

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
- The identifiability control is confounded. The flag is joint, but a coast-only episode identifies mu without m. The non-identifiable subset is also small (about 80 episodes). *(Fixed later: per-parameter flags. The reversed mu pattern turned out to be a weak-baseline artifact; see "Stronger ceiling".)*

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
| observables MLP (sklearn, underfit; not a ceiling) | 1.00 | 1.00 | 0.88 | 1.00 | 0.64 | 0.18 |

Late spans (step ≥ 25), split by each parameter's own flag:

| | log_m id | log_m non-id | mu id | mu non-id |
|---|---|---|---|---|
| trained ridge | 0.67 | 0.39 | 0.21 | 0.31 |
| observables MLP | 0.75 | 0.51 | 0.20 | 0.29 |

Reading:
- **Mass is the headline result.** A linear probe on the trained model gets log_m R² 0.61. A linear probe on the stated numbers gets 0.02. So the model has computed a nonlinear function of the text (mass from dynamics) and stores it linearly. *("Close to the nonlinear ceiling of 0.64" was wrong: the real ceiling is about 0.80.)*
- **Acceleration is not evidence of computation.** a is nearly linear in the stated values (Δv/Δt with a mostly fixed Δt), so the observables ridge already reaches 0.75. The trained MLP (0.85) is near the ceiling (0.88).
- ~~mu is limited by the data, not the model.~~ **Superseded:** the sklearn MLP was underfit. The GPU ceiling reaches mu ≥ 0.67, so the model is far below what the data allow.
- The log_m id > non-id gap appears in the baseline too, so it comes from the data: non-identifiable episodes still carry partial information (stiction bounds, the parameter range). *(The reversed mu gap was an artifact of the weak sklearn baseline: with the GPU ceiling, mu is higher on identifiable episodes.)*
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
| observables MLP (sklearn, underfit) | 1.00 | 1.00 | 0.88 | 1.00 | 0.63 (0.64) | 0.18 |

Late spans, log_m: trained ridge 0.43 on identifiable vs 0.24 on non-identifiable episodes (ceiling 0.74 vs 0.52).

Reading:
- **First cross-channel result: the information is still there, but the model extracts less of it.** The `nat` baseline for log_m (0.63) matches `num` (0.64), so rounding to 0.1 removed almost no mass information. (Confirmed later with the GPU ceiling: identical to 3 decimals.) Yet the trained model's linear log_m drops from 0.61 to 0.39. This gap between information present and information extracted is exactly what the project measures.
- Mass is still computed: a linear probe gets 0.39 on the trained model vs 0.02 on the stated values.
- Every variable decodes worse at the span-end token, including stated ones (x 0.37). Possible causes:
  - (a) Language makes extraction harder.
  - (b) The probe position: x and v sit at varying positions inside templated sentences, while in `num` they sit at fixed offsets.
  - (c) The compute budget: same steps and model size, but the model must also learn the templates, and loss was still falling slowly.
  - (b) and (c) must be ruled out before claiming (a). *(Resolved: (b) was ruled out by mean pooling. (c) explains part of the gap. At equal training with 3 seeds, a real gap of about 0.1 remains.)*
- Random-init is about 0 everywhere, unlike `num` (about 0.5 for x and v). In `num`, a fixed format makes position a proxy for time; in `nat`, variable-length sentences break that.

### Probe-position check: mean pooling over each span (2026-10-02, local)

Same checkpoints, with features averaged over the span's tokens instead of read at its last token. Logs: `probe_nat_run1_mean.log`, `probe_num_run2_mean.log`.

| trained ridge, log_m | end of span | mean over span |
|---|---|---|
| `num` (run 2) | 0.61 | 0.67 |
| `nat` (run 3) | 0.39 | 0.48 |

Reading:
- **Probe position does not explain the `nat` vs `num` gap.** Pooling raises both channels by a similar amount, and the gap stays at about 0.2 (at 15k steps).
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
- So the `nat` vs `num` gap on log_m (0.09–0.12) is well above noise. The mu gap (0.09–0.13) was only marginally above it. *(Resolved by the 3-seed runs below.)*

### Run 7: `rel`, 45k steps on RunPod (2026-10-02): negative control

Same settings as runs 5–6. 2074 s. Validation loss plateaued at 0.0341 by step 15k and ended at 0.0346 (slight overfitting). Checkpoint `ckpt/rel_45k.pt`, logs `train_rel_45k.log`, `probe_rel_45k_end.log`, `probe_rel_45k_mean.log`, `ceiling_rel.log`. The observables are the parsed comparison codes (`code_observables`).

| | log_m end | log_m mean | mu end | mu mean |
|---|---|---|---|---|
| trained ridge | 0.19 | 0.21 | 0.16 | 0.20 |
| observables ridge | 0.18 | 0.18 | 0.16 | 0.16 |
| ceiling 5k / 50k episodes | 0.24 / 0.27 | | 0.22 / 0.25 | |

Reading:
- **Negative control passed.** When the text carries little mass information, the model doesn't invent it: it stays below the `rel` ceiling (0.27). Compare `num`, where the trained probe (0.64) far exceeds the linear observables baseline (0.02).
- Identifiable and non-identifiable episodes score about the same with `rel`, as expected.
- **Fraction of the available mass information the model extracts** (mean-pooled linear probe ÷ ceiling, seed 0): `num` 0.72/0.80 ≈ 0.90, `nat` 0.60/0.80 ≈ 0.75, `rel` 0.21/0.27 ≈ 0.79. This ratio is a candidate headline metric per channel. The ceilings are lower bounds, so the ratios are upper bounds. (`rel`'s ceiling is from 50k episodes, the others from 200k.)

### Seeds: `num` and `nat`, 3 seeds each at 45k steps (2026-10-02, RunPod)

Seeds 0, 1, 2 (weight init and batch order; same data). Checkpoints `ckpt/{num,nat}_45k.pt` (seed 0), `ckpt/{num,nat}_s{1,2}.pt`. Logs `{num,nat}_s{1,2}_{train,end,mean}.log`. Validation loss: `num` 0.2133 / 0.2032 / 0.2035, `nat` 0.0833 / 0.0836 / 0.0844.

Trained ridge, best layer, mean ± sd over 3 seeds:

| | log_m end | log_m mean | mu end | mu mean |
|---|---|---|---|---|
| `num` | 0.658 ± 0.019 | 0.707 ± 0.011 | 0.296 ± 0.015 | 0.385 ± 0.038 |
| `nat` | 0.527 ± 0.023 | 0.613 ± 0.017 | 0.189 ± 0.007 | 0.298 ± 0.045 |
| gap (t = gap / SE) | 0.131 (t 7.5) | 0.094 (t 8.3) | 0.107 (t 11.5) | 0.086 (t 2.6) |

Reading:
- **The `num` vs `nat` gap is real for both hidden parameters.** log_m loses about 0.09–0.13 and mu about 0.09–0.11 when the trajectory is told in words. Every gap is well beyond seed noise, except mean-pooled mu, which is weaker (t 2.6; mu is noisy under mean pooling, sd about 0.04).
- Seed noise is about 0.01–0.02 for log_m and 0.01–0.05 for mu. This matches the `sym` estimate (run 6).

### Run 8: `qual`, 45k steps on RunPod (2026-10-02)

2064 s, validation loss 0.0312. Checkpoint `ckpt/qual_45k.pt`, logs `qual45k_train.log`, `qual45k_end.log` (with ceiling), `qual45k_mean.log`. The observables are the parsed bin codes.

| | log_m end | log_m mean | mu end | mu mean |
|---|---|---|---|---|
| trained ridge | 0.78 | 0.89 | 0.21 | 0.33 |
| observables ridge | 0.89 | | 0.23 | |
| ceiling 50k | 0.90 | | 0.44 | |

- `qual` **states** a mass bin, so its mass ceiling (0.90) is higher than `num`'s (0.80). The model reaches about 0.98 of it, but that is mostly reading the stated bin, not computing mass.
- mu (never stated): 0.33 vs a ceiling of 0.44, a ratio of about 0.75.

### Generalization: `ood-extrap` with the existing iid models (2026-10-02)

Probes fit on the train part (m ≤ 5) and scored on m in (5, 8]. Scores are 1 − MSE/Var_train. Logs: `num45k_extrap_mean.log`, `nat45k_extrap_mean.log`, `lm_extrap.log`.

| log_m on m > 5 (mean pooling) | `num` 45k | `nat` 45k |
|---|---|---|
| trained ridge | −0.45 | −0.93 |
| trained MLP | −0.33 | −0.43 |
| observables ridge | −3.18 | −3.18 |
| ceiling 50k (trained on m ≤ 5) | −0.28 | −0.28 |

Next-token loss per token on **moving spans**, with heavy-mass relative to normal-mass in brackets: `num` 0.266 → 0.292 (1.10×), `nat` 0.090 → 0.101 (1.12×), `sym` 1.14×, `rel` 1.13×, `qual` 1.36×. Stuck spans are about 9× easier (`num`: 0.027 vs 0.243 iid), and heavy carts are stuck more. So loss over all tokens *falls* on heavy masses (`num` 0.220 → 0.165), which is an artifact.

Reading:
- **Probing hidden parameters beyond the training range is ill-posed.** Every decoder fit on m ≤ 5 fails on m > 5, including the ceiling. Extrapolation should be tested through model behavior (moving-span loss, interventions), not probes.
- Still, a linear probe on the `num` representation extrapolates far better than a linear probe on the stated values (−0.45 vs −3.18). The mass direction is roughly linear somewhat beyond the training range. `nat` is weaker (−0.93).
- Moving-span loss degrades 10–14% on unseen masses for `num`/`nat`/`sym`/`rel`, and 36% for `qual`, whose "heavy" bin covers everything above 2.5. This doesn't yet separate extrapolation failure from intrinsically harder dynamics: that needs a reference model trained on m up to 8.

### Interchange interventions on mass: `num` 45k (2026-10-03, RunPod)

Logs (local, gitignored): `das_num45k.log` (k=1, all positions), `das_k1_last.log`, `das_k1_all_invm.log`, `das_k8_last.log`, `das_k8_last_end.log`, `das_k8_last_bid.log`, `das_k8_last_fc.log`, `das_k8_span.log`, `das_k8_all.log`, `das_k32_all.log`. 400 test pairs; median |v_cf − v_A| at r1 is about 0.51. The clean model decodes v at r1 with median error 0.005 on generic pairs and 0.084 at force changes, closer to v_A than to v_cf in about 70–87% of pairs. So it uses mass information behaviourally.

**1. The ridge log_m probe direction is causally inert.** Patching along it does nothing beyond a random direction of the same norm: src slope 0.00 at every layer, and ×1–×100 dose-response is no better than random. The probe direction is decodable but not used. DAS directions have |cos| ≤ 0.13 with it.

**2. No 1-D variable carries mass.** At k = 1 (all positions or `last`), DAS from B's activations has src slope about 0 at every layer, the same as `das-shuf`. Its IIA gain (0.13 → about 0.3) equals `ablate`: it comes from erasing A's mass, not from transferring B's. Writing the true log m or 1/m at k = 1 also gives nothing (best src slope 0.16).

**3. An 8-D write channel that the model reads as mass** (`das-steer`, k = 8, `pos=last`):

| | L3 | L4 | L5 | L6 |
|---|---|---|---|---|
| src slope / r, all test pairs | 0.41 / 0.62 | 0.62 / 0.64 | 0.58 / 0.77 | 0.01 / 0.02 |
| src slope / r, m_B held out of DAS training | 0.35 / 0.65 | 0.66 / 0.88 | 0.54 / 0.85 | |
| effect (median) / iia | 0.43 / 0.50 | 0.60 / 0.61 | 0.33 / 0.44 | 0 / 0.20 |
| target CE, source B / B′ (clean 1.116) | 0.373 / 0.418 | 0.348 / 0.437 | 0.357 / 0.417 | |

It replicates with other pair sets (`BID=1`: L4 slope 0.59, r 0.70; held-out masses r 0.88). The code c depends only on m_B, but the counterfactual change depends on F's sign and size. So the model must combine the written value with F downstream: the subspace feeds its a = F/m computation. It interpolates to B masses never used in training (r 0.85–0.88).

**4. But the natural value doesn't transfer.**
- DAS from B's own activations finds nearly the same subspace (overlap with the steer subspace 0.94–0.96), yet src slope is only 0.00–0.15 at L3–L5.
- No B source choice helps:
  - mean over B's last span up to r0, or its last token (`SRC=end`);
  - B's prefix required to identify m_B (`BID=1`);
  - B's own last force-change span, the same situation as A (`BSRC=fc`).
- Diagnostic at L4 (a one-off script, not kept):
  - Natural activations in the subspace do carry mass: log m R² 0.41 (A's last span) and 0.48 (B's source span), vs 0.70–0.73 from all 256 dims.
  - But the learned steer code lies far off-manifold: its mean is 4.9 natural spreads from the natural mean, with up to 5× the natural per-dim spread.
- Reading: with `pos=last`, every earlier position still holds A's own mass at layer L and below, and later tokens attend to them. A natural-scale swap from B can't override this redundant copy; a 5× write can. The test is patching at all positions with k = 8 / 32 (below).

**5. The written value reaches predictions through attention.** With `pos=span` (only A's last prompt span patched; generated tokens untouched), `das-steer` still works: L4 slope 0.45, r 0.72; L5 slope 0.40, r 0.69; held-out masses r 0.75–0.85. The read side is still null there.

**6. Real interchange appears once every position and a wider subspace are patched** (`pos=all`, L4):

| L4, pos=all | k = 8 | k = 32 |
|---|---|---|
| `das` src slope / r | 0.19 / 0.26 | **0.31 / 0.53** |
| `das-shuf` src slope / r | 0.07 / 0.13 | 0.10 / 0.18 |
| `das` target CE, source B / B′ | 0.369 / 0.378 | 0.362 / 0.383 |
| `das-steer` src slope / r, effect, iia | 0.69 / 0.69, 0.63, 0.65 | 0.74 / 0.70, 0.83, 0.78 |

- L2–L3 stay near 0 (k = 32 L3: 0.18 / 0.27). L5 is weaker than L4 (k = 32: 0.13 / 0.34).

Seeds and width at L4, pos=all (logs `das_k32_all_L4_s{1,2}.log`, `das_k64_all_L4.log`). CE gap = target CE with source B′ minus with source B (source information actually used; 0 for any source-independent patch):

| | `das` src slope / r | `das` CE gap | `das-shuf` CE gap | `das-steer` CE gap |
|---|---|---|---|---|
| seed 0, k = 32 | 0.31 / 0.53 | 0.021 | 0.003 | 0.174 |
| seed 1, k = 32 | 0.11 / 0.30 | 0.008 | 0.000 | 0.101 |
| seed 2, k = 32 | 0.50 / 0.11 | 0.008 | 0.000 | 0.147 |
| seed 0, k = 64 | 0.39 / 0.54 | 0.036 | 0.008 | 0.196 |

- The read effect is positive in every seed and beats the shuffled control, but it is small and varies between seeds.
- The seed-2 slope is driven by a few outliers (r 0.11). The slope through 0 is outlier-sensitive, so read r and the CE gap first. A robust slope (Theil–Sen) would be better.
- Wider subspaces read more (k = 64 > 32 > 8), consistent with a distributed code.
- A random 64-D overwrite breaks decoding (only 87/400 parse).
- Held-out-mass rows (65 pairs) are noisy: `das-shuf` reaches 0.47 / 0.55 there at k = 32. Use the all-pairs row.

Reading:
- **The model carries mass in a distributed, redundant form,** spread over many dimensions and over every earlier position. No 1-D variable carries it, and the probe direction is not the causal one.
- **A real interchange from the model's own activations** (k = 32, every position, L4) transfers about a third of B's effect (slope 0.31, r 0.53), well above the shuffled control.
- **Writing the true value** into the same subspace transfers about 0.7. Part of that gap is the off-manifold amplitude the steer is free to use.
- **The operator reads the variable:** the effect depends on F per pair and interpolates to unseen mass values.
- **Caveats:**
  - Read effect replicated on 3 seeds in sign only (see the seeds table).
  - DAS can find directions with any model. The shuffled-source control and the source-difference metric guard against that, but a random-init-model control is not run.
  - The held-out mass band is a narrow interpolation test (m_B in [2, 3]), not the `ood-combo` separation of lookup from computation that decision 1 asks for (done in round 2, below).

### Interventions, round 2: robustness and lookup vs computation (2026-10-03, RunPod)

Logs (local, gitignored): `r2_s0.log`, `r2_s1.log`, `r2_s2.log` (k = 32), `r2_k64.log`, `r2_rnd.log` (random-init model), `r2_combo_iidmodel.log`, `r2_combo_combomodel.log`, `train_num_ood-combo.log`, `probe_num_ood-combo_mean.log`. All rows: L4, `pos=all`, Theil–Sen src slope. CE gap = target CE with source B′ minus with source B.

**Robustness** (1000 test pairs):

| | `das` CE gap | `das` slope / r | `das-shuf` gap | `das-steer` gap | `das-steer` slope |
|---|---|---|---|---|---|
| seed 0, k = 32 | +0.019 | 0.14 / 0.43 | +0.002 | +0.171 | 0.91 |
| seed 1, k = 32 | +0.010 | 0.01 / 0.20 | +0.001 | +0.100 | 0.67 |
| seed 2, k = 32 | +0.006 | 0.01 / 0.09 | −0.001 | +0.153 | 0.84 |
| seed 0, k = 64 | +0.035 | 0.28 / 0.51 | +0.004 | +0.214 | 1.00 |
| random-init model, k = 32 | +0.003 | (no decode) | 0.000 | +0.001 | |

- **The write effect replicates on all 3 seeds** and is absent in the random-init model, so DAS is not fitting noise.
- **The read effect is small.** Its CE gap is positive on every seed and above both controls, but the robust slope is near 0 for seeds 1–2. The round-1 seed-0 slope (0.31 through 0, 400 pairs) was partly outliers; the robust value is 0.14.

**Lookup vs computation** (decision 1). Model `ckpt/num_ood-combo.pt`: `num`, 45k steps, seed 0, trained on the `ood-combo` train part (never sees m in [3, 5] with |F| ≥ 7). Validation loss 0.2119. Pairs use `COMBO=1`: A and B are train-part episodes, and DAS trains on seen pairs only. 300 test pairs are held out (A's new force |F| ≥ 7 with m_B in [3, 5]: the counterfactual is the unseen combination), plus 300 seen pairs.

| k = 32, L4 | held-out combination: effect / iia | seen: effect / iia |
|---|---|---|
| `ood-combo` model, `das-steer` | **0.86 / 0.87** | 0.78 / 0.75 |
| `ood-combo` model, `ablate` | −0.01 / 0.24 | 0.20 / 0.37 |
| `ood-combo` model, `das` | 0.16 / 0.35 | 0.32 / 0.47 |
| iid model, `das-steer` | 0.88 / 0.88 | 0.79 / 0.78 |
| iid model, `ablate` | −0.14 / 0.21 | 0.32 / 0.43 |

Next-token loss on moving spans (ood-combo parts, 3000 / 1000 episodes):

| | train part | test part (held-out combination) |
|---|---|---|
| iid model (`num_45k`) | 0.2648 | 0.2624 |
| `ood-combo` model | 0.2571 | 0.2718 |

Reading:
- **The operator computes rather than looks up.** Given a mass it never saw together with a strong force, the `ood-combo` model's dynamics produce the right velocity (effect 0.86), as well as on seen pairs and as well as a model that did see the combination.
- **Behavioural generalization is close to complete.** On real held-out-combination episodes, moving-span loss is 4% above the iid model's. The held-out region is not intrinsically harder (the iid model's loss is the same on both parts).
- The read (`das`) effect is weak here too, as on the iid model.
- Caveat: one seed for the `ood-combo` model. The held-out region is an interpolation in m and in |F| separately (both ranges were seen, just not together).

### Interventions, round 3: CIs, `nat`, friction, `compose`, second `ood-combo` seed (2026-10-04, RunPod)

Logs (local, gitignored): `r3_ci_s{0,1,2}.log`, `r3_nat.log`, `r3_mu.log`, `r3_compose_ref.log`, `r3_compose_model.log`, `r3_combo_s1.log`, `train_num_compose.log`, `train_num_ood-combo_s1.log`, `probe_num_compose_mean.log`. Per-pair outputs: `out/*.npz` (gitignored). All rows: k = 32, `pos=all`. CIs: bootstrap for the CE gap, Theil–Sen for the slope.

**Mass in `num`, L4, 1000 test pairs:**

| seed | `das` CE gap [95% CI] | `das-shuf` gap | `das` slope [CI] | `das-steer` slope [CI] | `das-steer` gap |
|---|---|---|---|---|---|
| 0 | +0.018 [+0.015, +0.022] | +0.002 | 0.14 [0.11, 0.17] | 0.91 [0.88, 0.93] | +0.172 |
| 1 | +0.010 [+0.008, +0.013] | +0.001 | 0.01 [0.00, 0.02] | 0.67 [0.63, 0.70] | +0.099 |
| 2 | +0.007 [+0.004, +0.009] | −0.001 | 0.01 [0.00, 0.03] | 0.84 [0.82, 0.87] | +0.154 |

**Mass in `nat`** (`nat_45k`, 1000 test pairs; v stated to 0.1, so pairs need |v_cf − v_A| ≥ 0.15):

| | L3 | L4 | L5 |
|---|---|---|---|
| `das-steer` slope [CI] / effect / iia | 0.60 [0.57, 0.64] / 0.67 / 0.66 | 0.84 [0.81, 0.86] / 0.84 / 0.79 | 0.92 [0.90, 0.94] / 0.86 / 0.82 |
| `das-steer` CE gap (clean target CE 0.339) | +0.026 | +0.052 | +0.051 |
| `das` CE gap / slope | +0.000 / 0.00 | +0.002 / 0.00 | +0.001 / 0.00 |

**Friction** (`num_45k`, `TARGET=mu`, 1000 test pairs; median |v_cf − v_A| 0.22):

| | L3 | L4 | L5 |
|---|---|---|---|
| `das-steer` slope [CI] / effect / iia | 0.49 [0.45, 0.54] / 0.44 / 0.54 | 0.78 [0.74, 0.82] / 0.59 / 0.63 | 0.84 [0.81, 0.88] / 0.66 / 0.67 |
| `das-steer` CE gap | +0.034 | +0.051 | +0.065 |
| `das` CE gap (`das-shuf`) | +0.002 (+0.000) | +0.004 (+0.001) | +0.005 (+0.002) |

**Second `ood-combo` seed** (`ckpt/num_ood-combo_s1.pt`, validation loss 0.2108; `SPLIT=combo`, L4):

| | held-out combination: effect / iia | seen: effect / iia |
|---|---|---|
| `das-steer` | 0.87 / 0.85 | 0.72 / 0.71 |
| `ablate` | 0.00 / 0.19 | 0.14 / 0.38 |
| `das` | 0.05 / 0.25 | 0.18 / 0.40 |

This replicates seed 0 (held out 0.86 / 0.87). The src slope and CE gap are uninformative on held-out combo rows: B and B′ both have m in [3, 5], so v_cf(B) − v_cf(B′) is tiny. Read effect and IIA there.

**`compose`** (`ckpt/num_compose.pt`: `num`, 45k steps, seed 0, trained on frictionless episodes with forces plus episodes with friction and no force; validation loss 0.1410). `SPLIT=compose`, L4, 300 held-out pairs (A has friction and force) and 300 seen pairs (A frictionless):

| | held out: effect / iia / slope | seen: effect / iia / slope |
|---|---|---|
| `compose` model, `das-steer` | 0.82 / 0.74 / 0.85 | 0.91 / 0.90 / 0.87 |
| `compose` model, clean target CE / clean iia | 1.687 overall / 0.29 | 0.02 |
| iid model, `das-steer` | 0.87 / 0.72 / 0.92 | 0.91 / 0.86 / 0.92 |

Next-token loss per token on the `compose` parts (`probe_num_compose_mean.log`): train part moving 0.209, stuck 0.006; **test part moving 0.765, stuck 1.079**.

Reading:
- **Write effects generalize across channel and parameter.** The written mass works in `nat` as in `num` (L5 slope 0.92), and written friction works too (0.84). The model's dynamics read both hidden parameters from a 32-D L4–L5 subspace.
- **The read side stays weak everywhere.** It is statistically real for mass in `num` (all 3 CIs exclude 0 and the control), but small: slope 0.01–0.14. It is about 0 for `nat` and for mu. The parameters are not stored as swappable low-dimensional variables.
- **Computation, not lookup, on 2 seeds** (`ood-combo`).
- **No compositional reuse** (`compose`, 1 seed):
  - The model learned friction only from coasting and force only without friction, and fails on both together. Moving-span loss is 3.7× its training level, and it has no notion of stiction under a force.
  - The mass write still works on these episodes, but the counterfactual difference (F/m_B − F/m_A)·Δt doesn't involve mu. So this tests the F/m pathway only, which was trained.
  - The operator reuses its parts for new values (combo) but not for new combinations of mechanisms (compose).

## Dataset splits (built 2026-10-02)

`sim.SPLITS`, `sim.split_episodes(split, part, n, start_seed)`; `python train.py <ch> <n> <steps> <seed> <split>` trains on the train part only; `python probe.py <ckpt> <n> <pool> <split>` fits probes on the train part and scores them on the test part, and prints next-token loss on both parts.

| split | train part | test part | seeds kept (train / test) | m identifiable in test |
|---|---|---|---|---|
| `iid` | default | default | 100% / 100% | 83% |
| `ood-combo` | everything outside the test region | m in [3, 5] and some \|F\| ≥ 7 | 85% / 13% | 80% |
| `ood-extrap` | default (m 0.5–5) | m in (5, 8] | 100% / 100% | **45%** (heavy carts stick more) |
| `compose` | mu = 0, or F = 0 throughout (50/50) | mu > 0 and some F ≠ 0 | 100% / 100% | 83% (train part: 51%) |

- Scores on OOD splits are **1 − MSE / Var_train** (targets standardized with the train part's stats), not R². Plain R² on a narrow test region such as m in [3, 5] divides by that region's small variance and goes to −30 even for decent predictions. 1 means perfect; 0 means an error as large as the training spread. Predicting the train mean scores below 0 on a shifted region.
- `ood-extrap` needs no retraining: existing iid models never saw m > 5 (done; see Results). `ood-combo` is trained on 2 seeds (`ckpt/num_ood-combo.pt`, `ckpt/num_ood-combo_s1.pt`) and `compose` on 1 (`ckpt/num_compose.pt`); see "Interventions, rounds 2–3".
- Not built: the `cross-channel` split. It needs mixed-channel training in `train.py`, and probes fit on one channel and scored on another.

## Known issues and open questions

- Only `num` and `nat` have 3 seeds. `sym`, `rel` and `qual` have one seed each.
- The mu ceiling has not saturated even at 200k episodes (0.28 → 0.57 → 0.67). A closed-form or least-squares mu estimator from the stated trajectory would give a firmer reference.
- Moving-span loss on unseen masses rises 10–14%, but without a reference model trained on m up to 8, this can't be separated from intrinsically harder dynamics. That needs a new wide-mass training option (not built).
- `qual` repeats "It stays still." during long stuck stretches, which inflates token counts (a length confound). `qual` also reports "a gentle push starts" when the force changes within the same bin (intended information loss; note it in the paper).
- `rel` validation loss plateaus by 15k steps and rises slightly by 45k (mild overfitting).
- The citations in `project_idea.md` must be verified, especially arXiv 2607.27017, Paperlayer 2607.20058, and the PhysLang characterization.
- Related work missing from `project_idea.md`: Vafa et al. 2025 ("What has a foundation model found?", an inductive-bias probe on orbital mechanics; the closest prior work), Vafa et al. 2024 (world-model evaluation metrics), Li, Nye, Andreas 2021 (implicit entity state in text), and Othello-GPT.
- Interventions:
  - The DAS read effect is small (CE gap 0.007–0.018 across seeds, CIs exclude 0).
  - On held-out `combo` rows, the src slope and CE gap are uninformative (narrow source range). Use effect and IIA there.
  - The mass write test on `compose` doesn't involve mu, so it can't show composition. A mu write test on `compose` would.
  - `das-steer` at k = 8 failed to train once (`BSRC=fc`, L4: loss above `das`), so check training losses before reading a row.
  - Greedy decoding of `num` sometimes emits an unexpected step; about 5–10% of pairs are dropped per condition.

## Not built yet

- A mu write test on the `compose` model, and a second `compose` seed.
- Interventions for the `nat` `ood-combo` / `compose` designs (needs `nat` models trained on those splits).
- A wide-mass reference model for `ood-extrap`.
- The `cross-channel` split (mixed-channel training, with probes fit on one channel and scored on another).
- The `nat` ablations: `nat-noterm`, `nat-notime`, `nat-nocause`, `nat-short`.
- Phase 2: collisions.
