# Simulator and Verbalizer Spec (v1, matches the code as of 2026-10-02)

Goal: one simulator produces ground-truth trajectories `z_{1:T}`. Several verbalizers `g_i` turn the **same** trajectory into different texts `L_i`. Models train only on text. Every span of text links back to its ground-truth state, so we can probe and intervene.

---

## 1. Simulator (`sim.py`)

### Phase 1: one cart, applied force, friction (first paper)

1D track. Per-episode parameters:

| Param | Range | Notes |
|---|---|---|
| `m` | 0.5–5 kg, log-uniform | hidden in every channel except `qual` (binned) |
| `mu` | 0–0.3 | kinetic = static, for simplicity. Was 0–0.5; narrowed because stiction swallowed too many segments (71% identifiable) |
| `x0` | 0–10 m | |
| `v0` | −3–3 m/s | |
| `F(t)` | piecewise constant, each segment in [−10, 10] N, rounded to 0.1 N | 10% of episodes have 1 segment (control); otherwise 2–4 segments, and 30% of those get one coast segment (F = 0). Segments start on 0.1 s boundaries |
| `g` | 9.8 | fixed |

Dynamics (substeps of 0.01 s, recorded every 0.1 s, T = 5 s, so 50 recorded steps):

```
moving (v != 0):   a = (F - mu*m*g*sign(v)) / m
stuck  (v == 0):   if |F| <= mu*m*g: a = 0
                   else:             a = (F - mu*m*g*sign(F)) / m
if v crosses zero inside a substep: stop exactly at the crossing, then re-evaluate friction for the rest of the substep
```

Friction sign changes and stiction make the operator nonlinear. Without them, `f_physics` would be a linear map, and the model could fake it by interpolation.

**Identifiability** (`identifiable(tr)` returns `{"m": bool, "mu": bool}`):
- In the kinetic regime, a = F·(1/m) − g·s·mu, with s the direction of motion. Both parameters are identifiable iff two observed (F, s) rows are linearly independent: two force levels, or a reversal of motion.
- A coast row (F = 0, a = −g·s·mu) identifies mu alone and says nothing about m.
- Under a single constant force in one direction, m and mu are confounded.
- About 83% of episodes have m identifiable. The rest are the control. Non-identifiable episodes still carry partial information (stiction bounds, the parameter range), so probes there are not expected to reach 0.

### Phase 2: two carts with collisions (not built)

- Add a second cart and a restitution coefficient `e` in [0, 1].
- Collision update: `v1' = (m1 v1 + m2 v2 + m2 e (v2 - v1)) / (m1 + m2)`, and the symmetric formula for `v2'`.
- Collisions reveal the mass *ratio* even when no force is applied, a second route for inferring mass.

### Logged ground truth (per recorded step)

`t, x, v, a, F, friction, regime ("moving" | "stuck"), events (list of "force_change" | "start" | "stop")`, plus episode constants `m, mu` in `params`.

---

## 2. Verbalizers

Each verbalizer is a deterministic function of the trajectory, plus a per-channel template RNG. Each sentence type has 2–3 phrasings, so the model can't memorize a single template. Sentences come out in time order.

**When a sentence is emitted:** at every event (force change, start, stop), plus every 5 recorded steps (0.5 s) as a periodic report. Each emitted step is one span.

### Channels (real output, episode seed 11, first three spans)

| ID | Channel | Example |
|---|---|---|
| `num` | Explicit numeric; x and v to 0.01, F exact | `t=0.0 x=1.42 v=0.23 F=+7.1 . t=0.5 x=2.53 v=4.20 F=+7.1 . t=1.0 x=5.62 v=8.16 F=+7.1 .` |
| `nat` | Ordinary language; x and v to 0.1, F exact when it changes, time only as gaps | `A cart is at the 1.4 m mark, moving right at 0.2 m/s. A 7.1 N push to the right is applied. 0.5 s later, it is now moving right at 4.2 m/s, at the 2.5 m mark. …` |
| `qual` | Qualitative bins only (states a mass bin) | `A light cart moves to the right at a slow pace. A hard push to the right starts. It speeds up quickly and now moves to the right at a fast pace. …` |
| `rel` | Comparisons with the previous sentence only; no numbers, no times | `A cart is moving. A push acts on it with its motion. Then, it moves faster than before. It is further right than before. …` |
| `sym` | `num` under a fixed random character substitution (spaces included) | `pnaiabdnoiembgnaimkbfnjciobibpnaisbdnmiskbgneimabfnjciobi…` |

`sym` matters only for pretrained models, because it removes the meaning that English words and digits carry. For from-scratch models it should match `num`, and that equality is a sanity check. It passed (run 6).

### Qualitative bins (`qual`)

- mass: light < 1, medium 1–2.5, heavy > 2.5 kg (stated once, in the first sentence)
- push: gentle < 3, firm 3–7, hard > 7 N, with direction left/right; or "Nothing pushes it" / "The push stops"
- speed: still < 0.05, slow < 1, steady < 3, fast ≥ 3 m/s, with direction
- change since the previous sentence: speeds up slowly/quickly, slows down gently/sharply (threshold 1 m/s²), keeps its pace (|Δspeed| < 0.1)
- events: starts moving, halts / comes to a stop, stays still

### Relational codes (`rel`)

- speed: faster / slower / about as fast as before (threshold 0.05 m/s), and "now the other way" on reversal
- position: further right / further left / about where it was (threshold 0.05 m)
- push: starts (with/against the motion) / stops / reverses / grows stronger / grows weaker
- motion events: comes to rest, starts moving, stays at rest
- first sentence: moving or at rest, and push none / present / with / against the motion

### Ablation transforms (applied to `nat`, not built)

| ID | Transform |
|---|---|
| `nat-noterm` | Replace physics words (speed, push, …) with neutral words, or remove them |
| `nat-notime` | Replace time gaps with "then" / "later" |
| `nat-nocause` | Remove causal words |
| `nat-short` | Keep only every 3rd sentence, plus the events |

### Information map (the independent variable)

S = stated, I = inferable from the dynamics, B = binned, R = relative only, A = absent / not identifiable.

| Var | num / sym | nat | qual | rel |
|---|---|---|---|---|
| t | S | S (as gaps) | A (sentence order only) | A (sentence order only) |
| x | S (0.01) | S (0.1) | A | R |
| v | S (0.01) | S (0.1) | B | R |
| a | I | I | B (change word) | R |
| F | S | S (on change) | B | R |
| m | I | I | B | A (partial: ceiling log_m 0.27) |
| mu | I | I | A (partial) | A (partial) |

The headline probes are the **I** cells: variables never written in the text but recoverable from the dynamics. Probes on **S** cells only measure token copying, so they serve as baselines.

---

## 3. Data and model

- 100k training episodes (seeds 0..99999) per channel. Validation: 2000 episodes from seed 10M. Probes: seeds 20M+. Ceiling: seeds 30M+. Code vocab for the `qual`/`rel` baselines: seeds 40M+.
- Model: char-level GPT from scratch, 6 layers, d = 256, 8 heads, about 5M params. One episode per sequence; context length set from the data (576 tokens for `num`, 1472 for `nat`).
- One character tokenizer shared by all channels (no BPE), so the tokenizer is not a confound between channels.
- Text length differs by channel (mean characters: `num` 351, `qual` 712, `rel` 736, `nat` 874). Longer channels give the model more compute per episode, so this is a confound to report.

### Splits (`sim.SPLITS`, `sim.split_episodes(split, part, n, start_seed)`)

| Split | Train part | Test part | Tests |
|---|---|---|---|
| `iid` | default distribution | default distribution | baseline |
| `ood-combo` | everything outside the test region | m in [3, 5] and some \|F\| ≥ 7 | operator vs lookup (new combination of seen values) |
| `ood-extrap` | default (m 0.5–5) | m in (5, 8] | extrapolation to unseen masses |
| `compose` | mu = 0, or F = 0 throughout (50/50) | mu > 0 and some F ≠ 0 | compositional reuse |
| `cross-channel` | mixed-channel model; probe fit on channel i, scored on channel j (not built) | | channel-invariant state |

OOD scores are 1 − MSE/Var_train. Model behavior is judged by next-token loss on moving spans.

### Counterfactual pairs

`make_episode(seed, m=…)` (or `mu=…`, `F_segments=…`) regenerates an episode with one change and the same seed. Such whole-episode twins differ from t = 0, so their texts don't share a prefix.

`make_episode(seed, m_switch=[r0, m])` changes the mass at recorded step r0: the state at r0 is unchanged, and the dynamics from r0 on use the new mass. Its text is identical to the original up to step r0. These are the targets for interchange interventions (`intervene.py`): patch a mass representation from episode B into episode A's prefix (up to r0), let the model continue, and compare with the simulator's A-with-B's-mass.

---

## 4. Episode record

```json
{
  "id": "ep_000123",
  "params": {"m": 2.1, "mu": 0.18, "x0": 1.2, "v0": 0.0,
             "F_segments": [[0.0, 4.0], [2.3, 0.0]]},
  "traj": {"t": [...], "x": [...], "v": [...], "a": [...], "F": [...],
           "friction": [...], "regime": [...], "events": [[], ["force_change"], ...]},
  "identifiable": {"m": true, "mu": true},
  "texts": {
    "num":  {"text": "...", "spans": [[0, 28, 0], [29, 57, 5]]},
    "nat":  {"text": "...", "spans": [...]},
    "qual": {...}, "rel": {...}, "sym": {...}
  },
  "cf_of": null, "cf_change": null
}
```

Each `spans` entry is `[char_start, char_end, step_index]`. Token i + 1 is character i (BOS at 0), so the span ends at token `char_end`. Counterfactual twins have ids like `ep_000123_cf_m4.0`. Split episodes have ids like `ood-combo_test_00001234` and a `split` field.

### Parsers

Every channel has an inverse parser, checked in `_checks()` against the trajectory for 500 episodes:
- `parse_num`, `parse_sym`, `parse_nat` → numbers (`nat` rebuilds time from the gaps).
- `parse_qual` → bins, direction, change word, event.
- `parse_rel`, `parse_rel_first` → comparison codes.

They are used for the `qual`/`rel` observables baselines, and later for generation-based evaluation (give the model a prefix, let it generate, parse the text, compare with the simulator).

---

## 5. Build status

1. ✅ Simulator plus checks: closed-form kinematics, stiction, coasting stop time and distance, reversal, identifiability cases.
2. ✅ `num`, `nat`, `qual`, `rel`, `sym` verbalizers with parsers and round-trip checks.
3. ✅ Training on each channel; probes with baselines and ceiling; 3 seeds for `num` and `nat`.
4. ✅ Dataset splits; `ood-extrap` evaluated with the existing models.
5. ⬜ Training on `ood-combo` / `compose`; a wide-mass reference model.
6. 🟨 Interchange interventions on counterfactual pairs: `intervene.py` (`num` only; probe directions, DAS, steering, controls).
7. ⬜ `nat` ablations, `cross-channel` split.
8. ⬜ Phase 2 (collisions).
