# Simulator and Verbalizer Spec (v0)

Goal: one simulator produces ground-truth trajectories `z_{1:T}`. Several verbalizers `g_i` turn the **same** trajectory into different texts `L_i`. Models train only on text. Every token links back to its ground-truth state, so we can probe and intervene.

---

## 1. Simulator

### Phase 1: one cart, applied force, friction (first paper)

1D track. Per-episode parameters:

| Param | Range | Notes |
|---|---|---|
| `m` | 0.5–5 kg, log-uniform | hidden in most channels |
| `mu` | 0–0.3 | kinetic = static, for simplicity. Was 0–0.5; narrowed because stiction swallowed too many segments (71% identifiable) |
| `x0` | 0–10 m | |
| `v0` | -3–3 m/s | |
| `F(t)` | piecewise constant, 2–4 segments, each in [-10, 10] N | segment lengths random |
| `g` | 9.8 | fixed |

Dynamics (integrate at `dt = 0.01`, record every `0.1` s, `T = 5` s, so 50 recorded steps):

```
moving (v != 0):   a = (F - mu*m*g*sign(v)) / m
stuck  (v == 0):   if |F| <= mu*m*g: a = 0
                   else:             a = (F - mu*m*g*sign(F)) / m
if v changes sign within a substep while friction opposes it: set v = 0, become stuck
```

Friction sign changes and stiction make the operator nonlinear. Without them, `f_physics` is a linear map, and the model could fake it by interpolation.

**Identifiability (design constraint):**
- Under a single constant force, `a = F/m - mu*g`. Then `m` and `mu` **cannot both** be identified.
- Two force levels, or one coasting segment (`F = 0`, where deceleration is `mu*g` regardless of mass), make both identifiable.
- Most episodes therefore get at least 2 force segments or a coast.
- Keep about 10% deliberately non-identifiable episodes as a control. On those, probes *should* fail to recover `m` and `mu` separately.

### Phase 2: two carts with collisions (later)

- Add a second cart and a restitution coefficient `e` in [0, 1].
- Collision update: `v1' = (m1 v1 + m2 v2 + m2 e (v2 - v1)) / (m1 + m2)`, and the symmetric formula for `v2'`.
- Collisions reveal the mass *ratio* even when no force is applied. This gives a second, independent route for inferring mass.

### Logged ground truth (per recorded step)

`x, v, a, F, friction, regime (moving|stuck), events (list of force_change|start|stop)`, plus episode constants `m, mu`, plus the seed.

---

## 2. Verbalizers

Every verbalizer is a deterministic function `g_i(trajectory, surface_seed)`. `surface_seed` selects among about 5–20 templates per event type, so the model cannot memorize a single template. Sentences come out in time order.

**When a sentence is emitted:** at every event (force change, start, stop), plus every `k` steps (`k = 5`, or every 0.5 s) as a periodic report.

### Channels

| ID | Channel | Example |
|---|---|---|
| `num` | Explicit numeric | `t=0.0 x=1.20 v=0.00 F=+4.0 . t=0.5 x=1.38 v=0.71 F=+4.0 .` |
| `nat` | Ordinary language, rounded numbers | `A cart rests at 1.2 m. A 4 N push to the right begins. Half a second later it is moving at 0.7 m/s.` |
| `qual` | Qualitative bins only | `A heavy cart sits still. A gentle push to the right starts. It speeds up slowly.` |
| `rel` | Relational, comparing across time | `The push grows stronger than before. The cart now moves faster than it did earlier.` |
| `sym` | Symbolic control | `num` content rewritten with a random bijective vocabulary (`t=0.0` becomes `q7 0.0`, and so on) |

`sym` matters only for pretrained models, because it removes the meaning that English words carry. For from-scratch models it should match `num`, and that equality is itself a sanity check.

### Ablation transforms (applied to `nat`)

| ID | Transform |
|---|---|
| `nat-noterm` | Replace physics words (`velocity`, `mass`, `force`, `accelerate`) with neutral words (`pace`, `bulk`, `push`, `pick up`) or remove them |
| `nat-notime` | Replace timestamps with `then` / `later`, and use irregular emission gaps |
| `nat-nocause` | Remove causal words: `a push begins, so it speeds up` becomes `a push begins. it speeds up.` |
| `nat-short` | Keep only every 3rd sentence, plus the events |

### Qualitative bins (`qual`)

- mass: light < 1, medium 1–2.5, heavy > 2.5
- force: gentle < 3, firm 3–7, hard > 7
- speed: still, slow, steady, fast
- change: speeds up slowly/quickly, slows, stops, stays still

### Information map (the independent variable)

S = stated, I = inferable from the dynamics, B = binned, A = absent/not identifiable.

| Var | num | nat | qual | rel | nat-notime |
|---|---|---|---|---|---|
| x | S | S (0.1 m) | A | A | S |
| v | S | S (0.1 m/s) | B | relative only | S |
| a | I | I | B (change word) | relative only | A (no dt) |
| F | S | S | B | relative only | S |
| m | I | I | B | A | A |
| mu | I | I | A | A | A |

The headline probes are the **I** cells: variables never written in the text but recoverable from the dynamics. Probes on **S** cells only measure token copying, so they serve as baselines.

---

## 3. Dataset

- Size: about 500k episodes per channel to start. Model: GPT, 6–8 layers, about 20–50M params, trained from scratch.
- Tokenizer: one BPE shared across all channels, with digits split into separate tokens.
- Report tokens per episode for each channel. Longer channels give the model more compute per episode, so this length difference is a confound.

### Splits

| Split | Construction | Tests |
|---|---|---|
| `iid` | random 5% | baseline |
| `ood-combo` | hold out the region m in [3, 5] × \|F\| in [7, 10] from training | operator vs lookup |
| `ood-extrap` | m in (5, 8] | extrapolation |
| `compose` | train on (F != 0, mu = 0) and (coast-only, mu > 0); test on both nonzero together | compositional reuse |
| `cross-channel` | mixed-channel model; train probe on channel i, test on channel j | channel-invariant state |

### Counterfactual pairs

For each test episode, regenerate it with **one** change and the same seed: `m`, `mu`, or one force segment value. These pairs provide the target outputs for interchange interventions. Patch the `m` representation from episode B into episode A, then check whether the continuation matches the simulator run of A with B's mass.

---

## 4. Episode record

```json
{
  "id": "ep_000123",
  "params": {"m": 2.1, "mu": 0.18, "x0": 1.2, "v0": 0.0,
             "F_segments": [[0.0, 4.0], [2.3, 0.0]]},
  "traj": {"t": [...], "x": [...], "v": [...], "a": [...],
           "F": [...], "regime": [...], "event": [...]},
  "identifiable": true,
  "texts": {
    "nat": {"text": "...", "spans": [[0, 34, 0], [35, 71, 5]]}
  },
  "cf_of": null, "cf_change": null
}
```

Each `spans` entry is `[char_start, char_end, step_index]`. Probes read the last token of each span. Without this alignment, probing is impossible.

### Parsers (for generation-based evaluation)

Each channel needs an inverse parser `text -> partial z`:
- `num` and `nat`: regex.
- `qual`: maps words back to bins.
- `rel`: maps to signs of change.

Evaluation flow: give the model a prefix, let it generate, parse the generated text, then compare against the simulator.

---

## 5. Build order

1. Simulator plus unit checks: momentum and energy budget, stiction cases, a non-identifiable case.
2. Build the `num` and `qual` verbalizers and their parsers. Check round trips: `parse(g(z))` must agree with `z` within bin resolution.
3. Train on `num` alone, then run the probes. This tests the whole pipeline end to end.
4. Add `nat`, `rel`, `sym`, and the ablation transforms.
5. Run counterfactual pairs and interchange interventions.
6. Phase 2 (collisions).
