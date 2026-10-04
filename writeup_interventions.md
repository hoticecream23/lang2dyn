# Do language models trained on trajectory text use the physics they encode? — intervention results (draft)

Draft of the operator half of the project: interchange interventions on small char-level GPTs trained from scratch on verbalized 1-D cart dynamics. Numbers come from `PROJECT_STATE.md` (intervention sections, rounds 1–3); logs are local (`das_*.log`, `r2_*.log`, `r3_*.log`, `out/*.npz`). Status: first draft, 2026-10-04.

## 1. Question and setup

The state-recovery results show that models trained only on text compute unstated physical parameters and store them linearly decodable (`num`: log m probe R² 0.71 vs 0.02 from the stated numbers). Decodability does not show use. Here we ask whether the model's next-token computation *reads* a mass (or friction) variable, and whether that computation generalizes the way a physical operator should.

**Models.** 6-layer, d = 256 char-level GPTs (about 5M parameters), 100k episodes, 45k steps. Channels: `num` (numbers) and `nat` (the same trajectories in words). Seeds 0–2 for `num`.

**Counterfactual pairs.** Episode A is cut at a span where a new nonzero force starts (step r0). The next span's velocity then requires the new acceleration a = F/m − g·s·μ, so it cannot be extrapolated from the recent Δv. The target is the simulator run of A with episode B's mass (or friction) from r0 on (`m_switch` / `mu_switch`). The texts agree up to r0.

**Interventions** (model frozen; layer L; k-dim orthonormal subspace U; all positions unless stated). A's coordinates in U are overwritten with a value c:
- **interchange (DAS):** U is learned so the patched model predicts the counterfactual span; c is B's own activation (mean over its last span) projected on U.
- **write (steer):** U, α, β are learned; c = α·f(z_B) + β, with f = log m or μ. This writes the true value rather than B's representation.
- **controls:** ablation (c = mean), DAS trained with shuffled sources, a random subspace, the ridge probe direction, and the same procedure on a random-init model.

**Metrics** (held-out pairs):
- the greedy-decoded velocity at the next step: effect = (v_patched − v_clean)/(v_cf − v_A), and IIA;
- **source-difference metrics**: the Theil–Sen slope of v_patched(B) − v_patched(B′) against v_cf(B) − v_cf(B′), and the teacher-forced CE gap between sources B′ and B.

Erasing A's value already moves predictions toward a random B (regression to the mean), so effect and IIA alone overstate interchange success. The source-difference metrics are 0 for any source-independent change.

## 2. Results

**R1. The probe direction is causally inert.** Patching along the ridge log m direction (decodable, R² ≈ 0.7) changes predictions no more than a random direction of the same norm, at any layer, from ×1 to ×100. Learned directions have |cos| ≤ 0.13 with it.

**R2. There is no 1-D mass variable.** With k = 1, interchange and write both fail at every layer (source slope ≈ 0).

**R3. A write channel at L4–L5.** Writing the true value into a 32-D subspace makes the model produce the counterfactual velocity.

| model / target | layer | slope [95% CI] | effect / IIA | CE gap |
|---|---|---|---|---|
| `num` seed 0, mass | L4 | 0.91 [0.88, 0.93] | 0.85 / 0.78 | +0.172 |
| `num` seed 1, mass | L4 | 0.67 [0.63, 0.70] | 0.59 / 0.62 | +0.099 |
| `num` seed 2, mass | L4 | 0.84 [0.82, 0.87] | 0.77 / 0.73 | +0.154 |
| `nat`, mass | L5 | 0.92 [0.90, 0.94] | 0.86 / 0.82 | +0.051 |
| `num`, friction | L5 | 0.84 [0.81, 0.88] | 0.66 / 0.67 | +0.065 |
| random-init model, mass | L4 | (no decode) | | +0.001 |

- **The model combines the written value with F.** The written value depends only on z_B, but the required change depends on the sign and size of F in each pair.
- **It works through attention.** Patching only the prompt span (not the generated tokens) still works (slope 0.45, r 0.72 at k = 8; round-1 slope-through-0 metric).
- **It interpolates** to mass values never used in training (held-out band [2, 3] kg).

**R4. The natural value is not swappable.** Interchange from B's own activations transfers little:

| | CE gap [95% CI] | shuffled control | slope [CI] |
|---|---|---|---|
| `num` seed 0 | +0.018 [+0.015, +0.022] | +0.002 | 0.14 [0.11, 0.17] |
| `num` seed 1 | +0.010 [+0.008, +0.013] | +0.001 | 0.01 [0.00, 0.02] |
| `num` seed 2 | +0.007 [+0.004, +0.009] | −0.001 | 0.01 [0.00, 0.03] |
| `nat` (L4) | +0.002 | +0.000 | 0.00 |
| friction (L4) | +0.004 | +0.001 | 0.00 |

- The effect is statistically real for mass in `num`, but small. It grows with subspace width (k = 64: slope 0.28) and needs every position patched.
- The subspace the model reads (found by the write) overlaps the DAS subspace at 0.99, and B's natural activations in it do carry mass (R² 0.48 at k = 8). But the write uses about 5× the natural spread.
- **Reading:** the parameter is represented redundantly, across positions and many dimensions. The subspace the model reads can be driven, but a natural-scale swap within it can't override the remaining copies.

**R5. Computation, not lookup.** `num` models trained without the combination m ∈ [3, 5] with |F| ≥ 7 (`ood-combo`; 2 seeds). Writing m_B in that range under such a force:

| | held-out combination: effect / IIA | seen: effect / IIA | ablation, held out |
|---|---|---|---|
| `ood-combo` seed 0 | 0.86 / 0.87 | 0.78 / 0.75 | −0.01 / 0.24 |
| `ood-combo` seed 1 | 0.87 / 0.85 | 0.72 / 0.71 | 0.00 / 0.19 |
| iid model (saw it) | 0.88 / 0.88 | 0.79 / 0.78 | −0.14 / 0.21 |

On real held-out-combination episodes, the `ood-combo` model's moving-span loss is 4% above the iid model's (0.272 vs 0.262).

**R6. Weak compositional reuse.** `num` models (2 seeds) are trained on frictionless episodes with forces plus friction-only coasting episodes (`compose`), and tested on friction and force together.
- **Behaviour fails on both seeds.** Moving-span loss is 3.7× / 3.1× the training level (0.765 vs 0.209; 0.670 vs 0.218). The iid model scores 0.264 on both parts, so the test episodes are not intrinsically harder. Stuck-span loss rises from 0.006 to 1.08 / 0.93: the models never saw stiction holding a cart against a force.
- **Mass write:** it still works there (0.82), but that tests only the trained F/m pathway, since μ cancels in the mass counterfactual.
- **Friction write:** writing friction under a force is the untrained combination. Source slope [95% CI]:

| model | L4 | L5 |
|---|---|---|
| iid model | 0.65 [0.61, 0.69] | 0.78 [0.75, 0.82] |
| `compose` seed 0 | 0.36 [0.32, 0.40] | 0.56 [0.51, 0.60] |
| `compose` seed 1 | 0.32 [0.26, 0.38] | 0.24 [0.20, 0.29] |

The friction variable reaches the force dynamics, but at only 0.3–0.7× the strength of a model trained on the combination.

## 3. Interpretation

Next-token prediction on trajectory text yields an internal dynamics computation that reads hidden parameters: written values propagate correctly through F-dependent updates and interpolate to unseen values and unseen value combinations. The parameters are not stored as a single swappable variable. The probe-decodable direction is not the one the computation uses, and natural representations are distributed and redundant. The learned operator generalizes over *values* (lookup is ruled out), but only weakly over *mechanisms*. Friction and force learned in separate regimes are combined at reduced strength, and behaviour on the combination fails.

## 4. Caveats

- **Small models, one architecture, one simulator.** The k = 32 subspace is about 1/8 of the residual width.
- **The write effects use off-manifold amplitudes**: about 5× the natural spread at k = 8.
- **`compose` and `ood-combo` are single-channel (`num`), 2 seeds each.**
- **The held-out regions interpolate within the trained ranges** (a new combination, not new values). `ood-extrap` (new values) is tested only behaviourally.
- **DAS can find directions in any network.** The shuffled-source and random-init controls guard against this here.
- **Related work** (to verify before citing): Vafa et al. 2025 on inductive-bias probes of orbital mechanics; Othello-GPT; Li, Nye, Andreas 2021; Geiger et al. on DAS / causal abstraction.

## 5. Figures

Made by `python make_figures.py` from the intervention logs (local) into `figures/` (PDF and PNG).

1. `figures/fig1_pair.png`: an example counterfactual pair. A's velocity, the force change at r0, the simulator's target (A with 3 kg from r0), and the step the model decodes.
2. `figures/fig2_write_vs_read.png`: write vs read vs shuffled control, for 3 `num` seeds (mass), `nat` (mass) and friction. Left: source slope (Theil–Sen 95% CI). Right: CE gap (bootstrap 95% CI). (R3, R4.)
3. `figures/fig3_layers.png`: write slope by layer (L3–L5) for `nat` mass and `num` friction; the read is about 0 at every layer.
4. `figures/fig4_combo.png`: `ood-combo`. Write effect on held-out vs seen pairs for 2 seeds and the iid model, with ablation as baseline. (R5.)
5. `figures/fig5_compose.png`: `compose`. Left: moving-span loss on the train vs test part for the iid model and both `compose` seeds. Right: friction-write source slope at L4 / L5. (R6.)
