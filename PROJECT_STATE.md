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
- `identifiable(tr)`: in the kinetic regime a = F·(1/m) − g·s·mu, so m and mu are both identifiable iff two observed (F, s) rows are linearly independent.
- `make_episode(seed, **changes)` builds a counterfactual twin (for example `m=4.0`), with `cf_of` pointing to the base id.
- The surface template RNG is seeded with `f"{seed}-surface"`.
- Commands: `python sim.py` runs the self-checks. `python sim.py N out.jsonl` writes N episodes.

## Known issues and open questions

- `qual` repeats "It stays still." during long stuck stretches. This inflates token counts (the length confound). It could be collapsed into one sentence.
- `qual` reports "a gentle push starts" when the force changes within the same bin. This is intended information loss, but worth noting in the paper.
- The citations in `project_idea.md` must be verified, especially arXiv 2607.27017, Paperlayer 2607.20058, and the PhysLang characterization.
- Related work missing from `project_idea.md`: Vafa et al. 2025 ("What has a foundation model found?", inductive-bias probe on orbital mechanics; the closest prior work), Vafa et al. 2024 (world-model evaluation metrics), Li, Nye, Andreas 2021 (implicit entity state in text), and Othello-GPT.

## Not built yet

- Verbalizers: `nat`, `rel`, `sym`, and the ablations `nat-noterm`, `nat-notime`, `nat-nocause`, `nat-short`.
- Dataset splits: `iid`, `ood-combo`, `ood-extrap`, `compose`, `cross-channel`. The `ood-combo` and `ood-extrap` splits need param-region filtering in `sample_params` or at dataset-build time.
- Tokenizer (shared BPE, digits split), training script, probes, and the interchange intervention code.
- Phase 2: collisions.
