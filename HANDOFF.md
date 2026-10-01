# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md`. Read `Simulator_Verbalizer_Spec.md` and `sim.py` only for the parts the next task touches.

## Where we are

- The idea has been reviewed and the key design decisions are made (see `PROJECT_STATE.md`, "Decisions made").
- Spec v0 is written.
- The phase-1 simulator plus the `num` and `qual` verbalizers work, and `python sim.py` passes all checks (84% of random episodes identifiable).

## Next steps, in order

1. **`num`-only pipeline test.** Generate data, train a small GPT from scratch on `num` text, then fit linear probes at span-end tokens for x, v, a, m, mu. The goal is to confirm the pipeline end to end before building more channels.
   - Decide first: tokenizer (shared BPE with digits split, or char-level to start) and framework (nanoGPT-style PyTorch is the likely default).
   - Probe targets m and mu only on `identifiable == true` episodes. The non-identifiable episodes are the control.
2. Add the `nat`, `rel` and `sym` verbalizers, each with a parser and a round-trip check in `_checks()`.
3. Build the dataset splits (`ood-combo` holds out m in [3, 5] × |F| in [7, 10]).
4. Implement the interchange interventions using `make_episode(seed, m=...)` counterfactual twins.

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then start on next step 1 (the num-only pipeline test). Before writing code, ask me about the tokenizer and framework choice if it's still open.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (decisions, facts, issues, not-built list) and the "Where we are" and "Next steps" sections of this file.
