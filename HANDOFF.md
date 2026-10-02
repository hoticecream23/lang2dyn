# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md`. Read `Simulator_Verbalizer_Spec.md`, `sim.py`, `train.py` and `probe.py` only for the parts the next task touches.

## Where we are

- The idea has been reviewed and the key design decisions are made (see `PROJECT_STATE.md`, "Decisions made").
- Spec v0 is written.
- The phase-1 simulator plus the `num` and `qual` verbalizers work, and `python sim.py` passes all checks (84% of random episodes identifiable).
- The `num`-only pipeline runs end to end: char-level GPT, a 15-minute training run, probes. In run 1, the unstated variables a and m decode far above the random-init baseline. The probe ceiling and the identifiability control both still need fixing. Numbers are in `PROJECT_STATE.md`, "Results".
- `ckpt/num.pt` exists locally (gitignored). Regenerate it with `python train.py num 100000 5000`.

## Next steps, in order

1. **Make run 1's measurements trustworthy.**
   - Split `identifiable` into separate flags for m and mu in `sim.py`. A coast identifies mu without m.
   - Add an MLP probe and an observables baseline to `probe.py`. The baseline regresses the targets from ground-truth stated values (x, v, F history up to each span).
   - Train longer (for example 15k steps), since loss was still falling.
2. Add the `nat`, `rel` and `sym` verbalizers, each with a parser and a round-trip check in `_checks()`. `nat` will need a BPE tokenizer.
3. Build the dataset splits (`ood-combo` holds out m in [3, 5] × |F| in [7, 10]).
4. Implement the interchange interventions using `make_episode(seed, m=...)` counterfactual twins.

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then start on next step 1.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (decisions, facts, results, issues, not-built list) and the "Where we are" and "Next steps" sections of this file.
