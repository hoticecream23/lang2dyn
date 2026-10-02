# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md`. Read `Simulator_Verbalizer_Spec.md`, `sim.py`, `train.py` and `probe.py` only for the parts the next task touches.

## Where we are

- The idea has been reviewed and the key design decisions are made (see `PROJECT_STATE.md`, "Decisions made").
- Spec v0 is written.
- The phase-1 simulator plus the `num` and `qual` verbalizers work. `python sim.py` passes all checks.
- Identifiability is now flagged separately for m and mu.
- The `num`-only pipeline runs end to end. With an MLP probe, run 1 decodes the unstated variable a at R² 0.80 and log_m at 0.51. mu stays weak at 0.14. The observables baselines (the key comparison) have not finished a run yet.
- **We are moving training to cloud compute.** The local run 2 (15k steps) was stopped at step 7000. Only `ckpt/num_run1.pt` and `ckpt/num.pt` exist locally (identical, both gitignored).

## Next steps, in order

1. **Cloud setup.**
   - Done: checkpoint/resume in `train.py` (atomic save every 500 steps to `ckpt/<channel>_state.pt`; rerun the same command to resume) and `requirements.txt`.
   - Push the repo to a remote the cloud machine can pull from.
2. **Run 2 on cloud:** `python train.py num 100000 15000`, then `python probe.py ckpt/num.pt 3000`. Record the results in `PROJECT_STATE.md`.
   - The key check: does the trained model's probe clearly beat the observables baseline on a and log_m? If yes, the signal is real, so move to step 3. If no, fix the probing method first.
3. Add the `nat`, `rel` and `sym` verbalizers, each with a parser and a round-trip check in `_checks()`. `nat` will need a BPE tokenizer.
4. Build the dataset splits (`ood-combo` holds out m in [3, 5] × |F| in [7, 10]).
5. Implement the interchange interventions using `make_episode(seed, m=...)` counterfactual twins.

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then start on next step 1.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (decisions, facts, results, issues, not-built list) and the "Where we are" and "Next steps" sections of this file.
