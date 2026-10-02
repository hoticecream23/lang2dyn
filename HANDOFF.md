# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md`. Read `Simulator_Verbalizer_Spec.md`, `sim.py`, `train.py` and `probe.py` only for the parts the next task touches.

## Where we are

- The idea has been reviewed and the key design decisions are made (see `PROJECT_STATE.md`, "Decisions made").
- Spec v0 is written.
- The phase-1 simulator plus the `num`, `qual` and `nat` verbalizers work. `python sim.py` passes all checks.
- Identifiability is flagged separately for m and mu.
- **Run 2 (`num`, 15k steps, RunPod RTX 4090) passed the decision gate for mass.** A linear probe on the trained model decodes log_m at R² 0.61, while a linear probe on the stated numbers gets 0.02 and the nonlinear ceiling is 0.64. Acceleration and mu are not evidence of computation (see `PROJECT_STATE.md`, "Results").
- **Runs 3–4 (`nat`):** mass is computed from language too. At 45k steps it nearly matches `num` at 15k (mean pooling: log_m 0.60 vs 0.67). Probe position was ruled out as the cause of the gap; undertraining explains most of it.
- **Stronger ceiling** (`probe.ceiling`, GPU MLP): log_m about 0.80, mu at least 0.67. Both channels sit well below it, mu most of all.
- Training resumes from checkpoints. Local checkpoints: `ckpt/num_run1.pt`, `ckpt/num_run2.pt`, `ckpt/nat_run1.pt`, `ckpt/nat_45k.pt` (gitignored).

## Compute: RunPod

- SSH key: `~/.ssh/id_ed25519_runpod` (registered in RunPod settings).
- Code lives in `/workspace/lang2dyn` on the pod. The pod uses an externally managed Python, so install with `pip install --break-system-packages scikit-learn psutil`. torch comes with the template.
- Set `OMP_NUM_THREADS=8` (the pod sees 256 host cores but only has 8 vCPUs).
- Copy code: `git archive HEAD sim.py train.py probe.py requirements.txt | ssh -p <PORT> -i ~/.ssh/id_ed25519_runpod root@<IP> 'mkdir -p /workspace/lang2dyn && tar -x -C /workspace/lang2dyn'`
- Run long jobs inside `tmux` and copy results back with `scp -P <PORT>`.
- Use the "SSH over exposed TCP" address (root@IP -p PORT). The proxied `ssh.runpod.io` address can't copy files.
- The IP and port change with each pod. Delete pods when you're done (stopped pods still bill for disk).

## Next steps, in order

1. **Train `num` for 45k steps too** (RunPod, about 15 min), so both channels are compared at convergence. Probe with `end` and `mean` pooling. Run 4 showed most of the `nat` gap was undertraining.
   - Consider checking convergence directly: is `nat` still improving at 45k? Validation loss went 0.0869 → 0.0833.
2. Add the `rel` and `sym` verbalizers, plus a binned observables baseline so `qual` can be probed.
3. Optional: a closed-form mu estimator from the trajectory, to get a true ceiling for mu.
4. Build the dataset splits (`ood-combo` holds out m in [3, 5] × |F| in [7, 10]).
5. Implement the interchange interventions using `make_episode(seed, m=...)` counterfactual twins. The mass representation at layer 6 is the first target.

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then start on next step 1.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (decisions, facts, results, issues, not-built list) and the "Where we are", "Compute" and "Next steps" sections of this file.
