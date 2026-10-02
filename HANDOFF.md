# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md`. Read `Simulator_Verbalizer_Spec.md`, `sim.py`, `train.py` and `probe.py` only for the parts the next task touches.

## Where we are

- The idea has been reviewed and the key design decisions are made (see `PROJECT_STATE.md`, "Decisions made").
- Spec v0 is written.
- The phase-1 simulator plus the `num`, `qual` and `nat` verbalizers work. `python sim.py` passes all checks.
- Identifiability is flagged separately for m and mu.
- **Run 2 (`num`, 15k steps, RunPod RTX 4090) passed the decision gate for mass.** A linear probe on the trained model decodes log_m at R² 0.61, while a linear probe on the stated numbers gets 0.02 and the nonlinear ceiling is 0.64. Acceleration and mu are not evidence of computation (see `PROJECT_STATE.md`, "Results").
- **Run 3 (`nat`):** mass is still computed (linear 0.39 vs 0.02 baseline), but less is extracted than with `num` (0.61), despite the same information ceiling (0.63). Confounds are not yet ruled out.
- Training resumes from checkpoints. Local checkpoints: `ckpt/num_run1.pt`, `ckpt/num_run2.pt`, `ckpt/nat_run1.pt` (gitignored).

## Compute: RunPod

- SSH key: `~/.ssh/id_ed25519_runpod` (registered in RunPod settings).
- Code lives in `/workspace/lang2dyn` on the pod. The pod uses an externally managed Python, so install with `pip install --break-system-packages scikit-learn psutil`. torch comes with the template.
- Set `OMP_NUM_THREADS=8` (the pod sees 256 host cores but only has 8 vCPUs).
- Copy code: `git archive HEAD sim.py train.py probe.py requirements.txt | ssh -p <PORT> -i ~/.ssh/id_ed25519_runpod root@<IP> 'mkdir -p /workspace/lang2dyn && tar -x -C /workspace/lang2dyn'`
- Run long jobs inside `tmux` and copy results back with `scp -P <PORT>`.
- Use the "SSH over exposed TCP" address (root@IP -p PORT). The proxied `ssh.runpod.io` address can't copy files.
- The IP and port change with each pod. Delete pods when you're done (stopped pods still bill for disk).

## Next steps, in order

1. **Rule out confounds in the `nat` vs `num` gap** (run 3: log_m linear 0.39 vs 0.61, with the same 0.63 ceiling):
   - Probe position: also probe at the token right after each number, or average over the span's tokens.
   - Compute: train `nat` longer (for example 45k steps), and check whether the gap closes.
2. Add the `rel` and `sym` verbalizers, plus a binned observables baseline so `qual` can be probed.
3. Optional: a closed-form mu estimator from the trajectory, to get a true ceiling for mu.
4. Build the dataset splits (`ood-combo` holds out m in [3, 5] × |F| in [7, 10]).
5. Implement the interchange interventions using `make_episode(seed, m=...)` counterfactual twins. The mass representation at layer 6 is the first target.

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then start on next step 1.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (decisions, facts, results, issues, not-built list) and the "Where we are", "Compute" and "Next steps" sections of this file.
