# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md`. Read `Simulator_Verbalizer_Spec.md`, `sim.py`, `train.py` and `probe.py` only for the parts the next task touches.

## Where we are

- The idea has been reviewed and the key design decisions are made (see `PROJECT_STATE.md`, "Decisions made").
- Spec v0 is written.
- The phase-1 simulator plus the `num`, `qual`, `nat`, `rel` and `sym` verbalizers work. `python sim.py` passes all checks.
- Identifiability is flagged separately for m and mu.
- **Run 2 (`num`, 15k steps, RunPod RTX 4090) passed the decision gate for mass.** A linear probe on the trained model decodes log_m at R² 0.61, while a linear probe on the stated numbers gets 0.02 and the nonlinear ceiling is 0.64. Acceleration and mu are not evidence of computation (see `PROJECT_STATE.md`, "Results").
- **Runs 3–5, the first channel comparison at equal training (45k steps, mean pooling):** log_m `num` 0.72 vs `nat` 0.60; mu 0.41 vs 0.32. Both channels have the same information ceiling (log_m 0.80), so language costs about 0.1 of extraction. Probe position was ruled out.
- **3 seeds each for `num` and `nat` confirm the language cost** (mean ± sd, mean pooling): log_m 0.707 ± 0.011 vs 0.613 ± 0.017, and mu 0.385 ± 0.038 vs 0.298 ± 0.045. At the end token the mu gap is clearer: 0.296 vs 0.189, t 11.5.
- **Run 7 (`rel`) passed the negative control:** the trained probe (log_m 0.21) stays below the `rel` ceiling (0.27). The fraction of available mass information extracted: `num` about 0.90, `nat` about 0.75, `rel` about 0.79.
- **Run 6 (`sym`) passed the sanity check** (log_m 0.70 vs `num` 0.72, mean pooling). It sets a noise floor: about 0.02 on log_m and up to 0.07 on mu. The `nat` mass gap is real; the mu gap needs more seeds.
- **Stronger ceiling** (`probe.ceiling`, GPU MLP): log_m about 0.80, mu at least 0.67. Both channels sit well below it, mu most of all.
- Training resumes from checkpoints. Local checkpoints: `ckpt/num_run1.pt`, `ckpt/num_run2.pt`, `ckpt/nat_run1.pt`, `ckpt/nat_45k.pt`, `ckpt/num_45k.pt`, `ckpt/sym_45k.pt`, `ckpt/rel_45k.pt`, `ckpt/qual_45k.pt`, `ckpt/{num,nat}_s{1,2}.pt` (gitignored).

## Compute: RunPod

- SSH key: `~/.ssh/id_ed25519_runpod` (registered in RunPod settings).
- Code lives in `/workspace/lang2dyn` on the pod. The pod uses an externally managed Python, so install with `pip install --break-system-packages scikit-learn psutil`. torch comes with the template.
- Set `OMP_NUM_THREADS=8` (the pod sees 256 host cores but only has 8 vCPUs).
- Copy code: `git archive HEAD sim.py train.py probe.py requirements.txt | ssh -p <PORT> -i ~/.ssh/id_ed25519_runpod root@<IP> 'mkdir -p /workspace/lang2dyn && tar -x -C /workspace/lang2dyn'`
- Run long jobs inside `tmux` and copy results back with `scp -P <PORT>`.
- Use the "SSH over exposed TCP" address (root@IP -p PORT). The proxied `ssh.runpod.io` address can't copy files.
- The IP and port change with each pod. Delete pods when you're done (stopped pods still bill for disk).

## Next steps, in order

1. Optional: a closed-form mu estimator from the trajectory, to get a true ceiling for mu.
2. **Generalization runs.** `ood-extrap` on the existing models is done: probes are ill-posed beyond the training range; moving-span loss degrades 10–14%. Next: train a reference `num` on m up to 8, to calibrate that loss ratio. Train `num` on `ood-combo` and `compose`, and judge them by moving-span loss on the test part (the probe scores are secondary).
3. Implement the interchange interventions using `make_episode(seed, m=...)` counterfactual twins. The mass representation at layer 6 is the first target.

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then start on next step 1.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (decisions, facts, results, issues, not-built list) and the "Where we are", "Compute" and "Next steps" sections of this file.
