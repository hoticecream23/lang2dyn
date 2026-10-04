# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md` (start with "Headline findings so far" and "Decisions made"). Read `Simulator_Verbalizer_Spec.md`, `sim.py`, `train.py`, `probe.py` and `intervene.py` only for the parts the next task touches.

## Where we are (2026-10-04)

- Simulator, all five channels (`num`, `nat`, `qual`, `rel`, `sym`), training, probing, ceilings and dataset splits are built and checked (`python sim.py` passes).
- **State recovery is measured for every channel.** Models compute mass from text (`num` log_m 0.707 ± 0.011 vs 0.02 for a linear probe on the stated numbers). Language costs about 0.1 of extraction (`nat` 0.613 ± 0.017). Models stay well below the ceiling, mu most of all.
- **Operators (interventions, `intervene.py`; see headline 6 and the three intervention sections in `PROJECT_STATE.md`):**
  - **Write:** writing a hidden parameter into a 32-D L4–L5 subspace makes the model produce the right counterfactual, F-dependent velocities. Mass in `num`: slope 0.67–0.91 on 3 seeds. Mass in `nat`: 0.92. Friction: 0.84. Random-init control: none.
  - **Read:** swapping another episode's natural activations transfers little. Real for mass in `num` (CIs exclude 0) but slope 0.01–0.14; about 0 for `nat` and mu.
  - **Computation, not lookup:** `ood-combo` models (2 seeds) handle unseen mass × force combinations.
  - **No compositional reuse:** the `compose` model fails on friction and force together (moving loss 3.7×, stuck loss 0.006 → 1.08).
- **Generalization:** `ood-extrap` loss rises 10–14% on unseen masses (no wide-mass reference yet).
- Logs (`*.log`) and per-pair outputs (`out/*.npz`) are local only (gitignored).

### Local checkpoints (`ckpt/`)

| file | what |
|---|---|
| `num.pt` | copy of `num_run1.pt` (default for `probe.py`) |
| `num_run1.pt` | `num`, 5k steps (laptop) |
| `num_run2.pt` | `num`, 15k steps |
| `nat_run1.pt` | `nat`, 15k steps |
| `{num,nat,sym,rel,qual}_45k.pt` | 45k steps, seed 0 (the standard setting) |
| `{num,nat}_s{1,2}.pt` | 45k steps, seeds 1 and 2 |
| `num_ood-combo.pt`, `num_ood-combo_s1.pt` | `num`, 45k steps, seeds 0 and 1, `ood-combo` train part |
| `num_compose.pt` | `num`, 45k steps, seed 0, `compose` train part |

## Compute: RunPod

- Pods used: `47.47.180.106:16564`, `47.47.180.126:11742` (2026-10-03), `81.27.69.179:33861` (2026-10-04). Terminate any still up. To start one: RTX 4090 (about $0.75/h), an official RunPod PyTorch template, 30 GB container disk, a 20 GB volume disk at `/workspace`, **TCP port 22 exposed**. Use the "SSH over exposed TCP" address (`root@IP -p PORT`); the proxied `ssh.runpod.io` address can't copy files. The IP and port change with every pod.
- SSH key: `~/.ssh/id_ed25519_runpod` (already registered in RunPod settings).
- Set up the pod:
  - `git archive HEAD sim.py train.py probe.py intervene.py requirements.txt | ssh -p <PORT> -i ~/.ssh/id_ed25519_runpod root@<IP> 'mkdir -p /workspace/lang2dyn && tar -x --no-same-owner -C /workspace/lang2dyn'`
  - Then on the pod: `pip install --break-system-packages scikit-learn psutil` (the Python is externally managed; torch comes with the template).
  - `git archive HEAD` only ships committed files. For uncommitted work, use `tar -c <files> | ssh ... 'tar -x ...'` from the working tree. Copy the needed checkpoints the same way (`ckpt/num_45k.pt` etc.).
- Run jobs inside `tmux` with `OMP_NUM_THREADS=8` (the pod sees 256 host cores but only has 8 vCPUs). Write an exit marker to a `status` file, and copy results back with `scp -P <PORT>`.
- Timing on a 4090, 45k steps: `num`/`sym` about 18 min, `nat`/`rel`/`qual` about 35–40 min. Probes take about 3–6 min each, plus about 4 min with the default ceiling. `intervene.py` takes about 3–4 min per layer at 2000/400 pairs; 3–4 runs can share the GPU (use `OMP_NUM_THREADS=2` each).
- **Terminate** pods when done; stopped pods still bill for disk.

## Working agreements

- Heavy jobs run on the pod, not the laptop (it runs on battery). Ask before any heavy local job. Smoke tests of a few seconds are fine locally.
- State the cost before each paid launch, and get a yes.
- Commit only when asked. Never add Claude co-author or contributor lines.

## Next steps, in order

1. **Composition, properly:**
   - Run a mu write test on the `compose` model (`SPLIT=compose TARGET=mu`). Friction's counterfactual effect under a force is exactly the untrained combination.
   - Train a second `compose` seed (`python train.py num 100000 45000 1 compose`).
   - Optionally train a `nat` `compose` model.
2. **Why is the read weak?** Swapping in natural activations transfers little, though writing works. Options:
   - Patch several layers at once.
   - Source from B at matched token roles (`num` fields) rather than a span mean.
   - Learn a nonlinear read map (B's activations to c).
3. **Paper-facing write-up** of headline 6. The intervention story is now complete enough to draft: probe ≠ causal, a write channel, computation vs lookup, no composition.
4. Optional:
   - A wide-mass reference model for `ood-extrap`.
   - A firmer mu ceiling (least-squares estimator).
   - The `nat` ablations.
   - Verifying the citations in `project_idea.md`.

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then propose a plan for the next step before writing code.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (headline findings, decisions, code facts, results, issues, not-built list) and the "Where we are", "Compute" and "Next steps" sections of this file.
