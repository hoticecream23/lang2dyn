# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md` (start with "Headline findings so far" and "Decisions made"). Read `Simulator_Verbalizer_Spec.md`, `sim.py`, `train.py`, `probe.py` and `intervene.py` only for the parts the next task touches.

## Where we are (2026-10-03)

- Simulator, all five channels (`num`, `nat`, `qual`, `rel`, `sym`), training, probing, ceilings and dataset splits are built and checked (`python sim.py` passes).
- **State recovery (the first half of the question) is measured for every channel.** Models compute mass from text (`num` log_m 0.707 ± 0.011 vs 0.02 for a linear probe on the stated numbers). Language costs about 0.1 of extraction at equal information (`nat` 0.613 ± 0.017; 3 seeds each). Models stay well below the ceiling, mu most of all (≈ 0.3–0.4 vs ≥ 0.67). The `sym` sanity check and the `rel` negative control pass.
- **Generalization:** only `ood-extrap` is tested so far. Moving-span next-token loss rises 10–14% on unseen masses; probes can't extrapolate, so they are the wrong tool there.
- **Operator question, first pass (`intervene.py`, `num` 45k; details in `PROJECT_STATE.md`):**
  - The log_m probe direction is causally inert, and no 1-D mass variable exists.
  - A 32-D DAS subspace at L4, patched at every position, transfers part of another episode's mass effect: seed 0 src slope 0.31, r 0.53 (shuffled control 0.10). It is positive but weaker in seeds 1–2, and k = 64 reads more.
  - Writing the true log m into that kind of subspace gives F-dependent counterfactual velocities (slope about 0.7, r up to 0.9 on unseen masses).
  - Reading: mass is held distributed and redundantly across positions, and the dynamics computation reads it.
- `intervene.py`, `sim.py`'s `m_switch`, the spec and this file are **not committed yet** (2026-10-03 session). Logs `das_*.log` are local only (gitignored).

### Local checkpoints (`ckpt/`)

| file | what |
|---|---|
| `num.pt` | copy of `num_run1.pt` (default for `probe.py`) |
| `num_run1.pt` | `num`, 5k steps (laptop) |
| `num_run2.pt` | `num`, 15k steps |
| `nat_run1.pt` | `nat`, 15k steps |
| `{num,nat,sym,rel,qual}_45k.pt` | 45k steps, seed 0 (the standard setting) |
| `{num,nat}_s{1,2}.pt` | 45k steps, seeds 1 and 2 |

## Compute: RunPod

- Pod used on 2026-10-03: `root@47.47.180.106 -p 16564` (terminate it if it is still up). To start one: RTX 4090 (about $0.75/h), an official RunPod PyTorch template, 30 GB container disk, a 20 GB volume disk at `/workspace`, **TCP port 22 exposed**. Use the "SSH over exposed TCP" address (`root@IP -p PORT`); the proxied `ssh.runpod.io` address can't copy files. The IP and port change with every pod.
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

1. Optional: a closed-form or least-squares mu estimator from the stated trajectory, as a firmer mu ceiling (the GPU ceiling hasn't saturated: 0.28 → 0.57 → 0.67 at 5k / 50k / 200k episodes).
2. **Generalization runs:**
   - A wide-mass reference model (train on m up to 8) to calibrate the 10–14% `ood-extrap` loss rise. Not built: it needs a new split or option in `sim.py` / `train.py`.
   - Train `num` on `ood-combo` and on `compose` (`python train.py num 100000 45000 0 ood-combo`). Probe with `python probe.py ckpt/num_ood-combo.pt 3000 mean ood-combo`. Judge mainly by next-token loss on moving spans (printed by `probe.py` for OOD splits).
3. **Interventions, next round** (`intervene.py`; read "Interchange interventions on mass" in `PROJECT_STATE.md` first):
   - Controls and metric: DAS on the random-init model (k = 32, all positions, L4); a robust src slope (Theil–Sen); more test pairs, since the read effect is small and varies between seeds.
   - Lookup vs computation (decision 1): train `num` on `ood-combo`, then run `intervene.py` with test pairs whose A or B mass lies in the held-out region (needs a split option in `make_pairs`).
   - `nat` (needs a `nat` pair builder and parser in `intervene.py`; it is `num`-only now), then mu and F as targets.

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then propose a plan for the next step before writing code.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (headline findings, decisions, code facts, results, issues, not-built list) and the "Where we are", "Compute" and "Next steps" sections of this file.
