# Handoff

Read this file first in a new chat, then `PROJECT_STATE.md` (start with "Headline findings so far", "Decisions made" and "Paper readiness"). Read `Simulator_Verbalizer_Spec.md`, `sim.py`, `train.py`, `probe.py` and `intervene.py` only for the parts the next task touches. The write-up draft is `writeup_interventions.md`, and the draft related work is `related_work.md`.

## Where we are (2026-10-04)

- Simulator, all five channels (`num`, `nat`, `qual`, `rel`, `sym`), training, probing, ceilings, dataset splits and interventions are built and checked (`python sim.py` passes).
- **State recovery (the first half of the question)** is measured for every channel.
  - Models compute mass from text: `num` log_m 0.707 ± 0.011, vs 0.02 for a linear probe on the stated numbers.
  - Language costs about 0.1 of extraction at equal information (`nat` 0.613 ± 0.017).
  - Models stay well below the ceiling, mu most of all.
- **Operators (the second half): interventions with `intervene.py`.** See headline 6 and the five intervention sections (round 1 "Interchange interventions on mass", then rounds 2–5) in `PROJECT_STATE.md`.
  - **The probe direction is causally inert**, and there is no 1-D mass variable.
  - **Write:** writing a hidden parameter into a 32-D mid-layer subspace makes the model produce the right counterfactual, F-dependent velocities.
    - Mass in `num`: slope 0.67–0.91 on 3 seeds.
    - Mass in `nat`: 0.92. Friction: 0.84.
    - No effect in a random-init model.
    - Still works when bounded to the natural range: slope 0.89 at 1.06× the natural spread.
  - **Read:** swapping in another episode's natural activations at one layer transfers little. It is real for mass in `num` (CIs exclude 0), but the slope is only 0.01–0.14, and about 0 for `nat` and mu. Swapping at four layers at once (L2–L5) transfers 0.47: mass is held redundantly across layers.
  - **Scale:** a 38M model (12 layers) shows the same pattern: write 0.81 at L8, single-layer read about 0.
  - **Computation, not lookup:** `ood-combo` models (2 seeds) handle unseen mass × force combinations (effect 0.86 / 0.87).
  - **Weak compositional reuse:** `compose` models (2 seeds) fail on friction and force together. Moving-span loss is 3.1–3.7× (stuck-span 0.006 → about 1), and written friction moves their force dynamics at only 0.3–0.7× the iid model's strength.
- **Generalization (behaviour):** `ood-extrap` moving-span loss rises 10–14% on unseen masses (no wide-mass reference yet).
- **Write-up:**
  - `writeup_interventions.md`: results R1–R7, interpretation, caveats.
  - Six figures in `figures/`, made by `python make_figures.py`, which needs the local logs.
  - `related_work.md`: citation audit of `project_idea.md`, verified bibliography, draft related-work section. Made by a subagent; 2 entries spot-checked.
- **Paper readiness:** workshop-ready now. For a main venue, the remaining gaps are breadth, 38M seeds, and positioning against Makelov et al. (see `PROJECT_STATE.md`).
- Logs (`*.log`), per-pair outputs (`out/*.npz`) and checkpoints (`ckpt/`) are local only (gitignored). `make_figures.py` reads the logs from the repo root.

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
| `num_compose.pt`, `num_compose_s1.pt` | `num`, 45k steps, seeds 0 and 1, `compose` train part |
| `num_L12d512.pt` | `num`, 38M parameters (12 layers, d = 512), 45k steps, seed 0 |

## Compute: RunPod

- **Pods used:** `47.47.180.106:16564`, `47.47.180.126:11742` (2026-10-03); `81.27.69.179:33861`, `81.27.69.179:22782`, `203.57.40.132:10164` (2026-10-04). Terminate any still up.
- **To start one:**
  - RTX 4090 (about $0.75/h), an official RunPod PyTorch template, 30 GB container disk, a 20 GB volume disk at `/workspace`, **TCP port 22 exposed**.
  - Use the "SSH over exposed TCP" address (`root@IP -p PORT`); the proxied `ssh.runpod.io` address can't copy files. The IP and port change with every pod.
  - SSH key: `~/.ssh/id_ed25519_runpod`, already registered in RunPod settings. The user sometimes pastes `-i ~/.ssh/id_ed25519`; that key doesn't exist, so use `id_ed25519_runpod`.
- **Set up the pod:**
  - `git archive HEAD sim.py train.py probe.py intervene.py requirements.txt | ssh -p <PORT> -i ~/.ssh/id_ed25519_runpod root@<IP> 'mkdir -p /workspace/lang2dyn && tar -x --no-same-owner -C /workspace/lang2dyn'`
  - `git archive HEAD` only ships committed files. For uncommitted work, use `tar -c <files> ckpt/<needed>.pt | ssh ... 'tar -x ...'` from the working tree; that also copies checkpoints.
  - Then on the pod: `pip install --break-system-packages scikit-learn psutil` (the Python is externally managed; torch comes with the template).
- **Running jobs:**
  - Run jobs inside `tmux`, writing an exit code to a `*.status` file. For several jobs, use a `chain.sh` in tmux that launches runs and waits on status files.
  - `nproc` shows the host's cores (48–256 so far), not the pod's vCPUs (8 on the first pod). Use `OMP_NUM_THREADS=8` for one job, or 2–4 each when sharing.
  - `intervene.py` `OUT=out/<name>.npz` saves per-pair outputs. Copy logs, `out/` and checkpoints back with `scp -P <PORT>`.
- **Monitoring:** poll from the laptop with short repeated SSH checks (a loop of `ssh -o ConnectTimeout=15 ... 'ls *.status'` every 60 s). A single long-lived SSH wait can die silently and never report.
  - Never `pkill -f <pattern>` over SSH with a pattern that also matches your own SSH command line; it kills the session. Kill by PID.
- **Timing on a 4090, 45k steps:**
  - training: `num`/`sym` about 18 min; `nat`/`rel`/`qual` about 35–40 min; the 38M `num` model (`SIZE=12,512,8 LR=6e-4`) about 78 min on a shared GPU;
  - probes: about 3–6 min each, plus about 4 min with the default ceiling;
  - `intervene.py`: about 3–4 min per layer at 2000/400 pairs, longer at 1000 test pairs and for `nat`. 3–5 runs can share the GPU.
- **Terminate** pods when done; stopped pods still bill for disk.

## Working agreements

- Heavy jobs run on the pod, not the laptop (it runs on battery). Ask before any heavy local job. Smoke tests of a few seconds are fine locally, and short CPU evaluations (about 1 min) have been accepted. Local torch has no CUDA.
- State the cost before each paid launch, and get a yes.
- Commit only when asked. Never add Claude co-author or contributor lines.
- Subagents (Agent tool) use `model: sonnet`.
- When jobs run while the user is away, send a push notification when they finish.

## Next steps, in order

1. **Paper draft (no compute).**
   - Merge `writeup_interventions.md` and `related_work.md` into one paper draft for a workshop (interpretability / world models): abstract, intro, setup, results R1–R7 with figures 1–6, related work, limitations.
   - State the write channel as sufficiency under intervention (Makelov et al. caveat). Cite the four-layer swap (0.47) as evidence that the natural representation uses the same pathway.
   - Before citing: drop "PhysLang" (not found), confirm the venue of Jin & Rinard "Learning Latent Causal Semantics from Text", and strip `utm_source` from the links in `project_idea.md`.
2. **Breadth for a main venue** (pod):
   - 38M model: seeds 1–2 (`SIZE=12,512,8 LR=6e-4 python train.py num 100000 45000 <seed>`, about 75 min each), then `LAYERS=8 python intervene.py ckpt/num_L12d512_s<seed>.pt 2000 1000 2000 32 all`.
   - Bounded writes (`CLAMP=3`) for `nat` mass and for friction, which currently have unbounded writes only. Four-layer swaps (`MULTI=2,3,4,5`) for `nat` and friction.
   - `nat` models on `ood-combo` / `compose` (about 40 min each), then the same `SPLIT=combo` / `SPLIT=compose` tests.
   - A second dynamics domain, or phase 2 (collisions; spec section "Phase 2").
3. Optional:
   - a wide-mass reference model for `ood-extrap`;
   - a firmer mu ceiling (least-squares estimator);
   - the `nat` ablations;
   - an intervention test that separates storage sites further (e.g. patching by position block).

## Prompt to paste into a new chat

> Continue the LI project in C:\Users\advay\Desktop\dev\LI. Read HANDOFF.md, then PROJECT_STATE.md. Then propose a plan for the next step before writing code.

## Maintenance

At the end of each session, update `PROJECT_STATE.md` (headline findings, decisions, code facts, results, issues, not-built list, paper readiness) and the "Where we are", "Compute" and "Next steps" sections of this file. If results change a write-up claim, update `writeup_interventions.md` and rerun `python make_figures.py`.
