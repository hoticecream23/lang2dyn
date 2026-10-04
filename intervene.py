"""Interchange interventions on a hidden parameter (mass or friction; the operator question, decision 1).

python intervene.py [ckpt=ckpt/num_45k.pt] [n_train=2000] [n_test=400] [n_probe=2000] [k=1] [pos=all|last|span]
k: dimension of the patched subspace (u below is then an orthonormal d x k basis; probe row only for k = 1).
pos: overwrite at every position (all), or only from A's last prompt span on (last: the span with the new force,
     and the decoded / target tokens), or only on that span (span: the decoded tokens are not patched, so any effect
     reaches them through attention to the patched prompt).
The channel (num or nat) comes from the checkpoint.

Pairs (A, B): A's text up to a span where a new nonzero force starts (step r0) is the prompt, so the next span's
velocity needs the new acceleration F/m - g s mu. The target is the simulator's counterfactual: A with B's value of
the target parameter (env TARGET=m|mu, default m) from step r0 on (sim m_switch / mu_switch). The texts agree up to
r0 and differ from the next emitted step r1.

Intervention at layer L along a unit direction u, at every position: overwrite A's coordinate h.u with c.
  das       u learned (model frozen) so the patched model predicts the counterfactual span (teacher-forced CE);
            c = c_B, B's own activations averaged over its last span up to r0 (where the probe reads mass), projected on u
  ablate    das's u, c = the mean c_B over training pairs (erases A's value, carries no source information)
  das-shuf  control: trained with c_B taken from another pair's B (source unrelated to the target)
  das-steer u learned with c = alpha * f(z_B) + beta (alpha, beta learned): the best any k-D variable linear in f(z)
            can do. f = log m for mass (env STEER=invm: 1 / m), mu itself for friction.
  probe     the ridge probe direction for the target, c = c_B
  random    a random unit direction, c = c_B
Scored on held-out pairs by greedy decoding the next span:
  effect = (v_patched - v_clean) / (v_cf - v_A) at step r1, median   (1 = the simulator's full effect)
  iia    = share of pairs whose decoded v is closer to v_cf than to v_A (the clean model gives the floor)
  src    = Theil-Sen slope [95% CI] and correlation of v_patched(B) - v_patched(B') against v_cf(B) - v_cf(B'), with
           B' a second source for the same A. Erasing A's value moves v toward the average, which already looks like a
           partial effect toward a random B; src is 0 for any source-independent change. Ideal slope 1.
  CE gap = target CE with source B' minus with source B (teacher-forced), mean [bootstrap 95% CI]: the source
           information actually used. Also given for held-out and seen pairs separately.
Env SPLIT (pair design; DAS always trains on non-held-out pairs):
  iid      held out = B's value in HOLD.
  combo    (alias COMBO=1; for a model trained on ood-combo) A and B are ood-combo train-part episodes; held out = the
           counterfactual is an unseen combination (m_B in [3, 5] and |F| >= 7 at r0). Test set half held out.
  compose  (for a model trained on compose) held out = A is a compose test-part episode (friction and force together);
           seen = A is a train-part episode (mu = 0). B is a frictionless train-part episode. Test set half held out.
Env RANDOM=1: use a random-init model with the checkpoint's shapes (control; it can't decode, so read the CE gap).
Env OUT=path.npz: save per-pair outputs (decoded v, target CE) for every condition.
"""
import math, os, random, sys
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import theilslopes
from sklearn.linear_model import RidgeCV
import sim
from probe import PROBE_SEED, residuals, targets
from train import BOS, GPT, PAD, load

PAIR_SEED = 50_000_000
TARGET = os.environ.get("TARGET", "m")
assert TARGET in ("m", "mu")
TCOL = {"m": 4, "mu": 5}[TARGET]  # column of probe.targets (log_m, mu)
HOLD = {"m": (2.0, 3.0), "mu": (0.1, 0.15)}[TARGET]  # source values never used to train DAS (SPLIT=iid)
STEER = {"logm": math.log, "invm": lambda m: 1 / m, "lin": lambda z: z}[os.environ.get("STEER", "logm" if TARGET == "m" else "lin")]
MIN_DV = {"num": 0.05, "nat": 0.15}  # keep pairs whose counterfactual changes v at r1 by at least this (above rounding)
SRC = os.environ.get("SRC", "mean")  # source activation: mean over B's last span, or its last token (end)
LAYERS = [int(x) for x in os.environ.get("LAYERS", "1,2,3,4,5,6").split(",")]
BID = os.environ.get("BID") == "1"  # keep only pairs whose B prefix (up to r0) identifies B's value
BSRC = os.environ.get("BSRC", "last")  # B's source span: its last span up to r0, or (fc) its last force-change span
SPLIT = "combo" if os.environ.get("COMBO") == "1" else os.environ.get("SPLIT", "iid")
assert SPLIT in ("iid", "combo", "compose") and (SPLIT != "combo" or TARGET == "m")
RANDOM = os.environ.get("RANDOM") == "1"
CH = "num"  # set from the checkpoint in main


def probe_dirs(model, stoi, n):
    """-> per layer 1.., the ridge probe direction for the target in raw activation space (mean-pooled spans)."""
    eps = sim.split_episodes("iid", "train", n, PROBE_SEED)
    y = targets(eps, CH)[0][:, TCOL]
    out = []
    for X in residuals(model, eps, stoi, CH, "mean")[1:]:
        sd = X.std(0) + 1e-6
        out.append(RidgeCV(alphas=np.logspace(-1, 4, 6)).fit((X - X.mean(0)) / sd, (y - y.mean()) / y.std()).coef_ / sd)
    return out


def overrides(seed, rng):
    """-> (A's param overrides, B's, held out?) for this seed under SPLIT, or None to skip the seed.
    For combo, "held" is decided per candidate later (it depends on r0), so it is None here."""
    if SPLIT == "iid":
        return {}, {}, None
    if SPLIT == "combo":
        ok = not sim.in_combo(sim.sample_params(random.Random(seed))) and not sim.in_combo(sim.sample_params(random.Random(seed + 1)))
        return ({}, {}, None) if ok else None
    held = rng.random() < 0.5
    ovA = sim.split_overrides("compose", "test" if held else "train", seed)
    return None if ovA is None else (ovA, {"mu": 0.0}, held)


def make_pairs(done):
    """done(pairs) -> bool: stop condition, or an int n (stop at n pairs)."""
    if isinstance(done, int):
        n, done = done, lambda ps: len(ps) == n
    rng, pairs = random.Random(0), []
    for seed in range(PAIR_SEED, 10 ** 9, 2):
        if done(pairs):
            return pairs
        if (o := overrides(seed, rng)) is None:
            continue
        ovA, ovB, held = o
        A, B = sim.make_episode(seed, **ovA), sim.make_episode(seed + 1, **ovB)
        zB = B["params"][TARGET]
        spans, cand = A["texts"][CH]["spans"], []
        for i, (_, s1, r0) in enumerate(spans[:-1]):
            if r0 < 10 or "force_change" not in A["traj"]["events"][r0] or A["traj"]["F"][r0] == 0:
                continue
            cf = sim.make_episode(seed, **ovA, **{TARGET + "_switch": [r0, zB]})
            r1, cft = spans[i + 1][2], cf["texts"][CH]
            j = next((k for k, (c0, _, _) in enumerate(cft["spans"]) if c0 > s1), None)  # nat text can be shorter
            if j is None:
                continue
            # nat states events at r0 (e.g. "starts moving"), which can depend on the switched value: skip those
            if (cft["spans"][j][2] == r1 and abs(cf["traj"]["v"][r1] - A["traj"]["v"][r1]) >= MIN_DV[CH]
                    and cft["text"][:s1 + 1] == A["texts"][CH]["text"][:s1 + 1]):
                cand.append(dict(r0=r0, r1=r1, cut=s1 + 1, target=cft["text"][cft["spans"][j][0]:cft["spans"][j][1]],
                                 vA=A["traj"]["v"][r1], vcf=cf["traj"]["v"][r1], A_start=spans[i][0] + 1))
        if not cand:
            continue
        p = rng.choice(cand)
        id_at = lambda r: sim.identifiable({k: v[:r + 1] for k, v in B["traj"].items()})[TARGET]
        if BSRC == "fc":  # B's source span: its last force-change span (as A's r0) where its prefix identifies z_B
            bs = [s for s in B["texts"][CH]["spans"] if "force_change" in B["traj"]["events"][s[2]]
                  and B["traj"]["F"][s[2]] != 0 and id_at(s[2])]
            if not bs:
                continue
            b0, b1, _ = bs[-1]
        else:
            if BID and not id_at(p["r0"]):
                continue  # B's own text up to r0 doesn't determine its value, so B's activations can't carry it
            b0, b1, _ = [s for s in B["texts"][CH]["spans"] if s[2] <= p["r0"]][-1]
        if SPLIT == "iid":
            held = HOLD[0] <= zB <= HOLD[1]
        elif SPLIT == "combo":
            held = 3 <= zB <= 5 and abs(A["traj"]["F"][p["r0"]]) >= 7
        t = A["traj"]["t"]
        pairs.append(dict(p, seed=seed, ov=ovA, A=A["texts"][CH]["text"][:p["cut"]], B=B["texts"][CH]["text"][:b1 + 1],
                          B_span=(b0, b1), zB=zB, tr1=t[p["r1"]], dt1=round(t[p["r1"]] - t[p["r0"]], 1), held=held))


def v_cf(p, z):
    """v at r1 of A with value z from r0, or nan if that counterfactual emits a different step after the prompt."""
    cf = sim.make_episode(p["seed"], **p["ov"], **{TARGET + "_switch": [p["r0"], z]})
    nxt = next((r for c0, _, r in cf["texts"][CH]["spans"] if c0 >= p["cut"]), None)
    return cf["traj"]["v"][p["r1"]] if nxt == p["r1"] else math.nan


@torch.no_grad()
def source_acts(model, pairs, stoi):
    """-> (n_layers + 1, n, d): B's activations averaged over its last span's tokens (s0 + 1 .. s1), or (SRC=end) at token s1."""
    out = []
    for p in pairs:
        _, resid = model(torch.tensor([[BOS] + [stoi[c] for c in p["B"]]]).cuda(), True)
        b0, b1 = p["B_span"]
        out.append(torch.stack([h[0, b0 + 1 if SRC == "mean" else b1:b1 + 1].float().mean(0) for h in resid]))
    return torch.stack(out, 1)


def overwrite(model, L, U, c, start):
    """Hook: at layer L's output, set the coordinates in orthonormal basis U (d, k) to c (n, k), at positions
    start[:, 0] <= t < start[:, 1]."""
    def hook(mod, inp, out):
        t = torch.arange(out.shape[1], device=out.device)[None]
        keep = ((t >= start[:, :1]) & (t < start[:, 1:])).float()[..., None]
        return out + keep * ((c[:, None] - out @ U) @ U.T)
    return model.blocks[L - 1].register_forward_hook(hook)


def train_das(model, L, ids, n_prompt, hB, start, k=1, z=None, steps=800, bs=32, seed=0):
    """ids: list of prompt + target token lists; loss on the target tokens only.
    c = hB @ U, or with z given (steer mode) c = alpha * z + beta (alpha, beta in R^k). -> U, (alpha, beta), final loss"""
    g = torch.Generator().manual_seed(seed)
    u = torch.nn.Parameter(torch.randn(hB.shape[-1], k, generator=g).cuda())
    ab = torch.nn.Parameter(torch.stack([torch.ones(k), torch.zeros(k)]).cuda())
    opt = torch.optim.Adam([u, ab], lr=1e-2)
    for _ in range(steps):
        idx = torch.randint(len(ids), (bs,), generator=g)
        un = torch.linalg.qr(u)[0]
        c = hB[idx.cuda()] @ un if z is None else z[idx.cuda(), None] * ab[0] + ab[1]
        loss = target_ce(model, ids, n_prompt, idx.tolist(), (L, un, c, start[idx.cuda()])).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.linalg.qr(u)[0].detach(), ab.detach(), loss.item()


def target_ce(model, ids, n_prompt, idx, patch=None):
    """-> per row mean CE on the target tokens (teacher-forced), with overwrite(model, *patch) active if given."""
    T = max(len(ids[i]) for i in idx)
    X = torch.full((len(idx), T), PAD)
    y = torch.full((len(idx), T - 1), -100)
    for b, i in enumerate(idx):
        X[b, :len(ids[i])] = torch.tensor(ids[i])
        y[b, n_prompt[i] - 1:len(ids[i]) - 1] = X[b, n_prompt[i]:len(ids[i])]
    h = overwrite(model, *patch) if patch else None
    with torch.autocast("cuda", torch.bfloat16):
        logits = model(X[:, :-1].cuda())
    if h:
        h.remove()
    y = y.cuda()
    return F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").sum(1) / (y != -100).sum(1)


@torch.no_grad()
def test_ce(model, ids, n_prompt, patch=None):
    """-> per test pair target CE (numpy); patch = (L, U, c, start) with one c / start row per pair."""
    out = []
    for i in range(0, len(ids), 64):
        idx = list(range(i, min(i + 64, len(ids))))
        p = patch and (patch[0], patch[1], patch[2][i:i + 64], patch[3][i:i + 64])
        out.append(target_ce(model, ids, n_prompt, idx, p))
    return torch.cat(out).cpu().numpy()


def span_done(text):
    """A decoded span is complete: num spans end ' .', nat spans end with the state sentence ('... mark.' / '... m/s.')."""
    return text.endswith(" .") if CH == "num" else text.endswith(("mark.", "m/s."))


@torch.no_grad()
def decode(model, prompts, itos, L=None, u=None, c=None, start=None):
    """Greedy decode one span per prompt, optionally with overwrite(L, u, c, start) active."""
    max_new = 40 if CH == "num" else 120
    outs = []
    for i in range(0, len(prompts), 64):
        P = prompts[i:i + 64]
        cur = torch.tensor([len(p) for p in P]).cuda()
        X = torch.full((len(P), max(map(len, P)) + max_new), PAD).cuda()
        for b, p in enumerate(P):
            X[b, :len(p)] = torch.tensor(p)
        h = overwrite(model, L, u, c[i:i + 64], start[i:i + 64]) if L else None
        done, text = [False] * len(P), [""] * len(P)
        for _ in range(max_new):
            nxt = model(X[:, :int(cur.max())])[torch.arange(len(P)), cur - 1].argmax(-1)
            for b in range(len(P)):
                if not done[b]:
                    ch = itos.get(int(nxt[b]), "")
                    text[b] += ch
                    done[b] = span_done(text[b]) or not ch
            X[torch.arange(len(P)), cur] = nxt
            cur += 1
            if all(done):
                break
        if h:
            h.remove()
        outs += [t.strip() for t in text]
    return outs


def decoded_v(outs, pairs):
    """-> v per pair, nan where the decode is unparsable or reports another time than r1"""
    v = []
    for o, p in zip(outs, pairs):
        try:
            if CH == "num":
                z = sim.parse_num(o)
                v.append(z["v"] if z["t"] == p["tr1"] else math.nan)
            else:
                z = sim.parse_nat(o)
                v.append(z["v"] if z.get("dt") == p["dt1"] and "v" in z else math.nan)
        except (AttributeError, ValueError, TypeError):
            v.append(math.nan)
    return np.array(v)


def boot_ci(x, n=1000):
    """-> mean, 2.5th and 97.5th percentiles of the bootstrap mean"""
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    means = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def report(name, v, v2, v_clean, pairs, vcf2, hold, extra=""):
    """v, v2: decoded v with source B and with the second source B' (v2 None: source-independent condition)"""
    vA, vcf = np.array([p["vA"] for p in pairs]), np.array([p["vcf"] for p in pairs])
    cells = []
    for tag, m in (("all", np.ones(len(pairs), bool)), ("held out", hold), ("seen", ~hold)):
        ok = m & ~np.isnan(v) & ~np.isnan(v_clean)
        eff = (v[ok] - v_clean[ok]) / (vcf[ok] - vA[ok])
        iia = np.mean(np.abs(v[ok] - vcf[ok]) < np.abs(v[ok] - vA[ok]))
        cell = f"{tag} n={ok.sum():3d} effect {np.median(eff):5.2f} iia {iia:.2f}"
        if v2 is not None:
            ok2 = ok & ~np.isnan(v2) & ~np.isnan(vcf2)
            dp, dc = v[ok2] - v2[ok2], vcf[ok2] - vcf2[ok2]
            if len(dp) >= 5 and dc.std() > 0:
                sl, _, lo, hi = theilslopes(dp, dc, 0.95)
                cell += f" src slope {sl:5.2f} [{lo:5.2f},{hi:5.2f}] r {np.corrcoef(dp, dc)[0, 1]:5.2f}"
        cells.append(cell)
    print(f"  {name:9s} " + " | ".join(cells) + extra, flush=True)


def main():
    global CH
    path, n_train, n_test, n_probe, k, pos = (sys.argv[1:] + [None] * 6)[:6]
    path, n_train, n_test, n_probe = path or "ckpt/num_45k.pt", int(n_train or 2000), int(n_test or 400), int(n_probe or 2000)
    k, pos = int(k or 1), pos or "all"
    assert pos in ("all", "last", "span")
    model, stoi, ck = load(path)
    CH = ck["channel"]
    assert CH in MIN_DV
    if RANDOM:
        torch.manual_seed(1)
        model = GPT(**ck["cfg"]).cuda().eval()
    for q in model.parameters():
        q.requires_grad_(False)
    itos = {i: c for c, i in stoi.items()}
    if SPLIT == "iid":
        pairs = make_pairs(n_train + n_test)
        train, test = [p for p in pairs[:n_train] if not p["held"]], pairs[n_train:]
    else:
        pairs = make_pairs(lambda ps: sum(p["held"] for p in ps) >= n_test // 2
                           and sum(not p["held"] for p in ps) >= n_train + n_test // 2)
        held, seen = [p for p in pairs if p["held"]], [p for p in pairs if not p["held"]]
        train, test = seen[:n_train], held[:n_test // 2] + seen[n_train:n_train + n_test // 2]
    hold = np.array([p["held"] for p in test])
    enc = lambda t: [stoi[c] for c in t]
    ids = [[BOS] + enc(p["A"]) + enc(p["target"]) for p in train]
    n_prompt = [1 + len(p["A"]) for p in train]
    starts = lambda ps: torch.tensor([[0 if pos == "all" else p["A_start"], 1 + len(p["A"]) if pos == "span" else 10 ** 6]
                                      for p in ps]).cuda()
    start_train, start_test = starts(train), starts(test)
    hB_train, hB_test = source_acts(model, train, stoi), source_acts(model, test, stoi)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(1))
    z_train = torch.tensor([STEER(p["zB"]) for p in train]).cuda()
    z_test = torch.tensor([STEER(p["zB"]) for p in test]).cuda()
    second = np.roll(np.arange(len(test)), 1)  # B' for test pair k is test pair k-1's B
    vcf2 = np.array([v_cf(p, test[j]["zB"]) for p, j in zip(test, second)])
    probes = probe_dirs(model, stoi, n_probe) if k == 1 else None
    pA = [[BOS] + enc(p["A"]) for p in test]
    ids_test, np_test = [a + enc(p["target"]) for a, p in zip(pA, test)], [len(a) for a in pA]
    saved = {}

    def ces(name, L, u, c, both=True):
        a = test_ce(model, ids_test, np_test, (L, u, c, start_test))
        saved[f"L{L}_{name}_ce"] = a
        if not both:
            return f" || target CE {a.mean():.3f}"
        b = test_ce(model, ids_test, np_test, (L, u, c[second], start_test))
        saved[f"L{L}_{name}_ce2"] = b
        g, lo, hi = boot_ci(b - a)
        parts = " ".join(f"{t} {(b - a)[m].mean():+.3f}" for t, m in (("held", hold), ("seen", ~hold)))
        return f" || target CE {a.mean():.3f}, source B' {b.mean():.3f}, gap {g:+.3f} [{lo:+.3f},{hi:+.3f}] ({parts})"

    def run(name, L, u, c):
        v = decoded_v(decode(model, pA, itos, L, u, c, start_test), test)
        saved[f"L{L}_{name}"] = v
        return v

    v_clean = decoded_v(decode(model, pA, itos), test)
    held_desc = {"iid": f"{TARGET}_B in {HOLD}", "combo": "combo region", "compose": "compose test part"}[SPLIT]
    print(f"{path}: channel {CH}, target {TARGET}, split {SPLIT}, k={k}, pos={pos}, src={SRC}, bid={BID}, bsrc={BSRC}, "
          f"random={RANDOM}, {len(train)} DAS train pairs, {len(test)} test pairs ({hold.sum()} held out: {held_desc}), "
          f"median |v_cf - v_A| {np.median([abs(p['vcf'] - p['vA']) for p in test]):.3f}")
    clean_ce = test_ce(model, ids_test, np_test)
    report("clean", v_clean, None, v_clean, test, vcf2, hold, f" || target CE {clean_ce.mean():.3f}")
    saved.update(clean=v_clean, clean_ce=clean_ce, held=hold, vA=[p["vA"] for p in test], vcf=[p["vcf"] for p in test], vcf2=vcf2)
    for L in LAYERS:
        u_das, _, loss = train_das(model, L, ids, n_prompt, hB_train[L], start_train, k)
        u_shuf, _, loss_shuf = train_das(model, L, ids, n_prompt, hB_train[L][perm], start_train, k)
        u_st, ab, loss_st = train_das(model, L, ids, n_prompt, hB_train[L], start_train, k, z_train)
        dirs = {"das": u_das, "das-shuf": u_shuf}
        overlap = lambda a, b: float(torch.linalg.svdvals(a.T @ b).max())  # 1 = the subspaces share a direction
        note = ""
        if k == 1:
            dirs["probe"] = torch.tensor(probes[L - 1] / np.linalg.norm(probes[L - 1]), dtype=torch.float32).cuda()[:, None]
            note = f"; |cos| with probe: das {overlap(u_das, dirs['probe']):.2f}, steer {overlap(u_st, dirs['probe']):.2f}"
        dirs["random"] = torch.linalg.qr(torch.randn(len(u_das), k, generator=torch.Generator().manual_seed(L)))[0].cuda()
        print(f"L{L}: train loss das {loss:.3f}, das-shuf {loss_shuf:.3f}, das-steer {loss_st:.3f}{note}; "
              f"overlap(das, steer) {overlap(u_das, u_st):.2f}")
        for name, u in dirs.items():
            c = hB_test[L] @ u
            report(name, run(name, L, u, c), run(name + "_B2", L, u, c[second]), v_clean, test, vcf2, hold, ces(name, L, u, c))
        c = (hB_train[L] @ u_das).mean(0).expand(len(test), k)
        report("ablate", run("ablate", L, u_das, c), None, v_clean, test, vcf2, hold, ces("ablate", L, u_das, c, False))
        c = z_test[:, None] * ab[0] + ab[1]
        report("das-steer", run("das-steer", L, u_st, c), run("das-steer_B2", L, u_st, c[second]), v_clean, test, vcf2,
               hold, ces("das-steer", L, u_st, c))
        if os.environ.get("OUT"):
            np.savez(os.environ["OUT"], **{key: np.asarray(val) for key, val in saved.items()})


if __name__ == "__main__":
    main()
