"""Interchange interventions on the model's mass variable (the operator question, decision 1). num channel only.

python intervene.py [ckpt=ckpt/num_45k.pt] [n_train=2000] [n_test=400] [n_probe=2000] [k=1] [pos=all|last]
k: dimension of the patched subspace (u below is then an orthonormal d x k basis; probe row only for k = 1).
pos: overwrite at every position (all), or only from A's last prompt span on (last: the span with the new force,
     and the decoded / target tokens), or only on that span (span: the decoded tokens are not patched, so any effect
     reaches them through attention to the patched prompt).

Pairs (A, B): A's text up to a span where a new nonzero force starts (step r0) is the prompt, so the next span's
velocity needs the new acceleration F/m - g s mu, i.e. the mass. The target is the simulator's counterfactual: A with
B's mass from step r0 on (sim m_switch). The texts agree up to r0 and differ from the next emitted step r1.

Intervention at layer L along a unit direction u, at every position: overwrite A's coordinate h.u with c.
  das       u learned (model frozen) so the patched model predicts the counterfactual span (teacher-forced CE);
            c = c_B, B's own activations averaged over its last span up to r0 (where the probe reads mass), projected on u
  ablate    das's u, c = the mean c_B over training pairs (erases A's value, carries no source information)
  das-shuf  control: trained with c_B taken from another pair's B (source unrelated to the target)
  das-steer u learned with c = alpha * log m_B + beta (alpha, beta learned; env STEER=invm uses 1 / m_B): the best
            any k-D variable linear in log m (or 1/m) can do
  probe     the ridge log_m probe direction, c = c_B
  random    a random unit direction, c = c_B
Scored on held-out pairs by greedy decoding the next span:
  effect = (v_patched - v_clean) / (v_cf - v_A) at step r1, median   (1 = the simulator's full mass effect)
  iia    = share of pairs whose decoded v is closer to v_cf than to v_A (the clean model gives the floor)
  src    = Theil-Sen slope and correlation of v_patched(B) - v_patched(B') against v_cf(B) - v_cf(B'), with B' a
           second source for the same A. Erasing A's mass moves v toward the average mass, which already looks like a
           partial effect toward a random B; src is 0 for any source-independent change. Ideal slope 1.
  CE gap = target CE with source B' minus with source B (teacher-forced): the source information actually used.
DAS training pairs exclude B masses in HOLD; test rows split by whether m_B is in HOLD.
Env COMBO=1 (for a model trained on the ood-combo split): A and B are both ood-combo train-part episodes, and a pair
is held out when the counterfactual is an unseen combination (m_B in [3, 5] and |F| >= 7 at r0). DAS trains on the
other pairs only; the test set is half held-out pairs, half others.
Env RANDOM=1: use a random-init model with the checkpoint's shapes (control; it can't decode, so read the CE gap).
"""
import math, os, random, sys
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import theilslopes
from sklearn.linear_model import RidgeCV
import sim
from probe import PROBE_SEED, residuals, targets
from train import BOS, GPT, PAD, load

PAIR_SEED = 50_000_000
LOG_M = 4  # column of probe.targets
MIN_DV = 0.05  # keep pairs whose counterfactual changes v at r1 by at least this
HOLD = (2.0, 3.0)  # B masses never used to train DAS
STEER = {"logm": math.log, "invm": lambda m: 1 / m}[os.environ.get("STEER", "logm")]  # das-steer variable: log m or 1/m
SRC = os.environ.get("SRC", "mean")  # source activation: mean over B's last span, or its last token (end)
LAYERS = [int(x) for x in os.environ.get("LAYERS", "1,2,3,4,5,6").split(",")]
BID = os.environ.get("BID") == "1"  # keep only pairs whose B prefix (up to r0) identifies m_B
BSRC = os.environ.get("BSRC", "last")  # B's source span: its last span up to r0, or (fc) its last force-change span
COMBO = os.environ.get("COMBO") == "1"
RANDOM = os.environ.get("RANDOM") == "1"


def probe_dirs(model, stoi, n):
    """-> per layer 1.., the ridge log_m direction in raw activation space (mean-pooled spans)."""
    eps = sim.split_episodes("iid", "train", n, PROBE_SEED)
    y = targets(eps, "num")[0][:, LOG_M]
    out = []
    for X in residuals(model, eps, stoi, "num", "mean")[1:]:
        sd = X.std(0) + 1e-6
        out.append(RidgeCV(alphas=np.logspace(-1, 4, 6)).fit((X - X.mean(0)) / sd, (y - y.mean()) / y.std()).coef_ / sd)
    return out


def make_pairs(done):
    """done(pairs) -> bool: stop condition, or an int n (stop at n pairs)."""
    if isinstance(done, int):
        n, done = done, lambda ps: len(ps) == n
    rng, pairs = random.Random(0), []
    for seed in range(PAIR_SEED, 10 ** 9, 2):
        if done(pairs):
            return pairs
        A, B = sim.make_episode(seed), sim.make_episode(seed + 1)
        if COMBO and (sim.in_combo(A["params"]) or sim.in_combo(B["params"])):
            continue  # both must be ood-combo train-part episodes
        spans, cand = A["texts"]["num"]["spans"], []
        for i, (_, s1, r0) in enumerate(spans[:-1]):
            if r0 < 10 or "force_change" not in A["traj"]["events"][r0] or A["traj"]["F"][r0] == 0:
                continue
            cf = sim.make_episode(seed, m_switch=[r0, B["params"]["m"]])
            r1, cft = spans[i + 1][2], cf["texts"]["num"]
            j = next(k for k, (c0, _, _) in enumerate(cft["spans"]) if c0 > s1)
            if cft["spans"][j][2] == r1 and abs(cf["traj"]["v"][r1] - A["traj"]["v"][r1]) >= MIN_DV:
                assert cft["text"][:s1 + 1] == A["texts"]["num"]["text"][:s1 + 1]
                cand.append(dict(r0=r0, r1=r1, cut=s1 + 1, target=cft["text"][cft["spans"][j][0]:cft["spans"][j][1]],
                                 vA=A["traj"]["v"][r1], vcf=cf["traj"]["v"][r1]))
        if cand:
            p = rng.choice(cand)
            id_at = lambda r: sim.identifiable({k: v[:r + 1] for k, v in B["traj"].items()})["m"]
            if BSRC == "fc":  # B's source span: its last force-change span (as A's r0) where its prefix identifies m_B
                bs = [s for s in B["texts"]["num"]["spans"] if "force_change" in B["traj"]["events"][s[2]]
                      and B["traj"]["F"][s[2]] != 0 and id_at(s[2])]
                if not bs:
                    continue
                b0, b1, _ = bs[-1]
            else:
                if BID and not id_at(p["r0"]):
                    continue  # B's own text up to r0 doesn't determine its mass, so B's activations can't carry it
                b0, b1, _ = [s for s in B["texts"]["num"]["spans"] if s[2] <= p["r0"]][-1]
            pairs.append(dict(p, seed=seed, A_start=spans[[s[2] for s in spans].index(p["r0"])][0] + 1, A=A["texts"]["num"]["text"][:p["cut"]], B=B["texts"]["num"]["text"][:b1 + 1],
                              B_span=(b0, b1), mA=A["params"]["m"], mB=B["params"]["m"], tr1=A["traj"]["t"][p["r1"]],
                              combo=3 <= B["params"]["m"] <= 5 and abs(A["traj"]["F"][p["r0"]]) >= 7))


def v_cf(p, m):
    """v at r1 of A with mass m from r0, or nan if that counterfactual emits a different step after the prompt."""
    cf = sim.make_episode(p["seed"], m_switch=[p["r0"], m])
    nxt = next(r for c0, _, r in cf["texts"]["num"]["spans"] if c0 >= p["cut"])
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


def train_das(model, L, ids, n_prompt, hB, start, k=1, logm=None, steps=800, bs=32, seed=0):
    """ids: list of prompt + target token lists; loss on the target tokens only.
    c = hB @ U, or with logm given (steer mode) c = alpha * logm + beta (alpha, beta in R^k). -> U, (alpha, beta), final loss"""
    g = torch.Generator().manual_seed(seed)
    u = torch.nn.Parameter(torch.randn(hB.shape[-1], k, generator=g).cuda())
    ab = torch.nn.Parameter(torch.stack([torch.ones(k), torch.zeros(k)]).cuda())
    opt = torch.optim.Adam([u, ab], lr=1e-2)
    for _ in range(steps):
        idx = torch.randint(len(ids), (bs,), generator=g)
        un = torch.linalg.qr(u)[0]
        c = hB[idx.cuda()] @ un if logm is None else logm[idx.cuda(), None] * ab[0] + ab[1]
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
    """Mean target CE over all test pairs; patch = (L, U, c, start) with one c / start row per pair."""
    out = []
    for i in range(0, len(ids), 64):
        idx = list(range(i, min(i + 64, len(ids))))
        p = patch and (patch[0], patch[1], patch[2][i:i + 64], patch[3][i:i + 64])
        out.append(target_ce(model, ids, n_prompt, idx, p))
    return float(torch.cat(out).mean())


@torch.no_grad()
def decode(model, prompts, itos, L=None, u=None, c=None, start=None, max_new=40):
    """Greedy decode one span (ending ' .') per prompt, optionally with overwrite(L, u, c, start) active."""
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
                    done[b] = text[b].endswith(" .") or not ch
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
            z = sim.parse_num(o)
            v.append(z["v"] if z["t"] == p["tr1"] else math.nan)
        except (AttributeError, ValueError):
            v.append(math.nan)
    return np.array(v)


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
                cell += f" src slope {theilslopes(dp, dc)[0]:5.2f} r {np.corrcoef(dp, dc)[0, 1]:5.2f}"
        cells.append(cell)
    print(f"  {name:9s} " + " | ".join(cells) + extra, flush=True)


def main():
    path, n_train, n_test, n_probe, k, pos = (sys.argv[1:] + [None] * 6)[:6]
    path, n_train, n_test, n_probe = path or "ckpt/num_45k.pt", int(n_train or 2000), int(n_test or 400), int(n_probe or 2000)
    k, pos = int(k or 1), pos or "all"
    assert pos in ("all", "last", "span")
    model, stoi, ck = load(path)
    if RANDOM:
        torch.manual_seed(1)
        model = GPT(**ck["cfg"]).cuda().eval()
    assert ck["channel"] == "num"
    for q in model.parameters():
        q.requires_grad_(False)
    itos = {i: c for c, i in stoi.items()}
    if COMBO:
        pairs = make_pairs(lambda ps: sum(p["combo"] for p in ps) >= n_test // 2
                           and sum(not p["combo"] for p in ps) >= n_train + n_test // 2)
        held, seen = [p for p in pairs if p["combo"]], [p for p in pairs if not p["combo"]]
        train, test = seen[:n_train], held[:n_test // 2] + seen[n_train:n_train + n_test // 2]
        hold = np.array([p["combo"] for p in test])
    else:
        pairs = make_pairs(n_train + n_test)
        train = [p for p in pairs[:n_train] if not HOLD[0] <= p["mB"] <= HOLD[1]]
        test = pairs[n_train:]
        hold = np.array([HOLD[0] <= p["mB"] <= HOLD[1] for p in test])
    enc = lambda t: [stoi[c] for c in t]
    ids = [[BOS] + enc(p["A"]) + enc(p["target"]) for p in train]
    n_prompt = [1 + len(p["A"]) for p in train]
    starts = lambda ps: torch.tensor([[0 if pos == "all" else p["A_start"], 1 + len(p["A"]) if pos == "span" else 10 ** 6]
                                      for p in ps]).cuda()
    start_train, start_test = starts(train), starts(test)
    hB_train, hB_test = source_acts(model, train, stoi), source_acts(model, test, stoi)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(1))
    logm_train = torch.tensor([STEER(p["mB"]) for p in train]).cuda()
    logm_test = torch.tensor([STEER(p["mB"]) for p in test]).cuda()
    second = np.roll(np.arange(len(test)), 1)  # B' for test pair k is test pair k-1's B
    vcf2 = np.array([v_cf(p, test[j]["mB"]) for p, j in zip(test, second)])
    probes = probe_dirs(model, stoi, n_probe) if k == 1 else None
    pA = [[BOS] + enc(p["A"]) for p in test]
    ids_test, np_test = [a + enc(p["target"]) for a, p in zip(pA, test)], [len(a) for a in pA]
    ce = lambda L, u, c: f" || target CE {test_ce(model, ids_test, np_test, (L, u, c, start_test)):.3f}"
    def ces(L, u, c):
        a, b = (test_ce(model, ids_test, np_test, (L, u, cc, start_test)) for cc in (c, c[second]))
        return f" || target CE {a:.3f}, source B' {b:.3f}, gap {b - a:+.3f}"
    v_clean = decoded_v(decode(model, pA, itos), test)
    print(f"{path}: k={k}, pos={pos}, src={SRC}, bid={BID}, bsrc={BSRC}, combo={COMBO}, random={RANDOM}, steer={os.environ.get('STEER', 'logm')}, {len(train)} DAS train pairs, {len(test)} test pairs ({hold.sum()} held out: {"combo region" if COMBO else f"m_B in {HOLD}"}), "
          f"median |v_cf - v_A| {np.median([abs(p['vcf'] - p['vA']) for p in test]):.3f}")
    report("clean", v_clean, None, v_clean, test, vcf2, hold, f" || target CE {test_ce(model, ids_test, np_test):.3f}")
    run = lambda L, u, c: decoded_v(decode(model, pA, itos, L, u, c, start_test), test)
    for L in LAYERS:
        u_das, _, loss = train_das(model, L, ids, n_prompt, hB_train[L], start_train, k)
        u_shuf, _, loss_shuf = train_das(model, L, ids, n_prompt, hB_train[L][perm], start_train, k)
        u_st, ab, loss_st = train_das(model, L, ids, n_prompt, hB_train[L], start_train, k, logm_train)
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
            report(name, run(L, u, c), run(L, u, c[second]), v_clean, test, vcf2, hold, ces(L, u, c))
        c = (hB_train[L] @ u_das).mean(0).expand(len(test), k)
        report("ablate", run(L, u_das, c), None, v_clean, test, vcf2, hold, ce(L, u_das, c))
        c = logm_test[:, None] * ab[0] + ab[1]
        report("das-steer", run(L, u_st, c), run(L, u_st, c[second]), v_clean, test, vcf2, hold, ces(L, u_st, c))


if __name__ == "__main__":
    main()
