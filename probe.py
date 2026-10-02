"""Probes on the residual stream at span-end tokens, against two kinds of baseline:
random-init model (same probe), and "observables" (probes fed the ground-truth stated values directly).
"ceiling" rows: a GPU MLP on observables from many fresh episodes, the best decoder we have from the stated values.

python probe.py [ckpt=ckpt/num.pt] [n_episodes=3000] [pool=end|mean] [split=iid]
split iid: probes fit on 75% of n episodes, scored on 25%. Other splits: fit on n episodes of the split's train part,
scored on n/3 episodes of its test part, so the scores measure whether representations generalize out of distribution.
"""
import functools, os, sys, time
import numpy as np, torch
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import sim
from train import GPT, PAD, encode, load

PROBE_SEED = 20_000_000
TARGETS = ["x", "v", "a", "F", "log_m", "mu"]
MAX_SPANS = 24  # observables history slots
STATED_DIGITS = {"num": 2, "nat": 1, "sym": 2}  # decimals each channel states x and v with
CODE_CHANNELS = ("qual", "rel")  # no numbers: observables are the parsed bins / comparison codes, one-hot
VOCAB_SEED = 40_000_000
CEIL_SEED = 30_000_000
# two sizes, to see whether the ceiling has saturated. CEIL="" skips it (smoke tests); CEIL=5000,50000,200000 adds sizes
CEIL_EPISODES = tuple(int(k) for k in os.environ.get("CEIL", "5000,50000").split(",") if k)


SCORE_VS_TRAIN = False  # set in main for OOD splits


def score(y_true, y_pred):
    """R^2 for iid. For OOD splits, 1 - MSE / Var_train: targets are standardized with the train part's stats, so this is
    1 - MSE. Plain R^2 on a narrow test region (e.g. m in [3, 5]) divides by that region's small variance and explodes."""
    if SCORE_VS_TRAIN:
        return 1 - float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    return r2_score(y_true, y_pred)


def ridge():
    return RidgeCV(alphas=np.logspace(-1, 4, 6), alpha_per_target=True)


def mlp():
    return MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True, max_iter=300, random_state=0)


def ceiling(channel, X_test, y_mean, y_sd, n_eps, split="iid"):
    """Stronger reference: a 2-layer GPU MLP on observables from n_eps fresh episodes, with early stopping.
    Still a lower bound on what the stated values allow, so check that it saturates as n_eps grows.
    -> test predictions, standardized with the probe split's target stats (y_mean, y_sd)."""
    eps = sim.split_episodes(split, "train", n_eps, CEIL_SEED)  # trained on the train part only, like the probes
    X, y = observables(eps, channel), (targets(eps, channel)[0] - y_mean) / y_sd
    x_mean, x_sd = X.mean(0), X.std(0) + 1e-6
    to = lambda a: torch.tensor(a, dtype=torch.float32).cuda()
    X, y, Xt = to((X - x_mean) / x_sd), to(y), to((X_test - x_mean) / x_sd)
    n_val = len(X) // 20
    torch.manual_seed(0)
    perm = torch.randperm(len(X), device="cuda")
    tr, va = perm[n_val:], perm[:n_val]  # ponytail: split by row, not episode; fine for early stopping only
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                              torch.nn.GELU(), torch.nn.Linear(512, y.shape[1])).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(200):
        for idx in tr[torch.randperm(len(tr), device="cuda")].split(1024):
            loss = torch.nn.functional.mse_loss(net(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = torch.nn.functional.mse_loss(net(X[va]), y[va]).item()
        if val < best - 1e-4:
            best, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        elif (bad := bad + 1) >= 10:
            break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(Xt).cpu().numpy()


def targets(eps, channel):
    """-> y (n_spans, len(TARGETS)), meta (episode index, step, m identifiable, mu identifiable)"""
    ys, meta = [], []
    for i, e in enumerate(eps):
        tr, p = e["traj"], e["params"]
        for _, _, r in e["texts"][channel]["spans"]:
            ys.append([tr["x"][r], tr["v"][r], tr["a"][r], tr["F"][r], np.log(p["m"]), p["mu"]])
            meta.append([i, r, e["identifiable"]["m"], e["identifiable"]["mu"]])
    return np.array(ys), np.array(meta)


def stated_codes(channel, sentence, first):
    """What a qual/rel sentence states, as a set of "key=value" strings, read with the channel's own parser."""
    if channel == "qual":
        z = sim.parse_qual(sentence)
    else:
        z = sim.parse_rel_first(sentence) if first else sim.parse_rel(sentence)
    return {f"{k}={v}" for k, v in z.items()}


@functools.cache
def code_vocab(channel):
    """All codes seen in 3000 episodes -> one-hot index. ponytail: codes never seen there are dropped."""
    vocab = set()
    for s in range(VOCAB_SEED, VOCAB_SEED + 3000):
        t = sim.make_episode(s)["texts"][channel]
        for i, (s0, s1, _) in enumerate(t["spans"]):
            vocab |= stated_codes(channel, t["text"][s0:s1], i == 0)
    return {c: i for i, c in enumerate(sorted(vocab))}


def code_observables(eps, channel):
    """Per span: the latest stated value of every key (so once-stated facts like the mass bin sit in fixed slots),
    then each sentence's codes, most recent first, zero-padded to MAX_SPANS slots."""
    vocab = code_vocab(channel)
    onehot = lambda cs: np.bincount([vocab[c] for c in cs if c in vocab], minlength=len(vocab))
    rows = []
    for e in eps:
        t, hist, latest = e["texts"][channel], [], {}
        for i, (s0, s1, _) in enumerate(t["spans"]):
            codes = stated_codes(channel, t["text"][s0:s1], i == 0)
            latest.update({c.split("=")[0]: c for c in codes})
            hist.append(np.concatenate([[1.0], onehot(codes)]))
            assert len(hist) <= MAX_SPANS
            rows.append(np.concatenate([onehot(latest.values()), *hist[::-1], np.zeros((MAX_SPANS - len(hist)) * (len(vocab) + 1))]))
    return np.array(rows, dtype=np.float32)


def observables(eps, channel):
    """Stated values (t, x, v, F) of every span so far, at the precision the channel states them, most recent first, zero-padded to MAX_SPANS slots.
    This is everything the channel gives the model, minus the formatting. qual/rel: see code_observables."""
    if channel in CODE_CHANNELS:
        return code_observables(eps, channel)
    rows, d = [], STATED_DIGITS[channel]
    for e in eps:
        tr, hist = e["traj"], []
        for _, _, r in e["texts"][channel]["spans"]:
            hist.append([1.0, tr["t"][r], round(tr["x"][r], d), round(tr["v"][r], d), tr["F"][r]])
            assert len(hist) <= MAX_SPANS
            rows.append(np.concatenate([np.ravel(hist[::-1]), np.zeros(5 * (MAX_SPANS - len(hist)))]))
    return np.array(rows)


@torch.no_grad()
def residuals(model, eps, stoi, channel, pool="end"):
    """-> feats[layer] (n_spans, d), in the same span order as targets().
    pool="end": the span's last token. pool="mean": mean over the span's tokens (s0+1 .. s1)."""
    feats = [[] for _ in range(model.cfg["layers"] + 1)]
    for i in range(0, len(eps), 64):
        chunk = eps[i:i + 64]
        _, resid = model(encode([e["texts"][channel]["text"] for e in chunk], stoi, model.cfg["block"]).cuda(), True)
        b_idx = [b for b, e in enumerate(chunk) for _ in e["texts"][channel]["spans"]]
        s0 = torch.tensor([s for e in chunk for s, _, _ in e["texts"][channel]["spans"]])
        s1 = torch.tensor([s for e in chunk for _, s, _ in e["texts"][channel]["spans"]])  # token s1 = last char of span
        for L, h in enumerate(resid):
            h = h.float()
            if pool == "mean":
                c = h.cumsum(1)
                f = (c[b_idx, s1] - c[b_idx, s0]) / (s1 - s0).cuda()[:, None]  # sum of tokens s0+1..s1
            else:
                f = h[b_idx, s1]
            feats[L].append(f.cpu().numpy())
    return [np.concatenate(f) for f in feats]


@torch.no_grad()
def lm_loss_on(model, eps, stoi, channel):
    """Mean next-token loss per token: over all tokens, and over the tokens of spans where the cart is moving / stuck.
    Stuck spans repeat the same values and are trivially predictable, so "moving" is the fair comparison across parts."""
    sums = {k: [0.0, 0] for k in ("all", "moving", "stuck")}
    for i in range(0, len(eps), 64):
        chunk = eps[i:i + 64]
        X = encode([e["texts"][channel]["text"] for e in chunk], stoi, model.cfg["block"]).cuda()
        X = X[:, :int((X != PAD).sum(1).max())]
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(X[:, :-1])
        ce = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), X[:, 1:], reduction="none")  # ce[b, t-1]: loss of token t
        masks = {"all": X[:, 1:] != PAD, "moving": torch.zeros_like(ce, dtype=torch.bool), "stuck": torch.zeros_like(ce, dtype=torch.bool)}
        for b, e in enumerate(chunk):
            for s0, s1, r in e["texts"][channel]["spans"]:
                masks["moving" if e["traj"]["regime"][r] == "moving" else "stuck"][b, s0:s1] = True  # tokens s0+1..s1
        for k, m in masks.items():
            sums[k][0] += ce[m].sum().item()
            sums[k][1] += int(m.sum())
    return {k: total / max(count, 1) for k, (total, count) in sums.items()}


def fit_eval(feats, y, train, test, make):
    """-> R^2 (n_feature_sets, n_targets), test predictions per feature set. Targets standardized (R^2 unchanged)."""
    ys = (y - y[train].mean(0)) / y[train].std(0)
    r2, preds = [], []
    for X in feats:
        Xs = (X - X[train].mean(0)) / (X[train].std(0) + 1e-6)
        p = make().fit(Xs[train], ys[train]).predict(Xs[test])
        r2.append([score(ys[test][:, j], p[:, j]) for j in range(len(TARGETS))])
        preds.append(p)
    return np.array(r2), preds, ys[test]


def table(name, r2):
    print(f"\n{name}  ({'1 - MSE/Var_train on the test part' if SCORE_VS_TRAIN else 'R^2 on held-out episodes'})")
    print("layer " + "".join(f"{t:>8}" for t in TARGETS))
    for L, row in enumerate(r2):
        print(f"{L:5d} " + "".join(f"{v:8.3f}" for v in row))


def main():
    path, n, pool, split = (sys.argv[1:] + [None] * 4)[:4]
    path, n, pool, split = path or "ckpt/num.pt", int(n or 3000), pool or "end", split or "iid"
    assert pool in ("end", "mean") and split in sim.SPLITS
    global SCORE_VS_TRAIN
    SCORE_VS_TRAIN = split != "iid"
    model, stoi, ck = load(path)
    channel = ck["channel"]
    print(f"{path}: channel {channel}, {n} episodes, pool={pool}, split={split} (model trained on split={ck.get('split', 'iid')})")
    torch.manual_seed(1)
    rand = GPT(**ck["cfg"]).cuda().eval()

    if split == "iid":
        eps, n_fit = sim.split_episodes("iid", "train", n, PROBE_SEED), int(0.75 * n)
    else:
        eps, n_fit = sim.split_episodes(split, "train", n, PROBE_SEED) + sim.split_episodes(split, "test", n // 3, PROBE_SEED), n
        for name, part in (("train part", eps[:n]), ("test part", eps[n:])):
            print(f"  LM loss on {name}: " + "  ".join(f"{k} {v:.4f}" for k, v in lm_loss_on(model, part, stoi, channel).items()))
    y, meta = targets(eps, channel)
    train, test = meta[:, 0] < n_fit, meta[:, 0] >= n_fit  # split by episode, not by span
    mt = meta[test]

    res = {}  # name -> (r2 per layer, preds per layer)
    t0 = time.time()
    feats = residuals(model, eps, stoi, channel, pool)
    runs = [("trained ridge", feats, ridge), ("trained mlp", feats, mlp),
            ("random-init ridge", residuals(rand, eps, stoi, channel, pool), ridge)]
    has_obs = channel in STATED_DIGITS or channel in CODE_CHANNELS
    if has_obs:
        runs += [("observables ridge", [observables(eps, channel)], ridge), ("observables mlp", [observables(eps, channel)], mlp)]
    for name, f, make in runs:
        r2, preds, yt = fit_eval(f, y, train, test, make)
        res[name] = (r2, preds)
        if len(r2) > 1:
            table(f"{name}, channel {channel}", r2)
        print(f"  [{name} done, {time.time() - t0:.0f}s]", flush=True)

    X_test = observables(eps, channel)[test] if has_obs else None
    for n_ceil in CEIL_EPISODES if has_obs else ():
        p = ceiling(channel, X_test, y[train].mean(0), y[train].std(0), n_ceil, split)
        res[f"ceiling {n_ceil // 1000}k eps"] = (np.array([[score(yt[:, j], p[:, j]) for j in range(len(TARGETS))]]), [p])
        print(f"  [ceiling {n_ceil} episodes done, {time.time() - t0:.0f}s]", flush=True)

    print("\nbest layer per target")
    print(f"{'':18s}" + "".join(f"{t:>8}" for t in TARGETS))
    for name, (r2, _) in res.items():
        print(f"{name:18s}" + "".join(f"{v:8.3f}" for v in r2.max(0)))

    # hidden params on late spans (dynamics observed), split by each param's own identifiability flag
    print("\nhidden params, spans with step >= 25, by identifiability of that param")
    late = mt[:, 1] >= 25
    for j, flag in ((TARGETS.index("log_m"), 2), (TARGETS.index("mu"), 3)):
        for name in ("trained ridge", "trained mlp", "observables mlp", f"ceiling {CEIL_EPISODES[-1] // 1000}k eps" if CEIL_EPISODES else ""):
            if name not in res:
                continue
            r2, preds = res[name]
            L = int(r2[:, j].argmax())
            cells = []
            for ok in (1, 0):
                mask = late & (mt[:, flag] == ok)
                cells.append(f"{'id' if ok else 'non-id'} n={mask.sum():5d} R^2 {score(yt[mask, j], preds[L][mask, j]):6.3f}")
            print(f"  {TARGETS[j]:6s} {name:18s} L{L}  " + "   ".join(cells))


if __name__ == "__main__":
    main()
