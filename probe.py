"""Linear probes on the residual stream at span-end tokens, trained model vs random-init baseline.

python probe.py [ckpt=ckpt/num.pt] [n_episodes=3000]
"""
import sys
import numpy as np, torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import sim
from train import GPT, encode, load

PROBE_SEED = 20_000_000
TARGETS = ["x", "v", "a", "F", "log_m", "mu"]


@torch.no_grad()
def collect(model, eps, stoi, channel):
    """-> feats[layer] (n_spans, d), y (n_spans, len(TARGETS)), meta (episode index, step, identifiable)"""
    feats, ys, meta = [[] for _ in range(model.cfg["layers"] + 1)], [], []
    for i in range(0, len(eps), 64):
        chunk = eps[i:i + 64]
        _, resid = model(encode([e["texts"][channel]["text"] for e in chunk], stoi, model.cfg["block"]).cuda(), True)
        b_idx, pos = [], []
        for b, e in enumerate(chunk):
            tr, p = e["traj"], e["params"]
            for _, s1, r in e["texts"][channel]["spans"]:
                b_idx.append(b)
                pos.append(s1)  # token s1 = last char of the span
                ys.append([tr["x"][r], tr["v"][r], tr["a"][r], tr["F"][r], np.log(p["m"]), p["mu"]])
                meta.append([i + b, r, e["identifiable"]])
        for L, h in enumerate(resid):
            feats[L].append(h[b_idx, pos].float().cpu().numpy())
    return [np.concatenate(f) for f in feats], np.array(ys), np.array(meta)


def fit_eval(feats, y, train, test):
    """R^2 per layer per target, plus test predictions per layer."""
    out, preds = [], []
    for X in feats:
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
        Xs = (X - mu) / sd
        probe = RidgeCV(alphas=np.logspace(-1, 4, 6), alpha_per_target=True).fit(Xs[train], y[train])
        p = probe.predict(Xs[test])
        out.append([r2_score(y[test][:, j], p[:, j]) for j in range(y.shape[1])])
        preds.append(p)
    return np.array(out), preds


def table(name, r2):
    print(f"\n{name}  (R^2 on held-out episodes)")
    print("layer " + "".join(f"{t:>8}" for t in TARGETS))
    for L, row in enumerate(r2):
        print(f"{L:5d} " + "".join(f"{v:8.3f}" for v in row))


def main():
    path, n = (sys.argv[1:] + [None] * 2)[:2]
    path, n = path or "ckpt/num.pt", int(n or 3000)
    model, stoi, ck = load(path)
    channel = ck["channel"]
    torch.manual_seed(1)
    rand = GPT(**ck["cfg"]).cuda().eval()

    eps = [sim.make_episode(s) for s in range(PROBE_SEED, PROBE_SEED + n)]
    results = {}
    for name, m in [("trained", model), ("random-init", rand)]:
        feats, y, meta = collect(m, eps, stoi, channel)
        train, test = meta[:, 0] < 0.75 * n, meta[:, 0] >= 0.75 * n  # split by episode, not by span
        r2, preds = fit_eval(feats, y, train, test)
        table(f"{name} model, channel {channel}", r2)
        results[name] = (r2, preds, y[test], meta[test])

    # hidden parameters: identifiable vs non-identifiable test episodes, late spans only (dynamics observed)
    r2, preds, yt, mt = results["trained"]
    print("\nhidden params, trained model, best layer per target, spans with step >= 25:")
    for j in (TARGETS.index("log_m"), TARGETS.index("mu")):
        L = int(r2[:, j].argmax())
        late = mt[:, 1] >= 25
        for label, mask in [("identifiable", late & (mt[:, 2] == 1)), ("non-identifiable", late & (mt[:, 2] == 0))]:
            print(f"  {TARGETS[j]:6s} layer {L}  {label:17s} n={mask.sum():5d}  R^2 {r2_score(yt[mask, j], preds[L][mask, j]):.3f}")


if __name__ == "__main__":
    main()
