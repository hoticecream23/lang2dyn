"""Char-level GPT trained from scratch on one verbalizer channel. One episode per sequence.

python train.py [channel=num] [n_episodes=100000] [steps=5000] [seed=0] [split=iid]
    -> ckpt/<channel>[_<split>][_s<seed>].pt   (split and seed suffixes only when not iid / 0)
split (see sim.SPLITS): trains and validates on that split's train part only.
seed sets weight init and batch order only; the training episodes are the same for every seed.
Rerun the same command after an interruption to resume from the last 500-step save.
"""
import math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import sim

PAD, BOS, EOS = 0, 1, 2
VAL_SEED = 10_000_000  # validation episodes: seeds VAL_SEED.. (train uses 0..n-1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, block=512, d=256, layers=6, heads=8):
        super().__init__()
        self.cfg = dict(vocab=vocab, block=block, d=d, layers=layers, heads=heads)
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(block, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, idx, return_resid=False):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        resid = [x]  # resid[0] = embeddings, resid[L] = output of block L
        for b in self.blocks:
            x = b(x)
            resid.append(x)
        logits = self.head(self.ln(x))
        return (logits, resid) if return_resid else logits


def encode(texts, stoi, block):
    """Token i+1 is char i of the text (BOS at 0), so a span ending at char s1 ends at token s1."""
    rows = [[BOS] + [stoi[c] for c in t] + [EOS] for t in texts]
    assert max(map(len, rows)) <= block
    out = torch.full((len(rows), max(map(len, rows))), PAD)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


def lm_loss(model, xb):
    xb = xb[:, :int((xb != PAD).sum(1).max())]
    logits = model(xb[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1), ignore_index=PAD)


def save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)  # atomic: a write cut off by preemption never corrupts the last good save


def load(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = GPT(**ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), {c: i + 3 for i, c in enumerate(ck["chars"])}, ck


def main():
    channel, n, steps, seed, split = (sys.argv[1:] + [None] * 5)[:5]
    channel, n, steps, seed, split = channel or "num", int(n or 100_000), int(steps or 5000), int(seed or 0), split or "iid"
    torch.manual_seed(seed)
    name = channel + (f"_{split}" if split != "iid" else "") + (f"_s{seed}" if seed else "")
    texts = lambda count, start: [e["texts"][channel]["text"] for e in sim.split_episodes(split, "train", count, start)]
    t0 = time.time()
    train_t, val_t = texts(n, 0), texts(2000, VAL_SEED)
    chars = sorted(set("".join(train_t)))
    stoi = {c: i + 3 for i, c in enumerate(chars)}
    # context length from the data, with headroom for longer held-out episodes (num ~576, nat/qual ~1536)
    block = 64 * math.ceil(1.25 * (max(map(len, train_t)) + 2) / 64)
    model = GPT(len(chars) + 3, block=block).cuda()
    X, V = encode(train_t, stoi, model.cfg["block"]), encode(val_t, stoi, model.cfg["block"]).cuda()
    print(f"{name}: {n} episodes, vocab {len(chars) + 3}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"data {time.time() - t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    lr = lambda s: min(1, s / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr)

    # resume: ckpt/<name>_state.pt is written every 500 steps and deleted when the run finishes
    os.makedirs("ckpt", exist_ok=True)
    state_path, start = f"ckpt/{name}_state.pt", 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cuda", weights_only=True)
        if (st["n"], st["steps"], st["chars"]) == (n, steps, chars):
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            torch.set_rng_state(st["rng"].cpu())  # same batch sequence as an uninterrupted run
            start = st["step"]
            print(f"resumed from step {start}", flush=True)
        else:
            print(f"ignoring {state_path}: saved by a run with different n/steps/vocab", flush=True)

    t0 = time.time()
    for step in range(start, steps + 1):
        if step % 500 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                val = sum(lm_loss(model, V[i:i + 250]).item() for i in range(0, len(V), 250)) / (len(V) // 250)
            print(f"step {step:5d}  val {val:.4f}  {time.time() - t0:.0f}s", flush=True)
            model.train()
            if step < steps:
                save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "rng": torch.get_rng_state(), "step": step, "n": n, "steps": steps, "chars": chars}, state_path)
        if step == steps:
            break
        xb = X[torch.randint(len(X), (64,))].cuda(non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = lm_loss(model, xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    save({"model": model.state_dict(), "cfg": model.cfg, "chars": chars, "channel": channel, "seed": seed, "split": split}, f"ckpt/{name}.pt")
    if os.path.exists(state_path):
        os.remove(state_path)
    print(f"saved ckpt/{name}.pt")


if __name__ == "__main__":
    main()
