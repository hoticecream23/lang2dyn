"""Char-level GPT trained from scratch on one verbalizer channel. One episode per sequence.

python train.py [channel=num] [n_episodes=100000] [steps=5000]   -> ckpt/<channel>.pt
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


def load(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = GPT(**ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), {c: i + 3 for i, c in enumerate(ck["chars"])}, ck


def main():
    channel, n, steps = (sys.argv[1:] + [None] * 3)[:3]
    channel, n, steps = channel or "num", int(n or 100_000), int(steps or 5000)
    torch.manual_seed(0)
    texts = lambda seeds: [sim.make_episode(s)["texts"][channel]["text"] for s in seeds]
    t0 = time.time()
    train_t, val_t = texts(range(n)), texts(range(VAL_SEED, VAL_SEED + 2000))
    chars = sorted(set("".join(train_t)))
    stoi = {c: i + 3 for i, c in enumerate(chars)}
    model = GPT(len(chars) + 3).cuda()
    X, V = encode(train_t, stoi, model.cfg["block"]), encode(val_t, stoi, model.cfg["block"]).cuda()
    print(f"{channel}: {n} episodes, vocab {len(chars) + 3}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"data {time.time() - t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    lr = lambda s: min(1, s / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr)
    t0 = time.time()
    for step in range(steps + 1):
        if step % 500 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                val = sum(lm_loss(model, V[i:i + 250]).item() for i in range(0, len(V), 250)) / (len(V) // 250)
            print(f"step {step:5d}  val {val:.4f}  {time.time() - t0:.0f}s", flush=True)
            model.train()
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

    os.makedirs("ckpt", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": model.cfg, "chars": chars, "channel": channel}, f"ckpt/{channel}.pt")
    print(f"saved ckpt/{channel}.pt")


if __name__ == "__main__":
    main()
