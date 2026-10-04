"""Figures for writeup_interventions.md, read from the intervention logs (local, gitignored) -> figures/*.pdf, *.png

python make_figures.py
"""
import os, re
import matplotlib.pyplot as plt
import numpy as np
import sim

# reference palette (dataviz skill, light mode): categorical slots in fixed order, text and axis tokens
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})
MARK = dict(ms=6, mec=SURFACE, mew=1.5)  # >= 8 px markers with a surface ring

ROW = re.compile(r"^  (\S+)\s+all n=\s*\d+ effect\s+(\S+) iia (\S+)(.*)$")
CELL = re.compile(r"(all|held out|seen) n=\s*\d+ effect\s+(\S+) iia (\S+)(?: src slope\s+(\S+) \[\s*(\S+),\s*(\S+)\] r\s+(\S+))?")
GAP = re.compile(r"gap ([+-]\S+) \[([+-]\S+),([+-]\S+)\]")


def parse(path):
    """-> {(layer, condition, group): dict(effect, iia, slope, lo, hi, gap, glo, ghi)}"""
    out, L = {}, 0
    for line in open(path, encoding="utf-8"):
        if m := re.match(r"^L(\d+)[:+]", line):  # "L4:", "L10:", "L2+3+4+5:" (a group is keyed by its first layer)
            L = int(m[1])
        if not ROW.match(line):
            continue
        name = line.split()[0]
        body, _, ce = line.partition("||")
        g = GAP.search(ce)
        for c in CELL.finditer(body):
            d = dict(effect=float(c[2]), iia=float(c[3]))
            if c[4]:
                d.update(slope=float(c[4]), lo=float(c[5]), hi=float(c[6]))
            if g and c[1] == "all":
                d.update(gap=float(g[1]), glo=float(g[2]), ghi=float(g[3]))
            out[(L, name, c[1])] = d
    return out


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_pair():
    """An example counterfactual pair: A's velocity, the mass switch at r0, and the counterfactual."""
    M_CF = 3.0
    seed, r0 = 50_000_000, None
    while r0 is None:  # A moving at r0 under a new force, light enough that m = 3 kg changes the motion visibly
        A = sim.make_episode(seed)
        tr = A["traj"]
        fc = [r for r in sim.emit_steps(tr) if 15 <= r <= 30 and "force_change" in tr["events"][r] and abs(tr["F"][r]) >= 5
              and "stuck" not in tr["regime"][r - 5:]]
        if fc and 0.8 <= A["params"]["m"] <= 1.5:
            cf = sim.make_episode(seed, m_switch=[fc[0], M_CF])
            if "stuck" not in cf["traj"]["regime"][fc[0]:]:
                r0 = fc[0]
        seed += 2
    cfs = {M_CF: cf}
    t = np.array(A["traj"]["t"])
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.axvspan(t[0], t[r0], color=BLUE, alpha=0.08, lw=0)
    ax.text(t[r0] / 2, 0.03, "prompt: A's text up to r0", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK2)
    ax.plot(t, A["traj"]["v"], color=BLUE, lw=2, label=f"A (m = {A['params']['m']:.2f} kg)")
    for m, cf in cfs.items():
        ax.plot(t[r0:], cf["traj"]["v"][r0:], color=ORANGE, lw=2, label=f"target: A with m = {m:.0f} kg from r0 (simulator)")
    emit = [r for r in sim.emit_steps(A["traj"]) if r <= r0]
    ax.plot(t[emit], np.array(A["traj"]["v"])[emit], "o", color=BLUE, **MARK)
    r1 = next(r for r in sim.emit_steps(A["traj"]) if r > r0)
    for traj, col in ((A["traj"], BLUE), (cfs[M_CF]["traj"], ORANGE)):
        ax.plot(t[r1], traj["v"][r1], "o", color=col, **MARK)
    ax.annotate("model decodes v here (step r1)", (t[r1], cfs[M_CF]["traj"]["v"][r1]), xytext=(18, -26), textcoords="offset points",
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axvline(t[r0], color=AXIS, lw=1)
    ax.text(t[r0], 0.97, f" r0: force changes to {A['traj']['F'][r0]:+.1f} N", transform=ax.get_xaxis_transform(), va="top", color=INK2)
    ax.set(xlabel="time (s)", ylabel="velocity (m/s)", title="A counterfactual pair")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2)
    save(fig, "fig1_pair")


def dot_rows(ax, rows, series, ylabel, ref=None):
    """rows: [(label, {series: (value, lo, hi)})]; one dot + CI bar per series, offset within each row."""
    x = np.arange(len(rows))
    for j, (s, col) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * 0.22
        for i, (_, vals) in enumerate(rows):
            if s in vals:
                v, lo, hi = vals[s]
                if lo == hi:  # a point estimate without a CI
                    ax.plot(x[i] + off, v, "o", color=col, label=s if i == 0 else None, **MARK)
                else:
                    ax.errorbar(x[i] + off, v, yerr=[[v - lo], [hi - v]], fmt="o", color=col, ecolor=col, elinewidth=1.5,
                                capsize=0, label=s if i == 0 else None, **MARK)
    if ref is not None:
        ax.axhline(ref, color=AXIS, lw=1)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def fig2_write_read():
    """Write (steer) vs read (interchange) vs shuffled control across models and targets."""
    conds = [("mass\nnum s0", "r3_ci_s0.log", 4), ("mass\nnum s1", "r3_ci_s1.log", 4), ("mass\nnum s2", "r3_ci_s2.log", 4),
             ("mass\nnat", "r3_nat.log", 5), ("friction\nnum", "r3_mu.log", 5)]
    series = [("write (true value)", BLUE), ("read (B's activations)", ORANGE), ("read, shuffled sources", MUTED)]
    names = {"write (true value)": "das-steer", "read (B's activations)": "das", "read, shuffled sources": "das-shuf"}
    slope, gap = [], []
    for label, f, L in conds:
        d = parse(f)
        slope.append((label, {s: (d[(L, n, "all")]["slope"], d[(L, n, "all")]["lo"], d[(L, n, "all")]["hi"]) for s, n in names.items()}))
        gap.append((label, {s: (d[(L, n, "all")]["gap"], d[(L, n, "all")]["glo"], d[(L, n, "all")]["ghi"]) for s, n in names.items()}))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    dot_rows(axes[0], slope, series, "source slope (1 = full effect)", ref=0)
    axes[0].set_title("Velocity change follows the source (Theil–Sen, 95% CI)")
    dot_rows(axes[1], gap, series, "CE gap, source B′ − B (nats/char)", ref=0)
    axes[1].set_title("Source information used (bootstrap 95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.text(0.5, -0.06, "k = 32 subspace, every position; L4 for num mass, L5 for nat mass and friction; 1000 held-out pairs each",
             ha="center", color=MUTED, fontsize=8)
    save(fig, "fig2_write_vs_read")


def fig3_layers():
    """Write slope by layer (L3-L5) for nat mass and num friction."""
    fig, ax = plt.subplots(figsize=(5, 3))
    for (label, f), col in zip((("nat, mass", "r3_nat.log"), ("num, friction", "r3_mu.log")), (BLUE, ORANGE)):
        d = parse(f)
        Ls = [3, 4, 5]
        v = np.array([d[(L, "das-steer", "all")]["slope"] for L in Ls])
        lo = np.array([d[(L, "das-steer", "all")]["lo"] for L in Ls])
        hi = np.array([d[(L, "das-steer", "all")]["hi"] for L in Ls])
        ax.fill_between(Ls, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(Ls, v, "-o", color=col, lw=2, label=f"write, {label}", **MARK)
        reads = [d[(L, "das", "all")]["slope"] for L in Ls]
        assert max(map(abs, reads)) < 0.02  # both reads are ~0; drawn once below
    ax.axhline(0, color=AXIS, lw=1)
    ax.text(3.02, 0.03, "read (B's activations), both: ≈ 0 at every layer", color=INK2, va="bottom")
    ax.set(xticks=[3, 4, 5], xticklabels=["L3", "L4", "L5"], xlabel="patched layer", ylabel="source slope", ylim=(-0.05, 1),
           title="The write channel strengthens with depth")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig3_layers")


def fig4_combo():
    """ood-combo: writing an unseen mass x force combination vs seen pairs, with ablation as baseline."""
    models = [("ood-combo\nseed 0", "r2_combo_combomodel.log"), ("ood-combo\nseed 1", "r3_combo_s1.log"), ("iid model\n(saw it)", "r2_combo_iidmodel.log")]
    rows = []
    for label, f in models:
        d = parse(f)
        rows.append((label, {"write, held-out combination": (d[(4, "das-steer", "held out")]["effect"],) * 3,
                             "write, seen pairs": (d[(4, "das-steer", "seen")]["effect"],) * 3,
                             "ablation, held-out combination": (d[(4, "ablate", "held out")]["effect"],) * 3}))
    fig, ax = plt.subplots(figsize=(5.6, 3))
    dot_rows(ax, rows, [("write, held-out combination", BLUE), ("write, seen pairs", AQUA), ("ablation, held-out combination", MUTED)],
             "effect (median; 1 = simulator)", ref=0)
    ax.set_ylim(-0.25, 1.05)
    ax.set_title("Unseen mass × force combinations are computed, not looked up")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig4_combo")


def fig5_compose():
    """compose: behaviour on friction + force episodes, and the friction-write slope."""
    # moving / stuck next-token loss per character on the compose parts (300 episodes each, CPU; see PROJECT_STATE round 4)
    loss = {"iid model": ((0.2644, 0.0130), (0.2641, 0.0441)),
            "compose seed 0": ((0.2065, 0.0059), (0.7211, 1.1265)),
            "compose seed 1": ((0.2147, 0.0073), (0.6358, 0.9232))}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    x = np.arange(len(loss))
    for j, (part, col) in enumerate((("train part (seen regimes)", AQUA), ("test part (friction + force)", ORANGE))):
        ax.bar(x + (j - 0.5) * 0.3, [v[j][0] for v in loss.values()], width=0.28, color=col, label=part)
    ax.set_xticks(x, list(loss))
    ax.set(ylabel="next-token loss, moving spans (nats/char)", title="Behaviour fails on the new combination", ylim=(0, 0.95))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    rows = []
    for label, f in (("iid model", "r4_cmu_ref.log"), ("compose\nseed 0", "r4_cmu_model.log"), ("compose\nseed 1", "r4_cmu_s1.log")):
        d = parse(f)
        rows.append((label, {f"L{L}": (d[(L, "das-steer", "all")]["slope"], d[(L, "das-steer", "all")]["lo"], d[(L, "das-steer", "all")]["hi"]) for L in (4, 5)}))
    dot_rows(axes[1], rows, [("L4", BLUE), ("L5", ORANGE)], "source slope, written friction", ref=0)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Written friction reaches force dynamics weakly")
    axes[1].legend(loc="upper right", title="patched layer", title_fontsize=8)
    save(fig, "fig5_compose")


def fig6_robustness():
    """On-manifold writes, multi-layer interchange, and the 38M model."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    ax = axes[0]
    rows = []
    for K in (1, 2, 3):
        d = parse(f"r5_clamp{K}.log")[(4, "das-steer", "all")]
        amp = float(re.search(r"spread / natural spread (\S+)", open(f"r5_clamp{K}.log", encoding="utf-8").read())[1])
        rows.append((f"±{K} sd\n{amp:.2f}×", {"write, bounded": (d["slope"], d["lo"], d["hi"])}))
    d = parse("r3_ci_s0.log")[(4, "das-steer", "all")]
    rows.append(("unbounded", {"write, bounded": (d["slope"], d["lo"], d["hi"])}))
    dot_rows(ax, rows, [("write, bounded", BLUE)], "source slope", ref=0)
    ax.set(ylim=(0, 1), title="Writes inside the natural range still work", xlabel="bound; spread of written values vs natural")
    ax.legend().remove()
    ax = axes[1]
    single, multi = parse("r3_ci_s0.log"), parse("r5_multi.log")
    rows = [("one layer\n(L4)", {s: (single[(4, n, "all")]["slope"], single[(4, n, "all")]["lo"], single[(4, n, "all")]["hi"])
                                 for s, n in (("read", "das"), ("read, shuffled", "das-shuf"))}),
            ("four layers\n(L2–L5)", {s: (multi[(2, n, "all")]["slope"], multi[(2, n, "all")]["lo"], multi[(2, n, "all")]["hi"])
                                       for s, n in (("read", "das"), ("read, shuffled", "das-shuf"))})]
    dot_rows(ax, rows, [("read", ORANGE), ("read, shuffled", MUTED)], "source slope", ref=0)
    ax.set(ylim=(-0.05, 1), title="Swapping natural values needs several layers")
    ax.legend(loc="upper left")
    ax = axes[2]
    big = parse("r5_big.log")
    Ls = [6, 8, 10]
    for name, col, lab in (("das-steer", BLUE, "write"), ("das", ORANGE, "read")):
        v = np.array([big[(L, name, "all")]["slope"] for L in Ls])
        lo, hi = np.array([big[(L, name, "all")]["lo"] for L in Ls]), np.array([big[(L, name, "all")]["hi"] for L in Ls])
        ax.fill_between(Ls, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(Ls, v, "-o", color=col, lw=2, label=lab, **MARK)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set(xticks=Ls, xticklabels=[f"L{L}" for L in Ls], xlabel="patched layer (of 12)", ylabel="source slope", ylim=(-0.05, 1),
           title="38M model: same pattern at mid-depth")
    ax.legend(loc="upper right")
    fig.text(0.5, -0.12, "num, mass, k = 32, every position, 1000 held-out pairs; Theil–Sen 95% CI", ha="center", color=MUTED, fontsize=8)
    save(fig, "fig6_robustness")


if __name__ == "__main__":
    for f in (fig1_pair, fig2_write_read, fig3_layers, fig4_combo, fig5_compose, fig6_robustness):
        f()
        print("made", f.__name__)
