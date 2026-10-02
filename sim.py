"""Phase-1 simulator (one cart, piecewise-constant force, friction + stiction) and num/qual verbalizers.

python sim.py                 run self-checks
python sim.py N out.jsonl     write N episodes (seeds 0..N-1)
"""
import itertools, json, math, random, re, sys

G = 9.8
SUB, DT = 10, 0.01   # substeps per recorded step; substep length (record every 0.1 s)
STEPS = 50           # recorded steps per episode (5 s)
K = 5                # periodic report every K recorded steps


def sign(v):
    return (v > 0) - (v < 0)


def accel(F, v, m, mu):
    fr = mu * m * G
    if v:
        return (F - fr * sign(v)) / m
    if abs(F) <= fr:
        return 0.0  # static friction holds
    return (F - fr * sign(F)) / m


def substep(x, v, F, m, mu, dt):
    a = accel(F, v, m, mu)
    v2 = v + a * dt
    if v and v2 * v < 0:  # velocity hits zero inside the step: stop exactly there, re-evaluate friction
        tz = -v / a
        return substep(x + v * tz + 0.5 * a * tz * tz, 0.0, F, m, mu, dt - tz)
    return x + v * dt + 0.5 * a * dt * dt, v2


def sample_params(rng):
    n = 1 if rng.random() < 0.1 else rng.randint(2, 4)  # n == 1 is mostly non-identifiable (control)
    starts = [0] + sorted(rng.sample(range(5, STEPS - 5), n - 1))
    F = [round(rng.uniform(-10, 10), 1) for _ in starts]
    if n > 1 and rng.random() < 0.3:
        F[rng.randrange(1, n)] = 0.0  # coast segment
    return {
        "m": round(math.exp(rng.uniform(math.log(0.5), math.log(5))), 3),
        "mu": round(rng.uniform(0, 0.3), 3),  # max static friction 14.7 N vs |F| <= 10 N
        "x0": round(rng.uniform(0, 10), 2),
        "v0": round(rng.uniform(-3, 3), 2),
        "F_segments": [[s / 10, f] for s, f in zip(starts, F)],  # [start time s, force N]
    }


def simulate(p):
    m, mu = p["m"], p["mu"]
    segs = [(round(s * 10), f) for s, f in p["F_segments"]]  # starts in recorded-step units
    x, v = p["x0"], p["v0"]
    tr = {k: [] for k in ("t", "x", "v", "a", "F", "friction", "regime", "events")}
    for r in range(STEPS):
        F = [f for s, f in segs if s <= r][-1]
        a = accel(F, v, m, mu)
        reg = "moving" if v or a else "stuck"
        ev = []
        if r and F != tr["F"][-1]:
            ev.append("force_change")
        if r and reg != tr["regime"][-1]:
            ev.append("start" if reg == "moving" else "stop")
        for k, val in zip(tr, (round(r / 10, 1), x, v, a, F, m * a - F, reg, ev)):
            tr[k].append(val)
        for _ in range(SUB):
            x, v = substep(x, v, F, m, mu, DT)
    return tr


def identifiable(tr):
    """-> {"m": bool, "mu": bool}. Kinetic regime gives a = F*(1/m) - g*s*mu, s = direction of motion.
    Both are identifiable iff two observed (F, s) rows are linearly independent.
    mu alone is also identifiable from any coast row (F = 0, a = -g*s*mu), which says nothing about m."""
    rows = {(F, sign(v) or sign(F)) for F, v, reg in zip(tr["F"], tr["v"], tr["regime"]) if reg == "moving"}
    both = any(F1 * s2 != F2 * s1 for (F1, s1), (F2, s2) in itertools.combinations(rows, 2))
    return {"m": both, "mu": both or any(F == 0 for F, _ in rows)}


def emit_steps(tr):
    return [r for r in range(STEPS) if r % K == 0 or tr["events"][r]]


def join(sentences):
    """[(sentence, step)] -> {"text", "spans": [[char_start, char_end, step]]}"""
    spans, pos = [], 0
    for s, r in sentences:
        spans.append([pos, pos + len(s), r])
        pos += len(s) + 1
    return {"text": " ".join(s for s, _ in sentences), "spans": spans}


# ---- num ----

def verbalize_num(tr):
    return join([(f"t={tr['t'][r]:.1f} x={tr['x'][r]:.2f} v={tr['v'][r]:.2f} F={tr['F'][r]:+.1f} .", r)
                 for r in emit_steps(tr)])


NUM_RE = re.compile(r"t=(\S+) x=(\S+) v=(\S+) F=(\S+) \.")


def parse_num(sentence):
    return dict(zip("txvF", map(float, NUM_RE.fullmatch(sentence).groups())))


# ---- qual ----

def mass_bin(m):
    return "light" if m < 1 else "medium" if m <= 2.5 else "heavy"


def force_bin(F):
    return "gentle" if abs(F) < 3 else "firm" if abs(F) <= 7 else "hard"


def speed_bin(v):
    s = abs(v)
    return "still" if s < 0.05 else "slow" if s < 1 else "steady" if s < 3 else "fast"


def side(u):
    return "right" if u > 0 else "left"


def level(v):
    b = speed_bin(v)
    return "is at rest" if b == "still" else f"moves to the {side(v)} at a {b} pace"


def force_clause(F, rng):
    if F == 0:
        return rng.choice(["Nothing pushes it.", "The push stops."])
    return rng.choice(["A {} push to the {} acts on it.", "A {} push to the {} starts."]).format(force_bin(F), side(F))


def verbalize_qual(tr, m, rng):
    out, prev = [], 0
    for r in emit_steps(tr):
        v, ev = tr["v"][r], tr["events"][r]
        if r == 0:
            c = [rng.choice(["A {} cart {}.", "There is a {} cart that {}."]).format(mass_bin(m), level(v)),
                 force_clause(tr["F"][0], rng)]
        else:
            c = [force_clause(tr["F"][r], rng)] if "force_change" in ev else []
            if "stop" in ev:
                c.append(rng.choice(["It comes to a stop.", "It halts."]))
            elif "start" in ev:
                c.append(f"It starts moving to the {side(tr['a'][r])}.")
            elif tr["regime"][r] == "stuck":
                c.append("It stays still.")
            else:
                ds = abs(v) - abs(tr["v"][prev])
                fast = abs(ds) / (tr["t"][r] - tr["t"][prev]) > 1.0  # m/s^2
                change = ("keeps its pace" if abs(ds) < 0.1 else
                          f"speeds up {'quickly' if fast else 'slowly'}" if ds > 0 else
                          f"slows down {'sharply' if fast else 'gently'}")
                c.append(f"It {change} and now {level(v)}.")
        out.append((" ".join(c), r))
        prev = r
    return join(out)


def parse_qual(sentence):
    """Sentence -> partial binned state. Keys present only when the sentence states them."""
    z = {}
    if mo := re.search(r"\b(light|medium|heavy) cart", sentence):
        z["mass"] = mo[1]
    if mo := re.search(r"\b(gentle|firm|hard) push to the (left|right)", sentence):
        z["force"] = (mo[1], mo[2])
    elif re.search(r"Nothing pushes|push stops", sentence):
        z["force"] = ("none", None)
    if "at rest" in sentence or "halts" in sentence or "to a stop" in sentence or "stays still" in sentence:
        z["speed"] = "still"
    elif mo := re.search(r"at a (slow|steady|fast) pace", sentence):
        z["speed"] = mo[1]
    if mo := re.search(r"moves to the (left|right) at a|starts moving to the (left|right)", sentence):
        z["dir"] = mo[1] or mo[2]
    if mo := re.search(r"speeds up (?:slowly|quickly)|slows down (?:gently|sharply)|keeps its pace", sentence):
        z["change"] = mo[0]
    if "starts moving" in sentence:
        z["event"] = "start"
    elif "halts" in sentence or "to a stop" in sentence:
        z["event"] = "stop"
    return z


# ---- nat ----
# Ordinary language with rounded numbers: states t (as gaps), x and v (0.1 precision), F (exact). Never states a, m, mu.

def num1(u):
    return f"{round(u, 1) + 0.0:.1f}"  # + 0.0 turns -0.0 into 0.0


def at_mark(x):
    return f"at the {num1(x)} m mark"


def motion(v):
    return f"moving {side(v)} at {num1(abs(v))} m/s"


def nat_force(F, first, rng):
    if F == 0:
        return "Nothing is pushing it." if first else rng.choice(["The push is removed.", "The push stops."])
    f, d = num1(abs(F)), side(F)
    if first:
        return rng.choice([f"A {f} N push to the {d} is applied.", f"A {f} N push to the {d} begins."])
    return rng.choice([f"The push changes to {f} N toward the {d}.", f"Now a {f} N push to the {d} acts on it."])


def nat_state(tr, r, rng):
    x, v, ev = tr["x"][r], tr["v"][r], tr["events"][r]
    if "stop" in ev:
        return f"It comes to rest {at_mark(x)}."
    if "start" in ev:
        return f"It starts moving {side(tr['a'][r])} from the {num1(x)} m mark."
    if tr["regime"][r] == "stuck" or v == 0:
        return rng.choice([f"It is at rest {at_mark(x)}.", f"It stays put {at_mark(x)}."])
    return rng.choice([f"It is {at_mark(x)}, {motion(v)}.", f"It is now {motion(v)}, {at_mark(x)}."])


def verbalize_nat(tr, rng):
    out, prev = [], 0
    for r in emit_steps(tr):
        if r == 0:
            c = [nat_state(tr, 0, rng).replace("It", "A cart", 1), nat_force(tr["F"][0], True, rng)]
        else:
            c = ([nat_force(tr["F"][r], False, rng)] if "force_change" in tr["events"][r] else []) + [nat_state(tr, r, rng)]
            dt = num1(tr["t"][r] - tr["t"][prev])
            c[0] = rng.choice([f"{dt} s later,", f"{dt} seconds later,", f"After {dt} more seconds,"]) + " " + c[0][0].lower() + c[0][1:]
        out.append((" ".join(c), r))
        prev = r
    return join(out)


def parse_nat(sentence):
    """Sentence -> {"dt"?, "x", "v", "F"?}. Time comes as the gap since the previous sentence."""
    z = {}
    if mo := re.match(r"(\d+\.\d) s(?:econds)? later,|After (\d+\.\d) more seconds,", sentence):
        z["dt"] = float(mo[1] or mo[2])
    if mo := re.search(r"(\d+\.\d) N (?:push )?(?:to|toward) the (left|right)", sentence):
        z["F"] = float(mo[1]) * (1 if mo[2] == "right" else -1)
    elif re.search(r"Nothing is pushing|push is removed|push stops", sentence):
        z["F"] = 0.0
    z["x"] = float(re.search(r"(-?\d+\.\d) m mark", sentence)[1])
    if mo := re.search(r"moving (left|right) at (\d+\.\d) m/s", sentence):
        z["v"] = float(mo[2]) * (1 if mo[1] == "right" else -1)
    elif re.search(r"at rest|to rest|stays put|starts moving", sentence):
        z["v"] = 0.0
    return z


# ---- rel ----
# Only comparisons with the previous sentence: faster/slower, further left/right, push stronger/weaker/reversed.
# No numbers, no times, no absolute speed or position. m and mu are not identifiable from this channel.

SAME_V, SAME_X = 0.05, 0.05  # changes smaller than this are reported as "about the same"


def cmp(d, eps):
    return 0 if abs(d) < eps else (1 if d > 0 else -1)


def rel_codes(tr, r, prev):
    """Ground-truth comparisons the rel sentence at step r states, relative to the sentence at step prev."""
    v, vp, F, Fp = tr["v"][r], tr["v"][prev], tr["F"][r], tr["F"][prev]
    z = {}
    if "force_change" in tr["events"][r]:
        z["push"] = ("stops" if F == 0 else "starts" if Fp == 0 else "reverses" if sign(F) != sign(Fp)
                     else "stronger" if abs(F) > abs(Fp) else "weaker")
        if z["push"] == "starts" and v:
            z["with"] = sign(F) == sign(v)
    if "stop" in tr["events"][r]:
        z["motion"] = "stops"
    elif "start" in tr["events"][r]:
        z["motion"] = "starts"
    elif tr["regime"][r] == "stuck" or v == 0:
        z["motion"] = "rest"
    else:
        z["speed"] = cmp(abs(v) - abs(vp), SAME_V)
        z["reversed"] = bool(vp) and sign(v) != sign(vp)
    if z.get("motion") != "rest":
        z["x"] = cmp(tr["x"][r] - tr["x"][prev], SAME_X)
    return z


def relation(F, v):
    return "" if not v else " with its motion" if sign(F) == sign(v) else " against its motion"


def verbalize_rel(tr, rng):
    out, prev = [], 0
    for r in emit_steps(tr):
        v, F = tr["v"][r], tr["F"][r]
        if r == 0:
            c = ["A cart is at rest." if v == 0 else "A cart is moving.",
                 "Nothing pushes it." if F == 0 else f"A push acts on it{relation(F, v)}."]
        else:
            z, c = rel_codes(tr, r, prev), [rng.choice(["Later,", "Then,", "Next,"])]
            push = {"stops": "the push stops.", "starts": f"a push starts{relation(F, v)}.",
                    "reverses": "the push reverses.", "stronger": "the push grows stronger.",
                    "weaker": "the push grows weaker."}
            if "push" in z:
                c.append(push[z["push"]])
            motion = {"stops": "it comes to rest.", "starts": "it starts moving.", "rest": "it stays at rest."}
            if "motion" in z:
                c.append(motion[z["motion"]])
            else:
                pace = {1: "faster than before", -1: "slower than before", 0: "about as fast as before"}[z["speed"]]
                c.append(f"it moves {pace}{', now the other way' if z['reversed'] else ''}.")
            if "x" in z:
                c.append({1: "It is further right than before.", -1: "It is further left than before.",
                          0: "It is about where it was."}[z["x"]])
            for i in range(2, len(c)):  # clauses after the first one start new sentences
                c[i] = c[i][0].upper() + c[i][1:]
        out.append((" ".join(c), r))
        prev = r
    return join(out)


def parse_rel(sentence):
    """Sentence (r > 0) -> the comparison codes it states, same keys as rel_codes."""
    z = {}
    for k, pat in [("stops", "push stops"), ("starts", "push starts"), ("reverses", "push reverses"),
                   ("stronger", "grows stronger"), ("weaker", "grows weaker")]:
        if pat in sentence:
            z["push"] = k
    if "push starts with its motion" in sentence:
        z["with"] = True
    elif "push starts against its motion" in sentence:
        z["with"] = False
    for k, pat in [("stops", "comes to rest"), ("starts", "starts moving"), ("rest", "stays at rest")]:
        if pat in sentence:
            z["motion"] = k
    if mo := re.search(r"moves (faster|slower|about as fast) ", sentence):
        z["speed"] = {"faster": 1, "slower": -1, "about as fast": 0}[mo[1]]
        z["reversed"] = "the other way" in sentence
    for k, pat in [(1, "further right"), (-1, "further left"), (0, "about where it was")]:
        if pat in sentence:
            z["x"] = k
    return z


def parse_rel_first(sentence):
    """The first rel sentence: whether the cart moves, and the push relative to the motion."""
    push = ("none" if "Nothing pushes" in sentence else "with" if "with its motion" in sentence
            else "against" if "against its motion" in sentence else "present")
    return {"moving": "A cart is moving" in sentence, "push0": push}


# ---- sym ----
# num text under a fixed random character substitution (spaces included). Same information, positions and
# lengths as num, no readable symbols. For a from-scratch char model this should match num (sanity check).

SYM_PLAIN = "0123456789.+-=txvF "
SYM_CODE = dict(zip(SYM_PLAIN, random.Random("sym-cipher").sample("abcdefghijklmnopqrs", len(SYM_PLAIN))))
SYM_DECODE = {v: k for k, v in SYM_CODE.items()}


def verbalize_sym(num):
    return {"text": "".join(SYM_CODE[c] for c in num["text"]), "spans": num["spans"]}


def parse_sym(sentence):
    return parse_num("".join(SYM_DECODE[c] for c in sentence))


def make_episode(seed, **changes):
    """changes (e.g. m=3.0) make a counterfactual twin: same sampled params, one value overridden."""
    p = {**sample_params(random.Random(seed)), **changes}
    tr = simulate(p)
    base = f"ep_{seed:06d}"
    return {
        "id": base + "".join(f"_cf_{k}{v}" for k, v in changes.items()),
        "params": p,
        "traj": tr,
        "identifiable": identifiable(tr),
        "texts": {"num": (num := verbalize_num(tr)), "qual": verbalize_qual(tr, p["m"], random.Random(f"{seed}-surface")),
                  "nat": verbalize_nat(tr, random.Random(f"{seed}-surface-nat")),
                  "rel": verbalize_rel(tr, random.Random(f"{seed}-surface-rel")), "sym": verbalize_sym(num)},
        "cf_of": base if changes else None,
        "cf_change": changes or None,
    }


# ---- dataset splits ----
# Every split has a "train" part (what the language model and probes are fit on) and a "test" part.
#   iid         both parts from the default distribution
#   ood-combo   test = m in [3, 5] with some |F| >= 7; train = everything else (an unseen combination of seen values)
#   ood-extrap  train = default (m in 0.5-5); test = m in (5, 8] (masses never seen)
#   compose     train = frictionless episodes (mu = 0) or force-free coasting (F = 0); test = friction and force together

SPLITS = ("iid", "ood-combo", "ood-extrap", "compose")


def in_combo(p):
    return 3 <= p["m"] <= 5 and max(abs(f) for _, f in p["F_segments"]) >= 7


def split_overrides(split, part, seed):
    """Param overrides that put this seed's episode into (split, part), or None if this seed belongs to the other part."""
    p = sample_params(random.Random(seed))
    test = part == "test"
    if split == "iid":
        return {}
    if split == "ood-combo":
        return {} if in_combo(p) == test else None
    if split == "ood-extrap":
        return {"m": round(random.Random(f"{seed}-extrap").uniform(5, 8), 3)} if test else {}
    if split == "compose":
        if test:
            return {} if p["mu"] > 0 and any(f for _, f in p["F_segments"]) else None
        return {"mu": 0.0} if random.Random(f"{seed}-compose").random() < 0.5 else {"F_segments": [[0.0, 0.0]]}
    raise ValueError(split)


def split_episodes(split, part, n, start_seed):
    """n episodes of (split, part), taking seeds start_seed, start_seed + 1, ... and skipping seeds in the other part."""
    out, seed = [], start_seed
    while len(out) < n:
        if (ov := split_overrides(split, part, seed)) is not None:
            e = make_episode(seed, **ov)
            e.update(id=f"{split}_{part}_{seed:08d}", cf_of=None, cf_change=None, split=(split, part))  # overrides here are not counterfactuals
            out.append(e)
        seed += 1
    return out


def _checks():
    ep = lambda **p: simulate({"x0": 0.0, "v0": 0.0, "mu": 0.0, "m": 1.0, "F_segments": [[0.0, 0.0]], **p})

    # frictionless constant force: x = v0 t + F/(2m) t^2
    tr = ep(m=2.0, v0=1.0, F_segments=[[0.0, 4.0]])
    assert all(abs(x - (t + t * t)) < 1e-9 for t, x in zip(tr["t"], tr["x"]))
    assert identifiable(tr) == {"m": False, "mu": False}  # single force level, one direction: confounded

    # stiction: |F| < mu m g, cart never moves
    tr = ep(m=2.0, mu=0.5, F_segments=[[0.0, 5.0]])
    assert set(tr["x"]) == {0.0} and set(tr["regime"]) == {"stuck"}

    # coasting with friction: stops after v0/(mu g), distance v0^2/(2 mu g)
    tr = ep(mu=0.2, v0=2.0)
    stop = 2.0 / (0.2 * G)
    assert abs(tr["x"][-1] - 2.0 * stop / 2) < 1e-9
    r = tr["regime"].index("stuck")
    assert tr["events"][r] == ["stop"] and tr["t"][r - 1] < stop <= tr["t"][r]
    assert identifiable(tr) == {"m": False, "mu": True}  # coast reveals mu, says nothing about m

    # strong opposing force: passes through zero and reverses, never stuck at a record
    tr = ep(mu=0.1, v0=2.0, F_segments=[[0.0, -10.0]])
    assert tr["v"][-1] < 0 and "stuck" not in tr["regime"]

    # force then coast: two independent (F, s) rows, identifiable
    assert identifiable(ep(mu=0.2, v0=1.0, F_segments=[[0.0, 4.0], [1.0, 0.0]])) == {"m": True, "mu": True}
    # two force levels, no coast: both identifiable through rank 2
    assert identifiable(ep(mu=0.1, v0=1.0, F_segments=[[0.0, 4.0], [1.0, 8.0]])) == {"m": True, "mu": True}

    # round trips over random episodes: spans align, parsers recover the verbalized state
    n_m = n_mu = 0
    for seed in range(500):
        e = make_episode(seed)
        tr, n_m, n_mu = e["traj"], n_m + e["identifiable"]["m"], n_mu + e["identifiable"]["mu"]
        assert e["identifiable"]["mu"] or not e["identifiable"]["m"]
        num, qual = e["texts"]["num"], e["texts"]["qual"]
        for s0, s1, r in num["spans"]:
            z = parse_num(num["text"][s0:s1])
            assert z["t"] == tr["t"][r] and z["F"] == tr["F"][r]
            assert abs(z["x"] - tr["x"][r]) <= 0.005 + 1e-9 and abs(z["v"] - tr["v"][r]) <= 0.005 + 1e-9
        prev = 0
        for s0, s1, r in qual["spans"]:
            z = parse_qual(qual["text"][s0:s1])
            ev = tr["events"][r]
            if "dir" in z:
                assert z["dir"] == side(tr["a"][r] if "start" in ev else tr["v"][r])
            if r > 0:
                assert z.get("event") == ("stop" if "stop" in ev else "start" if "start" in ev else None)
            if "change" in z:
                ds = abs(tr["v"][r]) - abs(tr["v"][prev])
                fast = abs(ds) / (tr["t"][r] - tr["t"][prev]) > 1.0
                assert z["change"] == ("keeps its pace" if abs(ds) < 0.1 else
                                       f"speeds up {'quickly' if fast else 'slowly'}" if ds > 0 else
                                       f"slows down {'sharply' if fast else 'gently'}")
            prev = r
            if "mass" in z:
                assert r == 0 and z["mass"] == mass_bin(e["params"]["m"])
            if "force" in z:
                F = tr["F"][r]
                assert z["force"] == (("none", None) if F == 0 else (force_bin(F), side(F)))
            if "speed" in z:
                assert z["speed"] == speed_bin(tr["v"][r]), (seed, r, qual["text"][s0:s1])
        assert "speed" in parse_qual(qual["text"][slice(*qual["spans"][0][:2])])
        nat, t = e["texts"]["nat"], 0.0
        for s0, s1, r in nat["spans"]:
            z = parse_nat(nat["text"][s0:s1])
            assert ("dt" in z) == (r > 0)
            t = round(t + z.get("dt", 0.0), 1)  # time is only given as gaps; rebuild it
            assert t == tr["t"][r]
            assert abs(z["x"] - tr["x"][r]) <= 0.05 + 1e-9 and abs(z["v"] - tr["v"][r]) <= 0.05 + 1e-9
            if r == 0 or "force_change" in tr["events"][r]:
                assert z["F"] == tr["F"][r]
            else:
                assert "F" not in z
        rel, prev = e["texts"]["rel"], 0
        v0, F0 = tr["v"][0], tr["F"][0]
        assert parse_rel_first(rel["text"][slice(*rel["spans"][0][:2])]) == {
            "moving": v0 != 0, "push0": "none" if F0 == 0 else "present" if not v0 else "with" if sign(F0) == sign(v0) else "against"}
        for s0, s1, r in rel["spans"][1:]:  # the first sentence states only rest/moving and push/no push
            assert parse_rel(rel["text"][s0:s1]) == rel_codes(tr, r, prev), (seed, r, rel["text"][s0:s1])
            prev = r
        sym = e["texts"]["sym"]
        assert sym["spans"] == num["spans"] and not set(sym["text"]) & set("0123456789. ")
        for s0, s1, r in sym["spans"]:
            assert parse_sym(sym["text"][s0:s1]) == parse_num(num["text"][s0:s1])

    # splits: parts are disjoint and land in the intended regions
    for split in SPLITS:
        tr_eps, te_eps = split_episodes(split, "train", 200, 0), split_episodes(split, "test", 200, 0)
        tr_p, te_p = [e["params"] for e in tr_eps], [e["params"] for e in te_eps]
        if split == "ood-combo":
            assert not any(map(in_combo, tr_p)) and all(map(in_combo, te_p))
        if split == "ood-extrap":
            assert all(0.5 <= p["m"] <= 5 for p in tr_p) and all(5 < p["m"] <= 8 for p in te_p)
        if split == "compose":
            assert all(p["mu"] == 0 or p["F_segments"] == [[0.0, 0.0]] for p in tr_p)
            assert all(p["mu"] > 0 and any(f for _, f in p["F_segments"]) for p in te_p)
            assert {p["mu"] == 0 for p in tr_p} == {True, False}  # both kinds of training episode occur
    assert split_episodes("ood-combo", "train", 5, 0)[0]["id"].startswith("ood-combo_train_")

    # counterfactual twin differs only in the overridden param
    a, b = make_episode(7), make_episode(7, m=4.0)
    assert {k for k in a["params"] if a["params"][k] != b["params"][k]} <= {"m"} and b["cf_of"] == a["id"]

    print(f"all checks passed; of 500 random episodes, m identifiable in {n_m}, mu in {n_mu}")
    print(make_episode(3)["texts"]["qual"]["text"])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        with open(sys.argv[2], "w") as f:
            for seed in range(int(sys.argv[1])):
                f.write(json.dumps(make_episode(seed)) + "\n")
    else:
        _checks()
