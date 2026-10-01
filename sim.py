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
    """Kinetic regime gives a = F*(1/m) - g*s*mu, s = direction of motion.
    m and mu are both identifiable iff two observed (F, s) rows are linearly independent."""
    rows = {(F, sign(v) or sign(F)) for F, v, reg in zip(tr["F"], tr["v"], tr["regime"]) if reg == "moving"}
    return any(F1 * s2 != F2 * s1 for (F1, s1), (F2, s2) in itertools.combinations(rows, 2))


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
    return z


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
        "texts": {"num": verbalize_num(tr), "qual": verbalize_qual(tr, p["m"], random.Random(f"{seed}-surface"))},
        "cf_of": base if changes else None,
        "cf_change": changes or None,
    }


def _checks():
    ep = lambda **p: simulate({"x0": 0.0, "v0": 0.0, "mu": 0.0, "m": 1.0, "F_segments": [[0.0, 0.0]], **p})

    # frictionless constant force: x = v0 t + F/(2m) t^2
    tr = ep(m=2.0, v0=1.0, F_segments=[[0.0, 4.0]])
    assert all(abs(x - (t + t * t)) < 1e-9 for t, x in zip(tr["t"], tr["x"]))
    assert not identifiable(tr)  # single force level, one direction: m and mu confounded

    # stiction: |F| < mu m g, cart never moves
    tr = ep(m=2.0, mu=0.5, F_segments=[[0.0, 5.0]])
    assert set(tr["x"]) == {0.0} and set(tr["regime"]) == {"stuck"}

    # coasting with friction: stops after v0/(mu g), distance v0^2/(2 mu g)
    tr = ep(mu=0.2, v0=2.0)
    stop = 2.0 / (0.2 * G)
    assert abs(tr["x"][-1] - 2.0 * stop / 2) < 1e-9
    r = tr["regime"].index("stuck")
    assert tr["events"][r] == ["stop"] and tr["t"][r - 1] < stop <= tr["t"][r]
    assert not identifiable(tr)  # coast alone says nothing about m

    # strong opposing force: passes through zero and reverses, never stuck at a record
    tr = ep(mu=0.1, v0=2.0, F_segments=[[0.0, -10.0]])
    assert tr["v"][-1] < 0 and "stuck" not in tr["regime"]

    # force then coast: two independent (F, s) rows, identifiable
    assert identifiable(ep(mu=0.2, v0=1.0, F_segments=[[0.0, 4.0], [1.0, 0.0]]))

    # round trips over random episodes: spans align, parsers recover the verbalized state
    n_id = 0
    for seed in range(500):
        e = make_episode(seed)
        tr, n_id = e["traj"], n_id + e["identifiable"]
        num, qual = e["texts"]["num"], e["texts"]["qual"]
        for s0, s1, r in num["spans"]:
            z = parse_num(num["text"][s0:s1])
            assert z["t"] == tr["t"][r] and z["F"] == tr["F"][r]
            assert abs(z["x"] - tr["x"][r]) <= 0.005 + 1e-9 and abs(z["v"] - tr["v"][r]) <= 0.005 + 1e-9
        for s0, s1, r in qual["spans"]:
            z = parse_qual(qual["text"][s0:s1])
            if "mass" in z:
                assert r == 0 and z["mass"] == mass_bin(e["params"]["m"])
            if "force" in z:
                F = tr["F"][r]
                assert z["force"] == (("none", None) if F == 0 else (force_bin(F), side(F)))
            if "speed" in z:
                assert z["speed"] == speed_bin(tr["v"][r]), (seed, r, qual["text"][s0:s1])
        assert "speed" in parse_qual(qual["text"][slice(*qual["spans"][0][:2])])

    # counterfactual twin differs only in the overridden param
    a, b = make_episode(7), make_episode(7, m=4.0)
    assert {k for k in a["params"] if a["params"][k] != b["params"][k]} <= {"m"} and b["cf_of"] == a["id"]

    print(f"all checks passed; {n_id}/500 random episodes identifiable")
    print(make_episode(3)["texts"]["qual"]["text"])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        with open(sys.argv[2], "w") as f:
            for seed in range(int(sys.argv[1])):
                f.write(json.dumps(make_episode(seed)) + "\n")
    else:
        _checks()
