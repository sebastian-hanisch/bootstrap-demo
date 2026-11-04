"""Verdikte (reine Funktionen in bst_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import bst_constants as C
import bst_evaluation as ev

S = ev.Settings(n=50, B=499)


def _res(perz, t=0.905, bca=0.907, basic=0.879, boott=0.929, right=0.086, left=0.008):
    cell = lambda c, r=0.05, l=0.05: {"cov": c, "right": r, "left": l, "width": 1.0, "degenerate": 0.0}
    return {"t": cell(t), "perz": cell(perz, right, left), "basic": cell(basic), "boott": cell(boott), "bca": cell(bca)}


def _a(perz=0.906, **kw) -> ev.Analysis:
    rows = tuple({"n": n, "res": _res(c)} for n, c in ((10, 0.796), (20, 0.860), (50, perz), (100, 0.916), (200, 0.932)))
    basis = ev.Analysis(label="Mittelwert log-normaler Daten (σ = 1,0)", stat="mean", sigma=1.0, truth=1.6487, B=499, R=500, rows=rows)
    return dataclasses.replace(basis, **kw)


def test_hauptverdikt_zahlen():
    kind, text = ev.verdict_main(_a(), S)
    assert kind == "info"
    for frag in ("Mittelwert log-normaler Daten (σ = 1,0), n = 50, B = 499, 500 Wiederholungen", "Abdeckung t-Intervall 0,905, Perzentil 0,906, umgekehrtes Perzentil (basic) 0,879, Bootstrap-t 0,929, BCa 0,907 (nominell 0,95)",
                 "steigt von 0,796 bei n = 10 auf 0,932 bei n = 200", "in 8,6 % der Läufe auf der einen Seite (Wahrheit über dem Intervall) und in 0,8 % auf der anderen", "der Perzentil-Bootstrap ist nicht besser"):
        assert frag in text, frag


def test_hauptverdikt_schwellen_und_t_vergleich():
    assert ev.verdict_main(_a(0.899), S)[0] == "warning" and ev.verdict_main(_a(0.9), S)[0] == "info"
    assert ev.verdict_main(_a(0.929), S)[0] == "info" and ev.verdict_main(_a(0.93), S)[0] == "success"
    assert "ist hier besser" in ev.verdict_main(_a(0.95), S)[1] and "nicht besser" in ev.verdict_main(_a(0.925), S)[1]
    # andere Statistik: keine t-Spalte, kein Bootstrap-t
    rows = tuple({"n": r["n"], "res": {m: r["res"][m] for m in ("perz", "basic", "bca")}} for r in _a().rows)
    kind, text = ev.verdict_main(dataclasses.replace(_a(), stat="median", rows=rows), dataclasses.replace(S, stat="median"))
    assert "t-Intervall" not in text and "Bootstrap-t" not in text


def _stats(sd=0.586, mx=0.0):
    cell = lambda c: {"cov": c}
    mk = lambda p, b, c: {"truth": 1.0, "res": {"perz": cell(p), "basic": cell(b), "bca": cell(c)}}
    return {"sigma": 1.0, "n": 50, "B": 499, "R": 500, "rows": {"mean": mk(0.882, 0.856, 0.910), "median": mk(0.946, 0.854, 0.944), "q90": mk(0.884, 0.748, 0.926), "sd": mk(sd, 0.614, 0.670)},
            "max": {"perz": cell(mx), "basic": cell(0.860)}, "max_same": 0.636}


def test_statistikverdikt():
    kind, text = ev.verdict_stats(_stats())
    assert kind == "warning"
    for frag in ("bei n = 50 (σ = 1,0, B = 499): Mittelwert 0,882 / 0,856 / 0,910; Median 0,946 / 0,854 / 0,944; 90-%-Quantil 0,884 / 0,748 / 0,926; Standardabweichung 0,586 / 0,614 / 0,670", "Maximum einer Gleichverteilung: Perzentil 0,000, basic 0,860",
                 "64 % der Bootstrap-Stichproben haben dasselbe Maximum"):
        assert frag in text, frag
    assert ev.verdict_stats(_stats(sd=0.86, mx=0.6))[0] == "info" and ev.verdict_stats(_stats(sd=0.849, mx=0.6))[0] == "warning"


def _rare():
    rows = [{"n": n, "perz": p, "wilson": w, "wald": wa, "p0": z, "degenerate": z, "left": 0.0, "right": z} for n, p, w, wa, z in
            ((30, 0.299, 0.951, 0.300, 0.700), (100, 0.694, 0.969, 0.694, 0.305), (300, 0.860, 0.972, 0.868, 0.028), (1000, 0.932, 0.946, 0.940, 0.0), (3000, 0.942, 0.958, 0.948, 0.0))]
    return {"pi": 0.0118, "rows": rows}


def test_seltenverdikt():
    kind, text = ev.verdict_rare(_rare())
    assert kind == "warning"
    for frag in ("π = 0,0118", "bei n = 30 / 100 / 300 / 3000: 0,299 / 0,694 / 0,860 / 0,942", "Wilson 0,951 / 0,969 / 0,972 / 0,958", "Wald 0,300 / 0,694 / 0,868 / 0,948", "in 70 % der Fälle kein Ereignis"):
        assert frag in text, frag
    ok = _rare()
    ok["rows"][0]["perz"] = 0.85
    assert ev.verdict_rare(ok)[0] == "info"


def test_zahl_der_ziehungen_verdikt():
    rows = [{"B": b, "perz": p, "bca": c, "sd_lo": lo, "sd_hi": hi} for b, p, c, lo, hi in ((19, 0.803, 0.770, 0.100, 0.156), (99, 0.880, 0.884, 0.048, 0.077), (999, 0.902, 0.910, 0.016, 0.026))]
    kind, text = ev.verdict_b({"stat": "mean", "sigma": 1.0, "n": 50, "R": 500, "rows": rows})
    assert kind == "info"
    for frag in ("bei B = 19 / 99 / 999: 0,803 / 0,880 / 0,902 (BCa 0,770 / 0,884 / 0,910)", "um 10,0 % bis 15,6 % der Breite bei B = 19", "um 4,8 % bis 7,7 % bei B = 99", "um 1,6 % bis 2,6 % bei B = 999"):
        assert frag in text, frag


def _dep(iid=0.468):
    cell = lambda c, w: {"cov": c, "width": w}
    return {"phi": 0.8, "n": 500, "R": 200, "B": 299, "var_factor": 25.0, "rows": {"t": cell(0.465, 0.289), "iid": cell(iid, 0.286), "l5": cell(0.773, 0.54), "l10": cell(0.846, 0.657), "l25": cell(0.890, 0.748), "l50": cell(0.879, 0.754)}}


def test_abhaengigkeitsverdikt():
    kind, text = ev.verdict_dep(_dep())
    assert kind == "warning"
    for frag in ("φ = 0,80, n = 500", "das t-Intervall deckt 0,465, der iid-Bootstrap 0,468", "l = 5: 0,773, l = 10: 0,846, l = 25: 0,890, l = 50: 0,879", "25,0-mal so groß", "2,6-mal so breit"):
        assert frag in text, frag
    assert ev.verdict_dep(_dep(0.85))[0] == "info" and ev.verdict_dep(_dep(0.849))[0] == "warning"
