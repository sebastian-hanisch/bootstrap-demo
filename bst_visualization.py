"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import bst_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(exponentformat="power")
    return lock_axes(fig)


def build_coverage(a):
    """Abdeckung gegen die Stichprobengröße (logarithmische Achse) für alle Intervalle; gestrichelt das Niveau 0,95."""
    ns = [r["n"] for r in a.rows]
    fig = go.Figure()
    methods = [m for m in C.METHODS if m in a.rows[0]["res"]]
    for m in methods:
        fig.add_trace(go.Scatter(x=ns, y=[r["res"][m]["cov"] for r in a.rows], mode="lines+markers", name=C.METHOD_LABELS[m], line=dict(color=C.METHOD_COLORS[m], dash="dot" if m == "t" else "solid")))
    fig.add_hline(y=0.95, line=dict(color=C.COLORS["ref"], dash="dash"), annotation_text="nominell 0,95", annotation_position="bottom right")
    fig.update_xaxes(title_text="Stichprobengröße n", type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Abdeckung", range=[max(0.0, min(0.95, min(r["res"][m]["cov"] for r in a.rows for m in methods)) - 0.05), 1.0])
    return _base(fig, 380)


def build_misses(a, n):
    """Fehltreffer links und rechts (Anteil der Läufe) je Intervall bei der gewählten Stichprobengröße."""
    row = next(r for r in a.rows if r["n"] == n)["res"]
    ms = [m for m in C.METHODS if m in row]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[C.METHOD_LABELS[m] for m in ms], y=[row[m]["right"] for m in ms], name="Wahrheit über dem Intervall", marker_color=C.COLOR_BAD))
    fig.add_trace(go.Bar(x=[C.METHOD_LABELS[m] for m in ms], y=[row[m]["left"] for m in ms], name="Wahrheit unter dem Intervall", marker_color=C.COLOR_OK))
    fig.add_hline(y=0.025, line=dict(color=C.COLORS["ref"], dash="dash"), annotation_text="je Seite 0,025", annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anteil der Läufe")
    return _base(fig, 320)


def build_stats(ex):
    """Abdeckung von Perzentil, basic und BCa je Statistik, dazu das Maximum einer Gleichverteilung."""
    names = [C.STATS[s] for s in C.STATS] + ["Maximum (Gleichverteilung)"]
    palette = {"perz": C.METHOD_COLORS["perz"], "basic": C.METHOD_COLORS["basic"], "bca": C.METHOD_COLORS["bca"]}
    fig = go.Figure()
    for m in ("perz", "basic", "bca"):
        ys = [ex["rows"][s]["res"][m]["cov"] for s in C.STATS] + [ex["max"][m]["cov"] if m in ex["max"] else None]
        fig.add_trace(go.Bar(x=names, y=[max(y, 1e-6) if y is not None else None for y in ys], name=C.METHOD_LABELS[m], marker_color=palette[m]))
    fig.add_hline(y=0.95, line=dict(color=C.COLORS["ref"], dash="dash"), annotation_text="nominell 0,95", annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Abdeckung", range=[0, 1.05])
    return _base(fig, 360)


def build_rare(ex):
    """Exakte Abdeckung von Perzentil-Bootstrap, Wilson und Wald gegen n bei dem seltenen Ereignis."""
    ns = [r["n"] for r in ex["rows"]]
    fig = go.Figure()
    for key, name, color, dash in (("perz", "Perzentil-Bootstrap", C.COLORS["perz"], "solid"), ("wald", "Wald", C.COLORS["wald"], "dot"), ("wilson", "Wilson", C.COLORS["wilson"], "solid")):
        fig.add_trace(go.Scatter(x=ns, y=[r[key] for r in ex["rows"]], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.add_trace(go.Scatter(x=ns, y=[r["p0"] for r in ex["rows"]], mode="lines+markers", name="Anteil ohne Ereignis (Breite 0)", line=dict(color=C.COLORS["ref"], dash="dashdot")))
    fig.add_hline(y=0.95, line=dict(color=C.COLORS["ref"], dash="dash"), annotation_text="nominell 0,95", annotation_position="bottom right")
    fig.update_xaxes(title_text="Beobachtungen n", type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Abdeckung bzw. Anteil", range=[0, 1.02])
    return _base(fig, 360)


def build_b(ex):
    """Abdeckung gegen B (links) und Streuung der Endpunkte relativ zur Breite (rechts)."""
    bs = [r["B"] for r in ex["rows"]]
    cov, sd = go.Figure(), go.Figure()
    cov.add_trace(go.Scatter(x=bs, y=[r["perz"] for r in ex["rows"]], mode="lines+markers", name="Perzentil", line=dict(color=C.METHOD_COLORS["perz"])))
    cov.add_trace(go.Scatter(x=bs, y=[r["bca"] for r in ex["rows"]], mode="lines+markers", name="BCa", line=dict(color=C.METHOD_COLORS["bca"])))
    cov.add_hline(y=0.95, line=dict(color=C.COLORS["ref"], dash="dash"), annotation_text="nominell 0,95", annotation_position="bottom right")
    sd.add_trace(go.Scatter(x=bs, y=[r["sd_lo"] for r in ex["rows"]], mode="lines+markers", name="unterer Endpunkt", line=dict(color=C.COLORS["wilson"])))
    sd.add_trace(go.Scatter(x=bs, y=[r["sd_hi"] for r in ex["rows"]], mode="lines+markers", name="oberer Endpunkt", line=dict(color=C.COLORS["perz"])))
    for fig, title, log in ((cov, "Abdeckung", False), (sd, "Streuung der Endpunkte / Breite", True)):
        fig.update_xaxes(title_text="Zahl der Ziehungen B", type="log", tickvals=bs, ticktext=[str(b) for b in bs])
        fig.update_yaxes(title_text=title, type="log" if log else "linear")
    return _base(cov, 340), _base(sd, 340)


def build_dep(ex):
    """Abdeckung bei abhängigen Daten: t-Intervall, iid-Bootstrap und Blockbootstrap je Blocklänge."""
    keys = ["t", "iid"] + [f"l{l}" for l in C.DEP_BLOCKS]
    names = ["t-Intervall", "iid-Bootstrap"] + [f"Block l = {l}" for l in C.DEP_BLOCKS]
    fig = go.Figure(go.Bar(x=names, y=[max(ex["rows"][k]["cov"], 1e-6) for k in keys], marker_color=[C.METHOD_COLORS["t"], C.METHOD_COLORS["iid"]] + [C.COLORS["block"]] * len(C.DEP_BLOCKS),
                           text=[f"{ex['rows'][k]['cov']:.3f}".replace(".", ",") for k in keys], textposition="outside"))
    fig.add_hline(y=0.95, line=dict(color=C.COLORS["ref"], dash="dash"), annotation_text="nominell 0,95", annotation_position="top right")
    fig.update_yaxes(title_text="Abdeckung", range=[0, 1.1])
    fig.update_layout(showlegend=False)
    return _base(fig, 340)


def build_exact(ex, th):
    """Exakte Bootstrap-Verteilung des Mittels (kleine Stichprobe) als Stäbe."""
    fig = go.Figure(go.Bar(x=list(ex["values"]), y=list(ex["probs"]), marker_color=C.COLORS["perz"], name="Wahrscheinlichkeit"))
    fig.add_vline(x=th, line=dict(color=C.COLORS["ref"], dash="dash"), annotation_text="Stichprobenmittel", annotation_position="top")
    fig.update_xaxes(title_text="Bootstrap-Mittelwert")
    fig.update_yaxes(title_text="Wahrscheinlichkeit")
    fig.update_layout(showlegend=False)
    return _base(fig, 300)
