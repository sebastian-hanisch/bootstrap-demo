"""Bootstrap: wann Resampling die Verteilung ersetzt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 15 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Der Bootstrap ersetzt die unbekannte Verteilung durch die Stichprobe selbst. Das trägt nur so weit, wie die Stichprobe die Verteilung kennt:
bei schiefen Daten und kleinem n deckt das Perzentil-Intervall nicht besser als das t-Intervall, bei seltenen Ereignissen ist es nahezu das Wald-Intervall, beim Maximum versagt es ganz, bei abhängigen
Daten braucht es Blöcke; Bootstrap-t kommt dem Niveau am nächsten und kostet Breite. Wahrheiten sind exakt bekannt, Abdeckungen gemessen.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import bst_constants as C
import bst_evaluation as ev
from bst_evaluation import Settings, de, pi_text, pr, prozent
from bst_layout import dunkles_design, kopf_css
from bst_pdf_export import build_pdf
from bst_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from bst_scenario import named_netz, parse_numbers
from bst_visualization import build_b, build_coverage, build_dep, build_exact, build_misses, build_rare, build_stats

st.set_page_config(page_title="Bootstrap: wann Resampling die Verteilung ersetzt – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Abdeckung messen …")
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Statistiken messen …")
def _stats(sigma, n, B, seed):
    return ev.stat_experiment(sigma, n, B, seed)


@st.cache_data(show_spinner="Exakte Abdeckung rechnen …")
def _rare(netz, p):
    return ev.rare_experiment(netz, p)


@st.cache_data(show_spinner="Zahl der Ziehungen messen …")
def _b(stat, sigma, n, seed):
    return ev.b_experiment(stat, sigma, n, seed)


@st.cache_data(show_spinner="Abhängige Daten messen …")
def _dep(phi, seed):
    return ev.dep_experiment(phi, seed)


@st.cache_data(show_spinner=False)
def _custom(text, stat, seed):
    return ev.custom_result(parse_numbers(text), stat, seed)


def _netz_label(key):
    name, n, edges = named_netz(key)
    return f"{name} ({len(edges)} Kanten)"


def _g(x):
    return f"{x:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in bst_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🥾 Bootstrap: wann Resampling die Verteilung ersetzt")
st.markdown(
    """
Der **Bootstrap** ersetzt die unbekannte Verteilung durch die **Stichprobe selbst** (jede Beobachtung mit Gewicht 1/n) und zieht daraus mit Zurücklegen neue Stichproben; die Verteilung der Statistik über diese Ziehungen soll die wahre Stichprobenverteilung ersetzen.
Das lohnt, wenn die Daten schon da sind und man die Unsicherheit einer **beliebigen Statistik** ohne Formel will. Es trägt aber nur so weit, wie die Stichprobe die Verteilung **kennt**: bei **schiefen Daten und kleinem n** deckt das Perzentil-Intervall nicht besser als das t-Intervall,
bei **seltenen Ereignissen** ist es nahezu das Wald-Intervall (Breite null ohne Ereignis), beim **Maximum** versagt es ganz, bei **abhängigen Daten** braucht es Blöcke. Gemessen wird die Abdeckung der 95-%-Intervalle; die Wahrheit ist exakt bekannt.

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Bootstrap", expanded=True):
    st.markdown(
        """
Aus einer Stichprobe x₁, …, xₙ zieht man B-mal n Werte **mit Zurücklegen** und berechnet jedes Mal die Statistik θ* (Mittelwert, Median, Quantil, …). Aus den B Werten θ* entstehen Intervalle zum Niveau 95 %:

1. **Perzentil:** das 2,5-%- bis 97,5-%-Quantil der θ*.
2. **Umgekehrtes Perzentil (basic):** an der Schätzung gespiegelt: [2θ̂ − q₉₇,₅, 2θ̂ − q₂,₅].
3. **Bootstrap-t:** man bootstrapt die studentisierte Größe t* = (θ* − θ̂)/SE* und liest die Quantile von t* aus (nur für den Mittelwert).
4. **BCa:** Perzentile, korrigiert um eine Verzerrung z₀ (Anteil θ* < θ̂) und eine Beschleunigung a (aus dem Jackknife).
5. **Zum Vergleich:** das klassische **t-Intervall** (Mittelwert), bei Bernoulli **Wilson** und **Wald**, bei abhängigen Daten der **gleitende Blockbootstrap** (Blöcke der Länge l werden zusammengesetzt statt einzelner Werte).

Die Daten sind **log-normale Dauern** exp(σZ) (schief; σ regelt die Schiefe), die **Ausfälle eines Netzes** (seltene Ereignisse) und ein **AR(1)-Prozess** (Daten mit Gedächtnis); ihre Wahrheit ist exakt bekannt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten und Statistik**")
    stat = st.selectbox("Statistik", options("stat_select"), format_func=C.STATS.get, key="stat_select", help="Welche Größe aus der Stichprobe geschätzt wird; Bootstrap-t gibt es nur für den Mittelwert.")
    sigma = st.select_slider("Schiefe σ der log-normalen Daten", options("sigma_select"), key="sigma_select", format_func=_g, help="Daten exp(σZ): je größer σ, desto schiefer und schwerer der rechte Schwanz.")
    n = st.select_slider("Stichprobengröße n (Experimente 2 und 4)", options("n_select"), key="n_select", help="Experiment 1 zeigt alle Größen; die Experimente 2 und 4 benutzen diese.")
    b = st.select_slider("Bootstrap-Ziehungen B (Experimente 1 und 2)", options("b_select"), key="b_select", help="Zahl der Ziehungen mit Zurücklegen je Stichprobe.")
    st.markdown("**Seltene Ereignisse und Abhängigkeit**")
    netz = st.selectbox("Netz (Experiment 3)", options("netz_select"), format_func=_netz_label, key="netz_select", help="Seltenes Ereignis: das Netz fällt aus; Ausfallwahrscheinlichkeit exakt aus der Aufzählung aller Kantenmengen.")
    p = st.select_slider("Haltewahrscheinlichkeit p je Kante", options("p_select"), key="p_select", format_func=_g, help="Je größer p, desto seltener der Ausfall.")
    phi = st.select_slider("Abhängigkeit φ der AR(1)-Daten (Experiment 5)", options("phi_select"), key="phi_select", format_func=_g, help="x_t = φ x_(t−1) + ε_t; φ = 0 sind unabhängige Daten.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"stat_select": stat, "sigma_select": float(sigma), "n_select": int(n), "b_select": int(b), "netz_select": netz, "p_select": float(p), "phi_select": float(phi), "seed_input": int(seed)})

settings = Settings(str(stat), float(sigma), int(n), int(b), str(netz), float(p), float(phi), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

row = next(r for r in a.rows if r["n"] == settings.n)["res"]
st.markdown("## 🎯 Ergebnis")
st.caption(f"{a.label}; Wahrheit {de(a.truth, 4)}; B = {a.B} Ziehungen, {a.R} Wiederholungen je Stichprobengröße; Intervalle zum Niveau 95 %.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wahrheit", de(a.truth, 4), help="Exakt aus der log-normalen Verteilung.")
m2.metric(f"Perzentil, n = {settings.n}", pr(row["perz"]["cov"]), delta="nominell 0,95", delta_color="off", help="Anteil der Wiederholungen, in denen das Perzentil-Intervall die Wahrheit enthält.")
m3.metric("BCa", pr(row["bca"]["cov"]), delta="nominell 0,95", delta_color="off", help="Abdeckung des BCa-Intervalls.")
m4.metric("t-Intervall" if "t" in row else "basic", pr(row["t"]["cov"]) if "t" in row else pr(row["basic"]["cov"]), delta="nominell 0,95", delta_color="off", help="Abdeckung des klassischen t-Intervalls (nur Mittelwert); sonst des umgekehrten Perzentil-Intervalls.")
st.plotly_chart(build_coverage(a), width="stretch", key="main_chart")
st.caption("Abdeckung der Intervalle gegen die Stichprobengröße: der Anteil der Wiederholungen, in denen das Intervall die Wahrheit enthält. Je näher an 0,95, desto besser.")

if st.button("📄 PDF der Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="bootstrap.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Darf man dem Bootstrap-Intervall trauen?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Abdeckung gegen die Stichprobengröße")
st.caption(f"Je Stichprobengröße {a.R} Stichproben aus der log-normalen Verteilung, je B = {a.B} Ziehungen. Die Fehltreffer zeigen die **Richtung**: bei schiefen Daten liegt die Wahrheit fast immer über dem Intervall (das Intervall ist zu niedrig).")
ms = [m for m in C.METHODS if m in a.rows[0]["res"]]
st.dataframe({"n": [r["n"] for r in a.rows], **{C.METHOD_LABELS[m]: [pr(r["res"][m]["cov"]) for r in a.rows] for m in ms},
              **{f"Breite {C.METHOD_LABELS[m]}": [de(r["res"][m]["width"], 3) for r in a.rows] for m in ("perz", "bca") + (("boott",) if "boott" in ms else ())}}, hide_index=True, width="stretch", key="cov_table")
st.plotly_chart(build_misses(a, settings.n), width="stretch", key="miss_chart")
st.caption(f"Fehltreffer bei n = {settings.n}: Anteil der Läufe, in denen die Wahrheit über beziehungsweise unter dem Intervall liegt (je Seite gewünscht 0,025).")

st.markdown("---")
st.subheader("🔬 Experiment 2: Welche Statistik?")
st.caption(f"Abdeckung von Perzentil, basic und BCa für Mittelwert, Median, 90-%-Quantil und Standardabweichung log-normaler Daten (n = {settings.n}, σ = {_g(settings.sigma)}) und für das **Maximum** einer Gleichverteilung auf (0, 1), {C.STAT_R} Wiederholungen.")
if st.button("▶ Messung starten", key="run_stats"):
    st.session_state["stats_done"] = True
if st.session_state.get("stats_done"):
    ex2 = _stats(settings.sigma, settings.n, settings.B, settings.seed)
    st.plotly_chart(build_stats(ex2), width="stretch", key="stats_chart")
    names = [C.STATS[s] for s in C.STATS] + ["Maximum (Gleichverteilung)"]
    st.dataframe({"Statistik": names, "Wahrheit": [de(ex2["rows"][s]["truth"], 4) for s in C.STATS] + [de(1.0, 4)],
                  **{C.METHOD_LABELS[m]: [pr(ex2["rows"][s]["res"][m]["cov"]) for s in C.STATS] + [pr(ex2["max"][m]["cov"]) if m in ex2["max"] else "–"] for m in ("perz", "basic", "bca")}}, hide_index=True, width="stretch", key="stats_table")
    kind, text = ev.verdict_stats(ex2)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Seltene Ereignisse")
ex3 = _rare(settings.netz, settings.p)
st.caption(f"Das Netz fällt mit Wahrscheinlichkeit π = {pi_text(ex3['pi'])} aus. Der Bootstrap einer Bernoulli-Stichprobe ist **exakt rechenbar** (die Zahl der Ausfälle in einer Ziehung ist binomial mit n und der geschätzten Rate), die Abdeckung also **ohne Simulation**: Summe über alle möglichen Ausfallzahlen.")
st.plotly_chart(build_rare(ex3), width="stretch", key="rare_chart")
st.dataframe({"n": [r["n"] for r in ex3["rows"]], "Perzentil-Bootstrap": [pr(r["perz"]) for r in ex3["rows"]], "Wilson": [pr(r["wilson"]) for r in ex3["rows"]], "Wald": [pr(r["wald"]) for r in ex3["rows"]],
              "ohne Ereignis (Breite 0)": [pr(r["p0"]) for r in ex3["rows"]], "Fehltreffer rechts (Intervall zu niedrig)": [pr(r["right"]) for r in ex3["rows"]]}, hide_index=True, width="stretch", key="rare_table")
kind, text = ev.verdict_rare(ex3)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Wie viele Ziehungen?")
st.caption(f"Abdeckung des Perzentil- und des BCa-Intervalls gegen die Zahl B der Ziehungen ({C.STATS[settings.stat]}, n = {settings.n}, σ = {_g(settings.sigma)}, {C.B_R} Wiederholungen) und Streuung der Endpunkte von Lauf zu Lauf bei denselben Daten ({C.B_ENDPOINT_RUNS} Bootstrap-Läufe, {C.B_ENDPOINT_DATASETS} Datensätze).")
if st.button("▶ Messung starten", key="run_b"):
    st.session_state["b_done"] = True
if st.session_state.get("b_done"):
    ex4 = _b(settings.stat, settings.sigma, settings.n, settings.seed)
    left, right = build_b(ex4)
    c1, c2 = st.columns(2)
    c1.plotly_chart(left, width="stretch", key="b_cov")
    c2.plotly_chart(right, width="stretch", key="b_sd")
    st.dataframe({"B": [r["B"] for r in ex4["rows"]], "Abdeckung Perzentil": [pr(r["perz"]) for r in ex4["rows"]], "Abdeckung BCa": [pr(r["bca"]) for r in ex4["rows"]],
                  "Streuung unterer Endpunkt / Breite": [prozent(r["sd_lo"], 1) for r in ex4["rows"]], "Streuung oberer Endpunkt / Breite": [prozent(r["sd_hi"], 1) for r in ex4["rows"]]}, hide_index=True, width="stretch", key="b_table")
    kind, text = ev.verdict_b(ex4)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Daten mit Gedächtnis")
st.caption(f"AR(1)-Daten x_t = φ x_(t−1) + ε_t mit φ = {_g(settings.phi)}, n = {C.DEP_N}, Wahrheit μ = 0, B = {C.DEP_B}, {C.DEP_R} Wiederholungen. Der iid-Bootstrap mischt die Werte und zerstört die Abhängigkeit; der **gleitende Blockbootstrap** zieht Blöcke der Länge l.")
if st.button("▶ Messung starten", key="run_dep"):
    st.session_state["dep_done"] = True
if st.session_state.get("dep_done"):
    ex5 = _dep(settings.phi, settings.seed)
    st.plotly_chart(build_dep(ex5), width="stretch", key="dep_chart")
    keys = ["t", "iid"] + [f"l{l}" for l in C.DEP_BLOCKS]
    st.dataframe({"Verfahren": ["t-Intervall", "iid-Bootstrap"] + [f"Block l = {l}" for l in C.DEP_BLOCKS], "Abdeckung": [pr(ex5["rows"][k]["cov"]) for k in keys], "mittlere Breite": [de(ex5["rows"][k]["width"], 3) for k in keys]},
                 hide_index=True, width="stretch", key="dep_table")
    kind, text = ev.verdict_dep(ex5)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: eine eigene kleine Stichprobe"):
    st.caption(f"Drei bis {C.CUSTOM_MAX_N} Zahlen (getrennt durch Leerzeichen, Semikolon oder Zeilenumbruch; Dezimalkomma oder -punkt). Es gilt die gewählte Statistik; alle Intervalle mit {C.CUSTOM_B} Ziehungen. Bei höchstens {C.CUSTOM_EXACT_N} Werten und dem Mittelwert zeigt die App die **exakte** Bootstrap-Verteilung (alle Multimengen).")
    text_in = st.text_area("Werte", "1\n2\n3", height=120, key="eigene_werte")
    try:
        cres = _custom(text_in, settings.stat, settings.seed)
    except ValueError as err:
        st.error(str(err))
    else:
        names = {m: C.METHOD_LABELS[m] for m in cres["ints"]}
        st.dataframe({"Intervall": list(names.values()), "untere Grenze": [de(lo, 4) for lo, hi in cres["ints"].values()], "obere Grenze": [de(hi, 4) for lo, hi in cres["ints"].values()],
                      "Breite": [de(hi - lo, 4) for lo, hi in cres["ints"].values()]}, hide_index=True, width="stretch", key="custom_table")
        st.caption(f"{C.STATS[settings.stat]} der Stichprobe: {de(cres['theta'], 4)} (n = {cres['n']}).")
        if cres["exact"] is not None:
            ex = cres["exact"]
            st.plotly_chart(build_exact(ex, cres["theta"]), width="stretch", key="custom_exact")
            st.caption(f"Exakte Bootstrap-Verteilung des Mittels ({len(ex['values'])} verschiedene Werte): Mittel {de(ex['mean'], 4)} (gleich dem Stichprobenmittel), Varianz {de(ex['var'], 4)}, exaktes Perzentil-Intervall [{de(ex['perz'][0], 4)}; {de(ex['perz'][1], 4)}]. Bei drei Beobachtungen ist es der gesamte Bereich der Daten.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Stichprobe kennt die Verteilung** | Bei schiefen Daten und kleinem n fehlen die großen Werte: das Perzentil-Intervall liegt zu niedrig und deckt nicht besser als das t-Intervall (Experiment 1, Preset »Schiefe Daten, kleines n«). | Stück 1 |
| **Die Statistik ist glatt** | Beim Maximum deckt das Perzentil-Intervall nie (0,000), bei der Standardabweichung und beim 90-%-Quantil schlecht (Experiment 2). | – |
| **Es gibt genug Ereignisse** | Bei seltenen Ereignissen ist der Bootstrap nahezu das Wald-Intervall und hat bei null Ereignissen die Breite null (Experiment 3). | Stück 14 |
| **Genug Ziehungen** | Mit B = 99 sinkt die Abdeckung um rund zwei Punkte, und die Endpunkte sind unscharf (Experiment 4). | – |
| **Die Daten sind unabhängig** | Bei Abhängigkeit deckt der iid-Bootstrap so schlecht wie das t-Intervall; Blöcke helfen, wenn sie länger sind als die Korrelationslänge (Experiment 5). | Warteschlangen 2 |
| **Die Intervalle sind gleichwertig** | Bootstrap-t kommt dem Niveau am nächsten (bei n = 10 nur 0,88 bis 0,90 statt 0,95), wird aber bei kleinem n sehr breit; BCa verbessert nur wenig; das umgekehrte Perzentil ist schlechter (Experiment 1). | – |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Die Daten sind **erfunden** (log-normal, Bernoulli, AR(1)), ihre Wahrheit ist exakt bekannt; die Netze liefern nur die Ausfallwahrscheinlichkeit der seltenen Ereignisse. Die Zufallszahlen kommen aus festen Strömen (SplitMix64, Box–Muller), nie aus einem einzelnen Seed.

**Grenzen.** Gemessen werden **nur einfache Statistiken** (Mittelwert, Median, 90-%-Quantil, Standardabweichung, Maximum), keine Quotienten, Differenzen zweier Gruppen oder Regressionen. Es gibt **keinen parametrischen Bootstrap**, keinen Bayes-Bootstrap und keine Kalibrierung (Doppel-Bootstrap). Bootstrap-t gibt es nur für den Mittelwert, mit der Standardabweichung der Ziehung als Standardfehler.
Es gibt nur eine schiefe Familie (log-normal) und einen abhängigen Prozess (AR(1)); der Blockbootstrap ist der gleitende, ohne automatische Blocklänge.

**Bekannte Probleme und Unsicherheiten.** Die Abdeckungen beruhen auf 500 Wiederholungen je Zelle (Standardfehler etwa 0,01 bis 0,015); zwei Läufe derselben Zelle mit anderem Seed weichen um bis zu 0,03 ab. Für das Mittel der Bernoulli-Daten ist die Abdeckung exakt gerechnet, nicht simuliert. Die Quellen (Efron 1979, Efron und Tibshirani, Davison und Hinkley, Künsch 1989) stehen im Mathematik-Expander **aus dem Gedächtnis**;
es wurden keine Abstracts gelesen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Bootstrap.** Aus $x_1,\dots,x_n$ ($F_n$ die empirische Verteilung, Gewicht $1/n$) werden $B$ Stichproben $x^{*b}$ mit Zurücklegen gezogen, $\theta^{*b}=\theta(x^{*b})$. Für den Mittelwert ist die Bootstrap-Verteilung exakt: $\mathbb E^*\bar x^*=\bar x$, $\operatorname{Var}^*\bar x^*=\hat\sigma^2/n$ mit $\hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$.

**Intervalle** (Niveau $1-\alpha$, $q_\gamma$ das $\gamma$-Quantil der $\theta^{*b}$). Perzentil $[q_{\alpha/2},q_{1-\alpha/2}]$; basic $[2\hat\theta-q_{1-\alpha/2},2\hat\theta-q_{\alpha/2}]$; Bootstrap-t: $t^*=(\theta^*-\hat\theta)/\mathrm{SE}^*$, Intervall $[\hat\theta-t^*_{1-\alpha/2}\widehat{\mathrm{SE}},\hat\theta-t^*_{\alpha/2}\widehat{\mathrm{SE}}]$;
BCa: $\gamma_1=\Phi\bigl(z_0+\frac{z_0+z_{\alpha/2}}{1-a(z_0+z_{\alpha/2})}\bigr)$, $\gamma_2$ entsprechend mit $z_{1-\alpha/2}$, $z_0=\Phi^{-1}\bigl(\#\{\theta^*<\hat\theta\}/B\bigr)$, $a=\frac{\sum(\bar\theta_{(\cdot)}-\theta_{(i)})^3}{6\,(\sum(\bar\theta_{(\cdot)}-\theta_{(i)})^2)^{3/2}}$ aus dem Jackknife $\theta_{(i)}$ (ohne die $i$-te Beobachtung).

**Seltene Ereignisse.** Bei Bernoulli-Daten mit $S$ Ereignissen in $n$ Beobachtungen ist $S^*\sim\mathrm{Binomial}(n,S/n)$, die Perzentil-Grenzen sind Binomialquantile $/n$; die Abdeckung ist $\sum_s\mathbb P(S=s)\,\mathbf 1[\pi\in I_s]$ mit $S\sim\mathrm{Binomial}(n,\pi)$ (exakt, ohne Simulation). Bei $S=0$ ist $S^*\equiv0$: Intervall $[0,0]$, Breite null, $\mathbb P(S=0)=(1-\pi)^n$.

**Maximum.** $\mathbb P(\max x^*=\max x)=1-(1-1/n)^n\to1-e^{-1}\approx0{,}632$; der Bootstrap zieht nie über $\max x$ hinaus, das Perzentil-Intervall liegt unterhalb des wahren oberen Endpunkts.

**Abhängige Daten.** Für ein stationäres Zeitreihenmittel ist $\operatorname{Var}\bar x\approx\frac1n\sum_h\gamma_h$ (Langzeitvarianz), bei AR(1) mit Rauschvarianz 1 $=\frac1n(1-\varphi)^{-2}$. Der iid-Bootstrap trifft nur $\gamma_0/n$. Gleitender Blockbootstrap: $\lceil n/l\rceil$ Blöcke der Länge $l$ aus den $n-l+1$ gleitenden Blöcken, aneinandergesetzt und auf $n$ gekürzt; sinnvoll für $l$ größer als die Korrelationslänge und $l\ll n$.

**Literatur (aus dem Gedächtnis, keine Abstracts gelesen).** Efron 1979, „Bootstrap methods: another look at the jackknife“ (Annals of Statistics); Efron und Tibshirani 1993, „An Introduction to the Bootstrap“; Davison und Hinkley 1997, „Bootstrap Methods and their Application“; Künsch 1989, „The jackknife and the bootstrap for general stationary observations“ (Annals of Statistics); Hall 1992, „The Bootstrap and Edgeworth Expansion“.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
