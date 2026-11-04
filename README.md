# Bootstrap: wann Resampling die Verteilung ersetzt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-bootstrap-demo.streamlit.app/)** -->

Fünfzehntes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [abbruchregeln-demo](https://github.com/sebastian-hanisch/abbruchregeln-demo) (Stück 14, Aufhören bei laufender Simulation) und Gegenstück zu den Stücken davor: Bis hier erzeugt die Simulation **neue** Daten aus einem Modell. Hier liegen die Daten **schon da** (ein Lauf, ein Log), und der Bootstrap soll die Unsicherheit einer beliebigen Statistik ohne Formel liefern. Die Demo misst, **wann** das trägt: wie oft die Intervalle die Wahrheit treffen, wenn diese exakt bekannt ist. Vehikel sind log-normale Dauern (schief, Wahrheit in geschlossener Form), die **Zuverlässigkeit der Netze der Reihe** (seltene Ereignisse, π exakt aus der Aufzählung) und ein AR(1)-Prozess (Daten mit Gedächtnis).

## Warum dieses Problem

Der Bootstrap ersetzt die unbekannte Verteilung durch die Stichprobe selbst, zieht daraus mit Zurücklegen neue Stichproben und liest die Verteilung der Statistik über diese Ziehungen als Ersatz für die wahre Stichprobenverteilung (Efron 1979, aus dem Gedächtnis). Das klingt nach einem Werkzeug ohne Bedingungen. Gemessen ist es ein Werkzeug mit **fünf bekannten Fallen**: kleines n bei schiefen Daten, Statistiken, die der Bootstrap nicht kann, seltene Ereignisse, zu wenige Ziehungen und abhängige Daten. Die Kernfrage der Demo: **Deckt das 95-%-Intervall in 95 % der Fälle?**

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat oder beim Bau dazukam:

- **Das Perzentil-Intervall ist bei kleinem n nicht besser als das t-Intervall,** sondern etwas schlechter: Mittelwert log-normaler Daten (σ = 1), n = 10: Perzentil **0,826**, t **0,862** (andere Ströme und eine unabhängige Kontrolle: etwa 0,80 und 0,84); bei n = 50 etwa gleichauf: 0,880 gegen 0,878. Die Vermutung „Bootstrap rettet schiefe Daten“ trägt nicht. Bootstrap-t kommt dem Niveau am nächsten (n = 10: 0,902 bei σ = 1, 0,876 bei σ = 1,5, nominell 0,95).
- **Die Fehltreffer liegen fast alle auf einer Seite:** beim Perzentil-Intervall (σ = 1, n = 10) liegt die Wahrheit in 16,4 % der Läufe über dem Intervall (andere Ströme: 18 bis 19 %) und in 1,0 % darunter. Das Intervall ist zu kurz, weil der Bootstrap die lange rechte Flanke nicht kennt.
- **Bootstrap-t zahlt für die Abdeckung mit der Breite:** n = 10, σ = 1: 0,902 bei mittlerer Breite 4,58 gegen 1,94 beim Perzentil; bei σ = 1,5: 0,876 bei 29,3 gegen 5,2 (die mittlere Breite von Bootstrap-t ist wegen weniger riesiger Intervalle nur grob bestimmt).
- **Nicht jede Statistik trägt:** Standardabweichung (n = 50) 0,586 beim Perzentil, 90-%-Quantil 0,884, Median 0,946; das **Maximum** einer Gleichverteilung deckt mit dem Perzentil-Intervall **nie** (0,000), weil in 63,6 % der Bootstrap-Stichproben dasselbe Maximum steht wie in der Stichprobe.
- **Seltene Ereignisse:** der Bootstrap für eine Wahrscheinlichkeit ist exakt über die Binomialverteilung berechenbar. Bei π = 0,0118 (Gitter 3×3, p = 0,95) und n = 30 hat die Stichprobe in **70,0 %** der Fälle null Ereignisse, die Breite ist null, die Abdeckung **0,300**. Wilson hält 0,951.
- **Zu wenige Ziehungen:** B = 19 verliert beim Perzentil-Intervall etwa 0,08 Abdeckung gegenüber B = 999, B = 99 etwa 0,02 (Kontrolle mit gemeinsamen Datensätzen; die Tabelle unten lässt die Zeilen auf verschiedenen Datensätzen laufen, ihre Differenz hat einen Standardfehler von etwa 0,02), die Endpunkte streuen von Lauf zu Lauf um 10 bis 15 % der Breite.
- **Abhängige Daten:** der iid-Bootstrap deckt bei AR(1) mit φ = 0,8 und n = 500 nur etwa die Hälfte (0,510), der gleitende Blockbootstrap mit Blocklänge 25 kommt auf 0,900, aber bei 2,6-facher Breite (die wahre Standardabweichung des Mittels ist etwa das 3-Fache der iid-Schätzung, daher deckt auch dieser nur etwa 0,9).
- **Kein scipy zur Laufzeit:** die inverse Normalverteilung (Acklam mit Halley-Schritt) und die t-Verteilung (endliche Reihe plus Bisektion) sind eigene Näherungen, gegen `scipy.stats` bis 1e-8 (relativ) geprüft; die Bootstrap-Stichproben kommen aus SplitMix64-Strömen.

## Modell

Daten x_1, …, x_n aus einer **log-normalen Verteilung** exp(σ Z) (σ = 0,5 / 1,0 / 1,5; Wahrheiten in geschlossener Form: Mittelwert e^(σ²/2), Median 1, 90-%-Quantil e^(1,2816 σ), Standardabweichung √(e^(σ²)(e^(σ²) − 1))), oder **Bernoulli-Daten** mit der Ausfallwahrscheinlichkeit π eines Netzes (Hantel, Gitter 3×3, 3×4; jede Kante hält mit Wahrscheinlichkeit p; π exakt aus der Aufzählung aller Kantenmengen: Gitter 3×3, p = 0,95: 0,01184; p = 0,9: 0,05302; Hantel, p = 0,9: 0,14969; p = 0,7: 0,48901), oder einen **AR(1)-Prozess** x_t = φ x_(t−1) + ε_t. Der 📐-Expander der App rechnet die Formeln vor.

## Methodik

**Intervalle** (Niveau 95 %, zweiseitig): **t** (nur Mittelwert), **Perzentil** [q_(α/2), q_(1−α/2)] der Bootstrap-Verteilung, **basic** [2θ̂ − q_(1−α/2), 2θ̂ − q_(α/2)], **Bootstrap-t** (studentisiert, nur Mittelwert) und **BCa** (Verzerrungskorrektur z₀ aus dem Anteil θ* < θ̂ mit halbem Gewicht für Gleichstände, Beschleunigung aus dem Jackknife). Für Bernoulli-Daten dazu **Wilson** und **Wald**, für abhängige Daten der **gleitende Blockbootstrap** (Blöcke der Länge l, n/l Blöcke mit Zurücklegen). **Experiment 1** (Kernfrage): je Stichprobengröße n = 10 bis 200 werden 500 Datensätze erzeugt, auf jeden der Bootstrap mit B Ziehungen angewendet und der Anteil der Intervalle gezählt, die die exakte Wahrheit enthalten. **Experiment 2:** dasselbe für Median, 90-%-Quantil, Standardabweichung (n = 50, 500 Wiederholungen) und das Maximum einer Gleichverteilung. **Experiment 3** (seltene Ereignisse) ist **exakt** und ohne Simulation: die Abdeckung des Perzentil-Bootstraps, von Wilson und von Wald folgt aus der Binomialverteilung (Bootstrap-Quantile als Quantile von Binomial(n, p̂)/n). **Experiment 4:** B = 19 bis 999 (Abdeckung in 500 Wiederholungen; Streuung der Endpunkte über 40 Datensätze mit je 10 Bootstrap-Läufen). **Experiment 5:** AR(1) mit n = 500, 200 Wiederholungen, B = 299, Blocklängen 5, 10, 25, 50.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Strom 39, `tests/test_claims.py` rechnet sie nach (Abdeckungen sind Anteile aus 500 Wiederholungen, also auf etwa ±0,015 genau, bei abhängigen Daten aus 200 auf etwa ±0,03; die Bänder der Tests sind weiter). Mittelwert log-normaler Daten, B = 499, nominell 0,95:

| n | t | Perzentil | basic | Bootstrap-t | BCa |
|---|---|---|---|---|---|
| **σ = 1,0** | | | | | |
| 10 | 0,862 | 0,826 | 0,802 | 0,902 | 0,844 |
| 20 | 0,886 | 0,872 | 0,836 | 0,920 | 0,876 |
| 50 | 0,878 | 0,880 | 0,850 | 0,906 | 0,884 |
| 100 | 0,916 | 0,916 | 0,890 | 0,934 | 0,918 |
| 200 | 0,938 | 0,924 | 0,924 | 0,944 | 0,928 |
| **σ = 1,5** | | | | | |
| 10 | 0,754 | 0,716 | 0,672 | 0,876 | 0,772 |
| 20 | 0,760 | 0,750 | 0,708 | 0,872 | 0,798 |
| 50 | 0,800 | 0,804 | 0,770 | 0,866 | 0,822 |
| 100 | 0,846 | 0,854 | 0,814 | 0,898 | 0,880 |
| 200 | 0,882 | 0,892 | 0,858 | 0,920 | 0,896 |

- **Weniger Schiefe ist leichter:** σ = 0,5, n = 10: Perzentil 0,880, t 0,922; n = 50: 0,922.
- **Breiten (σ = 1,0, n = 10):** Perzentil 1,94, Bootstrap-t 4,58 (2,4-fach). **σ = 1,5, n = 10:** Perzentil 5,22, Bootstrap-t 29,26 (5,6-fach; in einer Kontrollrechnung bis 7-fach, weil wenige riesige Intervalle den Mittelwert bestimmen); bei n = 20 etwa das 2,9-Fache (Kontrolle: 3,8).
- **Fehltreffer rechts** (Perzentil, σ = 1, n = 10): 0,164 gegen links 0,010.
- **Schiefe Daten, kleines n** (σ = 1,5, n = 10): das Perzentil-Intervall deckt nur etwa 70 % (0,716 in diesem Strom, 0,69 in anderen).
- **Andere Statistiken** (σ = 1, n = 50, B = 499; Perzentil / basic / BCa): Mittelwert 0,882 / 0,856 / 0,910; Median 0,946 / 0,854 / 0,944; 90-%-Quantil 0,884 / 0,748 / 0,926; Standardabweichung 0,586 / 0,614 / 0,670; Maximum einer Gleichverteilung: Perzentil 0,000, basic 0,860, in 63,6 % der Ziehungen dasselbe Maximum. Bei σ = 1,5 und n = 10: Standardabweichung 0,238, Median 0,966.
- **Standardabweichung und 90-%-Quantil gegen n** (B = 999, Perzentil, n = 10 / 20 / 50 / 100 / 200): Standardabweichung 0,412 / 0,536 / 0,582 / 0,682 / 0,742; 90-%-Quantil 0,688 / 0,860 / 0,880 / 0,928 / 0,932; BCa hilft beim Quantil (n = 50: 0,932 gegen 0,888), kaum bei der Standardabweichung (0,680).
- **Seltene Ereignisse, exakt** (Perzentil / Wilson / Wald; Anteil Stichproben ohne Ereignis):

| π | n = 30 | n = 100 | n = 300 | n = 1.000 | n = 3.000 |
|---|---|---|---|---|---|
| 0,01184 (3×3, p = 0,95) | 0,300 / 0,951 / 0,300; 70,0 % | 0,695 / 0,969 / 0,695; 30,4 % | 0,861 / 0,972 / 0,868; 2,8 % | 0,933 / 0,945 / 0,941 | 0,942 / 0,958 / 0,948 |
| 0,05302 (3×3, p = 0,9) | 0,801 / 0,928 / 0,801; 19,5 % | 0,898 / 0,956 / 0,898; 0,4 % | 0,941 / 0,949 / 0,948 | 0,948 / 0,944 / 0,948 | 0,948 / 0,954 / 0,948 |

  Bei häufigen Ereignissen trägt der Bootstrap: Hantel, p = 0,9 (π = 0,14969), n = 30: 0,942; Gitter 3×3, p = 0,7 (π = 0,48901), n = 30: 0,933.
- **Zahl der Ziehungen** (Mittelwert, σ = 1, n = 50; Perzentil / BCa; Streuung der Endpunkte in Anteilen der Breite, unterer / oberer Endpunkt): B = 19: 0,852 / 0,802, 0,103 / 0,154; B = 99: 0,886 / 0,892, 0,045 / 0,076; B = 999: 0,900 / 0,898, 0,015 / 0,026.
- **Abhängige Daten** (AR(1), n = 500; Abdeckung t / iid-Bootstrap / Blöcke 5 / 10 / 25 / 50): φ = 0: 0,960 / 0,950 / 0,960 / 0,950 / 0,915 / 0,900; φ = 0,5: 0,695 / 0,695 / 0,905 / 0,900 / 0,905 / 0,870; φ = 0,8: 0,535 / 0,510 / 0,810 / 0,855 / 0,900 / 0,890; φ = 0,95: 0,230 / 0,230 / 0,465 / 0,615 / 0,760 / 0,815. Mittlere Breite bei φ = 0,8: iid 0,284, Blöcke der Länge 25: 0,751; bei φ = 0,95: iid 0,528 gegen 2,457 für Blöcke der Länge 50 (4,65-fach).

## Ehrliche Grenzen

- **Nur einfache Statistiken** (Mittelwert, Median, 90-%-Quantil, Standardabweichung, Maximum); keine Quotienten, Regressionen oder Differenzen zweier Gruppen. Bootstrap-t gibt es nur für den Mittelwert, mit der Stichprobenstandardabweichung der Ziehung als Standardfehler.
- **Kein parametrischer Bootstrap, kein Bayes-Bootstrap, kein Doppel-Bootstrap, keine Kalibrierung** der Intervalle.
- **Eine schiefe Familie** (log-normal) und **ein abhängiger Prozess** (AR(1)); gleitender Blockbootstrap, kein stationärer Bootstrap, keine automatische Blocklänge.
- **Die Abdeckungen sind Anteile aus 500 (abhängige Daten: 200) Wiederholungen** und auf ±0,015 (±0,03) genau; ein anderer Strom verschiebt einzelne Zellen um bis zu 0,03 (in der Vormessung: Mittelwert bei n = 50, σ = 1 mit 0,906 gegen 0,883 bei anderem Strom).
- **Die Fehltreffer rechts und links stammen aus denselben Läufen** wie die Abdeckung.
- **Der Bootstrap für Bernoulli-Daten ist exakt gerechnet,** nicht simuliert (eine Eigenschaft der Zählstatistik); ein Bootstrap mit endlichem B hätte zusätzlich das Rauschen aus Experiment 4.
- **Quellen nur aus dem Gedächtnis** (Efron 1979; Efron und Tibshirani; Davison und Hinkley; Künsch 1989 für Blöcke); es wurde keine Quelle im Wortlaut gelesen. Die BCa-Formeln stehen im 📐-Expander so, wie sie hier implementiert und gegen `scipy.stats.bootstrap` geprüft sind.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Perzentil-, basic- und BCa-Intervall von Hand an kleinen Zahlen (inkl. Gleichstands-Gewicht), Jackknife, Bootstrap-Indizes und Blockindizes, Statistiken, Wahrheiten der Log-Normalverteilung, exakter Bernoulli-Bootstrap an Handwerten.
- **Orakel** (`tests/test_oracle_bst.py`): inverse Normalverteilung und t-Verteilung gegen `scipy.stats`; Binomialzähldichte und exakte Bernoulli-Abdeckung gegen `scipy.stats.binom`; exakte Bootstrap-Verteilung des Mittels für n ≤ 8 gegen Simulation; Bootstrap-Varianz des Mittels = σ̂²/n; **BCa und Perzentil gegen `scipy.stats.bootstrap`** (Endpunkte bis auf 4 % der Breite, der Rest ist Monte-Carlo-Rauschen); t-Intervall gegen `scipy.stats.t`; Maximum einer Gleichverteilung (63 % gleiches Maximum); AR(1)-Varianz des Mittels gegen die Langzeitformel, Blockbootstrap näher an ihr als der iid-Bootstrap.
- **Eigenschaften** (`tests/test_properties.py`): Abdeckung wächst mit n, Fehltreffer einseitig, schiefere Daten sind schwerer, Bootstrap-t deckt besser und ist breiter, basic = Spiegelung des Perzentil-Intervalls, Reihenfolge der Statistiken, seltene Ereignisse (Perzentil = Wald, Wilson besser), häufige Ereignisse nahe am Niveau, mehr Ziehungen = schärfere Endpunkte, Blockbootstrap, Reproduzierbarkeit.
- **Randfälle** (`tests/test_edge_cases.py`): zwei Beobachtungen, konstante Stichprobe, eine einzige Ziehung, Bootstrap-t ohne Streuung, Blocklänge größer als n, keine Ereignisse, exakte Verteilung nur bis n = 8, extremer Wert bei BCa, Eingabeprüfung der Spielwiese, Zahlenformate.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Ergebnis; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jeden Regler, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims** (`tests/test_claims.py`).

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `bst_algorithm.py` | Kern: Strom, Verteilungen, Bootstrap-Ziehung, fünf Intervalle, Jackknife, Blockbootstrap, exakter Bernoulli-Bootstrap |
| `bst_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `bst_evaluation.py` | Abdeckungsexperimente über feste Ströme, Verdikt-Logik |
| `bst_layout.py` | Fester Kopf (Stylesheet) |
| `bst_pdf_export.py` | PDF der aktuellen Einstellung |
| `bst_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufallsstrom-Knopf |
| `bst_scenario.py` | Netze, exakte Ausfallwahrscheinlichkeiten, Eingabeprüfung der Spielwiese |
| `bst_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Intervalle zum festen Umfang für Bernoulli-Folgen, Konfidenzfolgen | [abbruchregeln-demo](https://github.com/sebastian-hanisch/abbruchregeln-demo) (Stück 14) |
| Permutationstest und Bootstrap-Vergleich zweier Gruppen | [permutation-test-demo](https://github.com/sebastian-hanisch/permutation-test-demo) |
| Parametrischer Bootstrap, Doppel-Bootstrap, Kalibrierung | nicht gebaut |
| Stationärer Bootstrap, automatische Blocklänge | nicht gebaut |
| Regression, Quotienten, mehrere Gruppen | nicht gebaut |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
