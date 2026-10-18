"""Konfundierung: Beobachten ist nicht Eingreifen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der Linie Kausale Diagramme und Adjustierung. Eine erfundene Welt aus Standorten, Bestandsregel und Servicegrad; die Wahrheit steckt in einem
simulierten Strukturmodell, in dem die Intervention exakt gerechnet wird. Kern: kfd_model.py, Generator: kausal_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kfd_constants as C
from kausal_generator import normal_cdf
from kfd_evaluation import (
    Settings,
    analyse,
    coverage_experiment,
    de,
    mechanism_experiment,
    proxy_experiment,
    sign_map,
    signed,
    verdict_coverage,
    verdict_instance,
    verdict_map,
    verdict_mechanism,
    verdict_proxy,
)
from kfd_pdf_export import build_pdf
from kfd_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from kfd_visualization import (
    build_coverage,
    build_diagram,
    build_mechanism,
    build_proxy,
    build_scatter,
    build_sign_map,
    build_table,
    style_table,
)

st.set_page_config(page_title="Konfundierung – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _coverage():
    return coverage_experiment()


@st.cache_data(show_spinner=SPINNER)
def _map():
    return sign_map()


@st.cache_data(show_spinner=SPINNER)
def _mechanism():
    return mechanism_experiment()


@st.cache_data(show_spinner=SPINNER)
def _proxy():
    return proxy_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧭 Konfundierung: Beobachten ist nicht Eingreifen")
st.markdown(
    """
**Mehr Daten machen die falsche Antwort hier nur sicherer.** Dies ist das erste Stück der Linie **Kausale Diagramme und Adjustierung**. Ein Disponent gibt
den **hohen Sicherheitsfaktor** vor allem an Standorten mit stark schwankender Nachfrage, die ohnehin schlechter bedient werden. Vergleicht man danach die
Servicegrade mit und ohne hohen Faktor, sieht der Faktor in der Standardwelt schädlich aus, obwohl er hilft. Die Welt ist simuliert; die Simulation kennt die **wahre Wirkung**, weil
sie den Faktor auch festsetzen kann, statt ihn zu beobachten. In der Linie **Experimente und Wirkungsmessung** verteilt eine Münze den Faktor; hier tut es der Disponent.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Konfundierung", expanded=True):
    st.markdown(
        """
1. **Beobachten.** Man sieht, welcher Servicegrad bei welchem Faktor herauskam. Der naive Unterschied ist das Mittel mit hohem Faktor minus das Mittel ohne.
2. **Eingreifen.** Die Wirkung ist die Frage, was sich ändert, wenn man den Faktor **festsetzt**, ohne dass der Disponent mitwählt. Die Simulation rechnet beides.
3. **Konfundierung.** Dieselbe Schwankung beeinflusst, wer den hohen Faktor bekommt, **und** wie gut ein Standort bedient wird. Der naive Vergleich mischt
   beide Wirkungen, und die Mischung bleibt mit jeder weiteren Beobachtung dieselbe.
4. **Zwei Welten, dieselben Daten.** Dieselbe Verteilung der Beobachtungen kann aus verschiedenen Ursachen-Diagrammen kommen. Welches gilt, steht nicht in den
   Zahlen (Experiment 3).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Standorte**")
    n = st.slider("Standorte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Standorte in der Stichprobe stehen.")
    sigma = st.slider("Reststreuung σ (Punkte)", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider",
                      help="Streuung des Servicegrads, die weder Schwankung noch Faktor erklärt.")
    st.markdown("**Ursachen (simuliert)**")
    beta = st.slider("Einfluss der Schwankung β (Punkte je Einheit)", *bounds("beta_slider"), step=C.BETA_STEP, key="beta_slider",
                     help="Wie viele Punkte Servicegrad ein Standort je Einheit Schwankung verliert.")
    tau = st.slider("Wahre Wirkung τ des hohen Faktors (Punkte)", *bounds("tau_slider"), step=C.TAU_STEP, key="tau_slider",
                    help="Was der hohe Faktor in der Welt tatsächlich bewirkt, wenn man ihn festsetzt.")
    st.markdown("**Zuteilung**")
    gamma = st.slider("Auswahlstärke γ", *bounds("gamma_slider"), step=C.GAMMA_STEP, key="gamma_slider",
                      help="Wie stark der Disponent den hohen Faktor nach der Schwankung vergibt. 0: unabhängig von der Schwankung.")
    assign = st.selectbox("Wer verteilt den Faktor?", C.ASSIGN_CHOICES, key="assign_select", format_func=lambda v: C.ASSIGN_LABELS[v],
                          help="Bei Münzwurf bekommt jeder Standort den hohen Faktor mit derselben Wahrscheinlichkeit wie unter der Auswahl, unabhängig von der Schwankung.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "gamma_slider": round(float(gamma), 2), "beta_slider": round(float(beta), 1), "tau_slider": round(float(tau), 1),
                   "sigma_slider": round(float(sigma), 1), "assign_select": assign, "seed_input": int(seed)})

settings = Settings(int(n), round(float(gamma), 2), round(float(beta), 1), round(float(tau), 1), round(float(sigma), 1), assign, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Anteil hoher Faktor", f"{de(100 * a.share_treated, 0)} %", delta=f"{int(a.x.sum())} von {settings.n}", delta_color="off",
          help="Anteil der Standorte, die den hohen Sicherheitsfaktor bekommen haben.")
c2.metric("Mittel mit Faktor", f"{de(a.mean_treated, 1)}", delta="Servicegrad in Punkten", delta_color="off", help="Mittlerer Servicegrad der Standorte mit hohem Faktor.")
c3.metric("Mittel ohne Faktor", f"{de(a.mean_untreated, 1)}", delta="Servicegrad in Punkten", delta_color="off", help="Mittlerer Servicegrad der Standorte ohne hohen Faktor.")
c4.metric("Naiver Unterschied", f"{signed(a.naive, 2)}", delta=f"95-%-Intervall {de(a.lo, 1)} bis {de(a.hi, 1)}", delta_color="off",
          help="Mittel mit hohem Faktor minus Mittel ohne, in Punkten. Intervall nach Welch (Normalnäherung).")

tab_scatter, tab_table = st.tabs(["Streudiagramm", "Tabelle"])
reveal = st.session_state.get("reveal_truth", False)
with tab_scatter:
    st.plotly_chart(build_scatter(a, reveal), width="stretch", key="scatter_chart")
    st.caption("Jeder Punkt ist ein Standort. Gestrichelt: die Mittel der beiden Gruppen; ihr Abstand ist der naive Unterschied. Der hohe Faktor sitzt "
               "rechts, wo die Servicegrade ohnehin niedriger liegen.")
with tab_table:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Standorte der Stichprobe; die Schwankung ist standardisiert (0 = typischer Standort).")

if st.button("📄 PDF der aktuellen Stichprobe erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="konfundierung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Zeigt der Vergleich, was der Faktor bewirkt?")
st.toggle("🙈 Wahrheit enthüllen (wahre Wirkung und Ursachen im Streudiagramm)", key="reveal_truth")
if st.session_state.get("reveal_truth", False):
    r1, r2, r3 = st.columns(3)
    r1.metric("Wahre Wirkung (Faktor festgesetzt)", f"{signed(a.truth, 2)} Punkte")
    r2.metric("Fehler des naiven Vergleichs", f"{signed(a.naive - a.truth, 2)} Punkte")
    r3.metric("Grenzwert für n gegen unendlich", f"{signed(a.limit, 2)} Punkte",
              help="Wohin der naive Unterschied mit wachsender Stichprobe läuft (geschlossene Formel, siehe Mathematische Formulierung).")
verdict(verdict_instance(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Mehr Daten, gleicher Fehler")
st.caption(f"Standardwelt (γ = {de(C.DEFAULT_GAMMA, 1)}, β = {de(C.DEFAULT_BETA, 1)}, τ = {de(C.DEFAULT_TAU, 1)}, σ = {de(C.DEFAULT_SIGMA, 1)}), {C.EXP_REPS} feste "
           "Wiederholungen je Stichprobengröße, unabhängig von den Reglern. Gezeigt ist, wie oft das 95-%-Intervall die wahre Wirkung enthält.")
rows1 = _coverage()
st.plotly_chart(build_coverage(rows1), width="stretch", key="coverage_chart")
verdict(verdict_coverage(rows1))
st.dataframe(
    [{"Standorte n": r["n"], "Naiv: Mittel": de(r["naive_mean"], 2), "Naiv: falsches Vorzeichen": f"{de(100 * r['naive_wrong'], 0)} %",
      "Naiv: Deckung": f"{de(100 * r['naive_cover'], 0)} %", "Münzwurf: Deckung": f"{de(100 * r['coin_cover'], 0)} %",
      "Mit Schwankung gerechnet: Deckung": f"{de(100 * r['adj_cover'], 0)} %"} for r in rows1],
    hide_index=True, width="stretch")
st.caption("„Mit Schwankung gerechnet“ heißt hier: die Schwankung als Kontrollvariable in einer Regression. Warum das gilt, wann es nicht gilt und was man "
           "stattdessen darf, steht in den Folgestücken **Das Diagramm lesen: d-Separation** und **Backdoor-Adjustierung**.")

st.subheader("🔬 Experiment 2: Wie stark muss die Auswahl sein?")
st.caption(f"Grenzwert des naiven Unterschieds (geschlossene Formel) für τ = {de(C.DEFAULT_TAU, 1)} über Auswahlstärke γ und Einfluss der Schwankung β. "
           f"Darunter, wie oft der Vergleich mit {C.EXP_MAP_N} Standorten das falsche Vorzeichen zeigt ({C.EXP_REPS} feste Wiederholungen je Zelle).")
rows2 = _map()
st.plotly_chart(build_sign_map(rows2), width="stretch", key="map_chart")
verdict(verdict_map(rows2))
std_cell = next(r for r in rows2 if r["gamma"] == 1.0 and r["beta"] == 4.0)
st.markdown(
    "**Anteil der Stichproben mit falschem Vorzeichen** (n = %d)\n\n" % C.EXP_MAP_N
    + "| β \\ γ | " + " | ".join(de(g, 2) for g in C.EXP_MAP_GAMMAS) + " |\n|---|" + "---|" * len(C.EXP_MAP_GAMMAS) + "\n"
    + "\n".join("| " + de(b, 1) + " | " + " | ".join(
        de(100 * next(r['wrong'] for r in rows2 if r['gamma'] == g and r['beta'] == b), 0) + " %" for g in C.EXP_MAP_GAMMAS) + " |" for b in C.EXP_MAP_BETAS)
)
st.caption(f"In der Standardwelt (γ = 1, β = 4) zeigt der Vergleich in {de(100 * std_cell['wrong'], 0)} % der Stichproben einen Schaden. Fehlt die Auswahl "
           "(γ = 0) oder die Wirkung der Schwankung (β = 0), verschwindet der Fehler vollständig.")

st.subheader("🔬 Experiment 3: Dieselben Daten, zwei Wahrheiten")
st.caption(f"{C.EXP_MECH_RUNS} Läufe mit je {C.EXP_MECH_N} Standorten je Diagramm, Standardwelt. Diagramm A: die Schwankung bestimmt den Faktor. Diagramm B: der "
           "Faktor bestimmt die Schwankung (zum Beispiel, weil der hohe Faktor Nachbarn anzieht). Beide erzeugen dieselbe gemeinsame Verteilung.")
res3 = _mechanism()
which = st.radio("Welches Diagramm gilt?", ("A", "B"), key="diagram_choice", horizontal=True,
                 format_func=lambda v: "A: Schwankung → Faktor" if v == "A" else "B: Faktor → Schwankung")
d1, d2 = st.columns([1, 1])
with d1:
    st.plotly_chart(build_diagram(which), width="stretch", key="diagram_chart")
with d2:
    st.plotly_chart(build_mechanism(res3, which), width="stretch", key="mechanism_chart")
verdict(verdict_mechanism(res3))
if which == "A":
    st.markdown("In **Diagramm A** ist die Schwankung die gemeinsame Ursache. Die Regression mit der Schwankung trifft die Wahrheit, der naive Vergleich verfehlt sie.")
else:
    st.markdown("In **Diagramm B** liegt die Schwankung auf dem Weg vom Faktor zum Servicegrad. Der naive Vergleich trifft die Wahrheit, und die Regression mit der "
                "Schwankung rechnet einen Teil der Wirkung weg.")
st.caption("Dieselben Zahlen, anderer Pfeil: Welche Rechnung richtig ist, entscheidet das Diagramm, nicht die Stichprobe. Wie man ein Diagramm liest, zeigt das "
           "Folgestück **Das Diagramm lesen: d-Separation**.")

st.subheader("🔬 Experiment 4: Wenn die Kontrolle wackelt")
st.caption(f"Standardwelt mit einer Stichprobe von {C.EXP_PROXY_N:,} Standorten".replace(",", ".") + ". Die Schwankung wird nur mit Messfehler beobachtet; "
           "die Reliabilität r ist ihr Signalanteil (1 = fehlerfrei).")
res4 = _proxy()
rows4 = res4["rows"]
st.plotly_chart(build_proxy(rows4, res4["naive"], res4["truth"]), width="stretch", key="proxy_chart")
verdict(verdict_proxy(res4))
st.caption("Eine ungenau gemessene Kontrolle rechnet den Fehler nur teilweise heraus. Wie stark ein unbeobachteter Einfluss sein müsste, um ein Ergebnis zu kippen, "
           "und wie man das prüft, zeigt das Folgestück **Sensitivität und Falsifikation**.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
p_low = float(normal_cdf(-2.0 * C.DEFAULT_GAMMA))
p_low2 = float(normal_cdf(-2.0 * 2.0))
r9 = next(r for r in rows4 if abs(r["r"] - 0.9) < 1e-9)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Schwankung ist die einzige gemeinsame Ursache** | Weitere verborgene Einflüsse auf Faktor und Servicegrad bleiben im Fehler; mit Daten allein sind sie nicht zu sehen. | Sensitivität und Falsifikation |
| **Die Schwankung ist ohne Messfehler beobachtet** | Bei Reliabilität 0,9 bleiben {de(100 * r9['left'], 0)} % des Fehlers übrig (Experiment 4). | Sensitivität und Falsifikation |
| **Es gibt überall Standorte mit und ohne hohen Faktor** | Bei γ = 1 hat ein Standort mit Schwankung −2 nur {de(100 * p_low, 0)} % Wahrscheinlichkeit auf den hohen Faktor, bei γ = 2 rund {de(100 * p_low2, 3)} %: Für die Ränder fehlen Vergleichsstandorte. | Schätzen |
| **Die Wirkung ist überall gleich** | Im Modell ist τ konstant. Wirkt der Faktor je nach Standort verschieden, mittelt ein einziger Unterschied darüber hinweg. | – |
| **Linear und normal verteilt** | Die geschlossene Formel gilt für dieses Modell; schiefe Verteilungen oder Schwellen im Servicegrad verändern die Zahlen. | – |
| **Der Disponent wählt nach einer Schwelle** | Die Auswahlregel ist erfunden; ein wirklicher Disponent hat weitere Gründe. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell (Diagramm A).** Die Schwankung $Z \sim N(0,1)$ ist die gemeinsame Ursache. Der hohe Faktor $X \in \{0,1\}$ gilt genau dann, wenn
$a + \gamma Z + \eta > 0$ mit $\eta \sim N(0,1)$ (Probit-Schwelle). Der Servicegrad ist $Y = b - \beta Z + \tau X + \sigma\varepsilon$ mit $\varepsilon \sim N(0,1)$
und dem Basiswert $b = 80$. Hier ist $a = 0$, also die Hälfte der Standorte mit hohem Faktor.

**Wahre Wirkung.** Mit dem do-Operator $\mathrm{do}(X = x)$ ersetzt man die Gleichung von $X$ durch den festen Wert $x$:
$E[Y \mid \mathrm{do}(X=1)] - E[Y \mid \mathrm{do}(X=0)] = \tau$. Im linearen Modell ist das die Summe der Pfadprodukte von $X$ nach $Y$.

**Naiver Unterschied.** $E[Y \mid X=1] - E[Y \mid X=0] = \tau - \beta\,(E[Z \mid X=1] - E[Z \mid X=0])$. Mit $S = a + \gamma Z + \eta \sim N(a, 1+\gamma^2)$ und
$X = [S > 0]$ ist $E[Z \mid X=1] = \tfrac{\gamma}{\sqrt{1+\gamma^2}}\,\tfrac{\varphi(c)}{1-\Phi(c)}$ und $E[Z \mid X=0] = -\tfrac{\gamma}{\sqrt{1+\gamma^2}}\,\tfrac{\varphi(c)}{\Phi(c)}$
mit $c = -a/\sqrt{1+\gamma^2}$. Der Fehler hängt nicht von der Stichprobengröße ab: Er verschwindet nur für $\gamma = 0$ oder $\beta = 0$.

**Zweites Diagramm.** Man zieht erst $X$ mit $P(X=1) = \Phi(a/\sqrt{1+\gamma^2})$, dann $Z$ aus seinem Gesetz gegeben $X$ (abgeschnittene Normalverteilung) und
dann $Y$ aus derselben Gleichung. Die gemeinsame Verteilung von $(Z, X, Y)$ ist dieselbe wie in Diagramm A; in Diagramm B gibt es keine gemeinsame Ursache, und
die Wirkung ist der naive Unterschied.

**Deckung.** Das 95-%-Intervall des Unterschieds nach Welch enthält die Wahrheit mit Wahrscheinlichkeit nahe 95 %, wenn der Schätzer unverzerrt ist (Münzwurf). Ist er
um $\delta$ verzerrt, sinkt die Deckung mit wachsendem $n$ gegen 0, weil der Standardfehler schrumpft und $\delta$ bleibt.

**Messfehler der Kontrolle.** Mit $Z^* = Z + u$ und Reliabilität $r = \mathrm{Var}(Z)/\mathrm{Var}(Z^*)$ rechnet die Regression von $Y$ auf $X$ und $Z^*$ den Einfluss von
$Z$ nur zu einem Anteil heraus, der kleiner als $r$ ist (maßgeblich ist die Reliabilität innerhalb der Gruppen des Faktors); der Rest bleibt im Koeffizienten von $X$.

**Literatur.** Das Beispiel ist erfunden; die Begriffe (do-Operator, Konfundierung, Simpson-Umkehr) stehen bei J. Pearl, *Causality*, Cambridge University Press 2009,
und E. H. Simpson, *The interpretation of interaction in contingency tables*, Journal of the Royal Statistical Society B 13, 1951 (aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
