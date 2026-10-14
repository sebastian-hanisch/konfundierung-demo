# Konfundierung: Beobachten ist nicht Eingreifen – Streamlit-Demo

Erstes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Vehikel ist die Welt der Linie *Experimente und Wirkungsmessung*: ein Sicherheitsfaktor der Bestandsregel an Standorten, gemessen am Servicegrad. Dort verteilt eine Münze den Faktor; hier hat ein Disponent ihn nach der Nachfrageschwankung der Standorte vergeben.

## Kernfrage

Ein Disponent gibt den hohen Faktor eher an Standorten mit stark schwankender Nachfrage, die ohnehin schlechter bedient werden. Der Vergleich der Servicegrade mit und ohne hohen Faktor zeigt dann einen Schaden, obwohl die wahre Wirkung positiv ist. Wie weit liegt der naive Vergleich daneben, wächst die Sicherheit mit mehr Daten, und lässt sich an den Zahlen erkennen, welches Ursachen-Diagramm gilt?

## Modell

Strukturmodell mit drei Größen (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie):

- **Schwankung** Z ~ N(0, 1), die gemeinsame Ursache.
- **Hoher Faktor** X = 1 genau dann, wenn a + γZ + η > 0 mit η ~ N(0, 1) (Probit-Schwelle). Hier a = 0, also die Hälfte der Standorte.
- **Servicegrad** Y = 80 − βZ + τX + σε. Wahre Wirkung τ = 3,5 Punkte, β = 4, σ = 5, γ = 1 in der Standardwelt.

Die Wahrheit ist die Wirkung beim Festsetzen von X (do-Operator): im linearen Modell die Summe der Pfadprodukte, hier τ. Die Simulation rechnet sie zusätzlich als gekoppelten Lauf mit denselben Zufallszahlen. Alles ist erfunden.

## Methodik

- **Geschlossene Form des naiven Unterschieds:** τ − β·(E[Z | X=1] − E[Z | X=0]) mit den abgeschnittenen Normalmitteln (inverse Mills-Quote). Der Grenzwert hängt nicht von n ab.
- **Schätzer:** Welch-Unterschied der Mittelwerte; Regression von Y auf X und Z (Normalgleichungen, vektorisiert über Wiederholungen); 95-%-Intervall in Normalnäherung.
- **Zweites Diagramm:** dieselbe gemeinsame Verteilung von (Z, X, Y) auf anderem Weg (erst X, dann Z aus seinem Gesetz gegeben X, dann Y); Kolmogorov-Smirnov-Tests auf Z | X und Y | X.
- **Experimente** laufen über feste Zufallsströme (SplitMix64): 400 Wiederholungen je Zelle, nie ein einzelner Seed.

## Befunde (gemessen, keine Behauptungen)

Standardwelt (γ = 1, β = 4, τ = 3,5, σ = 5), wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Wohin läuft der naive Vergleich? | Gegen **−1,01 Punkte**, nicht gegen +3,5. Geschlossene Form und Monte Carlo (N = 300 000) stimmen überein. | `test_standard_world_closed_form_limit_is_minus_one` |
| Standard-Stichprobe (400 Standorte, Seed 35) | Naiver Unterschied −1,10 Punkte, 95-%-Intervall −2,2 bis 0,0; 53 % der Standorte mit hohem Faktor, Mittel 80,8 gegen 81,9. Das Intervall verfehlt +3,5 und zeigt in die falsche Richtung. | `test_standard_sample_shows_a_harm_where_the_effect_is_positive` |
| Deckt das Intervall mit mehr Daten besser? | **Schlechter.** Deckung der wahren Wirkung: 27 % bei n = 50, **0 %** ab n = 200 (n = 200, 1 000, 5 000). | `test_experiment_1_naive_coverage_falls_to_zero` |
| Wie oft das falsche Vorzeichen? | 69 % bei n = 50, 90 % bei n = 200, 99,5 % bei n = 1 000, 100 % bei n = 5 000. | `test_experiment_1_wrong_sign_share_grows_to_one` |
| Und bei Münzwurf? | Deckung 94 bis 96 % für alle n, Mittel der Schätzung +3,5. | `test_experiment_1_coin_and_adjusted_cover_about_ninety_five_percent` |
| Mit der Schwankung als Kontrolle gerechnet? | Ebenfalls 94 bis 96 % (Regression mit Z); die Rechnung ist erlaubt, wenn Z die einzige gemeinsame Ursache ist und ohne Messfehler beobachtet wird. | `test_experiment_1_coin_and_adjusted_cover_about_ninety_five_percent` |
| Wie stark muss die Auswahl sein? | In 5 von 16 Zellen mit γ > 0 und β > 0 ist der Grenzwert negativ (γ = 1, β = 4: −1,0; γ = 2, β = 4: −2,2; γ = 0,5, β = 6: −0,8). Ohne Auswahl (γ = 0) oder ohne Einfluss der Schwankung (β = 0) ist er in jeder Zelle +3,5. | `test_experiment_2_five_of_sixteen_mixed_cells_are_negative`, `test_experiment_2_no_error_without_selection_or_influence` |
| Falsches Vorzeichen bei n = 200 | 86 % (γ = 1, β = 4), 6 % (γ = 1, β = 2), 18 % (γ = 2, β = 2), 22 % (γ = 0,5, β = 4), ab γ = 2, β = 4 praktisch immer. | `test_experiment_2_wrong_sign_shares_at_two_hundred_sites` |
| Gibt es ein zweites Diagramm mit denselben Daten? | Ja: Faktor → Schwankung → Servicegrad. Von 48 Tests auf gleiche Verteilung liegen 5 unter 0,05 (kleinster p-Wert 0,003); bei gleicher Verteilung sind rund 5 % zu erwarten. In Diagramm B ist der naive Unterschied die Wahrheit (−1,01), in Diagramm A ist es +3,5. | `test_experiment_3_forty_eight_tests_and_the_two_truths` |
| Hilft eine verrauschte Kontrolle? | Nur teilweise. Reliabilität 0,9: Schätzung 2,82, 15 % des Fehlers bleiben; 0,5: 0,74 (60 %); 0,2: −0,41 (86 %). | `test_experiment_4_remaining_error_by_reliability` |
| Überlappung | Bei γ = 1 hat ein Standort mit Schwankung −2 nur 2,3 % Wahrscheinlichkeit auf den hohen Faktor, bei γ = 2 rund 0,003 %. | `test_overlap_numbers_of_the_assumptions_table` |
| Mini-Instanz von Hand | 16 Standorte in zwei Gruppen: naiv −0,75, je Gruppe +4 und +5, standardisiert +4,5. | `test_mini_instance_naive_difference_and_group_means`, `test_mini_instance_effects_within_groups_and_standardized_difference` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Schwankung ist die einzige gemeinsame Ursache** | Weitere verborgene Einflüsse bleiben im Fehler; mit Daten allein sind sie nicht zu sehen. | Sensitivität und Falsifikation |
| **Die Schwankung ist ohne Messfehler beobachtet** | Bei Reliabilität 0,9 bleiben 15 % des Fehlers übrig. | Sensitivität und Falsifikation |
| **Überall gibt es Standorte mit und ohne hohen Faktor** | Bei starker Auswahl fehlen Vergleichsstandorte an den Rändern. | Schätzen |
| **Die Wirkung ist überall gleich** | Im Modell ist τ konstant; wirkt der Faktor je Standort verschieden, mittelt ein einziger Unterschied darüber hinweg. | – |
| **Linear und normal verteilt** | Die geschlossene Formel gilt für dieses Modell. | – |
| **Der Disponent wählt nach einer Schwelle** | Die Auswahlregel ist erfunden. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Standorte. Die „Wahrheit“ ist die Rechnung im simulierten Modell.

## Bezug zu den Nachbarn im Portfolio

- [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo): derselbe Servicegrad aus der Bestandsregel bei zufälliger Zuteilung; seine Deckung von 94 bis 96 % ist hier die Münzwurf-Spalte. Dort wird zugeteilt, hier nicht.
- [off-policy-demo](https://github.com/sebastian-hanisch/off-policy-demo): Auswertung aus Logdaten mit bekannter Protokoll-Wahrscheinlichkeit und Kontext; verdeckte Einflüsse stehen dort als Grenze, hier ist sie das Thema.
- [synthetic-control-demo](https://github.com/sebastian-hanisch/synthetic-control-demo): Parallelität der Trends als Annahme; hier geht es um die Auswahl nach einem beobachteten Merkmal.

## Tests

Mini-Instanz von Hand, geschlossene Form gegen Monte Carlo und scipy (`truncnorm`, `ttest_ind`, `ks_2samp`), Generator (SplitMix64 gegen eine unabhängige Fassung, Strukturmodell, Intervention, Pfadsummen), Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Slider-Grenzen, Wahrheit, Diagramm-Schalter, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell, do-Wirkung) |
| `kfd_model.py` | geschlossene Form, Schätzer, zweites Diagramm, KS-Test |
| `kfd_evaluation.py` | Kennzahlen, Experimente, Verdikte |
| `kfd_visualization.py`, `kfd_pdf_export.py` | Figuren, PDF |
| `kfd_constants.py`, `kfd_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Mehr als eine gemeinsame Ursache, Mediator, Collider, Instrumente:** eigene Stücke der Linie.
- **Adjustierungsregeln und ihre Voraussetzungen:** hier nur als Spalte gezeigt, nicht erklärt.
- **Heterogene Effekte, zeitliche Abfolge, Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
