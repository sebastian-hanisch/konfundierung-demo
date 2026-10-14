"""PDF-Export der aktuellen Stichprobe (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import kfd_constants as C
from kfd_evaluation import Analysis, de

_REPLACE = {"→": "->", "–": "-", "—": "-", "γ": "gamma", "β": "beta", "τ": "tau", "σ": "sigma", "−": "-", "€": "EUR", "≥": ">="}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Konfundierung: Beobachten ist nicht Eingreifen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    zuteilung = "Münze" if s.assign == "muenzwurf" else "Disponent wählt nach der Schwankung"
    pdf.multi_cell(0, 5, _s(f"Standorte n = {s.n}, Auswahlstärke gamma = {de(s.gamma, 2)}, Einfluss der Schwankung beta = {de(s.beta, 1)}, wahre Wirkung tau = "
                            f"{de(s.tau, 1)}, Streuung sigma = {de(s.sigma, 1)}, Zuteilung: {zuteilung}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Anteil mit hohem Faktor: {de(100 * a.share_treated, 0)} %. Mittel mit hohem Faktor {de(a.mean_treated, 1)}, ohne {de(a.mean_untreated, 1)} Punkte.",
        f"Naiver Unterschied: {de(a.naive, 2)} Punkte (95-%-Intervall {de(a.lo, 2)} bis {de(a.hi, 2)}).",
        f"Wahre Wirkung im Modell: {de(a.truth, 2)} Punkte; Intervall deckt sie: {'ja' if a.covers else 'nein'}.",
        f"Grenzwert des naiven Unterschieds für n gegen unendlich: {de(a.limit, 2)} Punkte.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Standort", "Schwankung", "Faktor", "Servicegrad"], [28, 36, 30, 36]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        row = [f"S{i + 1}", de(a.z[i], 2), "hoch" if a.x[i] == 1.0 else "normal", de(a.y[i], 1)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if s.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {s.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s(f"Servicegrad = {C.BASE:g} - beta * Schwankung + tau * Faktor + Rauschen. Quelle: konfundierung-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
