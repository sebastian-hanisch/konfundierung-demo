"""Plotly-Figuren und die Tabelle der Stichprobe. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import kfd_constants as C
from kfd_evaluation import Analysis, de

GL_FROM = 1500                       # ab so vielen Punkten Scattergl


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({
        "Standort": [f"S{i + 1}" for i in range(k)],
        "Schwankung": a.z[:k],
        "Faktor": np.where(a.x[:k] == 1.0, "hoch", "normal"),
        "Servicegrad (Punkte)": a.y[:k],
    })


def style_table(df: pd.DataFrame):
    return df.style.format({"Schwankung": lambda v: de(v, 2), "Servicegrad (Punkte)": lambda v: de(v, 1)})


def build_scatter(a: Analysis, reveal: bool) -> go.Figure:
    """Schwankung gegen Servicegrad, nach Faktor gefärbt; gestrichelt die Mittel je Gruppe, bei „Wahrheit enthüllen“ die beiden strukturellen Geraden."""
    s = a.settings
    mark = go.Scattergl if s.n >= GL_FROM else go.Scatter
    fig = go.Figure()
    for flag, label, color in ((0.0, "normaler Faktor", C.COLOR_UNTREATED), (1.0, "hoher Faktor", C.COLOR_TREATED)):
        sel = a.x == flag
        fig.add_trace(mark(x=a.z[sel], y=a.y[sel], mode="markers", name=label, marker=dict(size=7 if s.n < GL_FROM else 5, color=color, opacity=0.65),
                           hovertemplate="Schwankung %{x:.2f}<br>Servicegrad %{y:.1f}<extra></extra>"))
    xs = np.array([min(-3.0, float(a.z.min())), max(3.0, float(a.z.max()))])
    for flag, mean, color in ((0.0, a.mean_untreated, C.COLOR_UNTREATED), (1.0, a.mean_treated, C.COLOR_TREATED)):
        if np.isfinite(mean):
            fig.add_trace(go.Scatter(x=xs, y=[mean, mean], mode="lines", line=dict(color=color, width=2, dash="dash"), name=f"Mittel {'hoch' if flag else 'normal'}",
                                     hoverinfo="skip"))
    if reveal:
        for shift, label in ((0.0, "Wahrheit: ohne hohen Faktor"), (s.tau, "Wahrheit: mit hohem Faktor")):
            fig.add_trace(go.Scatter(x=xs, y=C.BASE - s.beta * xs + shift, mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot" if shift == 0 else "solid"),
                                     name=label, hoverinfo="skip"))
    fig.update_xaxes(title="Schwankung der Nachfrage (0 = typischer Standort)")
    fig.update_yaxes(title="Servicegrad (Punkte)")
    return _fixed(fig, 420)


def build_diagram(which: str) -> go.Figure:
    """Das Diagramm A (Schwankung -> Faktor) oder B (Faktor -> Schwankung); Servicegrad hängt in beiden an Schwankung und Faktor."""
    pos = {"Schwankung": (0.5, 1.0), "Faktor": (0.0, 0.0), "Servicegrad": (1.0, 0.0)}
    edges = [("Schwankung", "Faktor"), ("Schwankung", "Servicegrad"), ("Faktor", "Servicegrad")] if which == "A" else \
            [("Faktor", "Schwankung"), ("Schwankung", "Servicegrad"), ("Faktor", "Servicegrad")]
    fig = go.Figure()
    for src, dst in edges:
        (x0, y0), (x1, y1) = pos[src], pos[dst]
        shrink = 0.17
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        fig.add_annotation(x=x1 - dx / length * shrink, y=y1 - dy / length * shrink, ax=x0 + dx / length * shrink, ay=y0 + dy / length * shrink,
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="#555555", text="")
    fig.add_trace(go.Scatter(x=[p[0] for p in pos.values()], y=[p[1] for p in pos.values()], mode="markers+text", text=list(pos), textposition="middle center",
                             marker=dict(size=68, color="rgba(47,107,101,0.15)", line=dict(width=1.5, color=C.COLOR_TREATED)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.35, 1.35])
    fig.update_yaxes(visible=False, range=[-0.35, 1.35])
    return _fixed(fig, 240, showlegend=False)


def build_coverage(rows: list[dict]) -> go.Figure:
    """Deckung des 95-%-Intervalls gegen n (logarithmische Achse)."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    for key, label, color in (("naive_cover", "naiver Vergleich (Disponent wählt)", C.COLOR_WARN), ("adj_cover", "mit Schwankung als Kontrolle", C.COLOR_LINE),
                              ("coin_cover", "Münzwurf", C.COLOR_TREATED)):
        fig.add_trace(go.Scatter(x=ns, y=[100 * r[key] for r in rows], mode="lines+markers", name=label, line=dict(color=color, width=2.5)))
    fig.add_hline(y=95, line=dict(color="#888888", dash="dot"), annotation_text="versprochen: 95 %", annotation_position="bottom right")
    fig.update_xaxes(title="Standorte n", type="log", tickvals=ns)
    fig.update_yaxes(title="Deckung der wahren Wirkung (%)", range=[-3, 103])
    return _fixed(fig, 380)


def build_sign_map(rows: list[dict]) -> go.Figure:
    """Karte des Grenzwerts des naiven Unterschieds über Auswahlstärke (x) und Einfluss der Schwankung (y)."""
    gammas = sorted({r["gamma"] for r in rows})
    betas = sorted({r["beta"] for r in rows})
    z = [[next(r["limit"] for r in rows if r["gamma"] == g and r["beta"] == b) for g in gammas] for b in betas]
    text = [[de(v, 1) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=[de(g, 2) for g in gammas], y=[de(b, 1) for b in betas], text=text, texttemplate="%{text}", zmid=0.0,
                               colorscale=[[0.0, "#c0392b"], [0.5, "#f7f7f7"], [1.0, "#2F6B65"]], colorbar=dict(title=dict(text="Punkte", side="top"), nticks=4),
                               hovertemplate="γ %{x}, β %{y}: %{z:.2f}<extra></extra>"))
    fig.update_xaxes(title="Auswahlstärke γ", type="category")
    fig.update_yaxes(title="Einfluss der Schwankung β", type="category")
    return _fixed(fig, 360)


def build_mechanism(res: dict, which: str) -> go.Figure:
    """Naiver Unterschied, Regression mit Schwankung und Wahrheit im gewählten Diagramm."""
    d = res[which]
    truth = res["truth_A"] if which == "A" else res["truth_B"]
    labels = ["naiver Vergleich", "Regression mit Schwankung", "Wahrheit in diesem Diagramm"]
    values = [d["naive"], d["adjusted"], truth]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=[C.COLOR_WARN, C.COLOR_LINE, C.COLOR_TRUTH], text=[de(v, 2) for v in values], textposition="outside"))
    fig.update_yaxes(title="Unterschied in Punkten", zeroline=True)
    return _fixed(fig, 320, showlegend=False)


def build_proxy(rows: list[dict], naive: float, truth: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["r"] for r in rows], y=[r["estimate"] for r in rows], mode="lines+markers", name="Schätzung mit verrauschter Kontrolle",
                             line=dict(color=C.COLOR_LINE, width=2.5)))
    fig.add_hline(y=truth, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Wahrheit {de(truth, 1)}", annotation_position="top left")
    fig.add_hline(y=naive, line=dict(color=C.COLOR_WARN, dash="dash"), annotation_text=f"ohne Kontrolle {de(naive, 1)}", annotation_position="bottom left")
    fig.update_xaxes(title="Reliabilität der Kontrolle (1 = ohne Messfehler)", autorange="reversed")
    fig.update_yaxes(title="Geschätzte Wirkung (Punkte)")
    return _fixed(fig, 360)
