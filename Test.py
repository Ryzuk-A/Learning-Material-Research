"""
fly_rv_frontend.py
==================
 
EUR Swap Fly RV - Dash / Plotly view (front end)
------------------------------------------------
 
Drop-in tab for an existing multi-page Dash app. All calculations live in
`fly_rv_backend.py`; this module only builds the layout and callbacks.
 
Integration
~~~~~~~~~~~
    import fly_rv_frontend as flyrv
 
    # inside the host app's tab / page registry:
    tab_content = flyrv.layout()
 
    # once, after the Dash app object exists:
    flyrv.register_callbacks(app)
 
* Every component id is prefixed with "flyrv-" to avoid clashes.
* Colours and fonts live in THEME - remap them to the host app's palette.
* Data comes from `fly_rv_backend.get_rates(index)` (cached). Point
  `fly_rv_backend.load_spot_rates` at the real source and set
  `fly_rv_backend.USE_DEMO_DATA = False`. Groupings go in
  `fly_rv_backend.STRUCTURE_GROUPS`.
* Run this file directly for a standalone demo on synthetic data.
 
What the user can do
~~~~~~~~~~~~~~~~~~~~
* Switch floating index (6M EURIBOR default / ESTR), as-of date.
* Set the static regression lookback, rolling-beta window, z-score window,
  chart lookback, sigma band, residual model (static / rolling), C&R horizon.
* Browse flies by group, sort the summary table, click a row for detail.
* Add ad-hoc flies ("3s7s12s") to a Custom group.
* Override the level / curve-environment regressors per fly.
* Charts: fly level with stacked curve / level panels, z-scores with +/- band,
  actual vs fitted with residual, rolling vs static betas and R^2, scatter,
  plus a per-leg carry & roll breakdown for auditing.
"""
 
from __future__ import annotations
 
from typing import Dict, List, Optional
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme, Sign
from plotly.subplots import make_subplots
 
import fly_rv_backend as be
 
P = "flyrv-"  # id prefix
 
# =============================================================================
# THEME - remap to the host app's palette
# =============================================================================
THEME = {
    "font": 'system-ui, -apple-system, "Segoe UI", sans-serif',
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
    "s1": "#0d366b",   # dark blue: fly / residual z / beta level
    "s2": "#9b1c1c",   # dark red: curve env / fitted / beta curve
    "s3": "#52514e",   # dark grey: level / curve z / R^2
    "pos": "#0d366b",  # residual > 0 -> belly cheap -> RECEIVE
    "neg": "#9b1c1c",  # residual < 0 -> belly rich  -> PAY
    "pos_bg": "#cde2fb",
    "neg_bg": "#f9d3d2",
    "pos_bg_soft": "#e8f1fd",
    "neg_bg_soft": "#fcebea",
    "shade": "rgba(42,120,214,0.06)",
}
 
LOOKBACK_OPTIONS = [{"label": k, "value": v} for k, v in be.WINDOW_PRESETS.items()]
CHART_OPTIONS = LOOKBACK_OPTIONS + [{"label": "All", "value": 0}]
HORIZON_OPTIONS = [{"label": k, "value": v} for k, v in be.HORIZON_PRESETS.items()]
BAND_OPTIONS = [{"label": f"±{b:g}σ", "value": b} for b in (1.0, 1.5, 2.0, 2.5, 3.0)]
INDEX_OPTIONS = [{"label": v["label"], "value": k} for k, v in be.INDEX_CONFIG.items()]
CUSTOM_TAB = "__custom__"
 
 
# =============================================================================
# LAYOUT
# =============================================================================
 
def _control(label: str, component) -> html.Div:
    return html.Div(
        [html.Label(label, style={"fontSize": "11px", "color": THEME["ink2"],
                                  "marginBottom": "2px", "display": "block"}), component],
        style={"minWidth": "120px", "flex": "1 1 120px"},
    )
 
 
def _dd(id_, options, value, **kw):
    return dcc.Dropdown(id=P + id_, options=options, value=value, clearable=False,
                        style={"fontSize": "13px"}, **kw)
 
 
def _card_style():
    return {"background": THEME["surface"], "border": f"1px solid {THEME['border']}",
            "borderRadius": "8px", "padding": "12px"}
 
 
def _graph(id_):
    return dcc.Graph(id=P + id_, config={"displaylogo": False}, style={"height": "380px"})
 
 
def layout() -> html.Div:
    """Top-level component for the tab."""
    df = be.get_rates(be.DEFAULT_INDEX)
    groups = be.structure_groups()
    first_group = next(iter(groups))
    tenor_opts = _tenor_options(df)
 
    controls = html.Div(
        [
            _control("Floating index", _dd("index", INDEX_OPTIONS, be.DEFAULT_INDEX)),
            _control("As-of date", dcc.DatePickerSingle(
                id=P + "asof", date=df.index[-1].date(), min_date_allowed=df.index[0].date(),
                max_date_allowed=df.index[-1].date(), display_format="DD-MMM-YY",
                style={"fontSize": "13px"})),
            _control("Regression lookback", _dd("regwin", LOOKBACK_OPTIONS, 126)),
            _control("Rolling β window", _dd("rollwin", LOOKBACK_OPTIONS, 63)),
            _control("Z-score window", _dd("zwin", LOOKBACK_OPTIONS, 126)),
            _control("Chart lookback", _dd("chartwin", CHART_OPTIONS, 252)),
            _control("Band / signal", _dd("band", BAND_OPTIONS, 2.0)),
            _control("Residual model", _dd("mode", [
                {"label": "Static (in-sample)", "value": "static"},
                {"label": "Rolling", "value": "rolling"}], "static")),
            _control("C&R horizon", _dd("horizon", HORIZON_OPTIONS, 0.25)),
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "10px", **_card_style()},
    )
 
    tabs = dcc.Tabs(
        id=P + "group", value=first_group,
        children=[dcc.Tab(label=g, value=g) for g in groups]
        + [dcc.Tab(label="Custom", value=CUSTOM_TAB)],
        style={"marginTop": "12px"},
    )
 
    custom_row = html.Div(
        [
            dcc.Input(id=P + "custom-input", type="text", placeholder="Add fly e.g. 3s7s12s",
                      debounce=False, style={"width": "200px", "fontSize": "13px"}),
            html.Button("Add & view", id=P + "custom-add", n_clicks=0),
            html.Button("Clear custom", id=P + "custom-clear", n_clicks=0),
            html.Span(id=P + "custom-msg", style={"color": THEME["neg"], "fontSize": "12px"}),
        ],
        style={"display": "flex", "gap": "8px", "alignItems": "center", "margin": "10px 0"},
    )
 
    table = dash_table.DataTable(
        id=P + "table",
        columns=_table_columns(),
        data=[],
        sort_action="native",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "600", "fontSize": "12px", "background": THEME["page"],
                      "color": THEME["ink2"], "whiteSpace": "normal", "height": "auto"},
        style_cell={"fontFamily": THEME["font"], "fontSize": "12px", "padding": "6px 8px",
                    "color": THEME["ink"], "fontVariantNumeric": "tabular-nums",
                    "textAlign": "right", "minWidth": "62px"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"}
                                for c in ("name", "regressors", "signal", "agreement")],
        tooltip_header=_table_tooltips(),
        tooltip_delay=300,
    )
 
    overrides = html.Div(
        [
            html.Span("Regressors:", style={"fontSize": "12px", "color": THEME["ink2"]}),
            _control("Level tenor", _dd("ov-level", tenor_opts, None)),
            _control("Curve short", _dd("ov-c1", tenor_opts, None)),
            _control("Curve long", _dd("ov-c2", tenor_opts, None)),
            html.Button("Apply", id=P + "ov-apply", n_clicks=0),
            html.Button("Reset to default", id=P + "ov-reset", n_clicks=0),
            html.Span(id=P + "ov-msg", style={"color": THEME["neg"], "fontSize": "12px"}),
        ],
        style={"display": "flex", "gap": "10px", "alignItems": "flex-end", "flexWrap": "wrap",
               "maxWidth": "900px"},
    )
 
    def chart_card(title_id, graph_id, extra=None):
        return html.Div([html.Div(id=P + title_id, style={"fontSize": "13px", "fontWeight": 600}),
                         extra if extra is not None else html.Div(), _graph(graph_id)],
                        style=_card_style())
 
    charts = html.Div(
        [
            chart_card("t-level", "fig-level", dcc.Checklist(
                id=P + "overlay", inline=True, value=["curve"],
                options=[{"label": " Curve environment", "value": "curve"},
                         {"label": " Level", "value": "level"}],
                style={"fontSize": "12px"}, inputStyle={"marginLeft": "10px"})),
            chart_card("t-z", "fig-z", dcc.Checklist(
                id=P + "zshow", inline=True, value=["resid", "fly"],
                options=[{"label": " Residual z", "value": "resid"},
                         {"label": " Fly z", "value": "fly"},
                         {"label": " Curve z", "value": "curve"}],
                style={"fontSize": "12px"}, inputStyle={"marginLeft": "10px"})),
            chart_card("t-fit", "fig-fit"),
            chart_card("t-beta", "fig-beta"),
            chart_card("t-scatter", "fig-scatter", dcc.RadioItems(
                id=P + "scatter-x", inline=True, value="fitted",
                options=[{"label": " Actual vs fitted", "value": "fitted"},
                         {"label": " Fly vs level", "value": "level"},
                         {"label": " Fly vs curve", "value": "curve"}],
                style={"fontSize": "12px"}, inputStyle={"marginLeft": "10px"})),
            html.Div([
                html.Div(id=P + "t-legs", style={"fontSize": "13px", "fontWeight": 600}),
                dash_table.DataTable(
                    id=P + "legs", columns=_leg_columns(), data=[],
                    style_as_list_view=True, style_table={"overflowX": "auto"},
                    style_header={"fontWeight": "600", "fontSize": "11px",
                                  "background": THEME["page"], "color": THEME["ink2"],
                                  "whiteSpace": "normal", "height": "auto"},
                    style_cell={"fontFamily": THEME["font"], "fontSize": "12px",
                                "padding": "5px 6px", "textAlign": "right",
                                "fontVariantNumeric": "tabular-nums"},
                    style_data_conditional=[{"if": {"filter_query": '{leg} = "Fly (2B-W1-W2)"'},
                                             "fontWeight": "700"}],
                ),
                html.Div(id=P + "legs-note", style={"fontSize": "11px", "color": THEME["ink2"],
                                                    "marginTop": "8px", "lineHeight": "1.5"}),
            ], style=_card_style()),
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))",
               "gap": "12px", "marginTop": "12px"},
    )
 
    return html.Div(
        [
            dcc.Store(id=P + "selected", data=None),
            dcc.Store(id=P + "overrides", data={}),
            dcc.Store(id=P + "custom", data=[]),
            html.Div([
                html.H3("EUR Swap Fly RV", style={"margin": "0"}),
                html.Div(id=P + "subtitle", style={"fontSize": "12px", "color": THEME["ink2"]}),
            ], style={"marginBottom": "10px"}),
            controls,
            tabs,
            custom_row,
            html.Div(id=P + "table-msg", style={"fontSize": "12px", "color": THEME["ink2"],
                                                "margin": "4px 0"}),
            html.Div(table, style=_card_style()),
            html.Div([
                html.H4(id=P + "detail-title", style={"margin": "16px 0 2px"}),
                html.Div(id=P + "detail-sub", style={"fontSize": "12px", "color": THEME["ink2"],
                                                     "marginBottom": "8px"}),
                overrides,
                html.Div(id=P + "cards", style={"display": "flex", "flexWrap": "wrap",
                                                "gap": "8px", "marginTop": "12px"}),
            ]),
            charts,
        ],
        style={"fontFamily": THEME["font"], "color": THEME["ink"], "background": THEME["page"],
               "padding": "16px"},
    )
 
 
def _tenor_options(df: pd.DataFrame) -> List[Dict]:
    quoted = be.tenor_columns(df)
    tmax = int(max(quoted))
    tenors = sorted(set(range(1, tmax + 1)) | {t for t in quoted if t >= 1})
    return [{"label": f"{be._fmt_tenor(t)}y", "value": float(t)} for t in tenors]
 
 
def _num(prec=2, signed=False):
    return Format(precision=prec, scheme=Scheme.fixed,
                  sign=Sign.positive if signed else Sign.default)
 
 
def _table_columns() -> List[Dict]:
    n = "numeric"
    return [
        {"name": "Fly", "id": "name"},
        {"name": "Regressed on", "id": "regressors"},
        {"name": "Fly (bp)", "id": "fly_bp", "type": n, "format": _num(1)},
        {"name": "Resid (bp)", "id": "resid_bp", "type": n, "format": _num(2, True)},
        {"name": "Resid Z", "id": "z_resid", "type": n, "format": _num(2, True)},
        {"name": "Resid / SE", "id": "resid_over_se", "type": n, "format": _num(2, True)},
        {"name": "Fly Z", "id": "z_fly", "type": n, "format": _num(2, True)},
        {"name": "Curve Z", "id": "z_curve", "type": n, "format": _num(2, True)},
        {"name": "β Level", "id": "beta_level", "type": n, "format": _num(3, True)},
        {"name": "β Curve", "id": "beta_curve", "type": n, "format": _num(3, True)},
        {"name": "R²", "id": "r2", "type": n, "format": _num(2)},
        {"name": "SE (bp)", "id": "se_bp", "type": n, "format": _num(2)},
        {"name": "Half-life (d)", "id": "half_life_d", "type": n, "format": _num(0)},
        {"name": "Carry (bp/yr)", "id": "carry_ann_bp", "type": n, "format": _num(1, True)},
        {"name": "Roll (bp/yr)", "id": "roll_ann_bp", "type": n, "format": _num(1, True)},
        {"name": "C&R (bp/yr)", "id": "cr_ann_bp", "type": n, "format": _num(1, True)},
        {"name": "C&R signal dir", "id": "cr_signal_dir_bp", "type": n, "format": _num(1, True)},
        {"name": "Signal", "id": "signal"},
        {"name": "Resid vs Fly Z", "id": "agreement"},
    ]
 
 
def _table_tooltips() -> Dict[str, str]:
    return {
        "regressors": "Factors the fly is regressed on: L = level tenor, C = curve environment.",
        "resid_bp": "Fly minus fitted (bp). Positive = belly cheap vs level & curve.",
        "z_resid": "Residual z-score over the z-score window. +: receive belly, -: pay belly.",
        "resid_over_se": "Static residual / regression standard error (note-style z).",
        "z_fly": "Z-score of the outright fly level over the z-score window.",
        "z_curve": "Z-score of the curve-environment regressor.",
        "beta_level": "Static beta to the level factor over the regression lookback.",
        "beta_curve": "Static beta to the curve factor over the regression lookback.",
        "half_life_d": "AR(1) mean-reversion half-life of the static residual (business days).",
        "carry_ann_bp": "Receive-belly 1:2:1 fly carry, annualised (bp of fly per year).",
        "roll_ann_bp": "Receive-belly 1:2:1 fly roll-down, annualised.",
        "cr_signal_dir_bp": "C&R for the trade the residual signals (receive or pay belly).",
        "agreement": "STRONG: residual & fly z same sign, both beyond band. "
                     "AGREE: same sign. DIVERGE: opposite signs.",
    }
 
 
def _leg_columns() -> List[Dict]:
    n = "numeric"
    return [
        {"name": "Leg", "id": "leg"},
        {"name": "Weight", "id": "weight", "type": n, "format": _num(0, True)},
        {"name": "S(0,T) %", "id": "spot_pct", "type": n, "format": _num(4)},
        {"name": "S(0,T-h) %", "id": "spot_rolled_pct", "type": n, "format": _num(4)},
        {"name": "SR(h,T) %", "id": "fwd_pct", "type": n, "format": _num(4)},
        {"name": "Fixing %", "id": "fixing_pct", "type": n, "format": _num(3)},
        {"name": "P(0,h)", "id": "df_h", "type": n, "format": _num(5)},
        {"name": "A(h,T)", "id": "fwd_annuity", "type": n, "format": _num(4)},
        {"name": "Carry (bp/h)", "id": "carry_h_bp", "type": n, "format": _num(2, True)},
        {"name": "Roll (bp/h)", "id": "roll_h_bp", "type": n, "format": _num(2, True)},
        {"name": "Carry (bp/yr)", "id": "carry_ann_bp", "type": n, "format": _num(2, True)},
        {"name": "Roll (bp/yr)", "id": "roll_ann_bp", "type": n, "format": _num(2, True)},
        {"name": "Approx carry (bp/h)", "id": "carry_approx_h_bp", "type": n,
         "format": _num(2, True)},
    ]
 
 
# =============================================================================
# HELPERS
# =============================================================================
 
def _params(index, asof, regwin, rollwin, zwin, band, mode, horizon) -> be.RVParams:
    return be.RVParams(reg_window=int(regwin), roll_window=int(rollwin), z_window=int(zwin),
                       band=float(band), resid_mode=mode, horizon=float(horizon), index=index,
                       as_of=pd.Timestamp(asof) if asof else None)
 
 
def _flies_for(group: str, custom: List[str]) -> List[str]:
    return list(custom or []) if group == CUSTOM_TAB else be.structure_groups().get(group, [])
 
 
def _z_styles(band: float) -> List[Dict]:
    """Diverging cell shading: blue = residual +ve (receive), red = -ve (pay)."""
    styles = []
    for col in ("z_resid", "resid_over_se", "z_fly", "z_curve"):
        styles += [
            {"if": {"filter_query": f"{{{col}}} >= {band}", "column_id": col},
             "backgroundColor": THEME["pos_bg"]},
            {"if": {"filter_query": f"{{{col}}} <= {-band}", "column_id": col},
             "backgroundColor": THEME["neg_bg"]},
            {"if": {"filter_query": f"{{{col}}} >= 1 && {{{col}}} < {band}", "column_id": col},
             "backgroundColor": THEME["pos_bg_soft"]},
            {"if": {"filter_query": f"{{{col}}} <= -1 && {{{col}}} > {-band}", "column_id": col},
             "backgroundColor": THEME["neg_bg_soft"]},
        ]
    styles += [
        {"if": {"filter_query": '{signal} = "RECEIVE"', "column_id": "signal"},
         "color": THEME["pos"], "fontWeight": "700"},
        {"if": {"filter_query": '{signal} = "PAY"', "column_id": "signal"},
         "color": THEME["neg"], "fontWeight": "700"},
        {"if": {"filter_query": '{agreement} = "DIVERGE"', "column_id": "agreement"},
         "color": THEME["ink2"], "fontStyle": "italic"},
        {"if": {"filter_query": '{agreement} = "STRONG"', "column_id": "agreement"},
         "fontWeight": "700"},
        {"if": {"state": "active"}, "backgroundColor": THEME["page"],
         "border": f"1px solid {THEME['s1']}"},
    ]
    return styles
 
 
def _base_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=50, r=16, t=24, b=36),
        font=dict(family=THEME["font"], size=11, color=THEME["ink2"]),
        paper_bgcolor=THEME["surface"], plot_bgcolor=THEME["surface"],
        hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=THEME["grid"], linecolor=THEME["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=THEME["grid"], linecolor=THEME["axis"], zeroline=False)
    return fig
 
 
def _empty_fig(msg: str) -> go.Figure:
    fig = _base_fig(go.Figure())
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=13, color=THEME["muted"]),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
 
 
def _window(series: pd.DataFrame, n: int) -> pd.DataFrame:
    return series if not n else series.iloc[-int(n):]
 
 
def _shade_reg(fig, start, end, rows=1):
    for r in range(1, rows + 1):
        fig.add_vrect(x0=start, x1=end, fillcolor=THEME["shade"], line_width=0, layer="below",
                      row=r, col=1)
 
 
def _line(x, y, name, color, dash=None, width=2):
    return go.Scatter(x=x, y=y, name=name, mode="lines",
                      line=dict(color=color, width=width, dash=dash),
                      hovertemplate="%{y:.2f}")
 
 
# =============================================================================
# FIGURES
# =============================================================================
 
def fig_level(res: Dict, n: int, overlays: List[str]) -> go.Figure:
    """Fly level, with curve environment / level as stacked panels (shared x)."""
    s = _window(res["series"], n)
    reg = res["reg"]
    c1, c2 = reg.curve
    panels = [("fly", f"{res['spec'].name} fly (bp)", THEME["s1"])]
    if "curve" in (overlays or []):
        panels.append(("curve", f"{be._fmt_tenor(c1)}s{be._fmt_tenor(c2)}s curve (bp)", THEME["s2"]))
    if "level" in (overlays or []):
        panels.append(("level", f"{be._fmt_tenor(reg.level)}y level (bp)", THEME["s3"]))
    heights = [0.55] + [0.45 / (len(panels) - 1)] * (len(panels) - 1) if len(panels) > 1 else [1]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=heights)
    for i, (col, name, color) in enumerate(panels, start=1):
        fig.add_trace(_line(s.index, s[col], name, color), row=i, col=1)
    _shade_reg(fig, *res["reg_window_dates"], rows=len(panels))
    return _base_fig(fig)
 
 
def fig_z(res: Dict, n: int, band: float, show: List[str]) -> go.Figure:
    """Z-scores with +/- band; markers where the residual z breaches the band."""
    s = _window(res["series"], n)
    fig = go.Figure()
    fig.add_hrect(y0=-band, y1=band, fillcolor=THEME["shade"], line_width=0, layer="below")
    for y in (band, -band):
        fig.add_hline(y=y, line=dict(color=THEME["muted"], width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    series = [("resid", "z_resid", "Residual z", THEME["s1"]),
              ("fly", "z_fly", "Fly z", THEME["s2"]),
              ("curve", "z_curve", "Curve z", THEME["s3"])]
    for key, col, name, color in series:
        if key in (show or []):
            fig.add_trace(_line(s.index, s[col], name, color, width=2 if key == "resid" else 1.5))
    if "resid" in (show or []):
        hit = s[s["z_resid"].abs() >= band]
        fig.add_trace(go.Scatter(
            x=hit.index, y=hit["z_resid"], mode="markers", name=f"|Resid z| ≥ {band:g}",
            marker=dict(size=8, color=np.where(hit["z_resid"] > 0, THEME["pos"], THEME["neg"]),
                        line=dict(color=THEME["surface"], width=2)),
            hovertemplate="%{y:.2f}"))
    fig.update_yaxes(title_text="σ")
    return _base_fig(fig)
 
 
def fig_fit(res: Dict, n: int) -> go.Figure:
    """Actual vs static fitted fly (top) and residual in bp (bottom)."""
    s = _window(res["series"], n)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.62, 0.38])
    fig.add_trace(_line(s.index, s["fly"], "Actual fly", THEME["s1"]), row=1, col=1)
    rolling = res["stats"].get("resid_mode") == "rolling"
    fitted = s["fly"] - s["resid_roll"] if rolling else s["fitted"]
    fig.add_trace(_line(s.index, fitted, "Fitted (rolling)" if rolling else "Fitted (static)",
                        THEME["s2"], dash="dash"), row=1, col=1)
    resid = s["resid"]
    fig.add_trace(go.Bar(x=s.index, y=resid, name="Residual (bp)",
                         marker_color=np.where(resid > 0, THEME["pos"], THEME["neg"]),
                         marker_line_width=0, hovertemplate="%{y:.2f}"), row=2, col=1)
    fig.update_layout(bargap=0.1)
    fig.update_yaxes(title_text="bp", row=1, col=1)
    fig.update_yaxes(title_text="resid bp", row=2, col=1)
    _shade_reg(fig, *res["reg_window_dates"], rows=2)
    return _base_fig(fig)
 
 
def fig_beta(res: Dict, n: int) -> go.Figure:
    """Rolling betas (top) and rolling R^2 (bottom); static values dashed."""
    s = _window(res["series"], n)
    st = res["stats"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.6, 0.4])
    fig.add_trace(_line(s.index, s["beta_level"], "β level (rolling)", THEME["s1"]), row=1, col=1)
    fig.add_trace(_line(s.index, s["beta_curve"], "β curve (rolling)", THEME["s2"]), row=1, col=1)
    fig.add_hline(y=st["beta_level"], line=dict(color=THEME["s1"], dash="dash", width=1),
                  row=1, col=1)
    fig.add_hline(y=st["beta_curve"], line=dict(color=THEME["s2"], dash="dash", width=1),
                  row=1, col=1)
    fig.add_trace(_line(s.index, s["r2_roll"], "R² (rolling)", THEME["s3"]), row=2, col=1)
    fig.add_hline(y=st["r2"], line=dict(color=THEME["s3"], dash="dash", width=1), row=2, col=1)
    fig.update_yaxes(title_text="β", row=1, col=1)
    fig.update_yaxes(title_text="R²", range=[0, 1], row=2, col=1)
    return _base_fig(fig)
 
 
def fig_scatter(res: Dict, xmode: str) -> go.Figure:
    """Regression-sample scatter, coloured by time; latest point highlighted.
 
    fitted : actual vs fitted with 45-degree line
    level  : fly vs level, line = alpha + bL*x + bC*curve_now
    curve  : fly vs curve, line = alpha + bL*level_now + bC*x
    """
    st = res["stats"]
    start, end = res["reg_window_dates"]
    s = res["series"].loc[start:end]
    a, bl, bc = st["alpha"], st["beta_level"], st["beta_curve"]
    if xmode == "level":
        x, xl = s["level"], "Level (bp)"
        xs = np.linspace(x.min(), x.max(), 50)
        ys = a + bl * xs + bc * st["curve_bp"]
    elif xmode == "curve":
        x, xl = s["curve"], "Curve environment (bp)"
        xs = np.linspace(x.min(), x.max(), 50)
        ys = a + bl * st["level_bp"] + bc * xs
    else:
        x, xl = s["fitted"], "Fitted fly (bp)"
        xs = np.linspace(min(x.min(), s["fly"].min()), max(x.max(), s["fly"].max()), 50)
        ys = xs
    t = np.arange(len(s))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=s["fly"], mode="markers", name="Regression sample",
        marker=dict(size=8, color=t, colorscale=[[0, "#b7d3f6"], [1, "#104281"]],
                    line=dict(color=THEME["surface"], width=1), showscale=False),
        text=s.index.strftime("%d-%b-%y"),
        hovertemplate="%{text}<br>x %{x:.2f}<br>fly %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Model",
                             line=dict(color=THEME["muted"], width=1.5, dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[x.iloc[-1]], y=[s["fly"].iloc[-1]], mode="markers", name="Latest",
        marker=dict(size=13, color=THEME["s2"], line=dict(color=THEME["surface"], width=2)),
        hovertemplate="Latest<br>x %{x:.2f}<br>fly %{y:.2f}<extra></extra>"))
    eq = (f"fly = {a:+.2f} {bl:+.3f}·L {bc:+.3f}·C   R² {st['r2']:.0%}   SE {st['se_bp']:.2f}bp")
    fig.add_annotation(text=eq, xref="paper", yref="paper", x=0, y=-0.26, showarrow=False,
                       xanchor="left", font=dict(size=11, color=THEME["ink2"]))
    fig.update_xaxes(title_text=xl)
    fig.update_yaxes(title_text="Fly (bp)")
    fig = _base_fig(fig)
    fig.update_layout(hovermode="closest", margin=dict(l=50, r=16, t=24, b=80))
    return fig
 
 
def _cards(st: Dict, band: float) -> List[html.Div]:
    def card(label, value, sub=None, color=None):
        return html.Div([
            html.Div(label, style={"fontSize": "11px", "color": THEME["ink2"]}),
            html.Div(value, style={"fontSize": "18px", "fontWeight": 600,
                                   "color": color or THEME["ink"]}),
            html.Div(sub or "", style={"fontSize": "11px", "color": THEME["muted"]}),
        ], style={**_card_style(), "padding": "8px 12px", "minWidth": "110px"})
 
    sig = st["signal"]
    sig_color = THEME["pos"] if sig == "RECEIVE" else THEME["neg"] if sig == "PAY" else None
    hl = st["half_life_d"]
    return [
        card("Signal", sig, f"band ±{band:g}σ", sig_color),
        card("Fly", f"{st['fly_bp']:.1f}bp", f"fitted {st['fitted_bp']:.1f}bp"),
        card("Residual", f"{st['resid_bp']:+.2f}bp", f"resid/SE {st['resid_over_se']:+.2f}"),
        card("Residual z", f"{st['z_resid']:+.2f}", st["agreement"].lower()),
        card("Fly z", f"{st['z_fly']:+.2f}"),
        card("Curve z", f"{st['z_curve']:+.2f}", f"curve {st['curve_bp']:.1f}bp"),
        card("β level", f"{st['beta_level']:+.3f}", f"rolling {st['beta_level_roll']:+.3f}"),
        card("β curve", f"{st['beta_curve']:+.3f}", f"rolling {st['beta_curve_roll']:+.3f}"),
        card("R²", f"{st['r2']:.0%}", f"rolling {st['r2_roll']:.0%}"),
        card("Half-life", "∞" if np.isinf(hl) else f"{hl:.0f}d", "AR(1) on residual"),
        card("Carry", f"{st['carry_ann_bp']:+.1f}bp/yr", "receive belly"),
        card("Roll", f"{st['roll_ann_bp']:+.1f}bp/yr", "receive belly"),
    ]
 
 
def _legs_rows(res: Dict) -> List[Dict]:
    cr = res["carry_roll"]
    spec = res["spec"]
    role = {spec.w1: "W1", spec.b: "Belly", spec.w2: "W2"}
    rows = [{**{k: v for k, v in leg.items() if k != "tenor"},
             "leg": f"{role[leg['tenor']]} {be._fmt_tenor(leg['tenor'])}y"} for leg in cr["legs"]]
    rows.append({"leg": "Fly (2B-W1-W2)", "carry_h_bp": cr["carry_h_bp"],
                 "roll_h_bp": cr["roll_h_bp"], "carry_ann_bp": cr["carry_ann_bp"],
                 "roll_ann_bp": cr["roll_ann_bp"], "carry_approx_h_bp": cr["carry_approx_h_bp"]})
    return rows
 
 
# =============================================================================
# CALLBACKS
# =============================================================================
 
def register_callbacks(app) -> None:
    """Attach all callbacks for this tab to `app`."""
 
    controls = [Input(P + k, "value") for k in
                ("index", "regwin", "rollwin", "zwin", "band", "mode", "horizon")]
    asof_in = Input(P + "asof", "date")
 
    # ---- summary table -----------------------------------------------------
    @app.callback(
        Output(P + "table", "data"),
        Output(P + "table", "style_data_conditional"),
        Output(P + "table-msg", "children"),
        Output(P + "subtitle", "children"),
        Input(P + "group", "value"),
        Input(P + "custom", "data"),
        Input(P + "overrides", "data"),
        asof_in, *controls,
    )
    def _update_table(group, custom, overrides, asof, index, regwin, rollwin, zwin, band,
                      mode, horizon):
        params = _params(index, asof, regwin, rollwin, zwin, band, mode, horizon)
        df = be.get_rates(index)
        as_of = min(pd.Timestamp(asof), df.index[-1]) if asof else df.index[-1]
        subtitle = (f"{be.INDEX_CONFIG[index]['label']} · as of {as_of:%d-%b-%Y} · "
                    f"FLY = 2·Belly − W1 − W2 (bp) · regression FLY = α + βL·Level + βC·Curve · "
                    f"+ residual = belly cheap → receive")
        flies = _flies_for(group, custom)
        if not flies:
            msg = ("No flies in Custom yet - add one above." if group == CUSTOM_TAB else
                   f"No structures in '{group}' yet - populate STRUCTURE_GROUPS in the back end.")
            return [], _z_styles(band), msg, subtitle
        summ = be.summarise(df, flies, params, overrides)
        if "error" not in summ:
            summ["error"] = np.nan
        errors = summ[summ["error"].notna()]
        ok = summ[summ["error"].isna()]
        data = ok.drop(columns=[c for c in ("as_of", "error") if c in ok], errors="ignore")
        data = data.replace([np.inf, -np.inf], np.nan)
        records = data.to_dict("records")
        for r in records:
            r["id"] = r["name"]
        note = ok["carry_assumption"].iloc[0] if len(ok) else ""
        msg = f"{len(ok)} flies · carry: {note}"
        if len(errors):
            msg += " · skipped: " + "; ".join(f"{e['name']} ({e['error']})"
                                               for _, e in errors.iterrows())
        return records, _z_styles(band), msg, subtitle
 
    # ---- selection + custom list -------------------------------------------
    @app.callback(
        Output(P + "selected", "data"),
        Output(P + "custom", "data"),
        Output(P + "custom-msg", "children"),
        Output(P + "custom-input", "value"),
        Input(P + "table", "active_cell"),
        Input(P + "custom-add", "n_clicks"),
        Input(P + "custom-input", "n_submit"),
        Input(P + "custom-clear", "n_clicks"),
        Input(P + "group", "value"),
        State(P + "custom-input", "value"),
        State(P + "custom", "data"),
        State(P + "selected", "data"),
    )
    def _select(active_cell, _add, _submit, _clear, group, text, custom, selected):
        trig = ctx.triggered_id
        custom = list(custom or [])
        if trig in (P + "custom-add", P + "custom-input"):
            try:
                name = be.parse_fly(text or "").name
            except ValueError as exc:
                return no_update, no_update, str(exc), no_update
            if name not in custom:
                custom.append(name)
            return name, custom, "", ""
        if trig == P + "custom-clear":
            return (None if selected in custom else selected), [], "", no_update
        if trig == P + "table" and active_cell and active_cell.get("row_id"):
            return active_cell["row_id"], no_update, "", no_update
        flies = _flies_for(group, custom)
        if selected in flies:
            return no_update, no_update, "", no_update
        return (flies[0] if flies else None), no_update, "", no_update
 
    # ---- regressor override controls follow the selected fly --------------
    @app.callback(
        Output(P + "ov-level", "value"),
        Output(P + "ov-c1", "value"),
        Output(P + "ov-c2", "value"),
        Input(P + "selected", "data"),
        Input(P + "overrides", "data"),
    )
    def _sync_overrides(selected, overrides):
        if not selected:
            return None, None, None
        reg = be.resolve_regressors(be.parse_fly(selected), overrides)
        return reg.level, reg.curve[0], reg.curve[1]
 
    @app.callback(
        Output(P + "overrides", "data"),
        Output(P + "ov-msg", "children"),
        Input(P + "ov-apply", "n_clicks"),
        Input(P + "ov-reset", "n_clicks"),
        State(P + "selected", "data"),
        State(P + "ov-level", "value"),
        State(P + "ov-c1", "value"),
        State(P + "ov-c2", "value"),
        State(P + "overrides", "data"),
        prevent_initial_call=True,
    )
    def _apply_overrides(_apply, _reset, selected, level, c1, c2, overrides):
        if not selected:
            return no_update, "Select a fly first."
        overrides = dict(overrides or {})
        if ctx.triggered_id == P + "ov-reset":
            overrides.pop(selected, None)
            return overrides, ""
        if None in (level, c1, c2) or c1 >= c2:
            return no_update, "Curve short tenor must be below curve long tenor."
        overrides[selected] = {"level": level, "curve": [c1, c2]}
        return overrides, ""
 
    # ---- detail panel --------------------------------------------------------
    @app.callback(
        Output(P + "detail-title", "children"),
        Output(P + "detail-sub", "children"),
        Output(P + "cards", "children"),
        Output(P + "t-level", "children"),
        Output(P + "t-z", "children"),
        Output(P + "t-fit", "children"),
        Output(P + "t-beta", "children"),
        Output(P + "t-scatter", "children"),
        Output(P + "t-legs", "children"),
        Output(P + "fig-level", "figure"),
        Output(P + "fig-z", "figure"),
        Output(P + "fig-fit", "figure"),
        Output(P + "fig-beta", "figure"),
        Output(P + "fig-scatter", "figure"),
        Output(P + "legs", "data"),
        Output(P + "legs-note", "children"),
        Input(P + "selected", "data"),
        Input(P + "overrides", "data"),
        Input(P + "chartwin", "value"),
        Input(P + "overlay", "value"),
        Input(P + "zshow", "value"),
        Input(P + "scatter-x", "value"),
        asof_in, *controls,
    )
    def _detail(selected, overrides, chartwin, overlay, zshow, xmode, asof, index, regwin,
                rollwin, zwin, band, mode, horizon):
        if not selected:
            e = _empty_fig("Select a fly in the table")
            return ("Select a fly", "", [], "", "", "", "", "", "", e, e, e, e, e, [], "")
        params = _params(index, asof, regwin, rollwin, zwin, band, mode, horizon)
        try:
            res = be.analyse_fly(be.get_rates(index), be.parse_fly(selected), params, overrides)
        except Exception as exc:  # surface the problem instead of crashing the tab
            e = _empty_fig(f"Cannot analyse {selected}: {exc}")
            return (selected, str(exc), [], "", "", "", "", "", "", e, e, e, e, e, [], "")
 
        st, reg = res["stats"], res["reg"]
        src = {"auto": "default (wings)", "narrow-map": "narrow-fly macro map",
               "override": "user override"}[reg.source]
        r0, r1 = res["reg_window_dates"]
        title = f"{st['name']}: regressed on {reg.label}"
        sub = (f"Curve environment source: {src} · static fit {r0:%d-%b-%y} → {r1:%d-%b-%y} "
               f"({st['n_obs']} obs, shaded on charts) · rolling β {rollwin}d · z-window {zwin}d · "
               f"residual model: {mode}")
        hl = next((k for k, v in be.HORIZON_PRESETS.items() if abs(v - float(horizon)) < 1e-9),
                  f"{float(horizon):g}y")
        c1, c2 = reg.curve
        curve_name = f"{be._fmt_tenor(c1)}s{be._fmt_tenor(c2)}s"
        legs_note = (
            f"Receive-belly 1:2:1 (DV01 weights −1 / +2 / −1). Horizon h = {hl}. "
            f"Carry = P(0,h)·(S(0,T) − Fixing)·h / A(h,T); Roll = S(0,T) − S(0,T−h); "
            f"annualised = value / h. Approx carry = SR(h,T) − S(0,T), shown as a check only. "
            f"Carry basis: {res['carry_roll']['carry_assumption']}."
        )
        return (
            title, sub, _cards(st, float(band)),
            f"Fly level · overlays in stacked panels (curve env: {curve_name})",
            f"Z-scores · {zwin}d window · ±{float(band):g}σ band",
            f"Actual vs fitted ({mode} betas) · residual",
            "Rolling betas & R² · dashed = static over regression lookback",
            "Regression sample scatter · latest highlighted",
            f"Carry & roll breakdown · as of {st['as_of']:%d-%b-%Y}",
            fig_level(res, chartwin, overlay),
            fig_z(res, chartwin, float(band), zshow),
            fig_fit(res, chartwin),
            fig_beta(res, chartwin),
            fig_scatter(res, xmode),
            _legs_rows(res),
            legs_note,
        )
 
 
# =============================================================================
# STANDALONE DEMO
# =============================================================================
 
if __name__ == "__main__":
    from dash import Dash
 
    demo = Dash(__name__, title="EUR Swap Fly RV")
    demo.layout = layout()
    register_callbacks(demo)
    demo.run(debug=False, port=8050)
