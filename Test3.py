"""
curve_rv_frontend.py
====================
 
EUR Swap Curve Rich/Cheap Surface - Dash / Plotly view (front end)
------------------------------------------------------------------
 
Drop-in tab. All calculations live in `curve_rv_backend.py`.
 
Integration
~~~~~~~~~~~
    import curve_rv_frontend as cvrv
    tab_content = cvrv.layout()          # inside the host app's tab registry
    cvrv.register_callbacks(app)         # once, after the Dash app exists
 
* Component ids are prefixed "cvrv-".
* Colours live in THEME (same dark blue / dark red scheme as the fly tab).
* Data: spot comes from `fly_rv_backend.load_spot_rates` (shared), forwards
  from `curve_rv_backend.load_forward_rates` (placeholder).
 
Components
~~~~~~~~~~
1. Controls  : index, as-of, model (PCA / Nelson-Siegel), lookback, fit mode,
               z-score window, band, display (z-score / residual bp).
2. Heatmap   : rows = tenor, columns = forward start. Each column is its own
               model fit. Blue = cheap (market > model -> receive), red = rich
               (pay). Solid fill |z| >= band, light fill |z| >= 1. Footer row:
               variance explained by PC1-3 (PCA) or fit RMSE and lambda (NS).
               Hover a cell for detail; click to plot that curve below.
3. Strip bars: residual (z or bp) along the forward strip, with optional
               markers for 3 days, 1 week and 1 month ago.
4. Curve     : market vs model curve for the selected column / strip, with
               residuals underneath.
"""
 
from __future__ import annotations
 
import threading
from typing import Dict, List, Optional
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots
 
import curve_rv_backend as be
import fly_rv_backend as spot_be
 
P = "cvrv-"
 
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
    "blue": "#0d366b",       # cheap / market
    "red": "#9b1c1c",        # rich / model
    "blue_soft": "#cde2fb",
    "red_soft": "#f9d3d2",
    "marker": "#52514e",
    "shade": "rgba(13,54,107,0.06)",
}
 
WINDOWS = {"6M": 126, "1Y": 252, "2Y": 504, "3Y": 756}
WIN_OPTS = [{"label": k, "value": v} for k, v in WINDOWS.items()]
BAND_OPTS = [{"label": f"±{b:g}σ", "value": b} for b in (1.0, 1.5, 2.0, 2.5, 3.0)]
INDEX_OPTS = [{"label": v["label"], "value": k} for k, v in spot_be.INDEX_CONFIG.items()]
MARKER_STYLE = {"3D": ("triangle-down", "3 days ago"), "1W": ("circle", "1 week ago"),
                "1M": ("star", "1 month ago")}
 
 
# =============================================================================
# LAYOUT
# =============================================================================
 
def _card(children, **style):
    return html.Div(children, style={"background": THEME["surface"],
                                     "border": f"1px solid {THEME['border']}",
                                     "borderRadius": "8px", "padding": "12px", **style})
 
 
def _control(label, comp):
    return html.Div([html.Label(label, style={"fontSize": "11px", "color": THEME["ink2"],
                                              "display": "block", "marginBottom": "2px"}), comp],
                    style={"minWidth": "120px", "flex": "1 1 120px"})
 
 
def _dd(id_, options, value, **kw):
    return dcc.Dropdown(id=P + id_, options=options, value=value, clearable=False,
                        style={"fontSize": "13px"}, **kw)
 
 
def _title(text_id, static=None):
    return html.Div(static, id=P + text_id if text_id else None,
                    style={"fontSize": "13px", "fontWeight": 600, "marginBottom": "6px"})
 
 
def layout() -> html.Div:
    """Top-level component for the tab."""
    spot = spot_be.get_rates(spot_be.DEFAULT_INDEX)
    curve_opts = [{"label": f"{lab} forward" if s else "Spot", "value": lab}
                  for lab, s in be.GRID_STARTS] + [{"label": "Forward strip",
                                                    "value": be.STRIP_ID}]
    controls = _card([
        _control("Floating index", _dd("index", INDEX_OPTS, spot_be.DEFAULT_INDEX)),
        _control("As-of date", dcc.DatePickerSingle(
            id=P + "asof", date=spot.index[-1].date(), min_date_allowed=spot.index[0].date(),
            max_date_allowed=spot.index[-1].date(), display_format="DD-MMM-YY")),
        _control("Model", _dd("model", [{"label": "PCA (3 PCs)", "value": "pca"},
                                        {"label": "Nelson-Siegel", "value": "ns"}], "pca")),
        _control("Lookback", _dd("lookback", WIN_OPTS, 252)),
        _control("Fit mode (PCA)", _dd("mode", [{"label": "Static", "value": "static"},
                                                {"label": "Rolling", "value": "rolling"}],
                                       "static")),
        _control("Z-score window", _dd("zwin", WIN_OPTS, 252)),
        _control("Band", _dd("band", BAND_OPTS, 2.0)),
        _control("Display", _dd("display", [{"label": "Z-score", "value": "z"},
                                            {"label": "Residual (bp)", "value": "bp"}], "z")),
    ], display="flex", flexWrap="wrap", gap="10px")
 
    heat = _card([
        _title("heat-title"),
        html.Div("Blue = cheap (market above model, receive) · Red = rich (pay) · "
                 "shading always by z: solid |z| ≥ band, light |z| ≥ 1 · hover for detail · "
                 "click a cell to plot its curve",
                 style={"fontSize": "11px", "color": THEME["ink2"], "marginBottom": "8px"}),
        dash_table.DataTable(
            id=P + "heat", data=[], columns=[],
            cell_selectable=True,
            style_table={"overflowX": "auto"},
            style_header={"fontWeight": 600, "fontSize": "12px", "background": THEME["page"],
                          "color": THEME["ink2"], "textAlign": "right",
                          "border": f"1px solid {THEME['grid']}"},
            style_cell={"fontFamily": THEME["font"], "fontSize": "12px", "padding": "5px 10px",
                        "textAlign": "right", "fontVariantNumeric": "tabular-nums",
                        "color": THEME["ink"], "border": f"1px solid {THEME['grid']}",
                        "minWidth": "64px"},
            tooltip_delay=200, tooltip_duration=None,
        ),
    ])
 
    bars = _card([
        _title("bars-title"),
        dcc.Checklist(id=P + "markers", inline=True, value=["3D", "1W", "1M"],
                      options=[{"label": f" {v[1]}", "value": k} for k, v in MARKER_STYLE.items()],
                      style={"fontSize": "12px"}, inputStyle={"marginLeft": "10px"}),
        dcc.Graph(id=P + "bars", config={"displaylogo": False}, style={"height": "380px"}),
    ], marginTop="12px")
 
    curve = _card([
        html.Div([
            _title("curve-title"),
            html.Div(_dd("curve", curve_opts, "Spot"), style={"width": "220px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "gap": "12px", "flexWrap": "wrap"}),
        dcc.Graph(id=P + "curve-fig", config={"displaylogo": False}, style={"height": "460px"}),
    ], marginTop="12px")
 
    return html.Div([
        dcc.Store(id=P + "point", data=None),
        html.Div([html.H3("EUR Curve Rich / Cheap", style={"margin": 0}),
                  html.Div(id=P + "subtitle", style={"fontSize": "12px", "color": THEME["ink2"]})],
                 style={"marginBottom": "10px"}),
        controls,
        html.Div(heat, style={"marginTop": "12px"}),
        bars,
        curve,
    ], style={"fontFamily": THEME["font"], "color": THEME["ink"], "background": THEME["page"],
              "padding": "16px"})
 
 
# =============================================================================
# HELPERS
# =============================================================================
 
_LOCK = threading.Lock()   # callbacks share one cached fit instead of racing
 
 
def _surface(index, asof, model, lookback, mode, zwin):
    as_of = str(pd.Timestamp(asof).date()) if asof else None
    with _LOCK:
        surf = be.run_surface(index, as_of, model, int(lookback),
                              mode if model == "pca" else "static")
    z = be.add_z(surf, int(zwin))
    return surf, z, be.snapshot(surf, z)
 
 
def _fmt(v: float, display: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    d = 2 if display == "z" else 1
    return f"({abs(v):.{d}f})" if v < 0 else f"{v:.{d}f}"
 
 
def _ten_label(t: float) -> str:
    return f"{t:g}Y"
 
 
def _base(fig, height):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=55, r=16, t=30, b=40),
                      font=dict(family=THEME["font"], size=11, color=THEME["ink2"]),
                      paper_bgcolor=THEME["surface"], plot_bgcolor=THEME["surface"],
                      legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(gridcolor=THEME["grid"], linecolor=THEME["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=THEME["grid"], linecolor=THEME["axis"], zeroline=False)
    return fig
 
 
def _empty(msg):
    fig = _base(go.Figure(), 380)
    fig.add_annotation(text=msg, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                       font=dict(size=13, color=THEME["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
 
 
def build_heatmap(snap: pd.DataFrame, surf: Dict, display: str, band: float):
    """(columns, data, styles, tooltips) for the heatmap DataTable."""
    cols = [lab for lab, _ in be.GRID_STARTS if lab in surf["curves"]]
    columns = [{"name": "Tenor", "id": "tenor"}] + [{"name": c, "id": c} for c in cols]
    grid = snap[snap["cid"] != be.STRIP_ID]
    data, styles, tips = [], [], []
    val_col = "z" if display == "z" else "resid_bp"
    for i, t in enumerate(be.GRID_TENORS):
        row, tip = {"tenor": _ten_label(t)}, {}
        for c in cols:
            hit = grid[(grid["cid"] == c) & (np.isclose(grid["tenor"], t))]
            if hit.empty:
                row[c] = ""
                continue
            r = hit.iloc[0]
            row[c] = _fmt(r[val_col], display)
            zv = r["z"]
            fill = None
            if not np.isnan(zv):
                if zv >= band:
                    fill = {"backgroundColor": THEME["blue"], "color": "#ffffff", "fontWeight": 700}
                elif zv <= -band:
                    fill = {"backgroundColor": THEME["red"], "color": "#ffffff", "fontWeight": 700}
                elif zv >= 1:
                    fill = {"backgroundColor": THEME["blue_soft"]}
                elif zv <= -1:
                    fill = {"backgroundColor": THEME["red_soft"]}
            if fill:
                # repeat for active/selected states so clicking keeps the shading
                for state in (None, "active", "selected"):
                    cond = {"row_index": i, "column_id": c}
                    if state:
                        cond["state"] = state
                    styles.append({"if": cond, **fill})
            label = "Spot" if c == "Spot" else f"{c} fwd"
            tip[c] = {"type": "markdown", "value": (
                f"**{label} {_ten_label(t)}**  \n"
                f"Market {r['market_bp'] / 100:.3f}% · Model {r['model_bp'] / 100:.3f}%  \n"
                f"Residual {r['resid_bp']:+.2f}bp · z {zv:+.2f}  \n"
                f"z 1W ago {r['z_1W']:+.2f} · 1M ago {r['z_1M']:+.2f}")}
        data.append(row)
        tips.append(tip)
    # footer: model diagnostics per column
    foot = {"tenor": "PC1-3 var" if surf["params"]["model"] == "pca" else "NS RMSE / λ"}
    for c in cols:
        info = surf["curves"][c]["info"]
        foot[c] = (f"{info['explained_total']:.1%}" if "explained_total" in info
                   else f"{info['rmse_bp']:.1f}bp / {info['lambda']:g}")
    data.append(foot)
    tips.append({})
    n = len(data) - 1
    base = [
        {"if": {"state": "active"}, "border": f"2px solid {THEME['ink']}",
         "backgroundColor": THEME["surface"]},
        {"if": {"state": "selected"}, "backgroundColor": THEME["surface"]},
        {"if": {"row_index": n}, "fontStyle": "italic", "color": THEME["ink2"],
         "backgroundColor": THEME["page"], "fontSize": "11px"},
        {"if": {"column_id": "tenor"}, "fontWeight": 600, "textAlign": "left",
         "backgroundColor": THEME["page"]},
    ]
    return columns, data, base + styles, tips
 
 
def fig_strip(surf: Dict, snap: pd.DataFrame, display: str, band: float,
              markers: List[str]) -> go.Figure:
    s = snap[snap["cid"] == be.STRIP_ID].sort_values("x")
    if s.empty:
        return _empty("Strip unavailable")
    col = "z" if display == "z" else "resid_bp"
    vals = s[col].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=s["label"], y=vals, name="Latest", showlegend=False,
        marker_color=THEME["blue"], marker_line_width=0,
        customdata=np.column_stack([s["resid_bp"], s["z"], s["market_bp"] / 100,
                                    s["model_bp"] / 100]),
        hovertemplate="%{x}<br>residual %{customdata[0]:+.2f}bp · z %{customdata[1]:+.2f}"
                      "<br>market %{customdata[2]:.3f}% · model %{customdata[3]:.3f}%"
                      "<extra></extra>"))
    for k in markers or []:
        sym, name = MARKER_STYLE[k]
        fig.add_trace(go.Scatter(
            x=s["label"], y=s[f"{'z' if display == 'z' else 'resid'}_{k}"], mode="markers",
            name=name, marker=dict(symbol=sym, size=9, color=THEME["marker"],
                                   line=dict(color=THEME["surface"], width=1)),
            hovertemplate=f"{name}: %{{y:+.2f}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    if display == "z":
        for y in (band, -band):
            fig.add_hline(y=y, line=dict(color=THEME["muted"], width=1, dash="dash"))
    fig.update_yaxes(title_text="z-score (σ)" if display == "z" else "residual (bp)")
    fig.update_layout(bargap=0.35, hovermode="closest")
    return _base(fig, 380)
 
 
def fig_curve(surf: Dict, cid: str, point: Optional[Dict]) -> go.Figure:
    c = surf["curves"].get(cid)
    if c is None:
        return _empty(f"{cid} unavailable")
    cdef = c["def"]
    cols = list(c["market"].columns)
    idx = [cdef.points.index(p) for p in cols]
    strip = cdef.kind == "strip"
    x = [cdef.labels[i] for i in idx] if strip else [cdef.x[i] for i in idx]
    mkt = c["market"].iloc[-1].to_numpy() / 100
    mdl = c["fitted"].iloc[-1].to_numpy() / 100
    res = c["resid"].iloc[-1].to_numpy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.68, 0.32])
    fig.add_trace(go.Scatter(x=x, y=mkt, name="Market", mode="lines+markers",
                             line=dict(color=THEME["blue"], width=2),
                             marker=dict(size=7, line=dict(color=THEME["surface"], width=1)),
                             hovertemplate="%{y:.3f}%<extra>Market</extra>"), row=1, col=1)
    model_name = "PCA (3 PCs)" if surf["params"]["model"] == "pca" else "Nelson-Siegel"
    fig.add_trace(go.Scatter(x=x, y=mdl, name=model_name, mode="lines+markers",
                             line=dict(color=THEME["red"], width=2, dash="dash"),
                             marker=dict(size=6),
                             hovertemplate="%{y:.3f}%<extra>Model</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=res, name="Residual (bp)", showlegend=False,
                         marker_color=np.where(res >= 0, THEME["blue"], THEME["red"]),
                         marker_line_width=0, hovertemplate="%{y:+.2f}bp<extra></extra>"),
                  row=2, col=1)
    if point and point.get("cid") == cid:
        key = tuple(point["key"])
        if key in cols:
            j = cols.index(key)
            fig.add_trace(go.Scatter(x=[x[j]], y=[mkt[j]], mode="markers", name="Selected",
                                     marker=dict(size=14, color="rgba(0,0,0,0)",
                                                 line=dict(color=THEME["ink"], width=2)),
                                     hoverinfo="skip"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1), row=2, col=1)
    fig.update_yaxes(title_text="rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="resid (bp)", row=2, col=1)
    fig.update_xaxes(title_text="forward point" if strip else "tenor (years)", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return _base(fig, 460)
 
 
# =============================================================================
# CALLBACKS
# =============================================================================
 
def register_callbacks(app) -> None:
    """Attach all callbacks for this tab to `app`."""
    ctrl = [Input(P + k, "value") for k in ("index", "model", "lookback", "mode", "zwin")]
    asof = Input(P + "asof", "date")
 
    @app.callback(Output(P + "mode", "disabled"), Input(P + "model", "value"))
    def _mode_enabled(model):
        return model != "pca"
 
    @app.callback(
        Output(P + "heat", "columns"), Output(P + "heat", "data"),
        Output(P + "heat", "style_data_conditional"), Output(P + "heat", "tooltip_data"),
        Output(P + "heat-title", "children"), Output(P + "subtitle", "children"),
        asof, *ctrl, Input(P + "display", "value"), Input(P + "band", "value"),
    )
    def _heat(asof_, index, model, lookback, mode, zwin, display, band):
        surf, z, snap = _surface(index, asof_, model, lookback, mode, zwin)
        cols, data, styles, tips = build_heatmap(snap, surf, display, float(band))
        mname = "PCA, 3 PCs, centred covariance" if model == "pca" else "Nelson-Siegel"
        what = "z-score of residual" if display == "z" else "residual (bp)"
        title = f"Rich / cheap surface · {what} · market − model"
        fit = mode if model == "pca" else "daily cross-section"
        sub = (f"{spot_be.INDEX_CONFIG[index]['label']} · as of {surf['as_of']:%d-%b-%Y} · "
               f"{mname} · lookback {lookback}d · fit {fit} · z-window {zwin}d · "
               f"one model per column")
        return cols, data, styles, tips, title, sub
 
    @app.callback(
        Output(P + "bars", "figure"), Output(P + "bars-title", "children"),
        asof, *ctrl, Input(P + "display", "value"), Input(P + "band", "value"),
        Input(P + "markers", "value"),
    )
    def _bars(asof_, index, model, lookback, mode, zwin, display, band, markers):
        surf, z, snap = _surface(index, asof_, model, lookback, mode, zwin)
        what = "z-score" if display == "z" else "residual (bp)"
        return (fig_strip(surf, snap, display, float(band), markers),
                f"Forward strip · {what} · bar = latest, markers = history")
 
    @app.callback(
        Output(P + "curve", "value"), Output(P + "point", "data"),
        Input(P + "heat", "active_cell"), Input(P + "bars", "clickData"),
        State(P + "heat", "data"),
        prevent_initial_call=True,
    )
    def _select(cell, click, data):
        if ctx.triggered_id == P + "heat" and cell:
            col = cell.get("column_id")
            row = cell.get("row")
            if col == "tenor" or row is None or row >= len(be.GRID_TENORS):
                return no_update, no_update
            start = dict(be.GRID_STARTS)[col]
            return col, {"cid": col, "key": list(be._key(start, be.GRID_TENORS[row]))}
        if ctx.triggered_id == P + "bars" and click:
            label = click["points"][0]["x"]
            for s, n in be.STRIP:
                if be.strip_label(s, n) == label:
                    return be.STRIP_ID, {"cid": be.STRIP_ID, "key": list(be._key(s, n))}
            return be.STRIP_ID, no_update
        return no_update, no_update
 
    @app.callback(
        Output(P + "curve-fig", "figure"), Output(P + "curve-title", "children"),
        asof, *ctrl, Input(P + "curve", "value"), Input(P + "point", "data"),
    )
    def _curve(asof_, index, model, lookback, mode, zwin, cid, point):
        surf, _, _ = _surface(index, asof_, model, lookback, mode, zwin)
        name = "Forward strip" if cid == be.STRIP_ID else (
            "Spot curve" if cid == "Spot" else f"{cid} forward curve")
        return fig_curve(surf, cid, point), f"{name} · market vs model · as of {surf['as_of']:%d-%b-%Y}"
 
 
# =============================================================================
# STANDALONE DEMO
# =============================================================================
 
if __name__ == "__main__":
    from dash import Dash
 
    demo = Dash(__name__, title="EUR Curve Rich / Cheap")
    demo.layout = layout()
    register_callbacks(demo)
    demo.run(debug=False, port=8051)
 
