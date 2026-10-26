"""
curve_rv_backend.py
===================
 
EUR Swap Curve Rich/Cheap Surface (back end)
--------------------------------------------
 
Fair-value model for every point on the spot curve, a grid of forward-starting
curves, and a non-overlapping forward strip. Two interchangeable models:
 
* PCA          : time-series fair value (how the curve has co-moved).
* Nelson-Siegel: cross-sectional fair value (a smooth curve shape today).
 
For every point:   residual = market - model   (bp)
                   z        = rolling z-score of the residual
                   residual > 0 -> point CHEAP vs model -> RECEIVE
                   residual < 0 -> point RICH  vs model -> PAY
 
Curves
~~~~~~
* Grid columns : one curve per forward start (Spot, 1M, 3M, 6M, 1Y, 2Y, 5Y,
                 7Y, 10Y); each holds forward swaps of tenor 1Y..30Y starting
                 at that date. Each column is modelled SEPARATELY (one PCA or
                 one NS fit per column).
* Forward strip: Spot 1y, 1y1y .. 9y1y, 10y2y, 12y3y, 15y5y, 20y5y, 25y5y,
                 30y10y, 40y10y. Non-overlapping, tiles 0-50y. Modelled as its
                 own curve.
 
PCA methodology (per curve)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
On rate LEVELS (bp), over the lookback window W (T x p matrix X):
    1. Mean-centre each point:          Y = X - mu        (NOT standardised)
    2. Covariance:                      C = Y'Y / (T - 1)
    3. Eigen-decomposition:             C = V D V'   (eigenvalues descending)
    4. Keep the first 3 eigenvectors:   V3 (p x 3)
    5. Reconstruct:                     X_hat = mu + Y V3 V3'
    6. Residual:                        e = X - X_hat
  Static mode : fit once on the window ending at the as-of date; the loadings
                are applied to the full history (out-of-sample before W).
  Rolling mode: refit every day on the trailing W days; e_t uses the fit
                ending at t. (Eigenvector sign flips do not matter: V3 V3' is
                sign-invariant.)
 
Nelson-Siegel methodology
~~~~~~~~~~~~~~~~~~~~~~~~~
Grid columns (yield form, x = tenor tau in years):
    y(tau) = b0 + b1 * g1(tau) + b2 * g2(tau)
    g1 = (1 - exp(-tau/lam)) / (tau/lam),   g2 = g1 - exp(-tau/lam)
Strip (instantaneous-forward form, averaged over each point [s, e]):
    f(t) = b0 + b1 * exp(-t/lam) + b2 * (t/lam) * exp(-t/lam)
    F(s,e) = 1/(e-s) * integral_s^e f(t) dt
           = b0 + b1 * lam (e^{-s/lam} - e^{-e/lam}) / (e-s)
                + b2 * [(s+lam) e^{-s/lam} - (e+lam) e^{-e/lam}] / (e-s)
  lam is fixed per curve: chosen by grid search to minimise total squared
  error over the lookback window (Diebold-Li style). Given lam, the betas are
  a daily cross-sectional OLS, so the model is linear and stable. The fit
  mode toggle does not apply to NS (each day is fitted on its own).
 
Z-score: z_t = (e_t - mean_W(e)) / std_W(e) over the z-score window.
"""
 
from __future__ import annotations
 
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple
 
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
 
import fly_rv_backend as spot_be   # reuse the spot rates placeholder & demo data
 
# =============================================================================
# 0. CONFIGURATION / PLACEHOLDERS
# =============================================================================
 
# Grid columns: (label, forward start in years)
GRID_STARTS: List[Tuple[str, float]] = [
    ("Spot", 0.0), ("1M", 1 / 12), ("3M", 0.25), ("6M", 0.5), ("1Y", 1.0),
    ("2Y", 2.0), ("5Y", 5.0), ("7Y", 7.0), ("10Y", 10.0),
]
# Grid rows: tenors in years
GRID_TENORS: List[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
 
# Forward strip: (start, tenor) in years. Non-overlapping, tiles 0-50y.
STRIP: List[Tuple[float, float]] = (
    [(0.0, 1.0)] + [(float(s), 1.0) for s in range(1, 10)]
    + [(10.0, 2.0), (12.0, 3.0), (15.0, 5.0), (20.0, 5.0), (25.0, 5.0),
       (30.0, 10.0), (40.0, 10.0)]
)
STRIP_ID = "Strip"
 
N_FACTORS = 3
NS_LAMBDA_GRID = np.round(np.linspace(0.25, 30.0, 120), 4)   # years
HISTORY_MARKERS: Dict[str, int] = {"3D": 3, "1W": 5, "1M": 21}  # business days back
MIN_OBS = 40   # minimum rows for a fit
 
 
def load_forward_rates(index: str = spot_be.DEFAULT_INDEX) -> Optional[pd.DataFrame]:
    """PLACEHOLDER - return daily forward swap rates, or None.
 
    Contract
    --------
    index   : DatetimeIndex (business days, ascending), aligned with spot.
    columns : forward points as (start, tenor) in years - either a 2-level
              MultiIndex of floats or labels such as "1Y5Y", "6M2Y", "1y1y".
    values  : forward par swap rates in PERCENT.
 
    Any (start, tenor) the grid or strip needs that is missing here is
    derived from the spot curve with `derive_forwards` (geometric formula).
 
    # ASK (integration): which forwards dataset feeds this, how are its
    #   forwards built (geometric formula on spot, or a curve build), and does
    #   it already contain the strip points 10y2y, 12y3y, 15y5y, 20y5y, 25y5y,
    #   30y10y, 40y10y? If not, confirm the geometric fallback below is the
    #   same convention the dataset uses.
    """
    if spot_be.USE_DEMO_DATA:
        return None          # demo: every forward is derived from spot
    raise NotImplementedError(
        "load_forward_rates(): connect this to the forwards dataset "
        "(see docstring for the expected DataFrame contract)."
    )
 
 
# =============================================================================
# 1. DATA ASSEMBLY
# =============================================================================
 
def parse_fwd_label(label) -> Tuple[float, float]:
    """(start, tenor) in years from "1Y5Y", "6M2Y", "1y1y", ("1Y","5Y") or (1, 5)."""
    if isinstance(label, tuple) and len(label) == 2:
        return tuple(_tenor_years(x) for x in label)  # type: ignore[return-value]
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([YyMm])", str(label))
    if len(parts) != 2:
        raise ValueError(f"Cannot parse forward label {label!r}")
    return tuple(float(v) / (12.0 if u.upper() == "M" else 1.0) for v, u in parts)  # type: ignore
 
 
def _tenor_years(x) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([YyMm]?)\s*", str(x))
    if not m:
        raise ValueError(f"Cannot parse tenor {x!r}")
    return float(m.group(1)) / (12.0 if m.group(2).upper() == "M" else 1.0)
 
 
def _key(s: float, t: float) -> Tuple[float, float]:
    return (round(float(s), 6), round(float(t), 6))
 
 
def spot_rates_at(spot: pd.DataFrame, tenors: Sequence[float]) -> pd.DataFrame:
    """Spot par rates (%) at arbitrary tenors, every date.
 
    >= 1y : natural cubic spline across quoted tenors (shared with the fly tab).
    <  1y : quoted sub-1y tenors if any; otherwise linear between the fixing
            (placed at the fixing tenor, 6m) and the 1y rate, flat below.
 
    # ASK (integration): confirm the short-end convention used by the
    #   forwards dataset for 1M/3M/6M starts.
    """
    tenors = sorted({float(t) for t in tenors if t > 0})
    out = pd.DataFrame(index=spot.index)
    long = [t for t in tenors if t >= 1.0]
    if long:
        out = out.join(spot_be.get_par_rates(spot, long))
    short = [t for t in tenors if t < 1.0]
    if short:
        quoted = spot_be.tenor_columns(spot)
        sub = [q for q in quoted if q < 1.0]
        if sub:
            knots = sub + [q for q in quoted if q >= 1.0]
            sp = CubicSpline(knots, spot[knots].to_numpy(), axis=1, bc_type="natural")
            vals = sp(short)
            for i, t in enumerate(short):
                out[t] = vals[:, i]
        else:
            fix_t = spot_be.INDEX_CONFIG[spot_be.DEFAULT_INDEX]["fixing_tenor"]
            fix = spot[spot_be.FIXING_COLUMN] if spot_be.FIXING_COLUMN in spot else spot[1.0]
            one = spot[1.0]
            for t in short:
                w = 0.0 if t <= fix_t else (t - fix_t) / (1.0 - fix_t)
                out[t] = (1 - w) * fix + w * one
    return out
 
 
def derive_forwards(spot: pd.DataFrame, points: Sequence[Tuple[float, float]]) -> pd.DataFrame:
    """Forward rates (%) for (start, tenor) points via the GEOMETRIC formula.
 
        F(s, s+n) = [ (1 + S_{s+n})^{s+n} / (1 + S_s)^s ]^{1/n} - 1
 
    with S the spot rate (annual compounding, par rates used as the curve).
    For s = 0 this returns the spot rate exactly.
 
    # ASK (integration): confirm this matches how the forwards dataset is
    #   built (zero vs par rates, compounding). Replace if it differs.
    """
    need = {s for s, _ in points if s > 0} | {s + n for s, n in points}
    S = spot_rates_at(spot, need) / 100.0
    out = {}
    for s, n in points:
        e = s + n
        if s == 0:
            out[_key(s, n)] = 100.0 * S[e]
        else:
            g = (1 + S[e]) ** e / (1 + S[s]) ** s
            out[_key(s, n)] = 100.0 * (g ** (1.0 / n) - 1.0)
    return pd.DataFrame(out, index=spot.index)
 
 
def _fwd_frame_keys(fwd: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in fwd.columns:
        s, t = parse_fwd_label(c)
        cols[c] = _key(s, t)
    return fwd.rename(columns=cols)
 
 
@dataclass(frozen=True)
class CurveDef:
    """One modelled curve: an id, its points and display info."""
    cid: str
    points: Tuple[Tuple[float, float], ...]     # (start, tenor)
    labels: Tuple[str, ...]
    x: Tuple[float, ...]                        # x-axis value per point
    kind: str                                   # "grid" | "strip"
    start: Optional[float] = None               # grid only
 
 
def _fmt(t: float) -> str:
    if t < 1:
        return f"{round(t * 12):d}m"
    return f"{t:g}y"
 
 
def strip_label(s: float, n: float) -> str:
    return f"Spot {_fmt(n)}" if s == 0 else f"{_fmt(s)}{_fmt(n)}"
 
 
def curve_defs(max_end: float) -> List[CurveDef]:
    """Grid columns + strip, keeping points with start + tenor <= max_end."""
    defs = []
    for label, s in GRID_STARTS:
        pts = tuple(_key(s, n) for n in GRID_TENORS if s + n <= max_end + 1e-9)
        defs.append(CurveDef(label, pts, tuple(_fmt(n) for _, n in pts),
                             tuple(n for _, n in pts), "grid", s))
    pts = tuple(_key(s, n) for s, n in STRIP if s + n <= max_end + 1e-9)
    defs.append(CurveDef(STRIP_ID, pts, tuple(strip_label(s, n) for s, n in pts),
                         tuple(s + n / 2 for s, n in pts), "strip"))
    return defs
 
 
@lru_cache(maxsize=4)
def get_panel(index: str = spot_be.DEFAULT_INDEX) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """All points (grid + strip) in % as one wide frame keyed by (start, tenor).
 
    Returns (panel, source) where source[(s,t)] is "data" or "derived".
    """
    spot = spot_be.get_rates(index)
    max_end = max(spot_be.tenor_columns(spot))
    pts = sorted({p for d in curve_defs(max_end) for p in d.points})
    fwd = load_forward_rates(index)
    have = pd.DataFrame(index=spot.index)
    if fwd is not None:
        fwd = _fwd_frame_keys(prepare_index(fwd)).reindex(spot.index).ffill()
        have = fwd[[p for p in pts if p in fwd.columns]]
    missing = [p for p in pts if p not in have.columns]
    derived = derive_forwards(spot, missing) if missing else pd.DataFrame(index=spot.index)
    panel = pd.concat([have, derived], axis=1)[pts]
    source = {p: ("data" if p in have.columns else "derived") for p in pts}
    return panel, source
 
 
def prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()
 
 
# =============================================================================
# 2. MODELS
# =============================================================================
 
def pca_fit(X: np.ndarray, k: int = N_FACTORS) -> Dict:
    """Centred-covariance PCA on X (T x p). Returns mu, V (p x k), eigenvalues
    (all, descending) and explained-variance ratios."""
    mu = X.mean(axis=0)
    Y = X - mu
    C = (Y.T @ Y) / (X.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)             # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    k = min(k, X.shape[1])
    return {"mu": mu, "V": evecs[:, :k], "evals": evals,
            "explained": evals / evals.sum(), "k": k}
 
 
def pca_reconstruct(X: np.ndarray, fit: Dict) -> np.ndarray:
    """X_hat = mu + (X - mu) V V'."""
    Y = X - fit["mu"]
    return fit["mu"] + Y @ fit["V"] @ fit["V"].T
 
 
def rolling_pca_fitted(X: np.ndarray, W: int, k: int = N_FACTORS) -> np.ndarray:
    """Rolling PCA reconstruction, vectorised. Row t uses the centred-covariance
    PCA fitted on rows t-W+1..t (NaN for t < W-1). Identical to looping
    pca_fit / pca_reconstruct over each window, but with cumulative sums and a
    batched eigh."""
    T, p = X.shape
    out = np.full_like(X, np.nan)
    if T < W:
        return out
    Z = X - X.mean(axis=0)                       # global shift for precision
    S1 = np.vstack([np.zeros(p), np.cumsum(Z, axis=0)])
    S2 = np.concatenate([np.zeros((1, p, p)),
                         np.cumsum(Z[:, :, None] * Z[:, None, :], axis=0)])
    t = np.arange(W - 1, T)
    s1 = S1[t + 1] - S1[t + 1 - W]               # (n, p)
    s2 = S2[t + 1] - S2[t + 1 - W]               # (n, p, p)
    mu = s1 / W
    C = (s2 - W * mu[:, :, None] * mu[:, None, :]) / (W - 1)
    _, V = np.linalg.eigh(C)                     # ascending eigenvalues
    V = V[:, :, ::-1][:, :, :k]                  # top k, (n, p, k)
    y = Z[t] - mu
    proj = np.einsum("npk,nqk,nq->np", V, V, y)
    out[t] = X.mean(axis=0) + mu + proj
    return out
 
 
def ns_loadings_yield(tau: np.ndarray, lam: float) -> np.ndarray:
    """Yield-form NS loadings (p x 3) at tenors tau."""
    x = tau / lam
    g1 = (1 - np.exp(-x)) / x
    g2 = g1 - np.exp(-x)
    return np.column_stack([np.ones_like(tau), g1, g2])
 
 
def ns_loadings_fwd_avg(s: np.ndarray, e: np.ndarray, lam: float) -> np.ndarray:
    """Instantaneous-forward NS loadings averaged over [s, e] (p x 3)."""
    n = e - s
    es, ee = np.exp(-s / lam), np.exp(-e / lam)
    g1 = lam * (es - ee) / n
    g2 = ((s + lam) * es - (e + lam) * ee) / n
    return np.column_stack([np.ones_like(s), g1, g2])
 
 
def _ns_loadings(cdef: CurveDef, lam: float) -> np.ndarray:
    s = np.array([p[0] for p in cdef.points])
    n = np.array([p[1] for p in cdef.points])
    if cdef.kind == "strip":
        return ns_loadings_fwd_avg(s, s + n, lam)
    return ns_loadings_yield(n, lam)
 
 
def ns_fit_lambda(X: np.ndarray, cdef: CurveDef) -> Tuple[float, float]:
    """Grid-search lam minimising total squared error over the window.
    Returns (lam, rmse_bp)."""
    best = (np.nan, np.inf)
    for lam in NS_LAMBDA_GRID:
        L = _ns_loadings(cdef, lam)
        P = L @ np.linalg.pinv(L)               # projection onto NS span
        sse = float(((X - X @ P.T) ** 2).sum())
        if sse < best[1]:
            best = (float(lam), sse)
    return best[0], math.sqrt(best[1] / X.size)
 
 
def ns_fitted(X: np.ndarray, cdef: CurveDef, lam: float) -> np.ndarray:
    """Daily cross-sectional OLS fit, all dates at once."""
    L = _ns_loadings(cdef, lam)
    return X @ (L @ np.linalg.pinv(L)).T
 
 
# =============================================================================
# 3. SURFACE
# =============================================================================
 
def rolling_z(df: pd.DataFrame, window: int) -> pd.DataFrame:
    m = df.rolling(window, min_periods=window).mean()
    sd = df.rolling(window, min_periods=window).std()
    return (df - m) / sd.where(sd > 1e-12)
 
 
def _model_curve(Xdf: pd.DataFrame, cdef: CurveDef, model: str, lookback: int,
                 mode: str) -> Tuple[pd.DataFrame, Dict]:
    """Fitted values (bp) for one curve over the full history."""
    X = Xdf.to_numpy()
    T = len(X)
    W = min(lookback, T)
    fitted = np.full_like(X, np.nan)
    info: Dict = {}
    if model == "pca":
        if mode == "rolling":
            fitted = rolling_pca_fitted(X, W)
        fit = pca_fit(X[T - W:])
        if mode != "rolling":
            fitted = pca_reconstruct(X, fit)
        info = {"explained": fit["explained"][: fit["k"]].tolist(),
                "explained_total": float(fit["explained"][: fit["k"]].sum()),
                "loadings": fit["V"]}
    else:
        lam, rmse = ns_fit_lambda(X[T - W:], cdef)
        fitted = ns_fitted(X, cdef, lam)
        info = {"lambda": lam, "rmse_bp": rmse}
    return pd.DataFrame(fitted, index=Xdf.index, columns=Xdf.columns), info
 
 
@lru_cache(maxsize=16)
def run_surface(index: str = spot_be.DEFAULT_INDEX, as_of: Optional[str] = None,
                model: str = "pca", lookback: int = 252, mode: str = "static") -> Dict:
    """Fit every curve. Cached - z-scores are applied separately (`add_z`).
 
    Returns {"as_of": Timestamp, "curves": {cid: {...}}, "source": {...}} where
    each curve holds def, market, fitted, resid (DataFrames in bp) and info.
    """
    panel, source = get_panel(index)
    if as_of:
        panel = panel.loc[: pd.Timestamp(as_of)]
    panel = panel * 100.0                     # % -> bp
    spot = spot_be.get_rates(index)
    curves = {}
    for cdef in curve_defs(max(spot_be.tenor_columns(spot))):
        cols = [p for p in cdef.points if p in panel.columns]
        Xdf = panel[cols].dropna()
        if len(cols) < N_FACTORS + 1 or len(Xdf) < MIN_OBS:
            continue
        fitted, info = _model_curve(Xdf, cdef, model, lookback, mode)
        curves[cdef.cid] = {"def": cdef, "market": Xdf, "fitted": fitted,
                            "resid": Xdf - fitted, "info": info}
    return {"as_of": panel.index[-1], "curves": curves, "source": source,
            "params": {"index": index, "model": model, "lookback": lookback, "mode": mode}}
 
 
def add_z(surface: Dict, z_window: int) -> Dict[str, pd.DataFrame]:
    """{cid: z-score DataFrame} for the surface's residuals."""
    return {cid: rolling_z(c["resid"], z_window) for cid, c in surface["curves"].items()}
 
 
def snapshot(surface: Dict, z: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Latest market / model / residual / z per point, plus history markers.
 
    Columns: cid, start, tenor, label, x, market_bp, model_bp, resid_bp, z,
             z_3D, z_1W, z_1M, resid_3D, resid_1W, resid_1M
    """
    rows = []
    for cid, c in surface["curves"].items():
        cdef = c["def"]
        zc = z[cid]
        for j, p in enumerate(c["market"].columns):
            i = cdef.points.index(p)
            r = {"cid": cid, "start": p[0], "tenor": p[1], "label": cdef.labels[i],
                 "x": cdef.x[i],
                 "market_bp": c["market"][p].iloc[-1], "model_bp": c["fitted"][p].iloc[-1],
                 "resid_bp": c["resid"][p].iloc[-1], "z": zc[p].iloc[-1]}
            for name, lag in HISTORY_MARKERS.items():
                ok = len(zc) > lag
                r[f"z_{name}"] = zc[p].iloc[-1 - lag] if ok else np.nan
                r[f"resid_{name}"] = c["resid"][p].iloc[-1 - lag] if ok else np.nan
            rows.append(r)
    return pd.DataFrame(rows)
 
 
# =============================================================================
# 4. SELF TEST
# =============================================================================
 
def self_test(verbose: bool = True) -> None:
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=(300, 8)), axis=0) @ rng.normal(size=(8, 8))
    # (a) eigen-decomposition matches SVD of the centred data
    fit = pca_fit(X, k=8)
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    assert np.allclose(np.sort(fit["evals"])[::-1], s ** 2 / (len(X) - 1))
    # (b) all PCs reconstruct exactly; 3 PCs leave orthogonal residuals
    assert np.allclose(pca_reconstruct(X, fit), X)
    f3 = pca_fit(X, k=3)
    e = X - pca_reconstruct(X, f3)
    assert np.allclose(e @ f3["V"], 0, atol=1e-8)
    # (c) integrated forward loadings: numeric integral check
    lam, s0, e0 = 2.3, 10.0, 12.0
    t = np.linspace(s0, e0, 20001)
    f = np.column_stack([np.ones_like(t), np.exp(-t / lam), (t / lam) * np.exp(-t / lam)])
    num = np.trapezoid(f, t, axis=0) / (e0 - s0)
    ana = ns_loadings_fwd_avg(np.array([s0]), np.array([e0]), lam)[0]
    assert np.allclose(num, ana, atol=1e-8), (num, ana)
    # (d) geometric forward: spot point returns spot; 1y1y reproduces compounding
    spot = spot_be.get_rates()
    fw = derive_forwards(spot, [(0.0, 5.0), (1.0, 1.0)])
    assert np.allclose(fw[_key(0, 5)], spot[5.0])
    s1, s2 = spot[1.0].iloc[-1] / 100, spot[2.0].iloc[-1] / 100
    assert abs(fw[_key(1, 1)].iloc[-1] / 100 - ((1 + s2) ** 2 / (1 + s1) - 1)) < 1e-12
    # (e) vectorised rolling PCA equals the explicit per-window loop
    W = 120
    fast = rolling_pca_fitted(X, W)
    for tt in (W - 1, 200, len(X) - 1):
        slow = pca_reconstruct(X[tt: tt + 1], pca_fit(X[tt - W + 1: tt + 1]))[0]
        assert np.allclose(fast[tt], slow, atol=1e-6), (tt, fast[tt] - slow)
    # (f) surface runs for both models
    for m in ("pca", "ns"):
        surf = run_surface(model=m)
        assert len(surf["curves"]) == len(GRID_STARTS) + 1, surf["curves"].keys()
    if verbose:
        sp = run_surface(model="pca")["curves"]["Spot"]["info"]
        print("self_test passed")
        print("  spot PCA explained:", [f"{v:.1%}" for v in sp["explained"]])
 
 
if __name__ == "__main__":
    self_test()
