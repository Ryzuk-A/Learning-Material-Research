"""
fly_rv_backend.py
=================
 
EUR Swap Fly Relative-Value Engine (back end)
---------------------------------------------
 
Pure-python calculation layer for a swap-fly RV view. No Dash imports here, so
everything can be unit-tested on its own and called from any front end.
 
What it does
~~~~~~~~~~~~
1.  Builds 1:2:1 spot swap flies from daily par swap rates.
2.  Regresses each fly on a LEVEL factor and a CURVE-ENVIRONMENT factor
    (levels-on-levels OLS), in the style of published rates RV research:
 
        FLY_t = alpha + beta_L * LEVEL_t + beta_C * CURVE_t + e_t       (all bp)
 
3.  Reports static betas (fit over the regression lookback) and rolling betas
    (fit over a separate rolling window), R^2, standard error, residual.
4.  Z-scores the residual, the outright fly level and the curve environment
    over a user-chosen rolling window.
5.  Computes carry and roll per leg off a cubic-spline curve built from the
    par rates, combines them into the fly, and annualises (bp / year).
 
Conventions (READ THESE - they drive every sign in the module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Input rates are in PERCENT (2.45 = 2.45%). Everything reported is in bp.
* Fly          : FLY = 100 * (2*S(B) - S(W1) - S(W2))           [bp]
                 Higher fly = belly yields high vs wings = belly CHEAP.
* Level        : LEVEL = 100 * S(L)       (default L = belly tenor)   [bp]
* Curve        : CURVE = 100 * (S(C2) - S(C1))   (C2 > C1, steepener +) [bp]
* Residual     : e_t = FLY_t - fitted_t
                 e > 0  -> belly cheap vs level & curve -> RECEIVE belly
                 e < 0  -> belly rich  vs level & curve -> PAY belly
  (Matches the RV notes: a fly "trading cheap versus level and curve" has a
  positive residual and the recommendation is to receive the belly.)
* Carry & roll : quoted for the RECEIVE-BELLY position (receive 2 units of
                 belly DV01, pay 1 unit of DV01 on each wing). Positive = the
                 receive-belly fly earns if the curve is unchanged.
 
Carry & roll methodology (per leg, receiver swap, horizon h in years)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Follows the standard "carry is certain, roll is curve-unchanged" framework.
 
  Carry (exact, primary):
        Carry_h = P(0,h) * (S(0,T) - F) * h / A(h,T)
 
      S(0,T)  spot par swap rate, tenor T
      F       current floating fixing (6m EURIBOR) - KNOWN for the first period
      h       horizon (year fraction); (S - F) * h is the net accrual received
      P(0,h)  discount factor to the horizon (turns forward value into PV)
      A(h,T)  PV annuity (DV01 per unit rate) of the residual swap h -> T
      Result is in RATE terms over the horizon. Dividing the PV accrual by the
      annuity expresses the cash as an equivalent yield move on the position.
 
  Carry (approximation, shown only as an audit check):
        Carry_h ~= SR(h,T) - S(0,T)
      Valid only for h <= fixing period. Never use it beyond.
 
  Roll (curve frozen, tenor shortens by h):
        Roll_h = S(0,T) - S(0,T-h)
 
  Annualised   = horizon value / h.             (3m value x 4)
  Fly C&R      = 2 * X(B) - X(W1) - X(W2) for X in {carry, roll}.
      Yield-space terms add without rescaling because the fly is DV01-weighted
      1:2:1, i.e. every leg is quoted per unit of its own DV01.
 
  Horizon rule : h must not exceed the fixing period (6m for 6m EURIBOR) for
                 carry to be "certain". If it does, or for an overnight index
                 (ESTR), the fixing is held unchanged (roll assumption) and
                 the result is flagged in `carry_assumption`.
 
Curve construction (single curve, simplified)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Natural cubic spline on par rates -> annual par grid 1y..maxT.
* Short end: tenors < 1y are simple money-market points P = 1/(1 + r t);
  if none are quoted, the fixing sets P(0, fixing tenor) = 1/(1 + F tau).
* Annual bootstrap (single curve, annual fixed leg):
        P(1) = 1 / (1 + S_1)
        P(n) = (1 - S_n * A_{n-1}) / (1 + S_n),   A_{n-1} = sum_{i<n} P(i)
* Natural cubic spline on continuously-compounded zero rates
  z(t) = -ln P(t) / t  gives P(t) at any t. Flat extrapolation both ends.
* Forward swap rate (annual fixed, schedule rolled back from maturity with a
  short front stub):  SR(s,e) = (P(s) - P(e)) / A(s,e)
* Simplifications: single curve (projection = discounting), year fractions
  from tenors (no calendars or day counts). Fine for relative-value carry;
  swap in a full curve library if exact P&L is needed.
"""
 
from __future__ import annotations
 
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
 
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
 
# =============================================================================
# 0. CONFIGURATION / PLACEHOLDERS
# =============================================================================
 
# Set False once `load_spot_rates` is wired to the real source.
USE_DEMO_DATA: bool = True
 
# Column in the rates DataFrame holding the floating fixing (in %).
FIXING_COLUMN: str = "FIX"
 
# Floating index definitions. `fixing_tenor` (years) is the period over which
# the current fixing is KNOWN - it caps the "certain carry" horizon.
INDEX_CONFIG: Dict[str, Dict] = {
    "EURIBOR6M": {"label": "6M EURIBOR", "fixing_tenor": 0.5},
    "ESTR": {"label": "ESTR", "fixing_tenor": 1.0 / 360.0},
}
DEFAULT_INDEX: str = "EURIBOR6M"
 
# Structure groupings - PLACEHOLDERS, left empty on purpose.
# Fill with fly strings, e.g. "2s3s4s", "5s10s15s", "2s5s10s".
STRUCTURE_GROUPS: Dict[str, List[str]] = {
    "1Y Gap Flies": [],
    "5Y Gap Flies": [],
    "Macro Flies": [],
}
 
# Used ONLY when USE_DEMO_DATA is True, so the view renders standalone.
DEMO_STRUCTURE_GROUPS: Dict[str, List[str]] = {
    "1Y Gap Flies": ["2s3s4s", "4s5s6s", "9s10s11s", "14s15s16s", "19s20s21s"],
    "5Y Gap Flies": ["5s10s15s", "10s15s20s", "15s20s25s", "20s25s30s"],
    "Macro Flies": ["2s5s10s", "2s10s30s", "5s10s30s", "10s20s30s"],
}
 
# Curve-environment mapping for NARROW flies (wing gap <= 1y). A 1y-gap wing
# spread (e.g. 2s4s) has too little variance to explain the fly, so these are
# mapped to a macro curve by belly tenor. Each entry: (max belly, (C1, C2)).
# PLACEHOLDER defaults - edit freely.
NARROW_FLY_CURVE_MAP: List[Tuple[float, Tuple[float, float]]] = [
    (10.0, (2.0, 10.0)),     # belly <= 10y  -> 2s10s
    (20.0, (10.0, 20.0)),    # belly <= 20y  -> 10s20s
    (30.0, (20.0, 30.0)),    # belly <= 30y  -> 20s30s
    (99.0, (30.0, 50.0)),    # beyond        -> 30s50s
]
NARROW_GAP_YEARS: float = 1.0
 
# Per-structure regressor overrides, keyed by fly name. Example:
#   {"5s10s15s": {"level": 10, "curve": (5, 20)}}
REGRESSOR_OVERRIDES: Dict[str, Dict] = {}
 
# Window presets in business days (shared with the front end).
WINDOW_PRESETS: Dict[str, int] = {
    "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756, "5Y": 1260,
}
HORIZON_PRESETS: Dict[str, float] = {"1M": 1.0 / 12.0, "3M": 0.25, "6M": 0.5}
 
 
# =============================================================================
# 1. DATA ACCESS
# =============================================================================
 
def load_spot_rates(index: str = DEFAULT_INDEX) -> pd.DataFrame:
    """PLACEHOLDER - return daily close par swap rates.
 
    Contract
    --------
    index   : DatetimeIndex, business days, ascending.
    columns : numeric tenors in YEARS (floats or strings like "10" / "10Y"),
              e.g. 1, 2, ..., 10, 12, 15, 20, 25, 30, 40, 50. Tenors < 1y
              are optional (treated as simple money-market rates).
              Plus FIXING_COLUMN ("FIX"): the current floating fixing
              (6M EURIBOR for "EURIBOR6M", overnight ESTR for "ESTR").
              If FIX is missing it is implied from the curve (less accurate).
    values  : par rates in PERCENT.
 
    Parameters
    ----------
    index : key of INDEX_CONFIG ("EURIBOR6M" default, "ESTR" optional).
    """
    if USE_DEMO_DATA:
        return make_demo_rates(index=index)
    raise NotImplementedError(
        "load_spot_rates(): connect this to the spot rates source "
        "(see the docstring for the expected DataFrame contract)."
    )
 
 
@lru_cache(maxsize=4)
def get_rates(index: str = DEFAULT_INDEX) -> pd.DataFrame:
    """Cached, cleaned rates for `index`. Clear with get_rates.cache_clear()."""
    return prepare_rates(load_spot_rates(index))
 
 
def prepare_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw rates frame to the module contract.
 
    - tenor column labels -> float years ("10Y", "10", 10 -> 10.0)
    - keeps FIXING_COLUMN as is
    - sorts by date, forward-fills gaps, drops rows with no tenor data
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    rename = {}
    for c in out.columns:
        if c == FIXING_COLUMN:
            continue
        t = _parse_tenor_label(c)
        if t is None:
            raise ValueError(f"Unrecognised rates column: {c!r}")
        rename[c] = t
    out = out.rename(columns=rename)
    tenor_cols = sorted(c for c in out.columns if c != FIXING_COLUMN)
    cols = tenor_cols + ([FIXING_COLUMN] if FIXING_COLUMN in out.columns else [])
    out = out[cols].sort_index().astype(float).ffill()
    return out.dropna(subset=tenor_cols, how="any")
 
 
def _parse_tenor_label(label) -> Optional[float]:
    if isinstance(label, (int, float, np.integer, np.floating)):
        return float(label)
    s = str(label).strip().upper()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([YM]?)", s)
    if not m:
        return None
    v = float(m.group(1))
    return v / 12.0 if m.group(2) == "M" else v
 
 
def tenor_columns(df: pd.DataFrame) -> List[float]:
    """Tenor columns (years) of a prepared rates frame."""
    return [c for c in df.columns if c != FIXING_COLUMN]
 
 
def get_par_rates(df: pd.DataFrame, tenors: Iterable[float]) -> pd.DataFrame:
    """Par rates (in %) at `tenors` for every date.
 
    Quoted tenors are taken as is; others are interpolated across the curve
    each day with a natural cubic spline on the quoted tenors >= 1y
    (vectorised over dates).
    """
    tenors = sorted({float(t) for t in tenors})
    quoted = tenor_columns(df)
    out = pd.DataFrame(index=df.index)
    missing = [t for t in tenors if t not in quoted]
    for t in tenors:
        if t in quoted:
            out[t] = df[t]
    if missing:
        knots = [t for t in quoted if t >= 1.0]
        spline = CubicSpline(knots, df[knots].to_numpy(), axis=1, bc_type="natural")
        vals = spline(np.clip(missing, knots[0], knots[-1]))
        for i, t in enumerate(missing):
            out[t] = vals[:, i]
    return out[tenors]
 
 
def get_fixing(df: pd.DataFrame, index: str = DEFAULT_INDEX) -> pd.Series:
    """Floating fixing in % per date (column FIX, else curve-implied)."""
    if FIXING_COLUMN in df.columns:
        return df[FIXING_COLUMN]
    # Fallback: implied from the curve (approximation).
    tau = INDEX_CONFIG[index]["fixing_tenor"]
    vals = []
    for _, row in df.iterrows():
        c = SwapCurve(row[tenor_columns(df)] / 100.0, fixing=None, fixing_tenor=None)
        p = c.df(tau)
        vals.append(100.0 * (1.0 / p - 1.0) / tau)
    return pd.Series(vals, index=df.index, name=FIXING_COLUMN)
 
 
# =============================================================================
# 2. STRUCTURE DEFINITIONS
# =============================================================================
 
def _fmt_tenor(t: float) -> str:
    return str(int(t)) if float(t).is_integer() else f"{t:g}"
 
 
@dataclass(frozen=True)
class FlySpec:
    """A 1:2:1 fly W1 / B / W2 (tenors in years)."""
    w1: float
    b: float
    w2: float
 
    @property
    def name(self) -> str:
        return f"{_fmt_tenor(self.w1)}s{_fmt_tenor(self.b)}s{_fmt_tenor(self.w2)}s"
 
    @property
    def tenors(self) -> Tuple[float, float, float]:
        return (self.w1, self.b, self.w2)
 
 
def parse_fly(text: str) -> FlySpec:
    """Parse "2s5s10s", "2/5/10", "2y5y10y", "2 5 10" -> FlySpec(2, 5, 10)."""
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(text))]
    if len(nums) != 3:
        raise ValueError(f"Fly needs exactly three tenors: {text!r}")
    w1, b, w2 = nums
    if not (w1 < b < w2):
        raise ValueError(f"Tenors must be increasing (W1 < B < W2): {text!r}")
    return FlySpec(w1, b, w2)
 
 
@dataclass(frozen=True)
class Regressors:
    """Regression factors for one fly."""
    level: float                       # tenor of the level factor
    curve: Tuple[float, float]         # (C1, C2) curve-environment tenors
    source: str = "auto"               # "auto" | "narrow-map" | "override"
 
    @property
    def label(self) -> str:
        c1, c2 = self.curve
        return f"L: {_fmt_tenor(self.level)}y | C: {_fmt_tenor(c1)}s{_fmt_tenor(c2)}s"
 
 
def default_regressors(spec: FlySpec) -> Regressors:
    """Default factors: level = belly; curve = wings, unless the fly is
    narrow (both wing gaps <= NARROW_GAP_YEARS), in which case the curve is
    mapped to a macro curve via NARROW_FLY_CURVE_MAP."""
    narrow = (spec.b - spec.w1) <= NARROW_GAP_YEARS and (spec.w2 - spec.b) <= NARROW_GAP_YEARS
    if narrow:
        for max_belly, curve in NARROW_FLY_CURVE_MAP:
            if spec.b <= max_belly:
                return Regressors(spec.b, curve, "narrow-map")
    return Regressors(spec.b, (spec.w1, spec.w2), "auto")
 
 
def resolve_regressors(spec: FlySpec, overrides: Optional[Dict[str, Dict]] = None) -> Regressors:
    """Default regressors, then REGRESSOR_OVERRIDES, then session `overrides`
    (later wins). An override dict may set "level" and/or "curve"."""
    reg = default_regressors(spec)
    for src in (REGRESSOR_OVERRIDES, overrides or {}):
        ov = src.get(spec.name)
        if ov:
            level = float(ov.get("level", reg.level))
            curve = tuple(float(x) for x in ov.get("curve", reg.curve))
            if curve[0] >= curve[1]:
                raise ValueError(f"Curve tenors must be increasing: {curve}")
            reg = Regressors(level, curve, "override")
    return reg
 
 
def structure_groups() -> Dict[str, List[str]]:
    """Active groupings (demo set when USE_DEMO_DATA)."""
    return DEMO_STRUCTURE_GROUPS if USE_DEMO_DATA else STRUCTURE_GROUPS
 
 
# =============================================================================
# 3. CURVE (cubic-spline single curve) + CARRY & ROLL
# =============================================================================
 
def _schedule(start: float, end: float, freq: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-leg payment times rolled back from `end` in steps of `freq`,
    with a short front stub. Returns (pay_times, accruals)."""
    times = []
    t = end
    while t > start + 1e-9:
        times.append(t)
        t -= freq
    times = np.array(sorted(times))
    accr = np.diff(np.concatenate([[start], times]))
    return times, accr
 
 
class SwapCurve:
    """Single discount/projection curve built from one day of par rates.
 
    Parameters
    ----------
    par          : pd.Series, index = tenor (years), values = par rate (DECIMAL).
    fixing       : current fixing (DECIMAL) or None.
    fixing_tenor : fixing period (years). Used as a short-end point only when
                   no sub-1y tenors are quoted and the period is >= 1m.
    """
 
    def __init__(self, par: pd.Series, fixing: Optional[float], fixing_tenor: Optional[float]):
        par = par.dropna().sort_index()
        mm = par[par.index < 1.0]
        swaps = par[par.index >= 1.0]
        if len(swaps) < 3:
            raise ValueError("Need at least three par tenors >= 1y to build the curve.")
 
        knots_t: List[float] = []
        knots_p: List[float] = []
 
        # Short end: money-market points
        for t, r in mm.items():
            knots_t.append(float(t))
            knots_p.append(1.0 / (1.0 + r * t))
        if mm.empty and fixing is not None and fixing_tenor and fixing_tenor >= 1.0 / 12.0:
            knots_t.append(float(fixing_tenor))
            knots_p.append(1.0 / (1.0 + fixing * fixing_tenor))
 
        # Annual par grid via natural cubic spline, then bootstrap
        n_max = int(math.floor(swaps.index.max()))
        grid = np.arange(1, n_max + 1, dtype=float)
        par_spline = CubicSpline(swaps.index.to_numpy(float), swaps.to_numpy(float), bc_type="natural")
        s_grid = par_spline(np.clip(grid, swaps.index.min(), swaps.index.max()))
        annuity = 0.0
        for n, s in zip(grid, s_grid):
            p = (1.0 - s * annuity) / (1.0 + s)
            if p <= 0:
                raise ValueError(f"Bootstrap produced non-positive DF at {n}y")
            annuity += p
            knots_t.append(n)
            knots_p.append(p)
 
        order = np.argsort(knots_t)
        self._t = np.array(knots_t)[order]
        self._z = -np.log(np.array(knots_p)[order]) / self._t
        self._zspline = CubicSpline(self._t, self._z, bc_type="natural")
        self.par_grid = pd.Series(s_grid, index=grid)
 
    # --- primitives ---------------------------------------------------------
    def zero(self, t: float) -> float:
        """Continuously-compounded zero rate (decimal), flat outside knots."""
        tc = min(max(t, self._t[0]), self._t[-1])
        return float(self._zspline(tc))
 
    def df(self, t: float) -> float:
        """Discount factor P(0, t)."""
        if t <= 0:
            return 1.0
        return math.exp(-self.zero(t) * t)
 
    def annuity(self, start: float, end: float) -> float:
        """PV annuity A(start, end) of an annual fixed leg (per unit rate)."""
        times, accr = _schedule(start, end)
        return float(sum(a * self.df(t) for t, a in zip(times, accr)))
 
    def swap_rate(self, start: float, end: float) -> float:
        """Forward par swap rate SR(start, end) in DECIMAL (single curve)."""
        return (self.df(start) - self.df(end)) / self.annuity(start, end)
 
 
def curve_for_date(df: pd.DataFrame, date, index: str = DEFAULT_INDEX) -> Tuple[SwapCurve, float]:
    """(SwapCurve, fixing in decimal) for one date of a prepared frame."""
    row = df.loc[date]
    tau = INDEX_CONFIG[index]["fixing_tenor"]
    fix = row.get(FIXING_COLUMN, np.nan) if FIXING_COLUMN in df.columns else np.nan
    fix = None if pd.isna(fix) else float(fix) / 100.0
    curve = SwapCurve(row[tenor_columns(df)] / 100.0, fixing=fix, fixing_tenor=tau)
    if fix is None:  # implied fallback
        fix = (1.0 / curve.df(max(tau, 1.0 / 12.0)) - 1.0) / max(tau, 1.0 / 12.0)
    return curve, fix
 
 
def carry_roll_from_inputs(
    s_T: float, s_T_minus_h: float, fixing: float, df_h: float, fwd_annuity: float, h: float
) -> Dict[str, float]:
    """Receiver-swap carry & roll from raw inputs (all rates DECIMAL).
 
        carry = df_h * (s_T - fixing) * h / fwd_annuity     (rate terms)
        roll  = s_T - s_T_minus_h                           (rate terms)
 
    Returns bp over the horizon and bp per annum (/ h).
    """
    carry = df_h * (s_T - fixing) * h / fwd_annuity
    roll = s_T - s_T_minus_h
    return {
        "carry_h_bp": 1e4 * carry,
        "roll_h_bp": 1e4 * roll,
        "carry_ann_bp": 1e4 * carry / h,
        "roll_ann_bp": 1e4 * roll / h,
    }
 
 
def leg_carry_roll(curve: SwapCurve, T: float, h: float, fixing: float) -> Dict[str, float]:
    """Carry & roll of a RECEIVER spot swap of tenor T over horizon h.
 
    Also returns the forward rate SR(h, T) and the approximate carry
    SR(h,T) - S(0,T) as an audit check against the exact formula.
    """
    s_T = curve.swap_rate(0.0, T)
    s_Th = curve.swap_rate(0.0, T - h)
    a_h = curve.annuity(h, T)
    p_h = curve.df(h)
    out = carry_roll_from_inputs(s_T, s_Th, fixing, p_h, a_h, h)
    fwd = curve.swap_rate(h, T)
    out.update({
        "tenor": T,
        "spot_pct": 100 * s_T,
        "spot_rolled_pct": 100 * s_Th,
        "fwd_pct": 100 * fwd,
        "fixing_pct": 100 * fixing,
        "df_h": p_h,
        "fwd_annuity": a_h,
        "carry_approx_h_bp": 1e4 * (fwd - s_T),
    })
    return out
 
 
def fly_carry_roll(
    spec: FlySpec, curve: SwapCurve, fixing: float, h: float, index: str = DEFAULT_INDEX
) -> Dict:
    """Fly carry & roll for the RECEIVE-BELLY 1:2:1 position (bp).
 
    Fly X = 2 * X(B) - X(W1) - X(W2), X in {carry, roll}, horizon and annualised.
    `legs` holds the per-leg breakdown for auditing.
    """
    weights = {spec.w1: -1.0, spec.b: 2.0, spec.w2: -1.0}
    legs = []
    tot = {"carry_h_bp": 0.0, "roll_h_bp": 0.0, "carry_ann_bp": 0.0, "roll_ann_bp": 0.0,
           "carry_approx_h_bp": 0.0}
    for T, w in weights.items():
        leg = leg_carry_roll(curve, T, h, fixing)
        leg["weight"] = w
        legs.append(leg)
        for k in tot:
            tot[k] += w * leg[k]
    tot["cr_ann_bp"] = tot["carry_ann_bp"] + tot["roll_ann_bp"]
    tot["cr_h_bp"] = tot["carry_h_bp"] + tot["roll_h_bp"]
    fix_tenor = INDEX_CONFIG[index]["fixing_tenor"]
    if h <= fix_tenor + 1e-12:
        tot["carry_assumption"] = "certain (horizon within fixing period)"
    else:
        tot["carry_assumption"] = "fixing held unchanged (horizon beyond fixing period)"
    tot["legs"] = legs
    tot["horizon"] = h
    return tot
 
 
# =============================================================================
# 4. REGRESSION, Z-SCORES, SIGNALS
# =============================================================================
 
@dataclass
class RVParams:
    """User-controlled parameters (business days unless stated)."""
    reg_window: int = 126          # static regression lookback
    roll_window: int = 63          # rolling-beta window
    z_window: int = 126            # rolling z-score window
    band: float = 2.0              # +/- sigma band / signal threshold
    resid_mode: str = "static"     # "static" (in-sample, note style) | "rolling"
    horizon: float = 0.25          # carry & roll horizon (years)
    index: str = DEFAULT_INDEX
    as_of: Optional[pd.Timestamp] = None
 
 
def fly_frame(par_pct: pd.DataFrame, spec: FlySpec, reg: Regressors) -> pd.DataFrame:
    """Fly, level and curve series in bp from par rates in %."""
    fly = 100.0 * (2.0 * par_pct[spec.b] - par_pct[spec.w1] - par_pct[spec.w2])
    level = 100.0 * par_pct[reg.level]
    curve = 100.0 * (par_pct[reg.curve[1]] - par_pct[reg.curve[0]])
    return pd.DataFrame({"fly": fly, "level": level, "curve": curve})
 
 
def ols(y: np.ndarray, X: np.ndarray) -> Dict:
    """OLS with intercept. X: (n, k) regressors WITHOUT constant.
 
    Returns alpha, betas, r2, se (residual std, dof-adjusted), n, resid.
    """
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ coef
    resid = y - fitted
    ssr = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "alpha": float(coef[0]),
        "betas": coef[1:].astype(float),
        "r2": 1.0 - ssr / sst if sst > 0 else np.nan,
        "se": math.sqrt(ssr / (n - k - 1)) if n > k + 1 else np.nan,
        "n": n,
        "resid": resid,
    }
 
 
def rolling_ols2(y: pd.Series, x1: pd.Series, x2: pd.Series, window: int) -> pd.DataFrame:
    """Rolling two-factor OLS (closed-form normal equations, vectorised).
 
    For each date t, fit y = a + b1 x1 + b2 x2 on the trailing `window` obs
    (inclusive of t). Returns alpha, beta_level, beta_curve, r2, resid_roll
    (y_t - fitted_t using the betas estimated through t).
    """
    r = lambda s: s.rolling(window, min_periods=window).mean()  # noqa: E731
    my, m1, m2 = r(y), r(x1), r(x2)
    s11 = r(x1 * x1) - m1 * m1
    s22 = r(x2 * x2) - m2 * m2
    s12 = r(x1 * x2) - m1 * m2
    s1y = r(x1 * y) - m1 * my
    s2y = r(x2 * y) - m2 * my
    syy = r(y * y) - my * my
    det = s11 * s22 - s12 * s12
    det = det.where(det.abs() > 1e-12)
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    a = my - b1 * m1 - b2 * m2
    r2 = (b1 * s1y + b2 * s2y) / syy.where(syy > 1e-12)
    resid = y - (a + b1 * x1 + b2 * x2)
    return pd.DataFrame({"alpha": a, "beta_level": b1, "beta_curve": b2,
                         "r2_roll": r2, "resid_roll": resid})
 
 
def rolling_z(s: pd.Series, window: int) -> pd.Series:
    """(x_t - mean) / std over the trailing `window` obs (sample std)."""
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - m) / sd.where(sd > 1e-12)
 
 
def half_life(resid: np.ndarray) -> float:
    """Mean-reversion half-life (obs) from AR(1): e_t = c + phi e_{t-1}.
    inf if phi outside (0, 1)."""
    e = np.asarray(resid, dtype=float)
    e = e[~np.isnan(e)]
    if len(e) < 10:
        return np.nan
    fit = ols(e[1:], e[:-1].reshape(-1, 1))
    phi = fit["betas"][0]
    return -math.log(2.0) / math.log(phi) if 0 < phi < 1 else float("inf")
 
 
def classify(z_resid: float, z_fly: float, band: float) -> Tuple[str, str]:
    """(signal, agreement).
 
    signal    : RECEIVE (z_resid >= band), PAY (<= -band), else NEUTRAL.
    agreement : STRONG  - same sign, both beyond band
                AGREE   - same sign
                DIVERGE - opposite signs
    """
    if np.isnan(z_resid):
        return "N/A", "N/A"
    signal = "RECEIVE" if z_resid >= band else ("PAY" if z_resid <= -band else "NEUTRAL")
    if np.isnan(z_fly):
        return signal, "N/A"
    if np.sign(z_resid) == np.sign(z_fly):
        agree = "STRONG" if min(abs(z_resid), abs(z_fly)) >= band else "AGREE"
    else:
        agree = "DIVERGE"
    return signal, agree
 
 
def analyse_fly(
    df: pd.DataFrame,
    spec: FlySpec,
    params: RVParams,
    overrides: Optional[Dict[str, Dict]] = None,
    par_pct: Optional[pd.DataFrame] = None,
    curve_bundle: Optional[Tuple[SwapCurve, float]] = None,
) -> Dict:
    """Full RV analysis for one fly.
 
    Parameters
    ----------
    df           : prepared rates frame (see load_spot_rates contract).
    spec         : the fly.
    params       : RVParams.
    overrides    : session regressor overrides {fly name: {"level", "curve"}}.
    par_pct      : optional pre-computed par rates (speeds up batch runs).
    curve_bundle : optional pre-built (SwapCurve, fixing) for the as-of date.
 
    Returns
    -------
    dict with
      "spec", "reg"      : FlySpec, Regressors
      "series"           : DataFrame (date index) - fly, level, curve, fitted,
                           resid_static, resid_roll, resid, z_resid, z_fly,
                           z_curve, z_level, alpha, beta_level, beta_curve, r2_roll
      "stats"            : dict of scalar outputs as of the last date
      "reg_window_dates" : (start, end) of the static regression sample
      "carry_roll"       : fly_carry_roll output
    """
    if params.as_of is not None:
        df = df.loc[: pd.Timestamp(params.as_of)]
    reg = resolve_regressors(spec, overrides)
    need = set(spec.tenors) | {reg.level, *reg.curve}
    if par_pct is None:
        par_pct = get_par_rates(df, need)
    else:
        par_pct = par_pct.loc[df.index[0]: df.index[-1], sorted(need)]
    fr = fly_frame(par_pct, spec, reg)
 
    # ---- static regression over the lookback ---------------------------------
    sample = fr.iloc[-params.reg_window:].dropna()
    fit = ols(sample["fly"].to_numpy(), sample[["level", "curve"]].to_numpy())
    b_l, b_c = fit["betas"]
    fr["fitted"] = fit["alpha"] + b_l * fr["level"] + b_c * fr["curve"]
    fr["resid_static"] = fr["fly"] - fr["fitted"]
 
    # ---- rolling regression --------------------------------------------------
    roll = rolling_ols2(fr["fly"], fr["level"], fr["curve"], params.roll_window)
    fr = fr.join(roll)
 
    fr["resid"] = fr["resid_roll"] if params.resid_mode == "rolling" else fr["resid_static"]
    fr["z_resid"] = rolling_z(fr["resid"], params.z_window)
    fr["z_fly"] = rolling_z(fr["fly"], params.z_window)
    fr["z_curve"] = rolling_z(fr["curve"], params.z_window)
    fr["z_level"] = rolling_z(fr["level"], params.z_window)
 
    last = fr.iloc[-1]
    resid_now = float(last["resid"])
    signal, agree = classify(float(last["z_resid"]), float(last["z_fly"]), params.band)
 
    # ---- carry & roll --------------------------------------------------------
    if curve_bundle is None:
        curve_bundle = curve_for_date(df, df.index[-1], params.index)
    cr = fly_carry_roll(spec, curve_bundle[0], curve_bundle[1], params.horizon, params.index)
    direction = 1.0 if signal == "RECEIVE" else (-1.0 if signal == "PAY" else np.nan)
 
    stats = {
        "name": spec.name,
        "regressors": reg.label,
        "reg_source": reg.source,
        "resid_mode": params.resid_mode,
        "as_of": fr.index[-1],
        "fly_bp": float(last["fly"]),
        "level_bp": float(last["level"]),
        "curve_bp": float(last["curve"]),
        "fitted_bp": float(last["fitted"]),
        "resid_bp": resid_now,
        "resid_over_se": float(last["resid_static"]) / fit["se"] if fit["se"] else np.nan,
        "z_resid": float(last["z_resid"]),
        "z_fly": float(last["z_fly"]),
        "z_curve": float(last["z_curve"]),
        "z_level": float(last["z_level"]),
        "alpha": fit["alpha"],
        "beta_level": float(b_l),
        "beta_curve": float(b_c),
        "r2": fit["r2"],
        "se_bp": fit["se"],
        "n_obs": fit["n"],
        "beta_level_roll": float(last["beta_level"]),
        "beta_curve_roll": float(last["beta_curve"]),
        "r2_roll": float(last["r2_roll"]),
        "half_life_d": half_life(sample["fly"].to_numpy() - (fit["alpha"] + sample[["level", "curve"]].to_numpy() @ fit["betas"])),
        "signal": signal,
        "agreement": agree,
        "carry_ann_bp": cr["carry_ann_bp"],
        "roll_ann_bp": cr["roll_ann_bp"],
        "cr_ann_bp": cr["cr_ann_bp"],
        "cr_signal_dir_bp": direction * cr["cr_ann_bp"],
        "carry_assumption": cr["carry_assumption"],
    }
    return {
        "spec": spec,
        "reg": reg,
        "series": fr,
        "stats": stats,
        "reg_window_dates": (sample.index[0], sample.index[-1]),
        "carry_roll": cr,
    }
 
 
def summarise(
    df: pd.DataFrame,
    flies: Sequence[str],
    params: RVParams,
    overrides: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """One row of headline stats per fly (for the summary table).
 
    Builds par rates and the as-of curve once and reuses them across flies.
    Flies that fail (bad tenor, too little history) get an "error" entry.
    """
    if params.as_of is not None:
        df = df.loc[: pd.Timestamp(params.as_of)]
    specs, rows = [], []
    for f in flies:
        try:
            specs.append(parse_fly(f))
        except ValueError as exc:
            rows.append({"name": f, "error": str(exc)})
    if not specs:
        return pd.DataFrame(rows)
    need = set()
    for s in specs:
        r = resolve_regressors(s, overrides)
        need |= set(s.tenors) | {r.level, *r.curve}
    par_pct = get_par_rates(df, need)
    bundle = curve_for_date(df, df.index[-1], params.index)
    p = RVParams(**{**params.__dict__, "as_of": None})
    for s in specs:
        try:
            rows.append(analyse_fly(df, s, p, overrides, par_pct, bundle)["stats"])
        except Exception as exc:  # keep the table alive
            rows.append({"name": s.name, "error": str(exc)})
    return pd.DataFrame(rows)
 
 
# =============================================================================
# 5. DEMO DATA (synthetic, for standalone testing only)
# =============================================================================
 
def make_demo_rates(index: str = DEFAULT_INDEX, years: int = 6, seed: int = 7) -> pd.DataFrame:
    """Synthetic EUR-like par curve: Nelson-Siegel factors following random
    walks plus mean-reverting tenor noise (so flies have RV to find)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-10-23"), periods=252 * years)
    n = len(dates)
    tenors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30, 40, 50], float)
    def ou(mean, kappa, vol):
        x = np.empty(n)
        x[0] = mean
        for i in range(1, n):
            x[i] = x[i - 1] + kappa * (mean - x[i - 1]) + rng.normal(0, vol)
        return x
 
    lvl = ou(2.7, 0.004, 0.035)
    slope = ou(-0.8, 0.006, 0.025)
    curv = ou(0.6, 0.006, 0.03)
    lam = 2.2
    x = tenors / lam
    f1 = (1 - np.exp(-x)) / x
    f2 = f1 - np.exp(-x)
    base = lvl[:, None] + slope[:, None] * f1 + curv[:, None] * f2
    # long-end hump down (EUR-style)
    base -= 0.012 * np.maximum(tenors - 20, 0)[None, :]
    noise = np.zeros((n, len(tenors)))
    phi = 0.97
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, 0.012, len(tenors))
    rates = base + noise
    out = pd.DataFrame(rates, index=dates, columns=tenors)
    short = lvl + slope + rng.normal(0, 0.01, n)
    out[FIXING_COLUMN] = short - (0.0 if index == "EURIBOR6M" else 0.15)
    return out.round(5)
 
 
# =============================================================================
# 6. SELF TEST
# =============================================================================
 
def self_test(verbose: bool = True) -> None:
    """Sanity checks on the maths. Raises AssertionError on failure."""
    # (a) Reproduce the published numeric example: 5y receiver, 6m horizon
    #     S(0,5)=1.023%, S(0,4.5)=0.928%, P(0,6m)=0.9995, dv01=4.45, F=0.319%
    #     -> carry 7.9bp, roll 9.5bp over 6m.
    ex = carry_roll_from_inputs(0.01023, 0.00928, 0.00319, 0.9995, 4.45, 0.5)
    assert abs(ex["carry_h_bp"] - 7.9) < 0.05, ex
    assert abs(ex["roll_h_bp"] - 9.5) < 1e-9, ex
    assert abs(ex["carry_ann_bp"] - 2 * ex["carry_h_bp"]) < 1e-12
 
    # (b) Curve reprices its own quoted par tenors (>= 1y)
    df = prepare_rates(make_demo_rates())
    d = df.index[-1]
    curve, fix = curve_for_date(df, d)
    for t in tenor_columns(df):
        if float(t).is_integer() and t >= 1:
            err = 1e4 * (curve.swap_rate(0, t) - df.loc[d, t] / 100)
            assert abs(err) < 1e-6, (t, err)
 
    # (c) Exact carry vs approximation agree closely for h <= fixing period
    leg = leg_carry_roll(curve, 10.0, 0.25, fix)
    assert abs(leg["carry_h_bp"] - leg["carry_approx_h_bp"]) < 0.5, leg
 
    # (d) Rolling OLS at the last date equals a direct OLS on the same window
    spec = parse_fly("2s5s10s")
    p = RVParams(reg_window=126, roll_window=126)
    res = analyse_fly(df, spec, p)
    s = res["stats"]
    assert abs(s["beta_level"] - s["beta_level_roll"]) < 1e-6
    assert abs(s["beta_curve"] - s["beta_curve_roll"]) < 1e-6
    assert abs(s["r2"] - s["r2_roll"]) < 1e-6
 
    # (e) Static z with z_window = reg_window equals resid / sample std
    p2 = RVParams(reg_window=126, z_window=126)
    s2 = analyse_fly(df, spec, p2)["stats"]
    se_ratio = s2["resid_bp"] / (s2["se_bp"] * math.sqrt((126 - 3) / 125))
    assert abs(s2["z_resid"] - se_ratio) < 1e-6, (s2["z_resid"], se_ratio)
 
    if verbose:
        print("self_test passed")
        print(f"  example carry {ex['carry_h_bp']:.2f}bp / roll {ex['roll_h_bp']:.2f}bp (6m)")
        print(f"  10y leg 3m: exact carry {leg['carry_h_bp']:.3f}bp vs approx {leg['carry_approx_h_bp']:.3f}bp")
 
 
if __name__ == "__main__":
    self_test()
