"""
Safe evaluation helpers for the calculation notebook tab.
Supports scalars, vectors (1-D arrays), matrices, engineering functions, and unit constants.
"""

from __future__ import annotations

import ast
import cmath  # noqa: F401 – available in notebook context
import math
import re
from typing import Any

import numpy as np

from .formula_parser import ALLOWED_CONSTANTS, ALLOWED_FUNCTIONS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arr(v):
    return np.asarray(v, dtype=float)


class _CallableFloat(float):
    """Float-like unit value that can also behave like a callable helper."""

    def __new__(cls, value: float, fn):
        obj = float.__new__(cls, value)
        obj._fn = fn
        return obj

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


# ── Engineering functions ─────────────────────────────────────────────────────

def _cfl_dt(dx, u, cfl=1.0):
    """CFL time-step: cfl * dx / |u|."""
    return float(cfl * abs(dx) / (abs(u) + 1e-300))

def _diffusion_dt(dx, D, f=0.5):
    """Explicit diffusion time step: f * dx² / D."""
    return float(f * dx**2 / (abs(D) + 1e-300))

def _fourier_number(D, dt, dx):
    """Fourier number: D * dt / dx²."""
    return float(D * dt / dx**2)

def _peclet(u, L, D):
    """Péclet number: |u| * |L| / D."""
    return float(abs(u) * abs(L) / (abs(D) + 1e-300))

def _cell_size(L, N):
    """Grid cell size: |L| / N."""
    return float(abs(L) / max(1, int(N)))

def _domain_points(L, dx):
    """Grid point count: round(|L| / dx) + 1."""
    return int(round(abs(L) / abs(dx))) + 1

def _von_mises(s11, s22, s33, s12=0.0, s23=0.0, s13=0.0):
    """Von Mises equivalent stress."""
    return float(np.sqrt(0.5 * (
        (s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
        + 6 * (s12**2 + s23**2 + s13**2)
    )))

def _shear_modulus(E, nu):
    """Shear modulus G = E / (2*(1+nu))."""
    return float(E / (2 * (1 + nu)))

def _lame_lambda(E, nu):
    """First Lamé parameter λ = E*nu / ((1+nu)*(1-2nu))."""
    return float(E * nu / ((1 + nu) * (1 - 2 * nu)))

def _hooke_1d(E, eps):
    """1-D Hooke's law: σ = E * ε."""
    return float(E * eps)

def _hooke_3d(E, nu, exx, eyy, ezz, exy=0.0, eyz=0.0, exz=0.0):
    """3-D isotropic Hooke → stress vector [sxx, syy, szz, sxy, syz, sxz]."""
    lam = _lame_lambda(E, nu)
    G = _shear_modulus(E, nu)
    tr = exx + eyy + ezz
    return np.array([lam*tr + 2*G*exx, lam*tr + 2*G*eyy, lam*tr + 2*G*ezz,
                     2*G*exy, 2*G*eyz, 2*G*exz])

def _plane_strain(E, nu, exx, eyy, exy=0.0):
    """Plane-strain stress state → [sxx, syy, txy]."""
    lam = _lame_lambda(E, nu)
    G = _shear_modulus(E, nu)
    return np.array([(lam + 2*G)*exx + lam*eyy,
                     lam*exx + (lam + 2*G)*eyy,
                     2*G*exy])

def _plane_stress(E, nu, exx, eyy, exy=0.0):
    """Plane-stress state → [sxx, syy, txy]."""
    C = E / (1 - nu**2)
    G = E / (2 * (1 + nu))
    return np.array([C*(exx + nu*eyy), C*(eyy + nu*exx), 2*G*exy])

def _plane_stress_ezz(nu, exx, eyy):
    """Out-of-plane strain in plane stress: -nu/(1-nu)*(exx+eyy)."""
    return float(-nu / (1 - nu) * (exx + eyy))


# ── Vector functions ──────────────────────────────────────────────────────────

def _dot(a, b):
    return float(np.dot(_arr(a).ravel(), _arr(b).ravel()))

def _cross(a, b):
    return np.cross(_arr(a), _arr(b))

def _norm(v):
    return float(np.linalg.norm(_arr(v)))

def _normalize(v):
    a = _arr(v)
    return a / (np.linalg.norm(a) + 1e-300)

def _length(v):
    return float(np.linalg.norm(_arr(v)))


# ── Matrix functions ──────────────────────────────────────────────────────────

def _det(M):
    return float(np.linalg.det(_arr(M)))

def _inv(M):
    return np.linalg.inv(_arr(M))

def _transpose(M):
    return _arr(M).T

def _trace(M):
    return float(np.trace(_arr(M)))

def _rank(M):
    return int(np.linalg.matrix_rank(_arr(M)))

def _eig(M):
    vals, _ = np.linalg.eig(_arr(M))
    result = np.real(vals)
    return float(result[0]) if result.size == 1 else result


# ── Builder functions ─────────────────────────────────────────────────────────

def _eye(n):
    return np.eye(int(n))

def _zeros(m, n=None):
    return np.zeros(int(m)) if n is None else np.zeros((int(m), int(n)))

def _ones(m, n=None):
    return np.ones(int(m)) if n is None else np.ones((int(m), int(n)))

def _shape(M):
    return list(np.asarray(M).shape)


# ── Math helpers ──────────────────────────────────────────────────────────────

def _sign(x): return float(np.sign(x))
def _log2(x): return float(math.log2(x))
def _degrees(x): return float(math.degrees(x))
def _radians(x): return float(math.radians(x))
def _clamp(x, lo, hi): return float(max(lo, min(hi, x)))
def _lerp(a, b, t): return float(a + t * (b - a))
def _factorial(n): return float(math.factorial(int(abs(n))))
def _gcd(a, b): return float(math.gcd(int(abs(a)), int(abs(b))))
def _lcm(a, b):
    g = math.gcd(int(abs(a)), int(abs(b)))
    return 0.0 if g == 0 else float(abs(int(a)) // g * abs(int(b)))


# ── Statistics ────────────────────────────────────────────────────────────────

def _sum(v): return float(np.sum(_arr(v)))
def _mean(v): return float(np.mean(_arr(v)))
def _std(v): return float(np.std(_arr(v)))
def _variance(v): return float(np.var(_arr(v)))
def _median(v): return float(np.median(_arr(v)))
def _prod(v): return float(np.prod(_arr(v)))
def _cumsum(v): return np.cumsum(_arr(v))
def _diff(v): return np.diff(_arr(v))
def _min_arr(v): return float(np.min(_arr(v)))
def _max_arr(v): return float(np.max(_arr(v)))


# ── Array builders ────────────────────────────────────────────────────────────

def _linspace(start, stop, num=50): return np.linspace(float(start), float(stop), int(num))
def _arange(start, stop=None, step=1.0):
    if stop is None: return np.arange(float(start))
    return np.arange(float(start), float(stop), float(step))
def _diag(v): return np.diag(_arr(v))
def _flatten(M): return np.asarray(M, dtype=float).ravel()
def _outer(a, b): return np.outer(_arr(a), _arr(b))


# ── Linear algebra extras ──────────────────────────────────────────────────────

def _solve(A, b): return np.linalg.solve(_arr(A), _arr(b))
def _lstsq(A, b):
    result, *_ = np.linalg.lstsq(_arr(A), _arr(b), rcond=None)
    return result
def _svd(M):
    U, s, Vt = np.linalg.svd(_arr(M))
    return s  # return singular values
def _norm_p(v, p=2):
    p = float(p)
    return float(np.linalg.norm(_arr(v).ravel(), ord=p))


# ── Dimensionless numbers ──────────────────────────────────────────────────────

def _reynolds(rho, u, L, mu):
    return float(abs(rho * u * L) / (abs(mu) + 1e-300))
def _mach(u, a):
    return float(abs(u) / (abs(a) + 1e-300))
def _prandtl(mu, cp, k):
    return float(abs(mu * cp) / (abs(k) + 1e-300))
def _nusselt_dittus(Re, Pr, heating=True):
    n = 0.4 if heating else 0.3
    return float(0.023 * abs(Re) ** 0.8 * abs(Pr) ** n)


# ── Special math ──────────────────────────────────────────────────────────────

def _erf(x):
    if np.isscalar(x):
        return math.erf(float(x))
    a = np.asarray(x)
    return np.array([math.erf(float(v)) for v in a.ravel()]).reshape(a.shape)

def _erfc(x):
    if np.isscalar(x):
        return math.erfc(float(x))
    a = np.asarray(x)
    return np.array([math.erfc(float(v)) for v in a.ravel()]).reshape(a.shape)

def _gamma_fn(x): return math.gamma(float(x))
def _lgamma_fn(x): return math.lgamma(float(x))
def _beta_fn(a, b): return math.exp(math.lgamma(float(a)) + math.lgamma(float(b)) - math.lgamma(float(a) + float(b)))


# ── Sorting ───────────────────────────────────────────────────────────────────

def _sort(v, reverse=False):
    a = _arr(v)
    return a[::-1] if reverse else np.sort(a)

def _argsort(v): return np.argsort(_arr(v)).tolist()
def _unique(v): return np.unique(_arr(v))
def _flip(v): return np.flip(_arr(v))


# ── Array ops ─────────────────────────────────────────────────────────────────

def _where(condition, x, y): return np.where(np.asarray(condition, dtype=bool), x, y)
def _reshape(v, *shape): return np.asarray(v).reshape(shape)
def _concat(*arrays): return np.concatenate([np.asarray(a) for a in arrays])
def _vstack(*arrays): return np.vstack([np.asarray(a) for a in arrays])
def _hstack(*arrays): return np.hstack([np.asarray(a) for a in arrays])
def _append_arr(a, b): return np.append(np.asarray(a), np.asarray(b))
def _clip(v, lo, hi): return np.clip(np.asarray(v, dtype=float), float(lo), float(hi))
def _roll(v, shift): return np.roll(_arr(v), int(shift))
def _repeat(v, n): return np.repeat(_arr(v), int(n))
def _tile(v, n): return np.tile(_arr(v), int(n))
def _full_arr(n, val): return np.full(int(n), float(val))
def _meshgrid(x, y): return np.meshgrid(_arr(x), _arr(y))


# ── Interpolation ─────────────────────────────────────────────────────────────

def _interp(x, xp, fp): return np.interp(np.asarray(x, dtype=float), _arr(xp), _arr(fp))
def _polyfit(x, y, deg): return np.polyfit(_arr(x), _arr(y), int(deg))
def _polyval(p, x): return np.polyval(_arr(p), np.asarray(x, dtype=float))


# ── Calculus / Signal ─────────────────────────────────────────────────────────

def _gradient_1d(f, dx=1.0): return np.gradient(_arr(f), float(dx))

def _trapz(y, x=None, dx=1.0):
    y_arr = _arr(y)
    _trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    if x is not None:
        return float(_trapz_fn(y_arr, _arr(x)))
    return float(_trapz_fn(y_arr, dx=float(dx)))

def _cumtrapz(y, dx=1.0): return np.cumsum(_arr(y)) * float(dx)

def _diff2(v, n=1):
    a = _arr(v)
    for _ in range(int(n)):
        a = np.diff(a)
    return a


# ── FFT ───────────────────────────────────────────────────────────────────────

def _fft(v):
    result = np.fft.fft(_arr(v))
    return np.abs(result)

def _fft_complex(v): return np.fft.fft(_arr(v))
def _ifft(v): return np.real(np.fft.ifft(np.asarray(v)))
def _fftfreq(n, dt=1.0): return np.fft.fftfreq(int(n), float(dt))
def _fftshift(v): return np.fft.fftshift(np.asarray(v))
def _real(v): return np.real(np.asarray(v))
def _imag(v): return np.imag(np.asarray(v))
def _angle(v): return np.angle(np.asarray(v))


# ── More statistics ───────────────────────────────────────────────────────────

def _percentile(v, p): return float(np.percentile(_arr(v), float(p)))
def _quantile(v, q): return float(np.quantile(_arr(v), float(q)))
def _corrcoef(x, y): return float(np.corrcoef(_arr(x), _arr(y))[0, 1])
def _cov_fn(x, y): return float(np.cov(_arr(x), _arr(y))[0, 1])

def _histogram_counts(v, bins=10):
    counts, _ = np.histogram(_arr(v), bins=int(bins))
    return counts

def _mode_fn(v):
    a = _arr(v)
    vals, counts = np.unique(a, return_counts=True)
    return float(vals[np.argmax(counts)])

def _zscore(v):
    a = _arr(v)
    s = float(np.std(a))
    return (a - float(np.mean(a))) / (s if s > 0 else 1.0)

def _normalize_vec(v):
    a = _arr(v)
    lo, hi = float(np.min(a)), float(np.max(a))
    return (a - lo) / (hi - lo) if hi > lo else a - lo


# ── More linear algebra ───────────────────────────────────────────────────────

def _pinv(M): return np.linalg.pinv(_arr(M))
def _cond(M): return float(np.linalg.cond(_arr(M)))
def _matrix_power(M, n): return np.linalg.matrix_power(_arr(M), int(n))
def _kron(A, B): return np.kron(_arr(A), _arr(B))

def _cholesky(M):
    try:
        return np.linalg.cholesky(_arr(M))
    except np.linalg.LinAlgError as e:
        raise ValueError(f"cholesky: {e}") from e

def _qr(M):
    Q, R = np.linalg.qr(_arr(M))
    return [Q, R]

def _eig_full(M):
    vals, _ = np.linalg.eig(_arr(M))
    return np.real(vals)


# ── More dimensionless numbers ────────────────────────────────────────────────

def _biot(h, L, k): return float(abs(h) * abs(L) / (abs(k) + 1e-300))
def _fourier_thermal(alpha, t, L): return float(abs(alpha) * abs(t) / (abs(L) ** 2 + 1e-300))
def _lewis(D, alpha): return float(abs(D) / (abs(alpha) + 1e-300))
def _weber(rho, v, L, sigma): return float(abs(rho) * v ** 2 * abs(L) / (abs(sigma) + 1e-300))
def _stokes(rho, mu, g_acc, d): return float(abs(rho) * abs(g_acc) * d ** 2 / (18 * abs(mu) + 1e-300))
def _grashof(g_acc, beta, dT, L, nu): return float(abs(g_acc) * abs(beta) * abs(dT) * abs(L) ** 3 / (nu ** 2 + 1e-300))


# ── Alloy composition conversions ─────────────────────────────────────────────

def _wt_to_mol(wt, M):
    """Convert weight fractions / wt% to mole fractions.

    wt : array of weight percents or fractions (need not sum to 100/1).
    M  : array of molar masses [g/mol], same length as wt.
    Returns mole-fraction array (sums to 1).
    """
    wt_arr = _arr(wt)
    M_arr  = _arr(M)
    n = wt_arr / M_arr
    return n / np.sum(n)

def _wt_to_mol2(wt_B, M_A, M_B):
    """Binary alloy scan: wt% of B (scalar or array) → mole fraction of B.

    wt_B : wt% of component B; balance (100 - wt_B) is component A.
    M_A  : molar mass of A [g/mol].
    M_B  : molar mass of B [g/mol].
    Works element-wise on arrays, e.g. wt_B = linspace(0, 100, 200).
    """
    wt_B_arr = _arr(wt_B)
    n_B = wt_B_arr / float(M_B)
    n_A = (100.0 - wt_B_arr) / float(M_A)
    return n_B / (n_A + n_B)


def _mol_to_wt(x, M):
    """Convert mole fractions to weight percent.

    x : array of mole fractions (need not sum to 1).
    M : array of molar masses [g/mol], same length as x.
    Returns weight-percent array (sums to 100).
    """
    x_arr = _arr(x)
    M_arr = _arr(M)
    w = x_arr * M_arr
    return w / np.sum(w) * 100.0


# ── Physical constants ────────────────────────────────────────────────────────

PHYSICAL_CONSTANTS: dict[str, float] = {
    "g_n":      9.80665,           # standard gravity m/s²
    "c_0":      299792458.0,       # speed of light m/s
    "R_gas":    8.314462618,       # universal gas constant J/(mol·K)
    "k_B":      1.380649e-23,      # Boltzmann constant J/K
    "N_A":      6.02214076e23,     # Avogadro number 1/mol
    "h_p":      6.62607015e-34,    # Planck constant J·s
    "sigma_SB": 5.670374419e-8,    # Stefan-Boltzmann W/(m²·K⁴)
    "mu_0":     1.25663706212e-6,  # vacuum permeability H/m
    "eps_0":    8.8541878128e-12,  # vacuum permittivity F/m
    "atm":      101325.0,          # standard atmosphere Pa
}


# ── Unit constants ────────────────────────────────────────────────────────────

UNIT_CONSTANTS: dict[str, float] = {
    "nm": 1e-9, "um": 1e-6, "us": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3,
    "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
    "sec": 1.0, "ms": 1e-3, "minute": 60.0, "min_unit": 60.0, "h": 3600.0,
    "hour": 3600.0, "day": 86400.0, "week": 604800.0, "month": 2_592_000.0,
    "kg": 1.0, "N": 1.0, "kN": 1e3, "MN": 1e6,
    "J": 1.0, "kJ": 1e3, "MJ": 1e6,
    "W": 1.0, "kW": 1e3, "MW": 1e6,
    "Hz": 1.0, "kHz": 1e3, "MHz": 1e6,
    # Pressure extras
    "bar": 1e5, "mbar": 100.0,
    "atm_unit": 101325.0,
    # Energy
    "kWh": 3.6e6,
    # Volume
    "L": 1e-3,
}


# ── Allowed functions registry ────────────────────────────────────────────────

_NOTEBOOK_FUNCTION_EXCLUSIONS = {"abs", "min", "max"}

ALLOWED_NOTEBOOK_FUNCTIONS: dict[str, Any] = {
    **{name: fn for name, fn in ALLOWED_FUNCTIONS.items() if name not in _NOTEBOOK_FUNCTION_EXCLUSIONS},
    "floor": math.floor, "ceil": math.ceil,
    # Math helpers
    "sign": _sign, "log2": _log2, "degrees": _degrees, "radians": _radians,
    "clamp": _clamp, "lerp": _lerp, "factorial": _factorial,
    "gcd": _gcd, "lcm": _lcm,
    # Engineering
    "cfl_dt": _cfl_dt, "diffusion_dt": _diffusion_dt,
    "fourier_number": _fourier_number, "peclet": _peclet,
    "cell_size": _cell_size, "domain_points": _domain_points,
    "von_mises": _von_mises, "shear_modulus": _shear_modulus,
    "lame_lambda": _lame_lambda, "hooke_1d": _hooke_1d,
    "hooke_3d": _hooke_3d, "plane_strain": _plane_strain,
    "plane_stress": _plane_stress, "plane_stress_ezz": _plane_stress_ezz,
    # Dimensionless numbers
    "reynolds": _reynolds, "mach": _mach, "prandtl": _prandtl,
    "nusselt_dittus": _nusselt_dittus,
    # Statistics
    "mean": _mean, "std": _std, "variance": _variance,
    "median": _median, "prod": _prod, "cumsum": _cumsum, "diff": _diff,
    # Array builders
    "linspace": _linspace, "arange": _arange, "diag": _diag,
    "flatten": _flatten, "outer": _outer,
    # Linear algebra extras
    "solve": _solve, "lstsq": _lstsq, "svd": _svd, "norm_p": _norm_p,
    # Vector
    "dot": _dot, "cross": _cross, "norm": _norm,
    "normalize": _normalize, "length": _length,
    # Matrix
    "det": _det, "inv": _inv, "transpose": _transpose,
    "trace": _trace, "rank": _rank, "eig": _eig, "eig_full": _eig_full,
    # Builders
    "eye": _eye, "zeros": _zeros, "ones": _ones, "shape": _shape,
    # Special math
    "erf": _erf, "erfc": _erfc, "gamma": _gamma_fn, "lgamma": _lgamma_fn, "beta": _beta_fn,
    # Sorting
    "sort": _sort, "argsort": _argsort, "unique": _unique, "flip": _flip,
    # Array ops
    "where": _where, "reshape": _reshape, "concat": _concat,
    "vstack": _vstack, "hstack": _hstack, "append": _append_arr,
    "clip": _clip, "roll": _roll, "repeat": _repeat, "tile": _tile,
    "full": _full_arr, "meshgrid": _meshgrid,
    # Interpolation
    "interp": _interp, "polyfit": _polyfit, "polyval": _polyval,
    # Calculus
    "gradient": _gradient_1d, "trapz": _trapz, "cumtrapz": _cumtrapz, "diff2": _diff2,
    # FFT
    "fft": _fft, "ifft": _ifft, "fftfreq": _fftfreq, "fftshift": _fftshift,
    "real": _real, "imag": _imag, "angle": _angle,
    # More statistics
    "percentile": _percentile, "quantile": _quantile,
    "corrcoef": _corrcoef, "cov": _cov_fn,
    "histogram": _histogram_counts, "mode": _mode_fn,
    "zscore": _zscore, "norm01": _normalize_vec,
    # More linalg
    "pinv": _pinv, "cond": _cond, "matrix_power": _matrix_power,
    "kron": _kron, "cholesky": _cholesky, "qr": _qr,
    # More dimensionless
    "biot": _biot, "fourier_thermal": _fourier_thermal,
    "lewis": _lewis, "weber": _weber, "stokes": _stokes, "grashof": _grashof,
    # Alloy composition
    "wt_to_mol": _wt_to_mol, "mol_to_wt": _mol_to_wt, "wt_to_mol2": _wt_to_mol2,
}

# ── Loop helpers ──────────────────────────────────────────────────────────────

MAX_LOOP_ITERATIONS = 100_000

def _safe_range(*args):
    """range() with a safety cap on iteration count."""
    r = range(*[int(float(a)) for a in args])
    if len(r) > MAX_LOOP_ITERATIONS:
        raise NotebookEvaluationError(
            f"Loop too large: {len(r):,} iterations (max {MAX_LOOP_ITERATIONS:,})"
        )
    return r

def _safe_enumerate(iterable, start=0):
    return enumerate(iterable, int(start))

def _safe_zip(*iterables):
    return zip(*iterables)

def _safe_len(obj):
    return len(obj)


def _is_allowed_method_call(node: ast.Call, allowed_names: set[str]) -> bool:
    """Return True only for the exact safe method form `<name>.copy()`."""
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "copy":
        return False
    if node.args or node.keywords:
        return False
    base = node.func.value
    return isinstance(base, ast.Name) and base.id in allowed_names

ALLOWED_NOTEBOOK_FUNCTIONS["range"]     = _safe_range
ALLOWED_NOTEBOOK_FUNCTIONS["enumerate"] = _safe_enumerate
ALLOWED_NOTEBOOK_FUNCTIONS["zip"]       = _safe_zip
ALLOWED_NOTEBOOK_FUNCTIONS["len"]       = _safe_len

# Backward-compatible alias
ALLOWED_SCALAR_FUNCTIONS = ALLOWED_NOTEBOOK_FUNCTIONS


# ── AST node allowlist ────────────────────────────────────────────────────────

_ALLOWED_NODE_LIST = [
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Attribute,
    ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.MatMult,
    ast.List, ast.Subscript, ast.Slice,
    ast.Dict, ast.ListComp, ast.comprehension,
    # Multi-dim indexing and tuple literals
    ast.Tuple,
    # Comparisons
    ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
    # Logical
    ast.BoolOp, ast.And, ast.Or,
    # Unary not / bitwise invert
    ast.Not, ast.Invert,
    # Ternary
    ast.IfExp,
]
if hasattr(ast, "Index"):          # removed in Python 3.9
    _ALLOWED_NODE_LIST.append(ast.Index)

ALLOWED_NOTEBOOK_NODES = tuple(_ALLOWED_NODE_LIST)

# Exec-mode block nodes (for/if + assignments)
_BLOCK_EXTRA = [
    ast.Module, ast.Expr,
    ast.Assign, ast.AugAssign, ast.Store,
    ast.For, ast.If,
    ast.Pass, ast.Break, ast.Continue,
    ast.In, ast.NotIn,
    ast.FloorDiv,
]
ALLOWED_BLOCK_NODES = ALLOWED_NOTEBOOK_NODES + tuple(_BLOCK_EXTRA)


# ── Error type ────────────────────────────────────────────────────────────────

class NotebookEvaluationError(ValueError):
    """Raised when a notebook expression cannot be evaluated safely."""


# ── Validator ─────────────────────────────────────────────────────────────────

class _NotebookValidator(ast.NodeVisitor):
    def __init__(self, allowed_names):
        self._allowed = set(allowed_names)

    def generic_visit(self, node):
        if not isinstance(node, ALLOWED_NOTEBOOK_NODES):
            raise NotebookEvaluationError(f"Unsupported syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id not in self._allowed:
            raise NotebookEvaluationError(f"Unknown symbol: {node.id!r}")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_NOTEBOOK_FUNCTIONS:
            pass
        elif _is_allowed_method_call(node, self._allowed):
            self.visit(node.func.value)
        else:
            raise NotebookEvaluationError("Only approved functions are allowed")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.arg is None:
                raise NotebookEvaluationError("**kwargs unpacking is not allowed")
            self.visit(kw.value)

    def visit_Attribute(self, node):
        if node.attr != "copy" or not isinstance(node.value, ast.Name) or node.value.id not in self._allowed:
            raise NotebookEvaluationError("Only approved attributes are allowed")
        self.visit(node.value)

    def visit_List(self, node):
        for elt in node.elts:
            self.visit(elt)

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Tuple(self, node):
        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node):
        for key in node.keys:
            if key is not None:
                self.visit(key)
        for value in node.values:
            self.visit(value)

    def _collect_comprehension_target(self, target):
        if isinstance(target, ast.Name):
            self._allowed.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_comprehension_target(elt)

    def visit_ListComp(self, node):
        for gen in node.generators:
            self.visit(gen.iter)
            self._collect_comprehension_target(gen.target)
            for if_clause in gen.ifs:
                self.visit(if_clause)
        self.visit(node.elt)

    def visit_Compare(self, node):
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_BoolOp(self, node):
        for val in node.values:
            self.visit(val)

    def visit_IfExp(self, node):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)


# ── Block validator (exec mode) ───────────────────────────────────────────────

_RESERVED_NAMES = (
    set(ALLOWED_NOTEBOOK_FUNCTIONS) | set(ALLOWED_CONSTANTS) |
    set(PHYSICAL_CONSTANTS) | set(UNIT_CONSTANTS) | {"tau", "deg", "inf"}
)

class _BlockValidator(ast.NodeVisitor):
    """Validates exec-mode blocks: for/if with body statements."""

    def __init__(self, allowed_names):
        self._allowed = set(allowed_names)

    def generic_visit(self, node):
        if not isinstance(node, ALLOWED_BLOCK_NODES):
            raise NotebookEvaluationError(f"Not allowed in blocks: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            if node.id in _RESERVED_NAMES:
                raise NotebookEvaluationError(f"{node.id!r} is a reserved name")
            return
        if node.id not in self._allowed:
            raise NotebookEvaluationError(f"Unknown symbol: {node.id!r}")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_NOTEBOOK_FUNCTIONS:
            pass
        elif _is_allowed_method_call(node, self._allowed):
            self.visit(node.func.value)
        else:
            raise NotebookEvaluationError("Only approved functions are allowed")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.arg is None:
                raise NotebookEvaluationError("**kwargs unpacking is not allowed")
            self.visit(kw.value)

    def visit_Attribute(self, node):
        if node.attr != "copy" or not isinstance(node.value, ast.Name) or node.value.id not in self._allowed:
            raise NotebookEvaluationError("Only approved attributes are allowed")
        self.visit(node.value)

    def visit_Assign(self, node):
        # Register assigned names so later statements can reference them
        for t in node.targets:
            if isinstance(t, ast.Name):
                if t.id not in _RESERVED_NAMES:
                    self._allowed.add(t.id)
            self.visit(t)
        self.visit(node.value)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            if node.target.id not in _RESERVED_NAMES:
                self._allowed.add(node.target.id)
        self.visit(node.target)
        self.visit(node.value)

    def visit_For(self, node):
        # Register loop target variable(s) so the body can reference them
        def _collect_targets(t):
            if isinstance(t, ast.Name):
                if t.id in _RESERVED_NAMES:
                    raise NotebookEvaluationError(f"{t.id!r} is a reserved name")
                self._allowed.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for elt in t.elts:
                    _collect_targets(elt)
        _collect_targets(node.target)
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_If(self, node):
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_List(self, node):
        for elt in node.elts:
            self.visit(elt)

    def _collect_comprehension_target(self, target):
        if isinstance(target, ast.Name):
            if target.id in _RESERVED_NAMES:
                raise NotebookEvaluationError(f"{target.id!r} is a reserved name")
            self._allowed.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_comprehension_target(elt)

    def visit_Tuple(self, node):
        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node):
        for key in node.keys:
            if key is not None:
                self.visit(key)
        for value in node.values:
            self.visit(value)

    def visit_ListComp(self, node):
        for gen in node.generators:
            self.visit(gen.iter)
            self._collect_comprehension_target(gen.target)
            for if_clause in gen.ifs:
                self.visit(if_clause)
        self.visit(node.elt)

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Compare(self, node):
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_BoolOp(self, node):
        for val in node.values:
            self.visit(val)

    def visit_IfExp(self, node):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_number(x: float) -> str:
    """Format a single finite number compactly."""
    f = float(x)
    if not math.isfinite(f):
        return str(f)
    a = abs(f)
    if a == 0.0:
        return "0"
    if a >= 1e4 or (a > 0 and a < 1e-2):
        s = f"{f:.12g}"
        return s.replace("e+", "e")
    if f == int(f) and a < 1e15:
        return str(int(f))
    return f"{f:.12g}"


def _format_value(value) -> str:
    """Format any notebook result (scalar, vector, matrix) for the gutter."""
    if isinstance(value, np.ndarray):
        value = np.squeeze(value)
        if value.ndim == 0:
            return _format_number(float(value))
        if value.ndim == 1:
            n = len(value)
            if n <= 6:
                return "[" + ", ".join(_format_number(x) for x in value) + "]"
            return (f"[{_format_number(value[0])}, {_format_number(value[1])}, ..."
                    f", {_format_number(value[-1])}]")
        if value.ndim == 2:
            r, c = value.shape
            if r <= 3 and c <= 4:
                rows = ["[" + ", ".join(_format_number(x) for x in row) + "]"
                        for row in value]
                return "[" + ", ".join(rows) + "]"
            return f"{r}×{c} matrix"
        return f"ndarray {value.shape}"
    if isinstance(value, list):
        try:
            return _format_value(np.asarray(value, dtype=float))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return _format_number(float(value))
    return str(value)


# ── Expression normalization & comment stripping ──────────────────────────────

# Regex that matches  <number><whitespace><unit>  e.g. "10 MPa", "5.0 mm", "1e3 Hz"
# Built once from UNIT_CONSTANTS keys so it stays in sync automatically.
_UNIT_NAMES_PATTERN = re.compile(
    r"(?<![.\w])(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*("
    + "|".join(re.escape(u) for u in sorted(UNIT_CONSTANTS, key=len, reverse=True))
    + r")(?!\w)"
)


def _apply_unit_shorthand(text: str) -> str:
    """Convert '10 MPa' → '(10*MPa)' for all known unit names."""
    return _UNIT_NAMES_PATTERN.sub(r"(\1*\2)", text)


def _strip_comment(line: str) -> str:
    """Remove # and // trailing comments."""
    for marker in ("//", "#"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line.strip()


def _normalize_expression(expression: str) -> str:
    expr = _strip_comment(expression).replace("^", "**")
    return _apply_unit_shorthand(expr)


def _normalize_block_line(line: str) -> str:
    """Normalize a block body line: strip comment + replace ^, preserve indentation."""
    if not line.strip():
        return line
    indent = line[: len(line) - len(line.lstrip())]
    content = _strip_comment(line.lstrip()).replace("^", "**")
    content = _apply_unit_shorthand(content)
    return indent + content if content else ""


def _needs_expression_continuation(expr_rows: list[dict]) -> bool:
    """True when a non-block expression has an unclosed bracket and needs more rows."""
    normalized = "\n".join(
        _normalize_block_line(str(row.get("expression", "") or ""))
        for row in expr_rows
    )
    try:
        ast.parse(normalized, mode="exec")
        return False
    except SyntaxError as exc:
        message = str(exc.msg or "")
        return "was never closed" in message or "unexpected EOF while parsing" in message


def _is_block_header(stripped: str) -> bool:
    """True if the line opens a for/if block."""
    return bool(
        re.match(r"^(for|if)\b", stripped) and stripped.rstrip().endswith(":")
    )


def _build_context(variables: dict) -> dict:
    return {
        **UNIT_CONSTANTS,
        **PHYSICAL_CONSTANTS,
        **ALLOWED_NOTEBOOK_FUNCTIONS,
        **ALLOWED_CONSTANTS,
        "tau": 2 * math.pi, "deg": math.pi / 180, "inf": math.inf,
        **variables,
    }


def _evaluate_block(block_rows: list[dict], variables: dict) -> tuple[dict, str]:
    """Execute a multi-line block (for/if) in exec mode.

    Returns (updated_variables, summary_string).
    """
    lines = [_normalize_block_line(str(r.get("expression", "") or "")) for r in block_rows]
    block_text = "\n".join(lines)

    context = _build_context(variables)

    try:
        parsed = ast.parse(block_text, mode="exec")
    except SyntaxError as exc:
        raise NotebookEvaluationError(f"Syntax error: {exc.msg}") from exc

    _BlockValidator(set(context.keys())).visit(parsed)
    compiled = compile(parsed, "<notebook_block>", "exec")
    scope = {"__builtins__": {}, **context}
    exec(compiled, scope, scope)  # noqa: S102

    # Build summary label
    header_stripped = lines[0].strip() if lines else ""
    if re.match(r"^for\b", header_stripped):
        # Extract iteration count from range() if present
        m = re.search(r"range\(([^)]+)\)", header_stripped)
        if m:
            try:
                iters = len(_safe_range(*[int(float(x.strip())) for x in m.group(1).split(",")]))
                summary = f"[done — {iters:,} iter]"
            except Exception:
                summary = "[done]"
        else:
            summary = "[done]"
    else:
        summary = "[done]"

    # Extract updated user variables
    updated = {
        k: v for k, v in scope.items()
        if k not in _RESERVED_NAMES and k not in {"tau", "deg", "inf"}
    }
    return updated, summary


def _group_segments(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group rows into ('single', [row]), ('continued', [rows]) or ('block', [rows]) segments."""
    segments: list[tuple[str, list[dict]]] = []
    i = 0
    while i < len(rows):
        raw = str(rows[i].get("expression", "") or "")
        stripped = _strip_comment(raw)

        if _is_block_header(stripped):
            block_rows = [rows[i]]
            j = i + 1
            pending_empty: list[dict] = []

            while j < len(rows):
                next_raw = str(rows[j].get("expression", "") or "")
                next_stripped = _strip_comment(next_raw)
                is_indented = bool(next_raw) and next_raw[0] in (" ", "\t")
                is_elif_else = bool(
                    re.match(r"^(elif\b.*:|else\s*:)", next_stripped)
                )

                if is_indented or is_elif_else:
                    block_rows.extend(pending_empty)
                    pending_empty = []
                    block_rows.append(rows[j])
                    j += 1
                elif not next_stripped:
                    pending_empty.append(rows[j])
                    j += 1
                else:
                    j -= len(pending_empty)
                    break

            segments.append(("block", block_rows))
            i = j
        elif stripped:
            expr_rows = [rows[i]]
            j = i + 1
            while j < len(rows) and _needs_expression_continuation(expr_rows):
                expr_rows.append(rows[j])
                j += 1
            if len(expr_rows) > 1:
                segments.append(("continued", expr_rows))
                i = j
            else:
                segments.append(("single", [rows[i]]))
                i += 1
        else:
            segments.append(("single", [rows[i]]))
            i += 1

    return segments


def _evaluate_continued_rows(seg_rows: list[dict], variables: dict) -> tuple[Any | None, str, str]:
    """Evaluate a multi-line literal/expression/assignment."""
    normalized = "\n".join(
        _normalize_block_line(str(row.get("expression", "") or ""))
        for row in seg_rows
    )
    try:
        parsed = ast.parse(normalized, mode="exec")
    except SyntaxError as exc:
        return None, "", f"Syntax error: {exc.msg}"

    context = _build_context(variables)
    try:
        if len(parsed.body) != 1:
            raise NotebookEvaluationError("Unsupported multiline expression")
        stmt = parsed.body[0]
        if isinstance(stmt, ast.Assign):
            _BlockValidator(set(context.keys())).visit(parsed)
            compiled = compile(parsed, "<notebook_continued>", "exec")
            scope = {"__builtins__": {}, **context}
            exec(compiled, scope, scope)  # noqa: S102
            target = stmt.targets[0] if stmt.targets else None
            if isinstance(target, ast.Name):
                value = scope.get(target.id)
                variables[target.id] = value
                return value, _format_value(value), ""
            return None, "", ""
        if isinstance(stmt, ast.Expr):
            expr_ast = ast.Expression(stmt.value)
            _NotebookValidator(set(context.keys())).visit(expr_ast)
            scope = {"__builtins__": {}, **context}
            value = eval(compile(expr_ast, "<notebook_continued_expr>", "eval"), scope, scope)  # noqa: S307
            return value, _format_value(value), ""
        raise NotebookEvaluationError("Unsupported multiline expression")
    except NotebookEvaluationError as exc:
        return None, "", str(exc)
    except Exception as exc:
        return None, "", f"Error: {exc}"


# ── Core evaluator ────────────────────────────────────────────────────────────

def _evaluate_expr(expression: str, variables: dict) -> Any:
    """Compile and safely eval a single expression with given variable context."""
    expr = _normalize_expression(expression)
    if not expr:
        raise NotebookEvaluationError("Empty expression")

    context: dict = {
        **UNIT_CONSTANTS,
        **PHYSICAL_CONSTANTS,
        **ALLOWED_NOTEBOOK_FUNCTIONS,
        **ALLOWED_CONSTANTS,
        "tau": 2 * math.pi, "deg": math.pi / 180, "inf": math.inf,
        **variables,
    }

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise NotebookEvaluationError(f"Syntax error: {exc.msg}") from exc

    _NotebookValidator(context.keys()).visit(parsed)
    compiled = compile(parsed, "<notebook>", "eval")
    scope = {"__builtins__": {}, **context}
    return eval(compiled, scope, scope)  # noqa: S307


def _evaluate_subscript_assign(stmt: str, variables: dict) -> None:
    """Execute a subscript assignment like `arr[i] = value` in exec mode.

    Mutates numpy arrays in-place so the change is visible through *variables*.
    """
    normalized = _normalize_block_line(stmt)
    context = _build_context(variables)
    try:
        parsed = ast.parse(normalized, mode="exec")
    except SyntaxError as exc:
        raise NotebookEvaluationError(f"Syntax error: {exc.msg}") from exc
    _BlockValidator(set(context.keys())).visit(parsed)
    compiled = compile(parsed, "<notebook>", "exec")
    exec(compiled, {"__builtins__": {}}, context)  # noqa: S102


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_notebook_rows(
    rows: list[dict],
    initial_context: dict | None = None,
) -> tuple[list[dict], dict[str, float], dict[str, list]]:
    """Evaluate notebook rows top-to-bottom.

    Variables carrying forward can be scalars, vectors, or matrices.
    The returned *variables* dict contains only scalar floats so that
    formula-panel NB-toggle bindings remain type-safe.

    *initial_context* can be used to seed variables from prior cells.
    """
    variables: dict[str, Any] = dict(initial_context or {})   # full (may hold arrays)
    evaluated_rows: list[dict] = []

    for seg_type, seg_rows in _group_segments(rows):
        if seg_type == "block":
            # ── for / if block ────────────────────────────────────────────────
            header_row = seg_rows[0]
            body_rows  = seg_rows[1:]

            header_out = dict(header_row)
            header_out["result"] = ""
            header_out["error"] = ""
            header_out["raw_result"] = None
            header_out["source_span"] = len(seg_rows)

            try:
                new_vars, summary = _evaluate_block(seg_rows, variables)
                variables.update(new_vars)
                header_out["result"] = summary
            except NotebookEvaluationError as exc:
                header_out["error"] = str(exc)
            except Exception as exc:
                header_out["error"] = f"Error: {exc}"

            evaluated_rows.append(header_out)
            for body_row in body_rows:
                out = dict(body_row)
                out["result"] = ""
                out["error"] = ""
                out["raw_result"] = None
                evaluated_rows.append(out)

        elif seg_type == "continued":
            header_row = seg_rows[0]
            body_rows = seg_rows[1:]

            header_out = dict(header_row)
            header_out["result"] = ""
            header_out["error"] = ""
            header_out["raw_result"] = None
            header_out["source_span"] = len(seg_rows)

            value, result, error = _evaluate_continued_rows(seg_rows, variables)
            header_out["result"] = result
            header_out["error"] = error
            header_out["raw_result"] = value
            evaluated_rows.append(header_out)

            for body_row in body_rows:
                out = dict(body_row)
                out["result"] = ""
                out["error"] = ""
                out["raw_result"] = None
                evaluated_rows.append(out)

        else:
            # ── single expression ─────────────────────────────────────────────
            row = seg_rows[0]
            raw = str(row.get("expression", "") or "").strip()
            updated = dict(row)
            updated["result"] = ""
            updated["error"] = ""
            updated["raw_result"] = None

            stripped = _strip_comment(raw)
            if not stripped:
                evaluated_rows.append(updated)
                continue

            # plot() calls are handled by the plots panel — skip evaluation
            if re.match(r"^plot\s*\(", stripped):
                updated["result"] = "📊 plot"
                evaluated_rows.append(updated)
                continue

            try:
                if "=" in stripped:
                    var_name, rhs = stripped.split("=", 1)
                    var_name = var_name.strip()
                    rhs = rhs.strip()
                    if "[" in var_name or "+=" in stripped or "-=" in stripped:
                        # Subscript assignment (arr[i] = v) or augmented assign — exec in place
                        _evaluate_subscript_assign(stripped, variables)
                        updated["result"] = ""
                    elif not var_name.isidentifier():
                        raise NotebookEvaluationError(f"Invalid variable name: {var_name!r}")
                    elif var_name in ALLOWED_NOTEBOOK_FUNCTIONS or var_name in ALLOWED_CONSTANTS or var_name in UNIT_CONSTANTS:
                        raise NotebookEvaluationError(f"{var_name!r} is a reserved name")
                    else:
                        value = _evaluate_expr(rhs, variables)
                        variables[var_name] = value
                        updated["result"] = _format_value(value)
                        updated["raw_result"] = value
                else:
                    value = _evaluate_expr(stripped, variables)
                    updated["result"] = _format_value(value)
                    updated["raw_result"] = value

            except NotebookEvaluationError as exc:
                updated["error"] = str(exc)
            except Exception as exc:
                updated["error"] = f"Error: {exc}"

            evaluated_rows.append(updated)

    # Only expose scalars to formula-plot NB toggle
    scalar_vars: dict[str, float] = {}
    # Expose 1-D numeric arrays for graph plotter
    array_vars: dict[str, list] = {}

    for k, v in variables.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            try:
                scalar_vars[k] = float(v)
            except (TypeError, ValueError):
                pass
        elif isinstance(v, np.ndarray) and v.size == 1:
            try:
                scalar_vars[k] = float(v.ravel()[0])
            except (TypeError, ValueError):
                pass
        elif isinstance(v, np.ndarray) and v.ndim == 1 and np.issubdtype(v.dtype, np.number):
            try:
                array_vars[k] = v.tolist()
            except (TypeError, ValueError):
                pass
        elif isinstance(v, list):
            try:
                arr = np.asarray(v, dtype=float)
                if arr.ndim == 1:
                    array_vars[k] = arr.tolist()
            except (TypeError, ValueError):
                pass

    return evaluated_rows, scalar_vars, array_vars


def evaluate_cell(
    source: str,
    context: dict | None = None,
) -> tuple[list[dict], dict[str, float], dict[str, list], dict]:
    """Evaluate a single notebook cell with an accumulated variable context.

    Returns (result_rows, scalar_vars, array_vars, full_vars) where
    full_vars includes all variables after evaluating this cell.
    """
    rows = [{"id": f"line_{i}", "expression": line} for i, line in enumerate(source.splitlines())]
    result_rows, scalar_vars, array_vars = evaluate_notebook_rows(rows, initial_context=context)

    # Rebuild full variable dict so caller can pass it as context to the next cell
    import numpy as np
    full_vars: dict[str, Any] = dict(context or {})
    for row in result_rows:
        expr = str(row.get("expression", "") or "").strip()
        stripped = expr
        for marker in ("//", "#"):
            idx = stripped.find(marker)
            if idx != -1:
                stripped = stripped[:idx]
        stripped = stripped.strip()
        if "=" in stripped:
            var_name = stripped.split("=", 1)[0].strip()
            if var_name.isidentifier() and var_name in scalar_vars:
                full_vars[var_name] = scalar_vars[var_name]
            elif var_name.isidentifier() and var_name in array_vars:
                full_vars[var_name] = np.asarray(array_vars[var_name])
    # Also copy any array variables that came from initial_context processing
    for k, v in array_vars.items():
        if k not in full_vars:
            full_vars[k] = np.asarray(v)
    for k, v in scalar_vars.items():
        if k not in full_vars:
            full_vars[k] = v

    return result_rows, scalar_vars, array_vars, full_vars
