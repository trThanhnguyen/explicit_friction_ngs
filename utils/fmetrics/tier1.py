"""fmetrics Tier 1: PSR (S-all / S-common + coverage), FISC (+ strict) + C, OCR (spec).

Every function returns per-sequence values with explicit N/A (value=None + reason) — never a
coerced zero (P10). All sums fp64 CPU (deterministic). Masks arrive already windowed (t >= t0).
"""
import numpy as np

from .masks import onset_conversion, onset_events, run_starts  # noqa: F401 (re-export for tests)


def _ratio(num: float, den: float, na_reason: str):
    if den <= 0.0:
        return None, na_reason
    return float(num / den), None


def psr(vT: np.ndarray, c: np.ndarray, m: np.ndarray, masstime_floor: float,
        n_window_frames: int = None):
    """S_s(mu) [m/s] on support c (all-contact or common). masstime_floor is the spec-
    sequence floor: total contact mass-time >= floor fraction of (garment mass x duration),
    where duration = the METRIC WINDOW (frames >= t0), not the raw rollout length — pre-window
    frames are structurally contact-zero in the windowed mask and must not inflate the
    denominator."""
    W = c * m[None, :]
    den = float(W.sum(dtype=np.float64))
    total = float(m.sum(dtype=np.float64)) * (n_window_frames if n_window_frames else c.shape[0])
    if den <= 0.0 or den < masstime_floor * total:
        return {'value': None,
                'na_reason': ('empty_contact' if den <= 0.0 else 'contact_masstime_below_floor'),
                'masstime': den, 'masstime_frac': den / max(total, 1e-30)}
    num = float((W * vT).sum(dtype=np.float64))
    return {'value': num / den, 'na_reason': None, 'masstime': den,
            'masstime_frac': den / max(total, 1e-30)}


def coverage(c_common: np.ndarray, c_all: np.ndarray, m: np.ndarray):
    """cov_s(mu) in [0,1] by construction (spec)."""
    v, why = _ratio(float((c_common * m[None, :]).sum(dtype=np.float64)),
                    float((c_all * m[None, :]).sum(dtype=np.float64)), 'empty_contact')
    return {'value': v, 'na_reason': why}


def fisc(A: np.ndarray, c_mu: np.ndarray, Z_mu: np.ndarray, m: np.ndarray,
         strict_keep: np.ndarray = None):
    """FISC_s(mu) and matched coverage C_s(mu) (spec). strict_keep (Tv,V) bool optionally
    restricts the match mask M to ring-verified correspondences (D-5 variant)."""
    M = A & c_mu
    if strict_keep is not None:
        M = M & strict_keep
    mw = m[None, :]
    num_f = float((M & Z_mu).astype(np.float64).__mul__(mw).sum())
    den_f = float(M.astype(np.float64).__mul__(mw).sum())
    num_c = den_f
    den_c = float(A.astype(np.float64).__mul__(mw).sum())
    f_val, f_why = _ratio(num_f, den_f, 'no_matched_active_masstime')
    c_val, c_why = _ratio(num_c, den_c, 'no_active_masstime_at_mu0')
    return {'fisc': f_val, 'fisc_na': f_why, 'C': c_val, 'C_na': c_why,
            'matched_masstime': den_f, 'active_masstime': den_c}


def strict_keep_mask(faceB0: np.ndarray, faceB_mu: np.ndarray, rings, k_ring: int,
                     k_max: int) -> np.ndarray:
    """(Tv,V) bool: canonical body-surface correspondence within k_ring face rings (D-5).
    Misses (-1) never match. Vectorized over the (sparse) A-support is the caller's job; this
    computes densely for simplicity at smoke scale."""
    from .geometry import ring_distance
    Tv, V = faceB0.shape
    out = np.zeros((Tv, V), dtype=bool)
    diff = faceB0 != faceB_mu
    out[~diff & (faceB0 >= 0)] = True                          # same face: always within ring 0
    tt, vv = np.nonzero(diff)
    for t, v in zip(tt, vv):
        out[t, v] = ring_distance(rings, int(faceB0[t, v]), int(faceB_mu[t, v]), k_max) <= k_ring
    return out


def ocr(A: np.ndarray, c_rho: np.ndarray, Z_rho: np.ndarray, m: np.ndarray,
        t0: int, delta_onset: int, counters: dict):
    """Onset-window conversion rate OCR_rho + onset coverage (spec, D-7).
    The SAME function serves treatment (rho = mu run) and sham (rho = the mu=0 run itself)."""
    Tv = A.shape[0]
    tt, vv = onset_events(A, t0, Tv, delta_onset, counters)
    n_events = int(tt.shape[0])
    if n_events == 0:
        return {'value': None, 'na_reason': 'no_onset_events', 'n_events': 0,
                'n_eligible': 0, 'onset_coverage': None}
    eligible, converted = onset_conversion(tt, vv, c_rho, Z_rho, delta_onset)
    w = m[vv].astype(np.float64)
    den = float(w[eligible].sum())
    cov_val, _ = _ratio(float(w[eligible].sum()), float(w.sum()), 'no_events')
    if den <= 0.0:
        return {'value': None, 'na_reason': 'no_eligible_onset_events', 'n_events': n_events,
                'n_eligible': 0, 'onset_coverage': cov_val}
    val = float(w[converted].sum() / den)
    return {'value': val, 'na_reason': None, 'n_events': n_events,
            'n_eligible': int(eligible.sum()), 'onset_coverage': cov_val}
