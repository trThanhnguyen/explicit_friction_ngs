"""fmetrics masks: contact set, run-length machinery, A/M/Z, common support, onset events.

All masks live in VELOCITY-SAMPLE frame space: arrays are (Tv, V) with Tv = T-1, index t pairs
pred[t] -> pred[t+1] (P6). c_t is contact of the REALIZED position pred[t] (spec / SC-6).
The metric window (t >= t0, spec P17) is applied here exactly once via `window_mask`, and every
downstream sum inherits it. Pure numpy/CPU, deterministic.
"""
import numpy as np


# ------------------------------------------------------------------ elementary masks

def contact_mask(sd_TV: np.ndarray, r_c: float, pinned_V: np.ndarray) -> np.ndarray:
    """c (T,V) bool: signed plane-gap <= r_c, pinned excluded (spec; operator gate mirror)."""
    c = sd_TV <= r_c
    c[:, pinned_V] = False
    return c


def window_mask(Tv: int, V: int, t0: int) -> np.ndarray:
    """(Tv,V) bool: True for velocity frames t in [t0, Tv-1] (P17 warm-up + init exclusion)."""
    w = np.zeros((Tv, V), dtype=bool)
    w[min(t0, Tv):] = True
    return w


def run_length_mask(valid: np.ndarray, K: int) -> np.ndarray:
    """(T,V) bool -> frames belonging to a consecutive True-run of TOTAL length >= K.
    Vectorized: fwd[t] = consecutive True count ending at t; bwd[t] = count starting at t;
    total run length at t = fwd[t] + bwd[t] - 1."""
    if K <= 1:
        return valid.copy()
    T = valid.shape[0]
    idx = np.arange(T)[:, None]
    last_false = np.maximum.accumulate(np.where(~valid, idx, -1), axis=0)
    fwd = idx - last_false                                    # 0 where invalid
    fwd = np.where(valid, fwd, 0)
    vrev = valid[::-1]
    idxr = np.arange(T)[:, None]
    last_false_r = np.maximum.accumulate(np.where(~vrev, idxr, -1), axis=0)
    bwd = np.where(vrev, idxr - last_false_r, 0)[::-1]
    total = fwd + bwd - 1
    return valid & (total >= K)


def run_starts(mask: np.ndarray) -> np.ndarray:
    """(T,V) bool: first frame of each maximal True-run."""
    prev = np.zeros_like(mask)
    prev[1:] = mask[:-1]
    return mask & ~prev


# ------------------------------------------------------------------ spec masks

def activity_mask(c0: np.ndarray, vT0: np.ndarray, v_act: float, K_act: int,
                  wmask: np.ndarray) -> np.ndarray:
    """A (Tv,V): paired-mu0 contact AND ||v^T,0|| > v_act, in runs >= K_act. A contact break (or
    a window edge) terminates a run — spec (symmetric to the Z rule)."""
    return run_length_mask(c0 & (vT0 > v_act) & wmask, K_act)


def stick_mask(c_mu: np.ndarray, vT_mu: np.ndarray, eps_v: float, K_stick: int,
               wmask: np.ndarray) -> np.ndarray:
    """Z (Tv,V): contact AND ||v^T,mu|| <= eps_v in runs >= K_stick; contact break splits runs;
    slow OUT-OF-CONTACT frames neither create nor extend a run (spec ✦✦✦✦)."""
    return run_length_mask(c_mu & (vT_mu <= eps_v) & wmask, K_stick)


def common_support(c_by_mu: dict, common_min: int) -> dict:
    """q_{i,t} = 1[sum_mu c >= common_min]; returns {'q': q, mu: c_common} (spec)."""
    mus = sorted(c_by_mu)
    stack = np.stack([c_by_mu[m] for m in mus], axis=0)
    q = stack.sum(axis=0) >= int(common_min)
    return {'q': q, **{m: c_by_mu[m] & q for m in mus}}


def onset_events(A: np.ndarray, t0: int, Tv: int, delta_onset: int, counters: dict):
    """Onset events (spec): (i, t_o) = starts of maximal A-runs with the full window
    (t_o, t_o+delta] inside [t0, Tv-1]. Truncated-out events are counted (P6)."""
    starts = run_starts(A)
    tt, vv = np.nonzero(starts)
    keep = tt + delta_onset <= Tv - 1
    counters['onset_truncated'] = counters.get('onset_truncated', 0) + int((~keep).sum())
    return tt[keep], vv[keep]


def onset_conversion(tt: np.ndarray, vv: np.ndarray, c_rho: np.ndarray, Z_rho: np.ndarray,
                     delta_onset: int):
    """For each event: eligible iff c^rho[i, t_o] (contact retained at onset in run rho);
    converted iff a Z-run STARTS within (t_o, t_o + delta] (spec). Returns (eligible,
    converted) bool arrays over events."""
    zs = run_starts(Z_rho)
    csum = np.cumsum(zs, axis=0)                              # starts up to and incl. t
    n = tt.shape[0]
    eligible = c_rho[tt, vv] if n else np.zeros(0, dtype=bool)
    if n == 0:
        return eligible, np.zeros(0, dtype=bool)
    hi = np.minimum(tt + delta_onset, zs.shape[0] - 1)
    starts_in_window = csum[hi, vv] - csum[tt, vv]            # (t_o, t_o+delta]
    return eligible, (starts_in_window > 0) & eligible
