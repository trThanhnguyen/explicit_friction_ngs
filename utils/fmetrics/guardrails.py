"""fmetrics guardrails (spec): penetration (proxy + oracle), contact mass-time, edge stretch,
root-frame freeze ratio, jitter, runtime. Reported next to every Tier-1/2 number.
"""
import numpy as np


def penetration(sd_TV: np.ndarray, t0: int, Tmin: int):
    """rate / depth mean / p95 / max over frames [t0, Tmin) for one signed-distance field."""
    sd = sd_TV[t0:Tmin]
    inside = sd < 0.0
    n = inside.size
    depths = -sd[inside]
    return {'pen_rate': float(inside.sum()) / max(n, 1),
            'pen_depth_mean': (float(depths.mean()) if depths.size else 0.0),
            'pen_depth_p95': (float(np.percentile(depths, 95)) if depths.size else 0.0),
            'pen_depth_max': (float(depths.max()) if depths.size else 0.0)}


def contact_masstime(c: np.ndarray, m: np.ndarray, h: float):
    """M_c = sum m c h [kg s] (windowed mask arrives from the caller)."""
    return float((c * m[None, :]).sum(dtype=np.float64) * h)


def edge_stretch(lc_TE: np.ndarray, lr_E: np.ndarray, t0: int, Tmin: int):
    """Strain percentiles ||e||/||e_rest|| - 1 (p50/p95) over frames [t0, Tmin)."""
    s = lc_TE[t0:Tmin] / lr_E[None, :] - 1.0
    return {'stretch_p50': float(np.percentile(s, 50)), 'stretch_p95': float(np.percentile(s, 95))}


def jitter_accel(pred_np: np.ndarray, t0: int, Tmin: int):
    """accel_q99: 2nd-difference norm q99 [m/frame^2] (popping detector; bench-table convention)."""
    p = pred_np[t0:Tmin].astype(np.float64)
    if p.shape[0] < 3:
        return {'accel_q99': None}
    acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1)
    return {'accel_q99': float(np.percentile(acc, 99))}


def freeze_ratio(vroot_by_mu: dict, free_mask: np.ndarray, m: np.ndarray,
                 free_mass_floor: float):
    """rho(mu) = mean_free ||v_root||(mu) / mean_free ||v_root||(0) on the COMMON free support
    (out of the contact band in ALL runs; spec). Absolute speeds reported beside the ratio;
    N/A below the support floor (P15)."""
    W = free_mask * m[None, :]
    wsum = float(W.sum(dtype=np.float64))
    total = float(m.sum(dtype=np.float64)) * free_mask.shape[0]
    out = {'free_mass_frac': wsum / max(total, 1e-30)}
    if wsum < free_mass_floor * total:
        out['abs_speed'] = {mu: None for mu in vroot_by_mu}
        out['rho'] = {mu: None for mu in vroot_by_mu}
        out['na_reason'] = 'free_support_below_floor'
        return out
    means = {mu: float((W * v).sum(dtype=np.float64) / wsum) for mu, v in vroot_by_mu.items()}
    base = means.get(0.0)
    out['abs_speed'] = means
    out['rho'] = {mu: (v / base if (base and base > 1e-12) else None) for mu, v in means.items()}
    if not (base and base > 1e-12):
        out['na_reason'] = 'mu0_free_speed_near_zero'
    return out
