"""fmetrics Tier 3: mollified-law Coulomb-graph residual (headline, D-2), exact-law residual
(supplement, binned by |v*|/eps_op), decomposition, dissipation, W_end (spec).

All quantities from derive_operator_quants (convention A, fp64). Branch classification follows
the OPERATOR's own decision: arrest iff factor == 0 (exact clamp), slip otherwise.

Mollified graph (what a bug-free operator satisfies exactly, spec):
  arrest: v+ = 0, ||f_t|| = m|v*|/h  (< mu*lam strictly), cone (||f_t||-mu*lam)_+ = 0
  slip:   ||f_t|| = mu*lam*s*,  f_t || -v+ exactly
Exact-Coulomb law (ideal, eps_op->0): slip force ON the cone — the implemented operator sits
below it by the factor s*, so the exact-law residual has the analytic (1-s*) deficit (E16).
"""
import numpy as np

from .io import derive_operator_quants, flog_slice

SP_RATIO_BINS = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf])  # |v*|/eps_op bins (supplement)


def residuals_sequence(flog: dict, v_mass: np.ndarray, mu: float, h: float, eps_op: float,
                       eps_v: float, eta: float, t0: int, Tmin: int,
                       lambda_min: float, F0_g: float = 9.81,
                       pred=None, obst=None, faces=None):
    """Stream all flog slices of one sequence -> pooled residual/dissipation accumulators.

    pred/obst/faces (torch, optional): enables the realized-step residual
    r_real = ||vT_realized^A - v+|| / eps_v in the operator's own frame (spec Q3).
    Returns a dict of weighted sums; report.py turns them into means/p95 (weights w = lambda
    primary, mass secondary — never the product, spec).
    """
    acc = {k: [] for k in ('r_moll', 'r_exact', 'r_cone', 'c_opp', 'w_lam', 'w_m',
                           'sp_ratio', 'r_real')}
    sums = dict(e_diss=0.0, e_diss_neg=0.0, n_ediss_viol=0, w_end_pos_mass=0.0, w_end_mass=0.0,
                n_samples=0, n_lambda_min=0, mu0_f_over_F0=[], arrest_consistency=[],
                T_op=0)
    T_pkl = len(flog['offsets']) - 1
    import torch
    fnorm_cache = {}

    def face_normals(k):
        if k not in fnorm_cache:
            from .geometry import face_normals_at
            fnorm_cache.clear()                               # keep one frame only
            fnorm_cache[k] = face_normals_at(obst[k], faces)
        return fnorm_cache[k]

    for k in range(max(t0 + 1, 2), min(T_pkl, Tmin)):
        sl = flog_slice(flog, k)
        sums['T_op'] += 1
        if sl['vertex'].size == 0:
            continue
        q = derive_operator_quants(sl, v_mass, h, eps_op)
        m, lam, fac = q['m'], q['lam'], q['factor']
        ft, ftn = q['f_t'], q['f_t_norm']
        vp, spp = q['v_plus'], q['sp_plus']
        s_star, sp = q['s_star'], q['sp_star']
        n = m.shape[0]
        sums['n_samples'] += n

        # --- energy (all samples, incl. lambda=0: E_diss=0 there) --------------------------
        e = q['e_diss']
        sums['e_diss'] += float(e.sum())
        viol = e < -1e-9
        sums['n_ediss_viol'] += int(viol.sum())
        sums['e_diss_neg'] += float(e[viol].sum())
        # W_end sign (slip samples with defined direction)
        w_end = (ft * vp).sum(axis=-1) * h
        slipping = (fac > 0.0) & (spp > eps_v) & (ftn > eta)
        sums['w_end_mass'] += float(m[slipping].sum())
        sums['w_end_pos_mass'] += float(m[slipping & (w_end > 1e-12)].sum())

        if mu == 0.0:
            # spec edge case 1: report ||f_t||/F0, F0 = m g; never divide by eta
            sums['mu0_f_over_F0'].extend((ftn / (m * F0_g)).tolist())
            continue

        # --- residuals on loaded contacts only ----------------------------------------------
        loaded = lam >= lambda_min
        sums['n_lambda_min'] += int((~loaded & (lam > 0)).sum())
        act = loaded & (lam > 0)
        if not act.any():
            continue
        mlam = mu * lam[act]
        F = mlam + eta
        w_l, w_m_ = lam[act], m[act]
        ftn_a, fac_a = ftn[act], fac[act]
        spp_a, sstar_a, sp_a = spp[act], s_star[act], sp[act]
        vp_a, ft_a = vp[act], ft[act]

        r_cone = np.maximum(ftn_a - mlam, 0.0) / F
        # opposition cosine, defined where both vectors are meaningful (edge case 2)
        defined = (spp_a > eps_v) & (ftn_a > eta)
        c_opp = np.full(ftn_a.shape, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            c_opp[defined] = -(ft_a[defined] * vp_a[defined]).sum(axis=-1) / (
                ftn_a[defined] * np.linalg.norm(vp_a[defined], axis=-1))
        opp_term = np.zeros(ftn_a.shape)
        opp_term[defined] = 1.0 - c_opp[defined]              # (1 + cos(f, v)) with cos(f,v) = -c_opp

        arrested = fac_a == 0.0
        # mollified-law branch residuals (headline)
        r_stick_m = np.sqrt((spp_a / eps_v) ** 2 + (np.maximum(ftn_a - mlam, 0.0) / F) ** 2)
        r_slip_m = np.sqrt(((ftn_a - mlam * sstar_a) / F) ** 2 + opp_term ** 2)
        r_moll = np.where(arrested, r_stick_m, np.minimum(r_stick_m, r_slip_m))
        # exact-law residuals (supplement; slip term unmollified)
        r_slip_e = np.sqrt(((ftn_a - mlam) / F) ** 2 + opp_term ** 2)
        r_exact = np.minimum(r_stick_m, r_slip_e)

        # arrest-branch consistency: ||f_t|| must equal m|v*|/h exactly (spec)
        if arrested.any():
            fd_arr = (w_m_[arrested] * sp_a[arrested]) / h
            sums['arrest_consistency'].extend(
                (np.abs(ftn_a[arrested] - fd_arr) / F[arrested]).tolist())

        acc['r_moll'].append(r_moll); acc['r_exact'].append(r_exact)
        acc['r_cone'].append(r_cone); acc['c_opp'].append(c_opp)
        acc['w_lam'].append(w_l); acc['w_m'].append(w_m_)
        acc['sp_ratio'].append(sp_a / eps_op)

        # realized-step residual (needs positions; operator frame of the logged face)
        if pred is not None and k < pred.shape[0]:
            nrm = face_normals(k)[torch.as_tensor(q['face'][act], device=pred.device)]
            cen_k = obst[k][faces].mean(dim=1)
            cen_km1 = obst[k - 1][faces].mean(dim=1)
            fidx = torch.as_tensor(q['face'][act], device=pred.device)
            vidx = torch.as_tensor(q['vertex'][act], device=pred.device)
            dreal = (pred[k][vidx] - pred[k - 1][vidx]) - (cen_k[fidx] - cen_km1[fidx])
            dreal_t = dreal - (dreal * nrm).sum(dim=-1, keepdim=True) * nrm
            v_real = (dreal_t / h).cpu().numpy().astype(np.float64)
            acc['r_real'].append(np.linalg.norm(v_real - vp_a, axis=-1) / eps_v)

    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in acc.items()}
    out.update(sums)
    return out


def summarize_residuals(rs: dict, h: float):
    """Weighted mean/p95 (lambda-weighted primary, mass secondary) + dissipation aggregates."""
    def wstats(x, w):
        ok = np.isfinite(x)
        x, w = x[ok], w[ok]
        if x.size == 0 or w.sum() <= 0:
            return {'mean': None, 'p95': None}
        order = np.argsort(x)
        xw, ww = x[order], w[order]
        cw = np.cumsum(ww) / ww.sum()
        return {'mean': float((x * w).sum() / w.sum()),
                'p95': float(xw[np.searchsorted(cw, 0.95)] if cw.size else np.nan)}

    res = {}
    for key in ('r_moll', 'r_exact', 'r_cone'):
        res[key + '_lamW'] = wstats(rs[key], rs['w_lam'])
        res[key + '_mW'] = wstats(rs[key], rs['w_m'])
    # exact-law residual binned by |v*|/eps_op (the analytic mollifier-hump exhibit)
    bins = []
    for lo, hi in zip(SP_RATIO_BINS[:-1], SP_RATIO_BINS[1:]):
        sel = (rs['sp_ratio'] >= lo) & (rs['sp_ratio'] < hi)
        bins.append({'lo': float(lo), 'hi': float(hi),
                     **wstats(rs['r_exact'][sel], rs['w_lam'][sel]),
                     'n': int(sel.sum())})
    res['r_exact_by_spratio'] = bins
    res['c_opp_min'] = float(np.nanmin(rs['c_opp'])) if np.isfinite(rs['c_opp']).any() else None
    res['r_real'] = wstats(rs['r_real'], np.ones_like(rs['r_real'])) if rs['r_real'].size else \
        {'mean': None, 'p95': None}
    res['arrest_consistency_max'] = (max(rs['arrest_consistency'])
                                     if rs['arrest_consistency'] else None)
    T_op = max(rs['T_op'], 1)
    res['D_s'] = rs['e_diss'] / T_op                          # J/frame over operator-active frames
    res['ediss_violation_n'] = rs['n_ediss_viol']
    res['ediss_violation_J'] = rs['e_diss_neg']
    res['w_end_pos_fraction'] = (rs['w_end_pos_mass'] / rs['w_end_mass']
                                 if rs['w_end_mass'] > 0 else None)
    res['n_lambda_min_excluded'] = rs['n_lambda_min']
    res['n_samples'] = rs['n_samples']
    if rs['mu0_f_over_F0']:
        arr = np.asarray(rs['mu0_f_over_F0'])
        res['mu0_f_over_F0'] = {'mean': float(arr.mean()), 'max': float(arr.max())}
    return res
