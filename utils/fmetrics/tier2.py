"""fmetrics Tier 2: CSA with the operator-faithful classifier (decision D-1).

pred stick = 1[f_d^op <= mu*lambda_n]     (convention A, from the friction log)
observed   = 1[||v^T|| <= eps_v]          (convention B, realized trajectory)
take-off   = lambda_n == 0 excluded from BOTH classes (lambda is Signorini-clamped, so
             "<= 0" is exactly "== 0"; matches Zeng Eq. 4's active set).

At mu=0 the predicted-stick class is EXACTLY empty (spec identity 3) — callers get the
per-class masses and an N/A BA, by construction, not by accident.

Alignment (P6): flog slice at pkl frame k pairs with velocity sample t = k-1.
Margin histogram: dimensionless M_rel = (f_d^op - mu*lam) / (f_d^op + mu*lam + eta) in (-1,1)
— an implementation choice for the PLOT only (documented; not a metric).
"""
import numpy as np

from .io import derive_operator_quants, flog_slice

MARGIN_BINS = np.linspace(-1.0, 1.0, 65)


def csa_sequence(flog: dict, v_mass: np.ndarray, vT: np.ndarray, mu: float, h: float,
                 eps_op: float, eps_v: float, eta: float, t0: int, Tmin: int,
                 orient_normals=None, up_axis: int = 1, lam_quartiles=None):
    """Pooled (this sequence) mass-weighted confusion + margin histogram (+ optional strata).

    vT: (Tv,V) convention-B slip speeds of THIS run. orient_normals: optional callable
    k -> (F,3) face normals at pkl frame k (convention A pose) for the orientation strata.
    Returns dict with confusion masses, BA (or None), margin histogram, strata confusions.
    """
    Tv = vT.shape[0]
    conf = np.zeros(4, dtype=np.float64)          # [TP_stick, FN_stick, TN_slip, FP_slip] masses
    hist = np.zeros(MARGIN_BINS.size - 1, dtype=np.float64)
    hist_stick = np.zeros_like(hist)
    strata = {}
    excluded_takeoff = 0.0
    T_pkl = len(flog['offsets']) - 1
    for k in range(max(t0 + 1, 2), min(T_pkl, Tmin)):
        t = k - 1                                  # velocity-sample index (P6)
        if t >= Tv:
            break
        sl = flog_slice(flog, k)
        if sl['vertex'].size == 0:
            continue
        q = derive_operator_quants(sl, v_mass, h, eps_op)
        lam, m = q['lam'], q['m']
        active = lam > 0.0
        excluded_takeoff += float(m[~active].sum())
        if not active.any():
            continue
        fd_op = q['f_d_op'][active]
        mlam = mu * lam[active]
        w = m[active]
        pred_stick = fd_op <= mlam
        obs_stick = vT[t, q['vertex'][active]] <= eps_v
        conf[0] += float(w[pred_stick & obs_stick].sum())
        conf[1] += float(w[pred_stick & ~obs_stick].sum())
        conf[2] += float(w[~pred_stick & ~obs_stick].sum())
        conf[3] += float(w[~pred_stick & obs_stick].sum())
        mrel = (fd_op - mlam) / (fd_op + mlam + eta)
        hw, _ = np.histogram(mrel, bins=MARGIN_BINS, weights=w)
        hist += hw
        hs, _ = np.histogram(mrel[obs_stick], bins=MARGIN_BINS, weights=w[obs_stick])
        hist_stick += hs
        if orient_normals is not None:
            n = orient_normals(k)[q['face'][active]]
            nz = n[:, up_axis]
            for name, sel in (('up', nz >= 0.5), ('side', np.abs(nz) < 0.5), ('down', nz <= -0.5)):
                _acc_conf(strata, f'orient_{name}', w[sel], pred_stick[sel], obs_stick[sel])
        if lam_quartiles is not None:
            qi = np.searchsorted(lam_quartiles, lam[active])
            for b in range(4):
                sel = qi == b
                _acc_conf(strata, f'lamq{b + 1}', w[sel], pred_stick[sel], obs_stick[sel])
    return {'confusion': conf, 'ba': balanced_accuracy(conf),
            'margin_hist': hist, 'margin_hist_stick': hist_stick,
            'excluded_takeoff_mass': excluded_takeoff,
            'strata': {k2: {'confusion': v, 'ba': balanced_accuracy(v)} for k2, v in strata.items()}}


def _acc_conf(strata: dict, key: str, w, pred, obs):
    c = strata.setdefault(key, np.zeros(4, dtype=np.float64))
    c[0] += float(w[pred & obs].sum()); c[1] += float(w[pred & ~obs].sum())
    c[2] += float(w[~pred & ~obs].sum()); c[3] += float(w[~pred & obs].sum())


def balanced_accuracy(conf: np.ndarray):
    """BA = 0.5(TPR_stick + TPR_slip), conditioning on the OBSERVED classes (sensitivity /
    specificity): TPR_stick = w(pred&obs)/w(obs stick), TPR_slip = w(~pred&~obs)/w(obs slip).
    None (N/A) when either an OBSERVED or a PREDICTED class carries zero mass — the mu=0 row is
    N/A via the predicted-stick class being exactly empty (spec, SC-3).
    conf = [w(pred&obs), w(pred&~obs), w(~pred&~obs), w(~pred&obs)]."""
    obs_stick = conf[0] + conf[3]
    obs_slip = conf[1] + conf[2]
    pred_stick = conf[0] + conf[1]
    pred_slip = conf[2] + conf[3]
    if min(obs_stick, obs_slip, pred_stick, pred_slip) <= 0.0:
        return None
    return float(0.5 * (conf[0] / obs_stick + conf[2] / obs_slip))
