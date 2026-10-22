#!/usr/bin/env python
"""Which garment x BODY-PART interaction does mu actually change the most?

The selection ranked whole (cell, mu-pair) panels on aggregate displacement. The owner
rejected the result: aggregate motion is not the same thing as a legible contact event, and the
ranking had no term for penetration or for whether the pose is presentable. This asks the question
the other way round -- semantically, per contact -- so candidates arrive with a NAME:

    "dress x thigh", "longsleeve x forearm", "tank x torso", ...

Method, reusing the shipped friction suite's own estimators so the answer cannot drift from
Sec. 5.2.1's numbers:
  * contact set, window and pinned mask: identical to an internal tool (sd_proxy <= r_c,
    t >= t0, pinned excluded).
  * body part per contact: the suite already caches `faceB`, the nearest BODY FACE per cloth vertex
    per frame. Map face -> its 3 body vertices -> SMPL skinning-weight argmax -> joint -> a readable
    region. This is the same labelling trick p19_hold_retention.py uses for the hood
    ("lbs_weights: head = 1.00"), generalised to all 24 joints.
  * per (arm, mu, cell, region): mass-time weight, mass-time-weighted stick@0.002 and mean slip S.

Outputs one long CSV; rank with --report.

Run:
  pixi run python tools/r3/mu_interaction_map.py --out <csv>
  pixi run python tools/r3/mu_interaction_map.py --out <csv> --report
"""
import argparse
import csv
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.fmetrics import masks as fmask                        # noqa: E402
from utils.fmetrics.io import discover_sequences, load_seq       # noqa: E402

EPS_STICK = 2e-3      # 1e-3 * l, IPC default at the measured bbox diagonal
R_C = 3e-3
T0 = 2
MUS = ['0.0', '0.1', '0.3', '0.6']

# arm -> {mu: eval directory}, and its analysis cache. Both are deliberately EMPTY in this
# distribution: the rollout trees this module's own analysis reads are not shipped. Only
# JOINT_REGION / JOINT_SIDE below are imported by the renderers.
ARMS = {}
CACHE = {}

# SMPL's 24 joints grouped into regions a caption can name.
JOINT_REGION = {
    0: 'pelvis', 1: 'thigh', 2: 'thigh', 3: 'torso', 4: 'shin', 5: 'shin', 6: 'torso',
    7: 'foot', 8: 'foot', 9: 'torso', 10: 'foot', 11: 'foot', 12: 'neck',
    13: 'shoulder', 14: 'shoulder', 15: 'head', 16: 'upper arm', 17: 'upper arm',
    18: 'forearm', 19: 'forearm', 20: 'hand', 21: 'hand', 22: 'hand', 23: 'hand',
}


# Left/right of each SMPL joint, so a region can be restricted to ONE limb. 'forearm' spans both
# arms, whose bounding box is ~1 m wide -- aiming a zoom at it gives a 3/4-body crop, not a close-up.
JOINT_SIDE = {**{j: 'L' for j in (1, 4, 7, 10, 13, 16, 18, 20, 22)},
              **{j: 'R' for j in (2, 5, 8, 11, 14, 17, 19, 21, 23)},
              **{j: 'C' for j in (0, 3, 6, 9, 12, 15)}}


def body_face_region(with_side=False):
    """(F,) readable region label id per SMPL body face, via skinning-weight argmax.

    with_side=True additionally returns (F,) side letters, so callers can score / crop one limb.
    """
    with open(REPO / 'ccraft_data/aux_data/body_models/smpl/SMPL_FEMALE.pkl', 'rb') as f:
        smpl = pickle.load(f, encoding='latin1')
    w = np.asarray(smpl['weights'])                       # (6890,24)
    faces = np.asarray(smpl['f']).astype(np.int64)         # (13776,3)
    vjoint = w.argmax(axis=1)                             # (6890,)
    regions = sorted(set(JOINT_REGION.values()))
    rid = {r: i for i, r in enumerate(regions)}
    vreg = np.array([rid[JOINT_REGION[int(j)]] for j in vjoint])
    # a face's region = the majority of its three vertices (ties -> first, deterministic)
    fr = vreg[faces]                                      # (F,3)
    out = np.empty(len(faces), dtype=np.int32)
    for i in range(len(faces)):
        vals, cnt = np.unique(fr[i], return_counts=True)
        out[i] = vals[cnt.argmax()]
    if not with_side:
        return out, regions
    vside = np.array([JOINT_SIDE[int(j)] for j in vjoint])
    fs = vside[faces]
    side = np.empty(len(faces), dtype='<U1')
    for i in range(len(faces)):
        vals, cnt = np.unique(fs[i], return_counts=True)
        side[i] = vals[cnt.argmax()]
    return out, regions, side


def cache_of(cache_dir, run_name, key, pred):
    tag = hashlib.sha256(pred[:8].tobytes() + pred[-1].tobytes()).hexdigest()[:16]
    return np.load(cache_dir / f"{run_name}__{key.replace('/', '__')}__{tag}.npz")


def build(out_csv):
    freg, regions = body_face_region()
    base = REPO / 'ccraft_data/eval'
    eval_root = REPO / 'ccraft_data/eval'
    rows = []
    for arm, runs in ARMS.items():
        cache_dir = base / CACHE[arm] / 'cache'
        seq_maps = {m: discover_sequences(eval_root, rn) for m, rn in runs.items()}
        keys = sorted(seq_maps['0.0'])
        for key in keys:
            garment = key.split('/')[1]
            mz = np.load(base / 'masses' / f'{garment}.npz')
            m_v = mz['v_mass'].astype(np.float64)
            for mu, rn in runs.items():
                s = load_seq(seq_maps[mu][key], key)
                cz = cache_of(cache_dir, rn, key, s.pred)
                Tv = s.T - 1
                wm = fmask.window_mask(Tv, s.V, T0)
                c = fmask.contact_mask(cz['sd_proxy'], R_C, s.pinned())[:Tv] & wm
                vT = cz['vT'][:Tv].astype(np.float64)
                faceB = cz['faceB'][:Tv]
                sd = cz['sd_oracle'][:Tv] if 'sd_oracle' in cz.files else cz['sd_proxy'][:Tv]
                reg = np.where(faceB >= 0, freg[np.clip(faceB, 0, None)], -1)
                W = c * m_v[None, :]
                for r_i, r_name in enumerate(regions):
                    sup = W * (reg == r_i)
                    den = float(sup.sum())
                    if den <= 0:
                        continue
                    rows.append(dict(
                        arm=arm, mu=mu, cell=key, garment=garment,
                        motion=key.split('/')[2], region=r_name, masstime=den,
                        stick=float((sup * (vT <= EPS_STICK)).sum() / den),
                        S=float((sup * vT).sum() / den),
                        pen_rate=float((sup * (sd[:, :] < 0)).sum() / den),
                    ))
            print(f'[map] {arm} {key} done', flush=True)
    cols = list(rows[0])
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'[map] wrote {out_csv} ({len(rows)} rows)')


def report(out_csv):
    import pandas as pd
    t = pd.read_csv(out_csv)
    pd.set_option('display.width', 240)

    def pooled(df, col):
        w = df['masstime'].values
        return float(np.average(df[col].values, weights=w)) if w.sum() > 0 else np.nan

    print('\n=== mu-response by GARMENT x BODY REGION (pooled over all 40 cells) ===')
    print(f"{'garment':11s} {'region':10s} {'mass-time':>10s}  "
          f"{'ppf stick mu0->0.6':>20s}  {'ours stick mu0->0.6':>21s}  {'ppf drop':>9s} {'ours drop':>10s}")
    out = []
    for (g, r), grp in t.groupby(['garment', 'region']):
        mt = grp[grp.arm == 'ppf']['masstime'].sum()
        row = {'garment': g, 'region': r, 'masstime': mt}
        for arm in ('ppf', 'ours'):
            a = grp[grp.arm == arm]
            for mu in ('0.0', '0.6'):
                row[f'{arm}_{mu}'] = pooled(a[a.mu.astype(str) == mu], 'stick')
            s0 = pooled(a[a.mu.astype(str) == '0.0'], 'S')
            s6 = pooled(a[a.mu.astype(str) == '0.6'], 'S')
            row[f'{arm}_drop'] = (s0 - s6) / s0 if s0 else np.nan
        row['resp_ppf'] = row['ppf_0.6'] - row['ppf_0.0']
        row['resp_ours'] = row['ours_0.6'] - row['ours_0.0']
        out.append(row)
    d = pd.DataFrame(out).sort_values('resp_ppf', ascending=False)
    tot = d['masstime'].sum()
    for _, r in d.iterrows():
        print(f"{r['garment']:11s} {r['region']:10s} {100*r['masstime']/tot:9.2f}%  "
              f"{r['ppf_0.0']:8.3f} -> {r['ppf_0.6']:7.3f}  "
              f"{r['ours_0.0']:9.3f} -> {r['ours_0.6']:7.3f}  "
              f"{r['ppf_drop']:8.2f} {r['ours_drop']:9.2f}   resp ppf {r['resp_ppf']:+.3f} "
              f"ours {r['resp_ours']:+.3f}")

    print('\n=== the same, pooled over garments: which BODY REGION carries the mu effect ===')
    for _, r in d.groupby('region').apply(
            lambda g: pd.Series({'mt': g.masstime.sum(),
                                 'resp_ppf': np.average(g.resp_ppf, weights=g.masstime),
                                 'resp_ours': np.average(g.resp_ours, weights=g.masstime)})
    ).sort_values('resp_ppf', ascending=False).reset_index().iterrows():
        print(f"  {r['region']:10s} mass-time {100*r['mt']/tot:6.2f}%   "
              f"resp ppf {r['resp_ppf']:+.3f}   resp ours {r['resp_ours']:+.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', action='store_true', help='skip the build, read the CSV')
    a = ap.parse_args()
    if not a.report:
        build(a.out)
    report(a.out)
