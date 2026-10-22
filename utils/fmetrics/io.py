"""fmetrics IO: run manifest, pairing assertions (P11/P18/P19), rollout/flog loading, and the
SINGLE source of unit conversions from the logged displacement form (P3).

Frame conventions (P6, asserted here once):
  * pkl frames: pred[0]=prev_pos, pred[1]=pos (init states), pred[k>=2] = operator output of
    rollout step k-2. obstacle aligned; force[k] and friction_log slice k belong to the step
    that produced pred[k].
  * velocity-sample index t in [0, T-2]: vel[t] = (pred[t+1]-pred[t])/h.  A friction_log slice
    at pkl frame k pairs with velocity sample t = k-1.
  * metric window: t >= t0 = 2 + warmup_steps and t <= T-2.

Derived operator quantities (velocity units; delta_T logged in meters):
  v*  = delta_T / h                 [m/s]   trial relative tangential velocity (convention A)
  v+  = factor * v*                 [m/s]
  j   = m (v+ - v*) = -(1-factor) m delta_T / h        [N*s]  friction impulse
  f_t = j / h                                          [N]
  E_diss = 0.5 m (|v*|^2 - |v+|^2) = 0.5 m (1-factor^2) |delta_T|^2 / h^2   [J]
  f_d^op = (m/h) sqrt(|v*|^2 + eps_op^2)               [N]    operator arrest demand
  s*  = |v*| / sqrt(|v*|^2 + eps_op^2)                 [-]    mollifier fraction
"""
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ------------------------------------------------------------------ sequence discovery / pairing

def discover_sequences(eval_root: Path, run_name: str) -> Dict[str, Path]:
    """{(split/garment/seq_id): pkl path} for one run dir."""
    base = Path(eval_root) / run_name / 'rollouts'
    out = {}
    for pkl in sorted(base.glob('vs_*/*/*.pkl')):
        key = f'{pkl.parent.parent.name}/{pkl.parent.name}/{pkl.stem}'
        out[key] = pkl
    if not out:
        raise FileNotFoundError(f'no rollout pkls under {base}')
    return out


def pair_runs(seq_maps: Dict[str, Dict[str, Path]]) -> List[str]:
    """Intersect sequence keys across runs; assert identical sets (P11 / e2c5a1b hazard)."""
    names = list(seq_maps)
    keys = set(seq_maps[names[0]])
    for n in names[1:]:
        if set(seq_maps[n]) != keys:
            missing = keys.symmetric_difference(seq_maps[n])
            raise ValueError(f'sequence sets differ between {names[0]} and {n}: {sorted(missing)[:6]} ...')
    return sorted(keys)


# ------------------------------------------------------------------ per-sequence loading

@dataclass
class SeqRollout:
    key: str
    pred: np.ndarray            # (T,V,3) f32
    obstacle: np.ndarray        # (T,Vb,3) f32
    cloth_faces: np.ndarray     # (Fc,3) i64
    obstacle_faces: np.ndarray  # (Fo,3) i64
    vertex_type: np.ndarray     # (V,1)
    force: Optional[np.ndarray]
    flog: Optional[dict]        # CSR arrays + v_mass/rest_pos (schema v1) or None
    flog_meta: Optional[dict]
    fail_reason: str
    wall_time: Optional[float]

    @property
    def T(self):
        return self.pred.shape[0]

    @property
    def V(self):
        return self.pred.shape[1]

    def pinned(self):
        from utils.common import NodeType
        return (self.vertex_type.reshape(-1) == NodeType.HANDLE)

    def body_hash(self):
        return hashlib.sha256(self.obstacle_faces.tobytes()).hexdigest()[:12]

    def garment_hash(self):
        h = hashlib.sha256(self.cloth_faces.tobytes())
        if self.flog is not None and 'rest_pos' in self.flog:
            h.update(np.ascontiguousarray(self.flog['rest_pos']).tobytes())
        return h.hexdigest()[:12]


def load_seq(path: Path, key: str = '') -> SeqRollout:
    with open(path, 'rb') as f:
        d = pickle.load(f)
    metrics = d.get('metrics', {}) or {}
    vt = d.get('vertex_type')
    if vt is None:
        # [solver reference] solver-reference pkls (ppf) carry no vertex_type; a per-garment sidecar
        # written from the paired GNN pkl provides pin parity (same 2 HANDLE verts excluded on
        # both arms — the ppf runs drove those same 2 handles, n_handles_driven=2).
        sc = Path(path).parent / '_vertex_type.npy'
        if not sc.exists():
            raise KeyError(f'{path}: no vertex_type in pkl and no sidecar {sc}')
        vt = np.load(sc)
    return SeqRollout(
        key=key or str(path),
        pred=np.asarray(d['pred'], dtype=np.float32),
        obstacle=np.asarray(d['obstacle'], dtype=np.float32),
        cloth_faces=np.asarray(d['cloth_faces'], dtype=np.int64),
        obstacle_faces=np.asarray(d['obstacle_faces'], dtype=np.int64),
        vertex_type=np.asarray(vt),
        force=(np.asarray(d['force'], dtype=np.float32) if 'force' in d else None),
        flog=d.get('friction_log'),
        flog_meta=d.get('friction_log_meta'),
        fail_reason=str(metrics.get('fail_reason', 'SUCCESS')),
        wall_time=(float(metrics['time']) if 'time' in metrics else None),
    )


def assert_paired(seqs: Dict[str, SeqRollout], key: str, obstacle_tol: float = 0.0) -> float:
    """Byte-level pairing invariants across the mu runs of ONE sequence (P11, V9).

    [solver reference] obstacle_tol > 0 relaxes ONLY the obstacle byte-identity check for solver
    arms whose realized body is pin-driven and hence mu-dependent (ppf: p99 ~2 mm at
    extremities). The max cross-run body deviation [m] is returned and must be reported next to
    every number of that arm. 0.0 (default) keeps the strict spec behavior.
    """
    names = list(seqs)
    a = seqs[names[0]]
    max_dev = 0.0
    for n in names[1:]:
        b = seqs[n]
        if a.V != b.V:
            raise ValueError(f'[{key}] vertex count differs: {names[0]}={a.V} vs {n}={b.V}')
        if not np.array_equal(a.cloth_faces, b.cloth_faces):
            raise ValueError(f'[{key}] cloth topology differs between {names[0]} and {n}')
        if a.body_hash() != b.body_hash():
            raise ValueError(f'[{key}] body topology differs between {names[0]} and {n} (P12)')
        Tmin = min(a.T, b.T)
        if not np.array_equal(a.obstacle[:Tmin], b.obstacle[:Tmin]):
            dev = float(np.linalg.norm(
                a.obstacle[:Tmin].astype(np.float64) - b.obstacle[:Tmin].astype(np.float64),
                axis=-1).max())
            if obstacle_tol <= 0.0 or dev > obstacle_tol:
                raise ValueError(f'[{key}] obstacle trajectories are NOT byte-identical '
                                 f'({names[0]} vs {n}, max dev {dev:.4f} m) — paired design broken')
            max_dev = max(max_dev, dev)
    return max_dev


def common_frames(seqs: Dict[str, SeqRollout], key: str, counters: dict) -> int:
    """P19: intersection of available frames; count truncations."""
    Ts = {n: s.T for n, s in seqs.items()}
    Tmin = min(Ts.values())
    if len(set(Ts.values())) > 1:
        counters['truncated_pairs'] = counters.get('truncated_pairs', 0) + 1
        counters.setdefault('truncated_detail', []).append((key, Ts))
    for n, s in seqs.items():
        if s.fail_reason != 'SUCCESS':
            counters['failed_rollouts'] = counters.get('failed_rollouts', 0) + 1
            counters.setdefault('failed_detail', []).append((key, n, s.fail_reason))
    return Tmin


# ------------------------------------------------------------------ friction log access

def flog_slice(flog: dict, k: int) -> dict:
    """Per-pkl-frame slice of the CSR log."""
    o = flog['offsets']
    s, e = int(o[k]), int(o[k + 1])
    return {kk: flog[kk][s:e] for kk in ('vertex', 'face', 'lambda_n', 'delta_T', 'factor', 'x_pre')
            if kk in flog}


def derive_operator_quants(sl: dict, v_mass: np.ndarray, h: float, eps_op: float) -> dict:
    """The single unit-conversion point (P3). All fp64. Input: one flog slice."""
    idx = sl['vertex'].astype(np.int64)
    m = v_mass[idx].astype(np.float64)                       # (n,) kg
    dT = sl['delta_T'].astype(np.float64)                    # (n,3) m
    fac = sl['factor'].astype(np.float64)                    # (n,)
    lam = sl['lambda_n'].astype(np.float64)                  # (n,) N
    v_star = dT / h                                          # (n,3) m/s
    sp = np.linalg.norm(v_star, axis=-1)                     # |v*|
    v_plus = v_star * fac[:, None]
    j = -(1.0 - fac)[:, None] * m[:, None] * dT / h          # N*s
    f_t = j / h                                              # N
    e_diss = 0.5 * m * (1.0 - fac ** 2) * (np.linalg.norm(dT, axis=-1) ** 2) / h ** 2
    reg = np.sqrt(sp ** 2 + eps_op ** 2)
    return dict(vertex=idx, face=sl['face'].astype(np.int64), m=m, lam=lam, factor=fac,
                v_star=v_star, v_plus=v_plus, sp_star=sp, sp_plus=sp * fac,
                f_t=f_t, f_t_norm=np.linalg.norm(f_t, axis=-1), j=j,
                e_diss=e_diss, f_d=m * sp / h, f_d_op=m * reg / h, s_star=sp / reg)
