#!/usr/bin/env python
"""The garment's FREE BOUNDARY -- hem, cuffs, neckline, armholes -- as named, ordered loops.

Why this exists: the panels that work
are the ones where what moves is the EDGE of the cloth, read against the body behind it. Panel 09 is
a cuff on a forearm; panel 12 is the dress hem across the pelvis. Both are legible for the same
reason -- a boundary is a 1-D curve with the body as its backdrop, so a few centimetres of slide
moves a line across a landmark. Interior cloth has no such landmark: it can slide 30 mm and the
silhouette is unchanged, which is exactly what made panels 10/11 blank.

So the selection quantity should not be "how far did the cloth move" but "how far did the cloth's
EDGE move, relative to the body it is drawn against". This module supplies the edge; the metric
lives in select_boundary_frames.py.

Two design points that are not arbitrary:

  * Loops are found from the TOPOLOGY (edges used by one face), never from geometry or a vertex
    list. Every eval garment is an open surface, so its free boundary is exactly its openings.
  * A loop's NAME is the SMPL region its own vertices sit nearest to, majority-voted over a span of
    frames rather than one. One frame is not enough: a dress hem passes thigh -> shin -> thigh in a
    dance sequence, and naming it from t=0 gives a label the caption cannot use. The frame-by-frame
    label is available separately (`nearest_region`) and IS the interesting one for a hem.
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.r3.mu_interaction_map import JOINT_REGION, JOINT_SIDE   # noqa: E402

SMPL_F = REPO / 'ccraft_data/aux_data/body_models/smpl/SMPL_FEMALE.pkl'


def smpl_vertex_labels():
    """(6890,) region string and side letter per SMPL body vertex, by skinning-weight argmax."""
    with open(SMPL_F, 'rb') as f:
        w = np.asarray(pickle.load(f, encoding='latin1')['weights'])
    vj = w.argmax(axis=1)
    return (np.array([JOINT_REGION[int(j)] for j in vj]),
            np.array([JOINT_SIDE[int(j)] for j in vj]))


def boundary_loops(faces, min_len=8):
    """Ordered vertex loops of the free boundary of a triangle mesh.

    An edge used by exactly one face is a boundary edge; boundary vertices have degree 2 within that
    edge set, so the walk is unambiguous. Returns a list of (L,) int arrays, longest first.
    """
    f = np.asarray(faces).astype(np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, cnt = np.unique(key, axis=0, return_counts=True)
    be = uniq[cnt == 1]
    adj = defaultdict(list)
    for a, b in be:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [v for v in adj[cur] if v != prev]
            nxt = [v for v in nxt if v not in seen] or [v for v in nxt if v == start]
            if not nxt:
                break
            v = nxt[0]
            if v == start:
                break
            loop.append(v)
            seen.add(v)
            prev, cur = cur, v
        if len(loop) >= min_len:
            loops.append(np.array(loop, np.int64))
    return sorted(loops, key=len, reverse=True)


def nearest_region(pts, body, vreg, vside):
    """Majority (region, side) of the body vertices nearest to `pts`, plus the vote fraction."""
    from scipy.spatial import cKDTree
    _, idx = cKDTree(body).query(pts)
    lab = np.array([f'{r}{s}' if s != 'C' else f'{r}C'
                    for r, s in zip(vreg[idx], vside[idx])])
    vals, cnt = np.unique(lab, return_counts=True)
    k = int(cnt.argmax())
    return str(vals[k]), float(cnt[k] / len(lab))


def name_loops(loops, cloth, body, frames=None):
    """Stable names for the loops of one garment.

    `cloth` (T,V,3) and `body` (T,6890,3). The label is the majority nearest-body region over
    `frames` (default: 8 frames spread over the sequence) -- see the module docstring for why one
    frame is not enough. Ties in the label get a numeric suffix so names stay unique.
    """
    vreg, vside = smpl_vertex_labels()
    T = min(len(cloth), len(body))
    if frames is None:
        frames = np.linspace(0, T - 1, min(8, T)).astype(int)
    out = []
    for lp in loops:
        votes = defaultdict(float)
        for t in frames:
            lab, frac = nearest_region(cloth[t][lp], body[t], vreg, vside)
            votes[lab] += frac
        lab = max(votes, key=votes.get)
        out.append(dict(idx=lp, label=lab, n=len(lp)))
    seen = defaultdict(int)
    for d in out:
        seen[d['label']] += 1
        if seen[d['label']] > 1:
            d['label'] = f"{d['label']}#{seen[d['label']]}"
    return out


def loop_scale(pts):
    """Diameter of a boundary loop, metres -- the natural yardstick for 'is this shift legible'.

    A 30 mm slide is a third of a cuff opening and a fifteenth of a dress hem. The zoom crop is
    framed on the loop, so the loop's own size is what the shift is read against on the page.
    """
    return float(np.linalg.norm(np.asarray(pts).max(0) - np.asarray(pts).min(0)))
