"""fmetrics geometry: the two correspondence conventions (spec, decision D-3) + Kabsch +
face-ring tables.

CONVENTION GUARD (D-3): quantities from the operator's frame (centroid correspondence) and the
trajectory observable (advected barycentric material point) are produced by DIFFERENT functions
returning DIFFERENTLY-NAMED containers and must never be mixed. Tier-1/guardrails consume
TrajQuants (B); Tier-2 predicted / Tier-3 consume the friction_log (A) via io.derive_operator_
quants. GPU is used only for deterministic geometry queries; all statistics are accumulated in
fp64 on CPU by the callers.
"""
from dataclasses import dataclass

import numpy as np
import torch


# ------------------------------------------------------------------ convention A (operator)

@torch.no_grad()
def plane_gap_sd(pred, obst, faces, chunk=24):
    """(T,V) signed nearest-face-CENTROID plane gap — the operator's/pen_rate_strict's proxy
    (mirrors utils.contact.node_face_signed_distance, batched over frames; logic follows
    an internal tool, reimplemented metric-side so validated diag tools
    stay untouched). sd < 0 = inside the body."""
    from pytorch3d.ops import knn_points
    from utils.cloth_and_material import FaceNormals
    f_normals = FaceNormals().to(pred.device)
    sds = []
    for i in range(0, pred.shape[0], chunk):
        p, b = pred[i:i + chunk], obst[i:i + chunk]
        cen = b[:, faces].mean(dim=2)
        _, idx, _ = knn_points(p, cen)
        fn = f_normals(b, faces.unsqueeze(0).expand(b.shape[0], -1, -1))
        nn_cen = torch.gather(cen, 1, idx.expand(-1, -1, 3))
        nn_fn = torch.gather(fn, 1, idx.expand(-1, -1, 3))
        sds.append(((p - nn_cen) * nn_fn).sum(dim=-1))
    return torch.cat(sds, dim=0)


@torch.no_grad()
def face_normals_at(obst_frame, faces):
    """(F,3) unit outward face normals of one body pose (convention A / flog face ids)."""
    from utils.cloth_and_material import FaceNormals
    return FaceNormals().to(obst_frame.device)(obst_frame.unsqueeze(0), faces.unsqueeze(0))[0]


# ------------------------------------------------------------------ convention B (trajectory)

@dataclass
class TrajQuants:
    """Convention-B per-sequence trajectory observables (velocity-sample frame space:
    index t in [0, T-2] pairs pred[t] -> pred[t+1]).  All numpy on CPU.

    vT      (T-1, V) f32  ||v^T|| body-relative tangential slip speed [m/s]
    faceB   (T-1, V) i32  closest body face id at time t (oracle; -1 = miss)
    sd_oracle (T, V) f32  warp point-triangle signed distance per frame [m] (<0 inside)
    """
    vT: np.ndarray
    faceB: np.ndarray
    sd_oracle: np.ndarray


@torch.no_grad()
def convention_b_quants(pred, obst, faces, h, max_dist=0.5) -> TrajQuants:
    """The advected-material-point slip observable (spec convention B).

    Per t: closest point of pred[t] on body[t] via the warp oracle (face, u, v; barycentric
    convention cp = u*p0 + v*p1 + (1-u-v)*p2, verified on warp 1.14); the SAME material point
    evaluated on body[t+1] gives the body velocity; n = face normal at t; project.
    Finite-gap bound (math audit E1): a rigidly co-moving vertex at gap g reads ||vT|| <= omega*g
    — tests pin the glued case."""
    from utils.contact_oracle import closest_point_query, eval_bary_points
    T, V = pred.shape[0], pred.shape[1]
    vT = np.zeros((T - 1, V), dtype=np.float32)
    faceB = np.full((T - 1, V), -1, dtype=np.int32)
    sd_all = np.zeros((T, V), dtype=np.float32)
    for t in range(T):
        sd, face, u, v = closest_point_query(pred[t], obst[t], faces, max_dist=max_dist)
        sd_all[t] = sd.cpu().numpy()
        if t >= T - 1:
            break
        p_t = eval_bary_points(obst[t], faces, face, u, v)             # (V,3) material point at t
        p_adv = eval_bary_points(obst[t + 1], faces, face, u, v)       # same point advected to t+1
        n = face_normals_at(obst[t], faces)[face.long().clamp(min=0)]  # (V,3), normal AT TIME t
        rel = (pred[t + 1] - pred[t]) - (p_adv - p_t)                  # relative displacement [m]
        rel_t = rel - (rel * n).sum(dim=-1, keepdim=True) * n          # tangent-plane projection
        vt = rel_t.norm(dim=-1) / h
        miss = face < 0
        vt = torch.where(miss, torch.zeros_like(vt), vt)               # misses carry no sample
        vT[t] = vt.cpu().numpy()
        faceB[t] = face.cpu().numpy()
    return TrajQuants(vT=vT, faceB=faceB, sd_oracle=sd_all)


# ------------------------------------------------------------------ root frame (D-4)

def kabsch_root(obst_np, ref_frame=1):
    """Per-frame rigid fit (R_t, r_t): x_ref ~= R_t^T (x_t - r_t) formulation of spec
    Fits obst[t] -> obst[ref_frame] with det(R)=+1 enforced (reflection guard).
    Returns (R (T,3,3), r (T,3)) float64 such that tilde_x = R[t].T @ (x - r[t]) maps frame-t
    world points into the reference body frame. Deterministic (CPU float64 SVD)."""
    P0 = obst_np[ref_frame].astype(np.float64)
    c0 = P0.mean(axis=0)
    Q0 = P0 - c0
    T = obst_np.shape[0]
    R = np.zeros((T, 3, 3)); r = np.zeros((T, 3))
    for t in range(T):
        Pt = obst_np[t].astype(np.float64)
        ct = Pt.mean(axis=0)
        H = (Pt - ct).T @ Q0                       # maps ref -> t
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R[t] = U @ D @ Vt                          # x_t ~= R[t] @ x_ref + trans
        r[t] = ct - R[t] @ c0
    return R, r


def root_frame_speeds(pred_np, R, r, h):
    """(T-1, V) ||v_root|| [m/s]: garment velocity in the body root frame (spec)."""
    T = pred_np.shape[0]
    tilde = np.einsum('tij,tvj->tvi', np.transpose(R, (0, 2, 1)),
                      pred_np.astype(np.float64) - r[:, None, :])
    return (np.linalg.norm(tilde[1:] - tilde[:-1], axis=-1) / h).astype(np.float64)[:T - 1]


# ------------------------------------------------------------------ face rings (D-5, P12)

def face_ring_table(faces_np, k_max=3):
    """Face k-ring distances on VERTEX-SHARED adjacency (D-5). Returns rings: list over faces of
    dict{face_id: ring_distance<=k_max}. Precomputed once per body topology (P12)."""
    F = faces_np.shape[0]
    v2f = {}
    for f in range(F):
        for v in faces_np[f]:
            v2f.setdefault(int(v), []).append(f)
    nbrs = [set() for _ in range(F)]
    for f in range(F):
        for v in faces_np[f]:
            nbrs[f].update(v2f[int(v)])
        nbrs[f].discard(f)
    rings = []
    for f in range(F):
        dist = {f: 0}
        frontier = {f}
        for k in range(1, k_max + 1):
            nxt = set()
            for g in frontier:
                nxt.update(nbrs[g])
            nxt -= dist.keys()
            for g in nxt:
                dist[g] = k
            frontier = nxt
            if not frontier:
                break
        rings.append(dist)
    return rings


def ring_distance(rings, fa, fb, k_max=3):
    """Ring distance between two faces, or k_max+1 if farther / either is a miss (-1)."""
    if fa < 0 or fb < 0:
        return k_max + 1
    return rings[fa].get(fb, k_max + 1)


def edge_rest_and_current(faces_np, rest_pos_np, pos_np):
    """Unique garment edges + rest/current lengths for the stretch guardrail (spec)."""
    e = np.concatenate([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [0, 2]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    lr = np.linalg.norm(rest_pos_np[e[:, 0]] - rest_pos_np[e[:, 1]], axis=-1)
    lc = np.linalg.norm(pos_np[:, e[:, 0]] - pos_np[:, e[:, 1]], axis=-1)  # (T, E)
    return e, lr, lc
