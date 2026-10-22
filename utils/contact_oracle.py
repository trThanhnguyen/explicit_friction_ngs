"""Independent point-body distance oracle — DIAGNOSTICS ONLY.

Signed closest point-to-triangle distance to the body mesh via Warp's
`mesh_query_point_sign_normal` (angle-weighted pseudonormal sign; SMPL body is closed, so the
sign is reliable away from exact surface degeneracies). This is deliberately a DIFFERENT geometry
routine from the nearest-face-centroid plane-gap proxy that drives the normal completion and
the pen_rate_strict metric: per the 2026-07-14 review, the correction must not validate itself
using the same approximation it manipulates.

NEVER used inside the friction/normal-completion operator — validation and reporting only.

API verified on warp 1.14.0 (this host) with a unit-cube probe: sign < 0 inside, closest point
from mesh_eval_position, exact distances. NOTE the cross-host warp-version gotcha (1.12 vs 1.14):
re-run this module's checks after any warp version bump.
"""
import torch
import warp as wp

wp.init()


@wp.kernel
def _signed_point_mesh_kernel(mesh_id: wp.uint64, pts: wp.array(dtype=wp.vec3),
                              max_dist: float,
                              out_d: wp.array(dtype=float), out_f: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh_id, pts[tid], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        out_d[tid] = q.sign * wp.length(pts[tid] - cp)
        out_f[tid] = q.face
    else:
        out_d[tid] = max_dist
        out_f[tid] = -1


@wp.kernel
def _closest_point_bary_kernel(mesh_id: wp.uint64, pts: wp.array(dtype=wp.vec3),
                               max_dist: float,
                               out_d: wp.array(dtype=float), out_f: wp.array(dtype=wp.int32),
                               out_u: wp.array(dtype=float), out_v: wp.array(dtype=float)):
    tid = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh_id, pts[tid], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        out_d[tid] = q.sign * wp.length(pts[tid] - cp)
        out_f[tid] = q.face
        out_u[tid] = q.u
        out_v[tid] = q.v
    else:
        out_d[tid] = max_dist
        out_f[tid] = -1
        out_u[tid] = 0.0
        out_v[tid] = 0.0


def signed_point_mesh_distance(points, mesh_verts, mesh_faces, max_dist=1.0):
    """Signed closest-point distance from `points` to the closed triangle mesh.

    :param points:     (N,3) float tensor (cuda)
    :param mesh_verts: (V,3) float tensor (cuda)
    :param mesh_faces: (F,3) long/int tensor (cuda)
    :param max_dist:   query radius; points with no hit inside it get d=max_dist, face=-1
    :return: (signed_dist (N,), closest_face (N,) int32)  — d<0 means inside the body.
    """
    pts = points.detach().float().contiguous()
    mesh = wp.Mesh(points=wp.from_torch(mesh_verts.detach().float().contiguous(), dtype=wp.vec3),
                   indices=wp.from_torch(mesh_faces.detach().reshape(-1).contiguous().int()))
    n = pts.shape[0]
    out_d = torch.empty(n, dtype=torch.float32, device=pts.device)
    out_f = torch.empty(n, dtype=torch.int32, device=pts.device)
    wp.launch(_signed_point_mesh_kernel, dim=n, device=str(pts.device),
              inputs=[mesh.id, wp.from_torch(pts, dtype=wp.vec3), float(max_dist),
                      wp.from_torch(out_d), wp.from_torch(out_f)])
    wp.synchronize_device(str(pts.device))
    return out_d, out_f


def closest_point_query(points, mesh_verts, mesh_faces, max_dist=1.0):
    """Signed closest-point query WITH the barycentric correspondence (fmetrics convention B).

    Returns (sd (N,), face (N,) int32, u (N,), v (N,)) where the closest point reconstructs as
      cp = u*p0 + v*p1 + (1-u-v)*p2   with (p0,p1,p2) = mesh_verts[mesh_faces[face]]
    (warp 1.14 barycentric convention, verified empirically on this host 2026-07-15 against
    wp.mesh_eval_position; the fmetrics unit-test gate re-asserts it). sd<0 = inside; misses get
    sd=max_dist, face=-1, u=v=0.

    Validation-and-metrics only — never drives the friction/normal-completion operator.
    """
    pts = points.detach().float().contiguous()
    mesh = wp.Mesh(points=wp.from_torch(mesh_verts.detach().float().contiguous(), dtype=wp.vec3),
                   indices=wp.from_torch(mesh_faces.detach().reshape(-1).contiguous().int()))
    n = pts.shape[0]
    out_d = torch.empty(n, dtype=torch.float32, device=pts.device)
    out_f = torch.empty(n, dtype=torch.int32, device=pts.device)
    out_u = torch.empty(n, dtype=torch.float32, device=pts.device)
    out_v = torch.empty(n, dtype=torch.float32, device=pts.device)
    wp.launch(_closest_point_bary_kernel, dim=n, device=str(pts.device),
              inputs=[mesh.id, wp.from_torch(pts, dtype=wp.vec3), float(max_dist),
                      wp.from_torch(out_d), wp.from_torch(out_f),
                      wp.from_torch(out_u), wp.from_torch(out_v)])
    wp.synchronize_device(str(pts.device))
    return out_d, out_f, out_u, out_v


def eval_bary_points(mesh_verts, mesh_faces, face_idx, u, v):
    """Evaluate material points (face, u, v) on a (possibly different-pose) mesh with the SAME
    topology: p = u*p0 + v*p1 + (1-u-v)*p2. Used to advect convention-B correspondences to the
    next body pose. face_idx=-1 (miss) rows return zeros — callers must mask them."""
    valid = (face_idx >= 0)
    f = face_idx.long().clamp(min=0)
    tri = mesh_verts[mesh_faces[f].long()]                       # (N,3,3)
    w2 = 1.0 - u - v
    p = tri[:, 0] * u.unsqueeze(-1) + tri[:, 1] * v.unsqueeze(-1) + tri[:, 2] * w2.unsqueeze(-1)
    return p * valid.unsqueeze(-1)
