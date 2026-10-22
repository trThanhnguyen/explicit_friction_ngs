"""
Cubic-barrier Coulomb friction helpers for the bolt-on operator inference-time projection.

Ported from HOOD utils/contact.py.
Only the two Arm-BO-relevant functions are included here; NCP / force-head helpers
live in HOOD only (not used in ContourCraft's architecture).
"""
import numpy as np
import torch

from utils.timeprof import PROF

# diagnostics side-channel (an internal tool): when set to a list, the
# normal-completion block appends one dict per example per step with the candidate subset
# (indices, pre/post positions). Default None => zero effect on the operator.
NE_DEBUG_SINK = None


def node_face_signed_distance(cloth_pos, body_pos, body_faces):
    """Per-cloth-vertex signed distance to the nearest body face (>0 = outside, <0 = penetrating).

    Mirrors the KNN geometry used in bolt_on_friction_projection: nearest face by CENTROID in the
    current frame, signed distance along that face's outward normal. Used by the penetration-guard
    tool (check_bolt_on_penetration.py) to compute collision_penalty_perc and pen_rate_strict.

    :param cloth_pos:  (Vc, 3) torch float tensor
    :param body_pos:   (Vb, 3) torch float tensor
    :param body_faces: (Fo, 3) torch long tensor (face vertex indices into body_pos)
    :return: sd (Vc,) signed distance per cloth vertex
    """
    from pytorch3d.ops import knn_points
    from utils.cloth_and_material import FaceNormals

    device = cloth_pos.device
    f_normals_f = FaceNormals().to(device)
    face_centroid = body_pos[body_faces].mean(dim=1)          # (Fo, 3)
    _, nn_idx, _ = knn_points(cloth_pos.unsqueeze(0), face_centroid.unsqueeze(0), return_nn=True)
    nn_idx = nn_idx[0]                                         # (Vc, 1)
    nn_cen = face_centroid[nn_idx[:, 0]]                       # (Vc, 3)
    fn = f_normals_f(body_pos.unsqueeze(0), body_faces.unsqueeze(0))[0]  # (Fo, 3)
    nn_fn = fn[nn_idx[:, 0]]                                   # (Vc, 3)
    sd = ((cloth_pos - nn_cen) * nn_fn).sum(dim=-1)            # (Vc,) signed distance
    return sd


def coulomb_block_soft_threshold(delta_c, delta_b, n_hat, lambda_n, m, mu, h, eps_v):
    """Pure block-soft-threshold Coulomb friction projection on the tangential RELATIVE displacement
    (cubic-barrier proposal), in HOOD's *displacement* units. No pyg/geometry dependency, so
    this is the directly unit-testable core.

    Inputs are (N,3) / (N,1) tensors (`mu` may be a float or (N,1)):
      delta_c : cloth tentative per-step displacement (pred_pos - pos)
      delta_b : body displacement at the contact (target_pos - pos of the nearest face/point)
      n_hat   : unit contact normal (outward body normal)
      lambda_n: barrier normal force magnitude (>= 0)
      m       : nodal mass (> 0);  h: timestep;  eps_v: static-velocity band
    The normal component and the body motion are preserved exactly; only the tangential cloth-minus-body
    part is shrunk by  factor = max(0, 1 - mu*lambda_n*h^2 / (m * ||Delta_T||_eps)),  with
    ||Delta_T||_eps = sqrt(||Delta_T||^2 + (eps_v*h)^2). Note the h^2 (displacement units).

    :return: (delta_c_new (N,3) new per-step displacement from pos, factor (N,1) in [0,1],
              slip_pre (N,1) ||Delta_T||, slip_post (N,1) ||Delta_T_new||)
    """
    delta_rel = delta_c - delta_b
    rel_n = (delta_rel * n_hat).sum(dim=-1, keepdim=True) * n_hat          # normal part (preserved)
    delta_T = delta_rel - rel_n                                            # tangential relative
    slip = torch.norm(delta_T, dim=-1, keepdim=True)
    reg = torch.sqrt(slip * slip + (eps_v * h) ** 2)                       # static-band regularized
    factor = torch.clamp(1.0 - mu * lambda_n * (h * h) / (m * reg), min=0.0)
    delta_T_new = delta_T * factor
    delta_c_new = delta_b + rel_n + delta_T_new                           # reassemble (normal+body kept)
    slip_post = torch.norm(delta_T_new, dim=-1, keepdim=True)
    return delta_c_new, factor, slip, slip_post


# ---------------------------------------------------------------------------
# Laplace hoop pressure p = tr(T+ K) as the elastic lambda_n
# channel. Two pure, geometry-library-free helpers so they are CPU/float64
# unit-testable without any PyG fixture (same discipline as
# coulomb_block_soft_threshold above). Neither touches autograd; callers must
# still detach, because the eval rollout is NOT under no_grad.
# ---------------------------------------------------------------------------

def _sym2x2_eig(a, b, d, eps=1e-30):
    """Closed-form eigendecomposition of the symmetric 2x2 [[a,b],[b,d]].

    There is no torch.linalg.eigh anywhere in this repo, and we do not
    want one: a batched 2x2 in closed form is both faster and free of autograd surprises.

    :return: (g1, g2, v1, v2) with g1 >= g2, v1/v2 (...,2) orthonormal eigenvectors.
    """
    half_tr = 0.5 * (a + d)
    # sqrt((a-d)^2/4 + b^2) -- written this way (not via the determinant) so the radicand is a sum
    # of squares and can never go negative through cancellation.
    disc = torch.sqrt(torch.clamp(0.25 * (a - d) ** 2 + b * b, min=0.0))
    g1 = half_tr + disc
    g2 = half_tr - disc

    # Eigenvector for g1. Every nonzero COLUMN of (A - g2 I) lies in g1's eigenspace, so take the
    # two columns  c0 = (a - g2, b)  and  c1 = (b, d - g2)  and keep whichever is longer.
    #
    # Using a single candidate is WRONG and fails silently: c0 alone vanishes exactly when b == 0
    # and d > a (a diagonal G whose LARGER eigenvalue sits on the y axis), and a naive fallback then
    # returns the x axis -- i.e. the eigenvector of the SMALLER eigenvalue. On a cylindrical sleeve
    # that mis-assigns the hoop tension on ~half the faces (the two diagonal splits of each quad),
    # silently reporting T_minor/R instead of T_major/R. Caught by an internal tool B10.
    c0x, c0y = a - g2, b
    c1x, c1y = b, d - g2
    n0 = c0x * c0x + c0y * c0y
    n1 = c1x * c1x + c1y * c1y
    take0 = n0 >= n1
    vx = torch.where(take0, c0x, c1x)
    vy = torch.where(take0, c0y, c1y)
    nrm = torch.sqrt(vx * vx + vy * vy)
    # Both columns vanish only when a == d and b == 0: G is isotropic, EVERY direction is an
    # eigenvector, and the x axis is as good as any.
    degen = nrm <= eps
    vx = torch.where(degen, torch.ones_like(vx), vx / torch.clamp(nrm, min=eps))
    vy = torch.where(degen, torch.zeros_like(vy), vy / torch.clamp(nrm, min=eps))
    v1 = torch.stack([vx, vy], dim=-1)
    v2 = torch.stack([-vy, vx], dim=-1)          # perpendicular, so v1 . v2 == 0 exactly
    return g1, g2, v1, v2


def principal_membrane_tension(triangles, Dm_inv, lame_mu, lame_lambda, thickness,
                               tensile_only=True):
    """Per-face TENSILE Cauchy membrane stress resultant T+ [N/m], as a world 3x3 tensor.

    This is the cloth half of the elastic normal-load channel. It mirrors
    criterions/postcvpr/mataug/stretching_energy.py exactly up to the Green strain, then departs
    from it in TWO ways that are easy to get wrong:

      1. The criterion's `S = mu*G + 0.5*lam*tr(G)*I` is NOT the second Piola-Kirchhoff stress. It
         is an algebraic half-stress that exists only so the energy density can be written
         tr(S^T G). Since psi = mu*||G||_F^2 + 0.5*lam*tr(G)^2, the true PK2 is
         dpsi/dG = 2*mu*G + lam*tr(G)*I = 2*S_criterion.  We use the TRUE one.
         mislabels S_criterion as PK2 -- do not follow it.
      2. Laplace's law p = T*kappa is a CURRENT-configuration statement, so T must be force per
         unit CURRENT length. The push-forward of the reference resultant N = t*S_PK2 is
         n = (1/J) F N F^T, whose principal values are (t/J)*c_i*s_i with c_i = 1 + 2*g_i the
         squared principal stretches and J = sqrt(c1*c2).

    Because G's eigenvectors are also eigenvectors of C = F^T F, the pushed-forward directions
    F v_i are mutually orthogonal (|F v_i|^2 = c_i), so T+ stays a well-formed symmetric tensor.

    :param triangles:    (F,3,3) world positions of each face's 3 vertices (CURRENT pose)
    :param Dm_inv:       (F,2,2) inverse rest shape matrix (sample['cloth'].Dm_inv)
    :param lame_mu:      scalar or (F,) Lame mu [Pa]
    :param lame_lambda:  scalar or (F,) Lame lambda [Pa]
    :param thickness:    cloth thickness [m] (the stretching criterion's Config.thickness)
    :param tensile_only: clamp principal tensions at 0 (buckled/compressive directions carry no
                         transverse load -- the closed-form analogue of Ando's PSD projection)
    :return: (T_plus (F,3,3) [N/m], J (F,) current/rest area ratio, T1 (F,), T2 (F,) [N/m])
    """
    from utils.cloth_and_material import get_shape_matrix

    Ds = get_shape_matrix(triangles)                            # (F,3,2)
    F_def = Ds @ Dm_inv                                         # (F,3,2)

    C = F_def.transpose(-1, -2) @ F_def                         # (F,2,2) right Cauchy-Green
    I2 = torch.eye(2, dtype=C.dtype, device=C.device)
    G = 0.5 * (C - I2)                                          # (F,2,2) Green strain

    g1, g2, v1, v2 = _sym2x2_eig(G[..., 0, 0], G[..., 0, 1], G[..., 1, 1])

    if not torch.is_tensor(lame_mu):
        lame_mu = torch.as_tensor(lame_mu, dtype=C.dtype, device=C.device)
    if not torch.is_tensor(lame_lambda):
        lame_lambda = torch.as_tensor(lame_lambda, dtype=C.dtype, device=C.device)
    mu_f = lame_mu.reshape(-1) if lame_mu.numel() > 1 else lame_mu.reshape(())
    la_f = lame_lambda.reshape(-1) if lame_lambda.numel() > 1 else lame_lambda.reshape(())

    tr_G = g1 + g2
    s1 = 2.0 * mu_f * g1 + la_f * tr_G                          # TRUE PK2 principal stresses [Pa]
    s2 = 2.0 * mu_f * g2 + la_f * tr_G

    # Squared principal stretches. Floored: c_i <= 0 means the face has folded through itself
    # (g_i <= -0.5), which is not a physical membrane state and must not produce a NaN J.
    c1 = torch.clamp(1.0 + 2.0 * g1, min=1e-12)
    c2 = torch.clamp(1.0 + 2.0 * g2, min=1e-12)
    J = torch.sqrt(c1 * c2)

    T1 = (thickness / J) * c1 * s1                              # Cauchy membrane tension [N/m]
    T2 = (thickness / J) * c2 * s2

    # Push the material eigen-directions forward to world and normalise (|F v_i| = sqrt(c_i)).
    w1 = (F_def @ v1.unsqueeze(-1)).squeeze(-1)                 # (F,3)
    w2 = (F_def @ v2.unsqueeze(-1)).squeeze(-1)
    w1 = w1 / torch.clamp(torch.norm(w1, dim=-1, keepdim=True), min=1e-20)
    w2 = w2 / torch.clamp(torch.norm(w2, dim=-1, keepdim=True), min=1e-20)

    T1c = torch.clamp(T1, min=0.0) if tensile_only else T1
    T2c = torch.clamp(T2, min=0.0) if tensile_only else T2
    T_plus = (T1c[..., None, None] * (w1[..., :, None] * w1[..., None, :])
              + T2c[..., None, None] * (w2[..., :, None] * w2[..., None, :]))   # (F,3,3)

    T_plus = torch.where(torch.isfinite(T_plus), T_plus, torch.zeros_like(T_plus))
    return T_plus, J, T1, T2


def body_shape_operator(verts, faces, vert_normals, min_det=1e-18):
    """Per-face shape operator K [1/m] of a triangle mesh, as a world 3x3 symmetric tensor.

    This is the body half of it. SIGN CONVENTION, pinned deliberately and guarded by
    a unit test (an internal tool):

        K is the differential of the Gauss map,  dn = K dx   (i.e. K = +Dn, NOT the classical -Dn)

    so a CONVEX sphere of radius R carrying OUTWARD normals yields eigenvalues **+1/R**, and hence a
    positive Laplace pressure p = tr(T+ K) on a limb. Getting this backwards makes subtract
    pressure on limbs and add it in armpits, which presents as " ~, no separation" -- a bug
    that looks like a negative result. Do not flip it without flipping the test.

    Per face: solve the 2x2 system  E S^T = D  in an orthonormal in-plane basis, where E holds the
    two edge vectors and D the corresponding vertex-normal differences, then symmetrise (the exact
    shape operator is self-adjoint; discretisation breaks that) and lift back to world. The face
    normal is in K's null space by construction.

    Deliberately NOT built on get_face_connectivity_combined / dihedral angles: that helper stores
    each shared edge as tuple(sorted(...)) and pairs faces in face-id order, so its per-edge sign is
    arbitrary. Harmless for bending (theta^2) and fatal for a signed curvature.

    :param verts:        (V,3) vertex positions
    :param faces:        (F,3) long
    :param vert_normals: (V,3) unit OUTWARD vertex normals, e.g. from VertexNormals()
    :return: K (F,3,3) symmetric, with K @ n_face == 0
    """
    tri = verts[faces]                                          # (F,3,3)
    nrm = vert_normals[faces]                                   # (F,3,3)

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    d1 = nrm[:, 1] - nrm[:, 0]
    d2 = nrm[:, 2] - nrm[:, 0]

    fn = torch.linalg.cross(e1, e2)
    fn = fn / torch.clamp(torch.norm(fn, dim=-1, keepdim=True), min=1e-20)

    # In-plane orthonormal basis. e1 lies in the plane by construction, so it is a safe seed.
    t1 = e1 / torch.clamp(torch.norm(e1, dim=-1, keepdim=True), min=1e-20)
    t2 = torch.linalg.cross(fn, t1)
    t2 = t2 / torch.clamp(torch.norm(t2, dim=-1, keepdim=True), min=1e-20)

    def _c(v):                                                  # world -> 2D in-plane coords
        return torch.stack([(v * t1).sum(-1), (v * t2).sum(-1)], dim=-1)

    E = torch.stack([_c(e1), _c(e2)], dim=-2)                   # (F,2,2), rows = edges
    D = torch.stack([_c(d1), _c(d2)], dim=-2)                   # (F,2,2), rows = dnormals

    det = E[..., 0, 0] * E[..., 1, 1] - E[..., 0, 1] * E[..., 1, 0]
    ok = det.abs() > min_det
    det_safe = torch.where(ok, det, torch.ones_like(det))
    E_inv = torch.stack([
        torch.stack([E[..., 1, 1], -E[..., 0, 1]], dim=-1),
        torch.stack([-E[..., 1, 0], E[..., 0, 0]], dim=-1),
    ], dim=-2) / det_safe[..., None, None]

    S = (E_inv @ D).transpose(-1, -2)                           # solves E S^T = D
    S = 0.5 * (S + S.transpose(-1, -2))                         # symmetrise

    B = torch.stack([t1, t2], dim=-1)                           # (F,3,2)
    K = B @ S @ B.transpose(-1, -2)                             # (F,3,3), null space = fn
    K = torch.where(ok[..., None, None], K, torch.zeros_like(K))
    K = torch.where(torch.isfinite(K), K, torch.zeros_like(K))
    return K


def refreshed_plane_gap(points, face_centroids, face_normals, K=4,
                        ambig_ratio=1.5, ambig_cos=0.0):
    """: the refreshed nearest-face PLANE-GAP PROXY for a small query set.

    Nearest body face by CENTROID (K candidates), gap = distance along the chosen (nearest) face's
    plane normal — deliberately the SAME convention as bolt_on_friction_projection's gate and the
    pen_rate_strict metric. This is NOT a true signed distance or an exact closest-point query
    (independent validation lives in utils/contact_oracle.py).

    Ambiguity flag: a query is ambiguous when another candidate face whose
    centroid distance is within `ambig_ratio` x the nearest carries a normal with
    dot(n_k, n_0) < ambig_cos (default 0.0 = conflicting/opposing constraint side: thin parts,
    wedges, crotch/armpit, opposing surfaces). Callers must NOT apply a normal correction at
    ambiguous vertices — skip and count instead (a wrong-side push is worse than no push).

    :param points:         (N,3) query positions
    :param face_centroids: (F,3) body-face centroids (NEXT pose, matching the operator's gate)
    :param face_normals:   (F,3) unit outward body-face normals (NEXT pose)
    :return: g (N,1) plane-gap, n_hat (N,3), face_idx (N,), ambig (N,) bool
    """
    from pytorch3d.ops import knn_points

    K_eff = min(int(K), face_centroids.shape[0])
    # The KNN is a DISCRETE correspondence choice. Querying with the detached view is BITWISE
    # identical in forward (.detach() shares storage) and gradient-identical (d2 only ever feeds
    # boolean comparisons, idx is an integer index) — but it keeps 's differentiable-operator
    # graph free of a pointless knn backward node. The live `points` are still used for `g` below,
    # which is the only place the gap can carry a gradient.
    d2, idx, _ = knn_points(points.detach().unsqueeze(0), face_centroids.unsqueeze(0), K=K_eff)
    d2, idx = d2[0], idx[0]                                                # (N,K), (N,K)
    n0 = face_normals[idx[:, 0]]                                           # (N,3)
    cen0 = face_centroids[idx[:, 0]]                                       # (N,3)
    g = ((points - cen0) * n0).sum(dim=-1, keepdim=True)                   # (N,1)
    if K_eff > 1:
        near = d2 <= d2[:, :1] * (ambig_ratio ** 2)                        # (N,K) near-equal candidates
        cosk = (face_normals[idx] * n0.unsqueeze(1)).sum(dim=-1)           # (N,K) dot with chosen normal
        ambig = ((cosk < ambig_cos) & near)[:, 1:].any(dim=1)              # (N,)
    else:
        ambig = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    return g, n0, idx[:, 0], ambig


def normal_completion_plane_gap(x_pre, x_t, cap, face_centroids, face_normals, mode,
                                iters=2, slack=1e-4, K=4, ambig_ratio=1.5, ambig_cos=0.0,
                                resid_tol=1e-6):
    """ normal completion: a LOCAL minimum-Euclidean-displacement correction for a
    fixed body-face halfspace, iterated after refreshing the face correspondence.

    Moves each friction-altered vertex along its refreshed nearest-face plane normal so that its
    plane-gap proxy returns to `target`:
      mode='preserve' (-T): target = g(x_pre) — the network's own clearance
      mode='preserve_clamp' (-TC): target = max(g(x_pre), slack) — Signorini-clamped per fixed face

    Guarantees are LOCAL only: exact restoration holds per fixed selected face; across
    correspondence refreshes there is a fixed iteration budget and NO convergence guarantee
    (residuals are counted, not assumed away). Safety semantics:
      * ambiguous correspondence at the pre-query or any refresh  => vertex REVERTED to x_t, counted;
      * cumulative correction norm exceeding `cap`                => vertex REVERTED to x_t, counted
        (skip, not clip: a correction that large signals bad geometry — do not apply a potentially
        wrong correction merely because it is capped).

    :param x_pre: (N,3) network tentative positions (pre-friction), NEXT-pose body frame
    :param x_t:   (N,3) positions after the tangential friction shrink
    :param cap:   (N,1) per-vertex max total correction norm
    :param face_centroids/face_normals: (F,3) NEXT-pose body face geometry (same as the gate)
    :return: (x_out (N,3), stats dict)
    """
    g_pre, _, f_pre, amb = refreshed_plane_gap(x_pre, face_centroids, face_normals,
                                               K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
    if mode == 'preserve':
        target = g_pre
    elif mode == 'preserve_clamp':
        target = torch.clamp(g_pre, min=slack)
    else:
        raise ValueError(f"normal_completion_plane_gap: unknown mode {mode!r}")

    x_cur = x_t.clone()
    amb_any = amb.clone()
    f_prev = f_pre
    n_switch = 0
    for _ in range(int(iters)):
        g_c, n_c, f_c, amb_c = refreshed_plane_gap(x_cur, face_centroids, face_normals,
                                                   K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
        amb_any |= amb_c
        n_switch += int((f_c != f_prev).sum())
        f_prev = f_c
        x_cur = x_cur + n_c * (target - g_c)
    # final refresh: residual + last-face-switch accounting (measurement only, no move)
    g_f, _, f_f, amb_f = refreshed_plane_gap(x_cur, face_centroids, face_normals,
                                             K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
    amb_any |= amb_f
    n_switch += int((f_f != f_prev).sum())

    total_norm = (x_cur - x_t).norm(dim=-1, keepdim=True)                  # (N,1)
    over_cap = (total_norm > cap).reshape(-1)
    revert = amb_any | over_cap
    x_out = torch.where(revert.unsqueeze(-1), x_t, x_cur)

    corrected = ~revert
    resid_bad = ((g_f - target).abs().reshape(-1) > resid_tol) & corrected
    dn = total_norm.reshape(-1)[corrected]
    stats = {
        'ne_candidates': int(x_t.shape[0]),
        'ne_corrected': int(corrected.sum()),
        'ne_skip_ambig': int(amb_any.sum()),
        'ne_skip_cap': int((over_cap & ~amb_any).sum()),
        'ne_residual': int(resid_bad.sum()),
        'ne_face_switch': n_switch,
        'ne_dn_sum': float(dn.sum()) if dn.numel() else 0.0,
        'ne_dn_max': float(dn.max()) if dn.numel() else 0.0,
    }
    return x_out, stats


def surface_march(x_hat, x_pre, face_centroids, face_normals, substeps=8,
                  K=4, ambig_ratio=1.5, ambig_cos=0.0, return_normals=False):
    """: a predictor-corrector OFFSET PATH from x_hat to x_pre that hugs the
    body surface, replacing the ambient straight chord the friction shrink walks along.

    Per substep: a linear chord PREDICTOR step, then a CORRECTOR that re-queries the nearest plane
    (refreshed_plane_gap) and resets the point's normal coordinate to the gap linearly interpolated
    g(x_hat) -> g(x_pre). The endpoints are pinned EXACTLY to x_hat (k=0) and x_pre (k=S), so on a
    single planar face the corrector is inert, the polyline IS the chord and L == |chord| —
    collapses to (the flat-limit regression gate). Across a crease the corrector reorients with
    the freshly selected face, so the path rounds the ridge at the interpolated clearance instead of
    cutting through the body.

    All geometry against the fixed NEXT-pose body (same convention as bolt_on's gate and).

    :param x_hat: (N,3) body-carried previous cloth position v_hat (slide-path start)
    :param x_pre: (N,3) frictionless prediction v_tentative (slide-path end)
    :param return_normals:. Additionally return per-SEGMENT outward normals
             n_seg (N,S,3): segment k's normal is the refreshed plane normal at its FAR node
             (interior corrector normals for nodes 1..S-1; the x_pre query's normal for node S).
             These are already computed by the march — this only stops discarding them. Default
             False keeps the return signature so every existing caller is untouched.
    :return: P (N,S+1,3) polyline, seg (N,S) segment lengths, L (N,1) arc length,
             chord (N,1) |x_pre-x_hat|, amb (N,) any-substep ambiguity, n_switch (N,) face changes
             along the march, g_hat (N,1), g_pre (N,1)[, n_seg (N,S,3) iff return_normals].
    """
    N = x_hat.shape[0]
    S = max(1, int(substeps))
    g_hat, _, f0, amb = refreshed_plane_gap(x_hat, face_centroids, face_normals,
                                            K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
    g_pre, n_end, fS, ambp = refreshed_plane_gap(x_pre, face_centroids, face_normals,
                                                 K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
    amb = amb | ambp
    nodes = [x_hat]
    faces = [f0]
    far_normals = []                                                  # normals at nodes 1..S
    for k in range(1, S):
        t = float(k) / S
        base = x_hat + t * (x_pre - x_hat)                            # (N,3) chord predictor
        g_c, n_c, f_c, amb_c = refreshed_plane_gap(base, face_centroids, face_normals,
                                                   K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
        amb = amb | amb_c
        g_target = g_hat + t * (g_pre - g_hat)                        # (N,1) lerped clearance
        nodes.append(base + n_c * (g_target - g_c))                  # corrector: reset normal coord
        faces.append(f_c)
        far_normals.append(n_c)
    nodes.append(x_pre)
    faces.append(fS)
    far_normals.append(n_end)
    P = torch.stack(nodes, dim=1)                                     # (N,S+1,3)
    fmat = torch.stack(faces, dim=1)                                 # (N,S+1)
    seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1)                        # (N,S)
    L = seg.sum(dim=1, keepdim=True)                                # (N,1)
    chord = (x_pre - x_hat).norm(dim=-1, keepdim=True)             # (N,1)
    n_switch = (fmat[:, 1:] != fmat[:, :-1]).sum(dim=1)            # (N,)
    if return_normals:
        return P, seg, L, chord, amb, n_switch, g_hat, g_pre, torch.stack(far_normals, dim=1)
    return P, seg, L, chord, amb, n_switch, g_hat, g_pre


# --------------------------------------------------------------------------------------------------
# TRIANGLE-WALK traversal -- the parameter-free path constructor replacing the
# S-substep march. Walking-on-triangles scheme (SplattingAvatar, Shao et al. CVPR 2024; lineage The
# Phong Surface, ECCV 2020) with ISOMETRIC hinge-unfold transport instead of their shape-agnostic
# re-expression: our Coulomb landing consumes ARC LENGTH, and their "right triangles on the
# hypotenuse" re-expression is metric-distorting on SMPL's skinny triangles (armpits). The crossed
# faces ARE the segments, so `path_substeps` ceases to exist. Derivation:
# Gates: an internal tool (G0 scalar-reference fuzz, G1 flat-limit, G3 b->0).
# --------------------------------------------------------------------------------------------------

_WALK_ADJ_CACHE = {}


def _walk_adjacency(faces):
    """Face adjacency + partner slots of a triangle mesh, cached per topology.

    adj[f, i]  = face sharing f's edge OPPOSITE vertex i (the edge (f[(i+1)%3], f[(i+2)%3])),
                 or -1 on a boundary / non-manifold edge (guarded downstream, cannot happen on
                 closed SMPL). The slot convention matches barycentric exit: coordinate i -> 0
                 crosses the edge opposite vertex i.
    slot[f, i] = the matching slot on that neighbor: adj[adj[f, i], slot[f, i]] == f. Used to
                 exclude the entry edge from the next clip (the standard re-cross guard).

    Cached by (device, F, checksum), NOT by tensor identity: topology is static across a rollout
    but PyG rebuilds the batch tensor every step, and a data_ptr key could alias a reused
    allocation. The checksum costs one small host sync per step, same class as the gate's
    .nonzero() sync at the op_normal candidate set.
    """
    key = (str(faces.device), int(faces.shape[0]), int(faces.sum().item()))
    hit = _WALK_ADJ_CACHE.get(key)
    if hit is not None:
        return hit
    f = faces.detach().long().cpu()
    F = f.shape[0]
    va = f[:, [1, 2, 0]].reshape(-1)                       # (3F,) slot i edge start v_{(i+1)%3}
    vb = f[:, [2, 0, 1]].reshape(-1)                       # (3F,) slot i edge end   v_{(i+2)%3}
    lo, hi = torch.minimum(va, vb), torch.maximum(va, vb)
    ekey = lo * (int(f.max()) + 1) + hi                    # unique per undirected edge
    order = torch.argsort(ekey, stable=True)
    _, counts = torch.unique_consecutive(ekey[order], return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    s2 = starts[counts == 2]                               # manifold interior edges only
    a_sl, b_sl = order[s2], order[s2 + 1]
    adj = torch.full((3 * F,), -1, dtype=torch.long)
    slot = torch.full((3 * F,), -1, dtype=torch.long)
    adj[a_sl] = b_sl // 3
    slot[a_sl] = b_sl % 3
    adj[b_sl] = a_sl // 3
    slot[b_sl] = a_sl % 3
    out = (adj.reshape(F, 3).to(faces.device), slot.reshape(F, 3).to(faces.device))
    _WALK_ADJ_CACHE[key] = out
    return out


def _walk_bary(p, v0, v1, v2):
    """Barycentric coordinates of p's IN-PLANE projection on triangle (v0,v1,v2), any 3D p.

    The Gram solve reads p only through dot products with the in-plane edges, so the off-plane
    component of p is annihilated by construction -- no explicit projection needed for the TEST
    (the crossing POINT is still built from an explicitly projected aim, to stay on the plane).
    Returns (bary (r,3), degen (r,) bool) -- degenerate faces flagged RELATIVELY (area^2 vs edge
    products), dtype-portable between the float32 operator and the float64 test oracle.
    """
    e0, e1, d = v1 - v0, v2 - v0, p - v0
    d00 = (e0 * e0).sum(-1)
    d01 = (e0 * e1).sum(-1)
    d11 = (e1 * e1).sum(-1)
    d20 = (d * e0).sum(-1)
    d21 = (d * e1).sum(-1)
    den = d00 * d11 - d01 * d01
    degen = ~(den.abs() > (d00 * d11).clamp(min=1e-30) * 1e-9)
    den_s = torch.where(degen, torch.ones_like(den), den)
    a = (d11 * d20 - d01 * d21) / den_s
    b = (d00 * d21 - d01 * d20) / den_s
    return torch.stack([1.0 - a - b, a, b], dim=-1), degen


def _walk_seat(x, f_init, tri, adj, slot_tbl, face_normals, max_cross=3, tol=1e-6,
               capturable=False):
    """D4 proxy-native seating: (face, foot, gap) decomposition of x, gate face as seed.

    foot = x - g*n on the CURRENT face's plane; barycentrics inside => the seat is exact
    (seat + g*n reproduces x to fp). Otherwise walk one edge from the bary-clamped point toward
    the foot and re-derive the foot on the entered face -- the foot is a MOVING target, so every
    face re-aims.

    EDGE SEATS: ~2.6% of gate pairs have their exact closest
    point on a CONVEX EDGE -- there the plane-foot of EVERY adjacent face falls outside that
    face, the chase oscillates across the shared edge, and the entry-exclusion turns the
    oscillation into a provable dead-end (no admissible exit). K-candidate re-seeding does not
    rescue them (verified: identical failures from all 4 gate candidates) because no interior
    foot EXISTS. The dead-end point IS the edge-closest point, so it is ACCEPTED as the seat
    (`edge` True): reproduction relaxes from exact to the in-plane overshoot O(gap*tan(theta/2))
    -- the same proxy class as the gate's own plane gap, and the OUTPUT path endpoints stay
    pinned to the true endpoints regardless (S4). Without this rule those rows all fall to the
    chord class and gate G4 (< 0.5% per fallback class) fails at 2.6%.

    Genuinely lost rows remain failures: cap exhaustion (never observed on SMPL -- failures were
    invariant to cap 3/8/16), a boundary edge, a degenerate face => ok=False: the caller's
    chord-fallback class (floor =), counted, never guessed.

    :param capturable:. Removes the loop's two HOST SYNCS -- the `alive.any` early break
             and the `acc_edge.any()` branch guard -- so the body is CUDA-graph capturable. Both
             are pure scheduling: every statement they guard is already masked by `alive` /
             `acc_edge` through torch.where, so an iteration with nothing alive writes each tensor
             back to itself. The trip count is already bounded (max_cross+1), so nothing truncates.
             Default False executes the shipped branches verbatim (short-circuit `or`, so the
             `bool` is not even evaluated when True). See

    :return: (face (N,), pt (N,3), g (N,1) plane gap on that face, ok (N,), n_cross (N,),
              edge (N,) bool -- ok rows seated on an edge dead-end rather than an interior foot)
    """
    N = x.shape[0]
    face = f_init.clone()
    ok = torch.ones(N, dtype=torch.bool, device=x.device)
    alive = ok.clone()
    ncross = torch.zeros(N, dtype=torch.long, device=x.device)
    entry = torch.full((N,), -1, dtype=torch.long, device=x.device)
    pt = x.clone()                                          # finite filler for failed rows
    g_out = x.new_zeros(N)
    edge = torch.zeros(N, dtype=torch.bool, device=x.device)
    cur = None
    for it in range(max_cross + 1):
        v = tri[face]                                       # (N,3,3)
        v0 = v[:, 0]
        n_f = face_normals[face]
        gp = ((x - v0) * n_f).sum(-1)                       # plane gap through v0 (== centroid plane)
        foot = x - gp.unsqueeze(-1) * n_f
        ba, degen = _walk_bary(foot, v0, v[:, 1], v[:, 2])
        inside = (ba >= -tol).all(-1) & ~degen
        fin = alive & inside
        pt = torch.where(fin.unsqueeze(-1), foot, pt)
        g_out = torch.where(fin, gp, g_out)
        bad = alive & degen
        ok = ok & ~bad
        alive = alive & ~inside & ~degen
        #`capturable` drops the `.any` host sync only -- same trip bound either way.
        if (it == max_cross) if capturable else (it == max_cross or not bool(alive.any())):
            break
        if cur is None:                                     # first hop starts at the clamped foot
            bc = ba.clamp(min=0.0)
            bc = bc / bc.sum(-1, keepdim=True).clamp(min=1e-30)
            cur = (bc.unsqueeze(-1) * v).sum(1)
            cur_b = bc
        else:
            cur_b, _ = _walk_bary(cur, v0, v[:, 1], v[:, 2])
        cand = (ba < -tol) & ((cur_b - ba) > 1e-30)
        cand = cand & ~((torch.arange(3, device=x.device).unsqueeze(0) == entry.unsqueeze(-1))
                        & (entry >= 0).unsqueeze(-1))
        t = torch.where(cand, cur_b / (cur_b - ba), torch.full_like(ba, float('inf')))
        t_exit, s = t.min(dim=-1)
        stuck = alive & ~torch.isfinite(t_exit)
        # the convex-edge dead-end: the only way out is the edge we came in through => the
        # current point (ON that edge) is the closest-point class itself. Accept it.
        acc_edge = stuck & (entry >= 0)
        if capturable or bool(acc_edge.any()):              # body is acc_edge-masked anyway
            pt = torch.where(acc_edge.unsqueeze(-1), cur, pt)
            g_out = torch.where(acc_edge, gp, g_out)
            edge = edge | acc_edge
        ok = ok & ~(stuck & (entry < 0))                    # stuck with no history: genuinely lost
        alive = alive & ~stuck
        q = cur + t_exit.clamp(0.0, 1.0).unsqueeze(-1) * (foot - cur)
        nxt = adj[face, s]
        boundary = alive & (nxt < 0)
        ok = ok & ~boundary
        alive = alive & ~boundary
        entry = torch.where(alive, slot_tbl[face, s], entry)
        face = torch.where(alive, nxt, face)
        cur = torch.where(alive.unsqueeze(-1), q, cur)
        ncross = ncross + alive.long()
    ok = ok & ~alive                                        # still off-face after the cap
    return face, pt, g_out.unsqueeze(-1), ok, ncross, edge


def _walk_trace(start_pt, start_face, target_pt, tri, adj, slot_tbl, face_normals,
                max_cross=16, active=None, capturable=False, iters=None):
    """The greedy target-pinned crossing loop -- the directly unit-testable walk core (G0).

    From start_pt (on start_face's plane) toward target_pt: in each face aim at the target's
    image -- the target hinge-ROTATED about every crossed edge into the current face's frame
    (Rodrigues from the two face normals; no acos, sign-exact) -- clip the in-plane ray against
    the face's edges (barycentric first-zero, entry edge excluded), cross, repeat. Terminates
    when the image's barycentrics go interior. Pure geometry: no KNN, no gaps, no lifting --
    that is triangle_walk_path's job, which is what keeps this loop scalar-mirrorable.

    :param active: optional (W,) bool -- rows to walk; inactive rows return ncross=0, ok=True.
    :param capturable:. Drops the `alive.any` host sync (pure scheduling -- every
             statement is `alive`-masked, so a dead iteration is a self-assignment) and makes the
             `faces_path` padding write PRESERVE the column instead of restamping it with the row's
             current face, which is what the shipped break leaves behind. Required for CUDA-graph
             capture. NOTE: `faces_path`'s padded columns are not a contract in either mode -- in
             the shipped loop they already depend on how deep the DEEPEST row of the batch went
              -- so the bitwise gate belongs at the `triangle_walk_path` boundary,
             which masks them with `(kk < ncross)`. Default False = the shipped loop verbatim.
    :param iters:, `capturable` only -- a fixed trip count <= max_cross+1, so the loop has
             a static length for capture. Rows still alive after `iters` come back in `over`;
             the caller must treat that as "the graph truncated" and redo the batch eagerly (it is
             NOT the over-cap fallback class unless iters == max_cross+1). None = max_cross+1.
    :return: (faces_path (W,max_cross+1) long, [:,0]=start_face, [:,j+1]=face entered at crossing
             j (start_face where unused); cross_pts (W,max_cross,3), start_pt-filled where unused;
             ncross (W,); aim_final (W,3) the terminal in-plane target image; ok (W,) False on
             degenerate/boundary/no-exit/over-cap/non-finite rows; over (W,) the over-cap subset
             of ~ok -- kept separate so the caller's fallback classes stay exact)
    """
    W = start_pt.shape[0]
    dev = start_pt.device
    alive = active.clone() if active is not None else torch.ones(W, dtype=torch.bool, device=dev)
    ok = torch.ones(W, dtype=torch.bool, device=dev)
    face = start_face.clone()
    cur = start_pt.clone()
    T = target_pt.clone()
    entry = torch.full((W,), -1, dtype=torch.long, device=dev)
    ncross = torch.zeros(W, dtype=torch.long, device=dev)
    aim_final = target_pt.clone()
    faces_path = start_face.unsqueeze(1).repeat(1, max_cross + 1)
    cross_pts = start_pt.unsqueeze(1).repeat(1, max_cross, 1)
    arange3 = torch.arange(3, device=dev).unsqueeze(0)
    n_it = (max_cross + 1) if iters is None else int(iters)
    for _it in range(n_it):
        v = tri[face]
        v0 = v[:, 0]
        n_f = face_normals[face]
        aim = T - ((T - v0) * n_f).sum(-1, keepdim=True) * n_f
        ba, degen = _walk_bary(aim, v0, v[:, 1], v[:, 2])
        inside = (ba >= -1e-6).all(-1) & ~degen
        aim_final = torch.where((alive & inside).unsqueeze(-1), aim, aim_final)
        ok = ok & ~(alive & degen)
        alive = alive & ~inside & ~degen
        #`capturable` drops the `.any` host sync; `n_it` is the shipped bound unless the
        # caller shortened it for a graph bucket (then `over` reports the truncation, see docstring).
        if (_it == n_it - 1) if capturable else (_it == max_cross or not bool(alive.any())):
            break
        cur_b, degc = _walk_bary(cur, v0, v[:, 1], v[:, 2])
        ok = ok & ~(alive & degc)
        alive = alive & ~degc
        cand = (ba < -1e-6) & ((cur_b - ba) > 1e-30)
        cand = cand & ~((arange3 == entry.unsqueeze(-1)) & (entry >= 0).unsqueeze(-1))
        t = torch.where(cand, cur_b / (cur_b - ba), torch.full_like(ba, float('inf')))
        t_exit, s = t.min(dim=-1)
        stuck = alive & ~torch.isfinite(t_exit)             # numerical corner: no admissible exit
        ok = ok & ~stuck
        alive = alive & ~stuck
        q = cur + t_exit.clamp(0.0, 1.0).unsqueeze(-1) * (aim - cur)
        nxt = adj[face, s]
        boundary = alive & (nxt < 0)
        ok = ok & ~boundary
        alive = alive & ~boundary
        # hinge unfold: rotate T about the crossed edge by the dihedral taking this face's plane
        # onto the neighbor's. Isometric -- the developed straight ray stays straight, so the
        # accumulated in-plane arc is the true unfolded length the Coulomb budget consumes.
        i1 = ((s + 1) % 3).view(-1, 1, 1).expand(-1, 1, 3)
        i2 = ((s + 2) % 3).view(-1, 1, 1).expand(-1, 1, 3)
        e0 = v.gather(1, i1).squeeze(1)
        e1 = v.gather(1, i2).squeeze(1)
        u = e1 - e0
        un = u.norm(dim=-1, keepdim=True)
        edge_bad = alive & (un.reshape(-1) <= 1e-12)
        ok = ok & ~edge_bad
        alive = alive & ~edge_bad
        u = u / un.clamp(min=1e-30)
        n_b = face_normals[nxt.clamp(min=0)]
        cth = (n_f * n_b).sum(-1, keepdim=True)
        sth = (torch.cross(n_f, n_b, dim=-1) * u).sum(-1, keepdim=True)
        w_ = T - e0
        T_rot = (e0 + w_ * cth + torch.cross(u, w_, dim=-1) * sth
                 + u * (u * w_).sum(-1, keepdim=True) * (1.0 - cth))
        T = torch.where(alive.unsqueeze(-1), T_rot, T)
        cp = cross_pts[:, _it]
        cross_pts[:, _it] = torch.where(alive.unsqueeze(-1), q, cp)
        # under `capturable` the loop runs past a row's termination, so the padding write
        # must preserve the column (what the shipped break leaves behind), not restamp it.
        faces_path[:, _it + 1] = torch.where(alive, nxt,
                                             faces_path[:, _it + 1] if capturable else face)
        entry = torch.where(alive, slot_tbl[face, s], entry)
        face = torch.where(alive, nxt, face)
        cur = torch.where(alive.unsqueeze(-1), q, cur)
        ncross = ncross + alive.long()
    over = alive.clone()                                    # over-cap: never reached an interior image
    ok = ok & ~alive
    ok = ok & torch.isfinite(aim_final).all(-1)
    return faces_path, cross_pts, ncross, aim_final, ok, over


# --------------------------------------------------------------------------------------------------
# the WALK FAST PATH. Gated by bolt_on.path_walk_fast (default False => nothing below is
# reachable and the walk is byte-identical to). It changes SCHEDULING ONLY -- no arithmetic,
# no tolerance, no ordering of masked writes -- so its output is bitwise the shipped walk's.
#
# WHY: the walk's +8.10 ms/frame is kernel LAUNCH COUNT, not work. ~3965 launches per call
# against the march's 209, issued from a Python loop over ~200 rows of 3 floats; each kernel finishes
# in ~1 us and the GPU then waits for the CPU. The cost is flat in N (100x the pairs = 1.7x the time)
# and dropping 67 % of the walking rows changes it by 0 %. So the lever is launches, not rows.
#
# TWO INDEPENDENT CHANGES, both bitwise:
#   A. Merge the two `_walk_seat` calls into one. They run the same loop on disjoint rows and every
#      op is row-independent, so concatenating halves the phase's launch count. No graph needed --
#      this survives even if every graph below is refused.
#   B. Replay both loops from CUDA graphs. Needs `capturable=True` (the host syncs are what torch
#      refuses capture on), static shapes (=> row bucketing) and pinned input addresses (=> the
#      moving body is copied into buffers owned here).
#
# REFUSAL IS ALWAYS SAFE. Every entry point returns None rather than raising, and the caller then
# runs the shipped eager loop. The fast path declines when: autograd is live on an input (the eval
# rollout is NOT under no_grad -- see the module header and runners/ccraft.py:951), the batch exceeds
# the largest bucket, the graph budget is spent, capture raised, or a shortened trace graph truncated.
# --------------------------------------------------------------------------------------------------

_WALK_FAST_STATIC = {}          # (dev,F,dtype) -> static topology+pose buffers, refreshed in place
_WALK_FAST_GRAPHS = {}          # key -> (graph, ins, outs)
_WALK_FAST_POISON = set()       # keys whose capture raised: never retried
_WALK_FAST_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
_WALK_FAST_TRACE_DEPTH = 9      # P1 witness cross_max over 840 steps: 1:39 2:94 3:537 4:145 5:14
                                # 6:9 7:1 12:1 => a 9-iteration graph resolves 839/840 steps; the
                                # 840th truncates, is detected by `over`, and is redone eagerly.
_WALK_FAST_MAX_GRAPHS = 24

#: module-level, so the wk_* stats CONTRACT is untouched (a new key there would reach the timeprof
# assembler and the witness scrapers). Read/reset by an internal tool
WALK_FAST_COUNTERS = {'seat_graph': 0, 'seat_eager': 0, 'trace_graph': 0, 'trace_eager': 0,
                      'trunc': 0, 'captures': 0, 'refused_grad': 0, 'refused_bucket': 0}


def _walk_fast_bucket(n):
    """Smallest static row count that holds n, or 0 if n is past the largest bucket."""
    for b in _WALK_FAST_BUCKETS:
        if n <= b:
            return b
    return 0


def _walk_fast_ok(*tensors):
    """May the fast path run on these inputs? CUDA-resident is the only structural requirement.

    AUTOGRAD. The eval
    rollout is NOT under `no_grad` and `x_pre` arrives WITH a grad_fn -- measured, 444/444 calls --
    so an earlier draft that refused whenever an input required grad refused every real invocation
    and bought nothing. Capturing autograd-recording ops is not viable, so the fast path instead
    DETACHES what enters the two captured loops (the caller does this, not here).

    That detachment is not assumed harmless, it is measured harmless for this arm: a backward hook
    on every grad-carrying tensor `triangle_walk_path` returns fired 0 times in 2220 hooks over a
    full rollout -- no backward traverses the walk's outputs, so cutting the gradient through the
    interior path nodes cannot change any number. What it WOULD change is a train-through arm that
    prices its action through the path builder, and that case is refused at config validation
    (`path_walk_fast` + `return_action=True`), not silently detached here.
    """
    if not torch.cuda.is_available():
        return False
    for t in tensors:
        if t is None or not t.is_cuda:
            return False
        if t.requires_grad:
            WALK_FAST_COUNTERS['refused_grad'] += 1     # informational: how often we detach
    return True


def _walk_fast_static(tri, face_normals, adj, slot_tbl):
    """Static input buffers for capture, refreshed IN PLACE each step.

    A captured graph pins input ADDRESSES, but `tri = obs_pos[obs_faces]` and `face_normals` are
    rebuilt every step (the body moves), and `_walk_adjacency`'s cache could in principle hand back
    a different allocation. Copying all four into buffers owned here makes every graph depend on
    nothing outside this module. ~1.3 MB for SMPL, device-to-device, four kernels.
    """
    key = (str(tri.device), int(tri.shape[0]), str(tri.dtype))
    st = _WALK_FAST_STATIC.get(key)
    if st is None:
        st = {'tri': torch.zeros(tri.shape, device=tri.device, dtype=tri.dtype),
              'nrm': torch.zeros(face_normals.shape, device=tri.device, dtype=face_normals.dtype),
              'adj': torch.zeros(adj.shape, device=tri.device, dtype=adj.dtype),
              'slot': torch.zeros(slot_tbl.shape, device=tri.device, dtype=slot_tbl.dtype)}
        _WALK_FAST_STATIC[key] = st
    st['tri'].copy_(tri)
    st['nrm'].copy_(face_normals)
    st['adj'].copy_(adj)
    st['slot'].copy_(slot_tbl)
    return st


def _walk_fast_capture(key, build, dev, ins):
    """Fetch-or-capture. `build()` runs the loop on `ins` + the static buffers and returns its
    output tuple; every replay refills those same output tensors.

    Warmup runs on a SIDE STREAM (the documented requirement -- capture on the default stream
    without it deadlocks or captures allocator traffic). Any failure poisons the key and returns
    None, so the caller falls back to eager permanently rather than retrying every step.
    """
    hit = _WALK_FAST_GRAPHS.get(key)
    if hit is not None:
        return hit
    if key in _WALK_FAST_POISON or len(_WALK_FAST_GRAPHS) >= _WALK_FAST_MAX_GRAPHS:
        return None
    try:
        torch.cuda.synchronize(dev)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(3):
                build()
        torch.cuda.current_stream(dev).wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = build()
    except Exception:                                       # capture is an optimisation, never a
        _WALK_FAST_POISON.add(key)                          # correctness dependency
        return None
    WALK_FAST_COUNTERS['captures'] += 1
    _WALK_FAST_GRAPHS[key] = (g, ins, out)
    return _WALK_FAST_GRAPHS[key]


def _walk_seat_fast(x, f_init, st, max_cross=3):
    """Graph-replayed `_walk_seat`. Returns its 6-tuple sliced to the real rows, or None to decline.

    Padded rows are left holding whatever the previous step wrote. That is safe because every op in
    the loop is row-independent and the outputs are sliced -- a garbage row cannot reach a real one
    -- and the buffers start zeroed, i.e. face 0 at the origin, which is real geometry for warmup.
    """
    dev = x.device
    M = int(x.shape[0])
    Wp = _walk_fast_bucket(M)
    if Wp == 0:
        WALK_FAST_COUNTERS['refused_bucket'] += 1
        return None
    key = ('seat', str(dev), int(st['tri'].shape[0]), str(x.dtype), Wp, int(max_cross))
    ent = _WALK_FAST_GRAPHS.get(key)
    if ent is None:
        ins = {'x': torch.zeros(Wp, 3, device=dev, dtype=x.dtype),
               'f': torch.zeros(Wp, dtype=torch.long, device=dev)}
        ent = _walk_fast_capture(
            key,
            lambda: _walk_seat(ins['x'], ins['f'], st['tri'], st['adj'], st['slot'], st['nrm'],
                               max_cross=max_cross, capturable=True),
            dev, ins)
        if ent is None:
            return None
    g, ins, out = ent
    ins['x'][:M] = x
    ins['f'][:M] = f_init
    g.replay()
    WALK_FAST_COUNTERS['seat_graph'] += 1
    return tuple(o[:M] for o in out)


def _walk_trace_fast(start_pt, start_face, target_pt, st, active, max_cross,
                     depth=None):
    """Graph-replayed `_walk_trace`. Returns its 6-tuple sliced to the real rows, or None to decline.

    `depth` shortens the trip count to what the population actually needs. A row still alive at the
    end means the graph ran out of iterations, which is NOT the over-cap fallback class -- so the
    only honest response is to decline and let the caller redo the batch with the shipped loop. That
    costs one `.any()` sync per step in exchange for eight fewer inside it.
    """
    dev = start_pt.device
    W = int(start_pt.shape[0])
    Wp = _walk_fast_bucket(W)
    if Wp == 0:
        WALK_FAST_COUNTERS['refused_bucket'] += 1
        return None
    depth = min(int(_WALK_FAST_TRACE_DEPTH if depth is None else depth), int(max_cross) + 1)
    key = ('trace', str(dev), int(st['tri'].shape[0]), str(start_pt.dtype), Wp, depth,
           int(max_cross))
    ent = _WALK_FAST_GRAPHS.get(key)
    if ent is None:
        ins = {'pt': torch.zeros(Wp, 3, device=dev, dtype=start_pt.dtype),
               'fc': torch.zeros(Wp, dtype=torch.long, device=dev),
               'tg': torch.zeros(Wp, 3, device=dev, dtype=start_pt.dtype),
               'ac': torch.zeros(Wp, dtype=torch.bool, device=dev)}
        ent = _walk_fast_capture(
            key,
            lambda: _walk_trace(ins['pt'], ins['fc'], ins['tg'], st['tri'], st['adj'], st['slot'],
                                st['nrm'], max_cross=max_cross, active=ins['ac'],
                                capturable=True, iters=depth),
            dev, ins)
        if ent is None:
            return None
    g, ins, out = ent
    ins['pt'][:W] = start_pt
    ins['fc'][:W] = start_face
    ins['tg'][:W] = target_pt
    ins['ac'].zero_()
    ins['ac'][:W] = active
    g.replay()
    if depth < int(max_cross) + 1 and bool(out[5][:W].any()):
        WALK_FAST_COUNTERS['trunc'] += 1
        return None
    WALK_FAST_COUNTERS['trace_graph'] += 1
    return tuple(o[:W] for o in out)


def triangle_walk_path(x_hat, x_pre, obs_pos, obs_faces, face_centroids, face_normals,
                       max_cross=16, K=4, ambig_ratio=1.5, ambig_cos=0.0, fast=False):
    """: the triangle-walk slide path from x_hat to x_pre -- surface_march's drop-in sibling.

    Seat both endpoints on the NEXT-pose body (proxy-native, D4), then walk seat->seat greedily:
    in each face, aim at the target's image -- the target hinge-UNFOLDED about every crossed edge
    into the current face's plane (straightest-geodesic transport; isometric, so the arc feeds the
    Coulomb budget undistorted) -- clip against the face's edges, cross, repeat. The crossed faces
    ARE the segments: no substep count exists. Interior node j = crossing point lifted by d_j along
    the junction's normal BISECTOR (rounds the ridge; flat limit exact), d_j = the RAW clearance
    lerp g(v_hat)->g(x_pre) at base-arc fraction (S2: march-identical schedule, negative allowed).
    Endpoints pinned EXACTLY to x_hat / x_pre, as in surface_march.

    FAST PATH: pairs whose two gate faces coincide -- the population majority
    -- are emitted directly as the single straight segment (the march's exact flat-limit result on
    a shared plane) and never enter seating or the crossing loop. Same for zero-length chords.

    Per-vertex FALLBACK classes (=> `amb` true => the caller reverts to the chord result):
    gate ambiguity (as the march), seat failure, > max_cross crossings, boundary/degenerate/
    non-finite geometry. Counted per class in the returned wk_* dict, never silently absorbed.

    Rows are ROW-PADDED to the batch max: trailing nodes repeat x_pre (zero-length segments) and
    trailing normals duplicate the last real one -- multi_normal_arc_landing is transparent to
    both (padded junction wrap = acos(1) = 0, so no spurious capstan mass either).

    :param fast: (bolt_on.path_walk_fast). Merge the two seat calls and replay both loops
             from CUDA graphs. Every returned VALUE is bitwise the default path's, and every
             component declines to eager rather than failing. The one non-value difference: the two
             loops run DETACHED (capture cannot record autograd), so the interior path nodes carry
             no grad_fn. Measured inconsequential for inference -- no backward traverses these
             outputs (0 firings / 2220 hooks over a full rollout) -- and refused outright for the
             train-through arm at config validation. Default False = byte-identical.

    :return: (P (N,S+1,3), seg (N,S), L (N,1), chord (N,1), amb (N,), n_switch (N,) true crossing
             count, g_hat (N,1), g_pre (N,1) -- the GATE gaps, march-contract parity --,
             n_seg (N,S,3) carrying-face normals, wk stats dict). S = batch max crossings + 1.
    """
    N = x_hat.shape[0]
    dev = x_hat.device
    g_hat, _, f0, amb0 = refreshed_plane_gap(x_hat, face_centroids, face_normals,
                                             K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
    g_pre, _, f1, amb1 = refreshed_plane_gap(x_pre, face_centroids, face_normals,
                                             K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
    amb = amb0 | amb1
    chord = (x_pre - x_hat).norm(dim=-1, keepdim=True)
    adj, slot_tbl = _walk_adjacency(obs_faces)
    tri = obs_pos[obs_faces]                                # (F,3,3)

    same = (f0 == f1) | (chord.reshape(-1) <= 1e-12)        # the fast path: no seats, no loop
    need = ~same & ~amb
    ncross = torch.zeros(N, dtype=torch.long, device=dev)
    faces_path = f0.unsqueeze(1).repeat(1, max_cross + 1)   # face 0 = start face for EVERY row
    cross_pts = x_pre.unsqueeze(1).expand(N, max_cross, 3).clone()
    aim_final = x_pre.clone()
    seat_h_pt = x_hat.clone()
    gh_eff = g_hat.clone()                                  # lerp endpoints; seat gaps where seated
    gp_eff = g_pre.clone()
    fail = torch.zeros(N, dtype=torch.bool, device=dev)
    wk = {'wk_pairs': int(N), 'wk_same_face': int(same.sum()), 'wk_seat_edge': 0, 'wk_fb_seat': 0,
          'wk_fb_cross': 0, 'wk_fb_geom': 0, 'wk_walked': 0, 'wk_cross_sum': 0, 'wk_cross_max': 0}

    idxw = need.nonzero().reshape(-1)
    if idxw.numel():
        W = idxw.numel()
        use_fast = bool(fast) and _walk_fast_ok(x_hat, x_pre, obs_pos, face_normals)
        st = None
        if use_fast:
            # ONE seat call on the concatenated endpoints. Bitwise (the loop is
            # row-independent), halves the phase's launches, and needs no graph machinery -- so it
            # still applies when every capture below is refused.
            with torch.no_grad():
                st = _walk_fast_static(tri.detach(), face_normals.detach(), adj, slot_tbl)
                # DETACHED: capture cannot record autograd, and no backward traverses
                # these outputs in the eval rollout (0 firings / 2220 hooks). Values are untouched
                # -- detach changes the graph, never a number -- so the bitwise gates still bind.
                xcat = torch.cat([x_hat[idxw], x_pre[idxw]], 0).detach()
                fcat = torch.cat([f0[idxw], f1[idxw]], 0)
                sout = _walk_seat_fast(xcat, fcat, st)
                if sout is None:
                    WALK_FAST_COUNTERS['seat_eager'] += 1
                    sout = _walk_seat(xcat, fcat, tri.detach(), adj, slot_tbl,
                                      face_normals.detach())
            fh, fp_ = sout[0][:W], sout[0][W:]
            ph, pp = sout[1][:W], sout[1][W:]
            gh, gp = sout[2][:W], sout[2][W:]
            okh, okp = sout[3][:W], sout[3][W:]
            egh, egp = sout[5][:W], sout[5][W:]
        else:
            fh, ph, gh, okh, _, egh = _walk_seat(x_hat[idxw], f0[idxw], tri, adj, slot_tbl,
                                                 face_normals)
            fp_, pp, gp, okp, _, egp = _walk_seat(x_pre[idxw], f1[idxw], tri, adj, slot_tbl,
                                                  face_normals)
        seat_ok = okh & okp
        wk['wk_fb_seat'] = int((~seat_ok).sum())
        wk['wk_seat_edge'] = int(((egh | egp) & seat_ok).sum())
        seat_same = seat_ok & (fh == fp_)                   # seats landed on one face after all
        walkw = seat_ok & ~seat_same
        faces_path[idxw, 0] = torch.where(seat_ok, fh, f0[idxw])
        seat_h_pt[idxw] = torch.where(seat_ok.unsqueeze(-1), ph, seat_h_pt[idxw])
        gh_eff[idxw] = torch.where(seat_ok.unsqueeze(-1), gh, gh_eff[idxw])
        gp_eff[idxw] = torch.where(seat_ok.unsqueeze(-1), gp, gp_eff[idxw])
        fail[idxw] |= ~seat_ok

        tout = None
        if use_fast:
            with torch.no_grad():                           # graph-replayed crossing loop
                tout = _walk_trace_fast(ph, fh, pp, st, walkw, max_cross)
        if tout is None:
            if use_fast:
                WALK_FAST_COUNTERS['trace_eager'] += 1
                with torch.no_grad():                   # keep the eager fallback grad-consistent
                    tout = _walk_trace(ph, fh, pp, tri.detach(), adj, slot_tbl,
                                       face_normals.detach(), max_cross=max_cross, active=walkw)
        if tout is None:
            tout = _walk_trace(ph, fh, pp, tri, adj, slot_tbl, face_normals,
                               max_cross=max_cross, active=walkw)
        fpw, cpw, ncw, aimw, okw, over = tout
        wk['wk_fb_cross'] = int((walkw & over).sum())
        wk['wk_fb_geom'] = int((walkw & ~okw & ~over).sum())
        walked_ok = walkw & okw
        wk['wk_walked'] = int(walked_ok.sum()) + int(seat_same.sum())
        wk['wk_cross_sum'] = int(ncw[walked_ok].sum())
        wk['wk_cross_max'] = int(ncw[walked_ok].max()) if int(walked_ok.sum()) else 0
        keep = walked_ok
        ncross[idxw] = torch.where(keep, ncw, torch.zeros_like(ncw))
        aim_final[idxw] = torch.where(keep.unsqueeze(-1), aimw, aim_final[idxw])
        kk = torch.arange(max_cross, device=dev).unsqueeze(0)
        live = keep.unsqueeze(1) & (kk < ncw.unsqueeze(1))                  # (W,max_cross)
        cross_pts[idxw] = torch.where(live.unsqueeze(-1), cpw, cross_pts[idxw])
        faces_path[idxw, 1:] = torch.where(live, fpw[:, 1:], faces_path[idxw, 1:])
        fail[idxw] |= walkw & ~walked_ok

    amb_out = amb | fail
    c_max = max(int(ncross.max()), 0)
    S = c_max + 1
    # base (in-plane) arc fractions for the clearance lerp -- computed BEFORE lifting, since the
    # lift depends on them. Padded slots repeat aim_final => zero base length beyond each row's C.
    jj = torch.arange(c_max, device=dev).unsqueeze(0)
    validj = jj < ncross.unsqueeze(1)                       # (N,c_max)
    qpts = torch.where(validj.unsqueeze(-1), cross_pts[:, :c_max],
                       aim_final.unsqueeze(1).expand(N, max(c_max, 1), 3)[:, :c_max])
    base = torch.cat([seat_h_pt.unsqueeze(1), qpts, aim_final.unsqueeze(1)], dim=1)
    cumb = torch.cumsum((base[:, 1:] - base[:, :-1]).norm(dim=-1), dim=1)   # (N,c_max+1)
    t_frac = cumb[:, :c_max] / cumb[:, -1:].clamp(min=1e-12)                # (N,c_max)
    if c_max:
        d_j = gh_eff + t_frac * (gp_eff - gh_eff)           # RAW clearance lerp (S2), (N,c_max)
        f_prev = faces_path[:, :c_max]
        f_next = faces_path[:, 1:c_max + 1]
        bis = face_normals[f_prev] + face_normals[f_next]   # junction BISECTOR lift (rounds ridge)
        bn = bis.norm(dim=-1, keepdim=True)
        knife = (bn.reshape(N, c_max) <= 1e-6) & validj     # opposing normals: knife edge
        amb_out = amb_out | knife.any(dim=1)
        # knife rows are reverted above; the prev-face normal keeps their VALUES sane so the
        # sm_arc_infl_* diagnostics (computed over all rows) are not polluted by a 1e30 lift.
        bis = torch.where(bn > 1e-6, bis / bn.clamp(min=1e-30), face_normals[f_prev])
        lifted = cross_pts[:, :c_max] + d_j.unsqueeze(-1) * bis
        interior = torch.where(validj.unsqueeze(-1), lifted,
                               x_pre.unsqueeze(1).expand(N, c_max, 3))
        interior = torch.where(torch.isfinite(interior), interior,
                               x_pre.unsqueeze(1).expand(N, c_max, 3))
        P = torch.cat([x_hat.unsqueeze(1), interior, x_pre.unsqueeze(1)], dim=1)
    else:
        P = torch.stack([x_hat, x_pre], dim=1)
    seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1)               # (N,S)
    L = seg.sum(dim=1, keepdim=True)
    ks = torch.arange(S, device=dev).unsqueeze(0).expand(N, S)
    n_seg = face_normals[faces_path.gather(1, torch.minimum(ks, ncross.unsqueeze(1)))]
    return P, seg, L, chord, amb_out, ncross, g_hat, g_pre, n_seg, wk


def _polyline_point_at_arc(P, seg, L, s):
    """Point on polyline P (N,S+1,3) at arc position s (N,1), clamped to [0, L]. Linear
    interpolation inside the containing segment (vectorized over N)."""
    N = P.shape[0]
    cum = torch.cumsum(seg, dim=1)                                   # (N,S) arc at nodes 1..S
    cum0 = torch.cat([torch.zeros(N, 1, device=P.device, dtype=P.dtype), cum], dim=1)  # (N,S+1)
    s = s.clamp(min=0.0)
    s = torch.minimum(s, L)                                          # (N,1)
    j = (cum0 <= s).sum(dim=1) - 1                                  # (N,) lower-node index
    j = j.clamp(0, seg.shape[1] - 1)
    ar = torch.arange(N, device=P.device)
    p0 = P[ar, j]                                                    # (N,3)
    p1 = P[ar, j + 1]                                                # (N,3)
    seg_j = seg[ar, j].clamp(min=1e-12)                             # (N,)
    local = ((s.reshape(-1) - cum0[ar, j]) / seg_j).clamp(0.0, 1.0)  # (N,)
    return p0 + (p1 - p0) * local.unsqueeze(-1)


def multi_normal_arc_landing(seg, cand):
    """: the landing-segment prox walk-back over a segmented path with
    per-segment Coulomb landing candidates. Pure tensor math, no geometry dependency — the
    directly unit-testable core (same discipline as coulomb_block_soft_threshold).

    The arrest condition uses the load WHERE THE ENDPOINT LANDS: with piecewise-constant load,
    s* solves  L - s* = b(s*)  (band-regularized upstream, inside `cand`). Segments are walked
    far-end-first (prox-from-L = least intervention; resolves non-uniqueness when the load grows
    toward the anchor). At segment k with arc interval [lo_k, hi_k]:

      * cand_k > hi_k         => every segment further right already rejected LEFTWARD, and this
                                 segment's cone zero-crossing lies right of it => the piecewise
                                 objective's minimum sits ON the junction: land at hi_k.
      * lo_k <= cand_k <= hi_k => interior acceptance: land at cand_k.
      * cand_k < lo_k          => the cone here pulls further left: continue.

    Because lo_0 = 0 and cand >= 0, segment 0 always accepts => the walk always lands (full
    stick emerges as cand_0 = 0). `exhausted` can only fire on non-finite candidates and is a
    NaN guard, not a physics outcome.

    Uniform candidates (all cand_k equal — the single-normal limit) reproduce the arc-cone
    landing EXACTLY: the containing segment interior-accepts the common value, and the junction
    branch can never fire (descending past segment k requires cand < lo_k = hi_{k-1}).

    :param seg:  (N,S) segment lengths (>= 0)
    :param cand: (N,S) per-segment landing candidates, each in [0, L]
    :return: (s_star (N,1), land_k (N,) landing segment index (-1 on exhaustion),
              junction (N,) bool landed on a junction, exhausted (N,) bool NaN-guard fallback)
    """
    N, S = seg.shape
    cum = torch.cumsum(seg, dim=1)                                   # (N,S) right edges
    cum0 = torch.cat([seg.new_zeros(N, 1), cum[:, :-1]], dim=1)      # (N,S) left edges
    s_star = seg.new_zeros(N)
    land_k = torch.full((N,), -1, dtype=torch.long, device=seg.device)
    junction = torch.zeros(N, dtype=torch.bool, device=seg.device)
    done = torch.zeros(N, dtype=torch.bool, device=seg.device)
    for k in range(S - 1, -1, -1):
        c_k, hi, lo = cand[:, k], cum[:, k], cum0[:, k]
        acc_j = (~done) & (c_k > hi)
        acc_i = (~done) & (c_k >= lo) & (c_k <= hi)
        s_star = torch.where(acc_j, hi, torch.where(acc_i, c_k, s_star))
        land_k = torch.where(acc_j | acc_i, torch.full_like(land_k, k), land_k)
        junction |= acc_j
        done |= acc_j | acc_i
    return s_star.unsqueeze(-1), land_k, junction, ~done


def geometry_aware_transport(x_hat, x_pre, x_t, cap, factor, b, band,
                             face_centroids, face_normals, mode,
                             arc_cone='off', substeps=8, arc_cap=3.0,
                             iters=2, slack=1e-4, K=4, ambig_ratio=1.5, ambig_cos=0.0,
                             resid_tol=1e-6, return_tangential=False,
                             path_normals='single', path_capstan='off', drag_k=0.0,
                             mu=0.0, h=None, m_sub=None,
                             drive_gi=None, gi_proj=None, lam_primary=None,
                             path_builder='march', obs_pos=None, obs_faces=None,
                             walk_max_cross=16, walk_fast=False):
    """: geometry-aware friction transport =
    (surface-marched tangential walk-back) o (the UNCHANGED endpoint restore).

    The friction shrink is realized by walking BACK along the marched surface path (surface_march)
    to the retained arc length, then re-applying normal_completion_plane_gap at the landing point q.
    Per-vertex FALLBACK to the exact chord result on march ambiguity or pathological arc
    inflation => the robustness floor is (never worse by construction, modulo the arc-cone
    mu-response which is a separate knob).

    :param x_t: (N,3) the chord landing (fallback output + displacement stats only)
    :param factor: (N,1) chord cone factor in [0,1]; arc_cone='off' retains factor*L of arc
    :param b:      (N,1) Coulomb budget mu*lambda_n*h^2/m; arc_cone='on' removes exactly b of arc
    :param band:   scalar eps_v*h static-velocity regularizer (mirrors the chord cone's band)
    :param return_tangential:. Also return the landing point of the TANGENTIAL
             transport alone, i.e. before the endpoint restore (per-vertex: the marched `q`, or the
              chord `x_t` on reverted vertices). The composite action objective prices its
             dissipation there, because the restore moves along the LOCAL surface normal and motion
             along the local normal does no frictional work. Default False => the return signature
             and every existing caller are unchanged.
    :param path_normals:. 'single' (default) = the byte-identical walk-back
             (one λ, one budget for the whole arc). 'multi' = per-SEGMENT Coulomb landing: the
             gravity+inertia drive vector is re-projected onto each march segment's own normal
             (the elastic channel stays a frozen primary-frame scalar), giving per-segment budgets
             b_k = mu*λ_k*h²/m, and the landing is the far-end-first landing-segment prox
             (multi_normal_arc_landing). Requires arc_cone='on' plus the mu/h/m_sub/drive_gi/
             gi_proj/lam_primary kwargs below.
    :param path_capstan:, 'multi' only. 'on' amplifies each segment's budget by
             the accumulated junction wrap e^{mu*Σ Δθ_j} over the slid portion (discrete belt/
             capstan surcharge — corners resist sliding through changing contact normals). 'off'
             (default) = per-segment loads only (the isolation arm).
    :param mu, h, m_sub: scalar Coulomb coefficient / timestep, (N,1) nodal mass — used to build
             the per-segment budgets with EXACTLY the caller's b expression (mu*λ*(h*h)/m), so the
             uniform-normal limit is bitwise v17.
    :param drive_gi: (N,3) gravity+inertia drive vector G_gi (energy-gradient convention).
    :param gi_proj: (N,1) its primary-frame projection (G_gi·n̂) — passed so the per-segment load
             is the DELTA form λ_k = clamp(lam_primary + G_gi·n_k − gi_proj, 0), which is bitwise
             the caller's λ_n wherever n_k equals the primary normal.
    :param lam_primary: (N,1) the caller's PRE-CLAMP primary-frame λ scalar (grav+inertia+elastic).
    :param path_builder:. 'march' (default, byte-identical) = surface_march at
             `substeps`. 'walk' = the parameter-free triangle walk (triangle_walk_path): crossed
             faces are the segments, `substeps` is ignored, per-class fallbacks land in the same
             per-vertex revert (floor =) and are counted under wk_*. Landing / completion /
             arc-cap guard / wave seat below are the SAME code for both builders.
    :param obs_pos, obs_faces: NEXT-pose body vertices (Vo,3) and faces (Fo,3); required by (and
             only read under) path_builder='walk' — the walk needs the actual triangles, not just
             centroid/normal summaries.
    :param walk_max_cross: 'walk' only — SAFETY cap on edge crossings per pair (never-binds class,
             like path_arc_cap; exceed => chord fallback, counted wk_fb_cross). Not a resolution.
    :param walk_fast:, 'walk' only — merged seating + CUDA-graph replay of both walk loops.
             Scheduling only: bitwise-identical output, declines to the eager loop on any obstacle.
    :return: (out (N,3)[, q_eff (N,3)], stats): stats carry the ne_* baseline-completion counters
             PLUS the surface-specific sm_* counters (sm_marched / sm_fallback_* / sm_arc_infl_*)
             PLUS, under 'multi', the mn_* counters (mn_active / mn_landing_moved / mn_junction_land
             / mn_fullstick / mn_lambda_spread_max / mn_wrap_sum / mn_capstan_gain_max).
    """
    if path_normals not in ('single', 'multi'):
        raise ValueError(f"geometry_aware_transport: unknown path_normals {path_normals!r}")
    if path_builder not in ('march', 'walk'):
        raise ValueError(f"geometry_aware_transport: unknown path_builder {path_builder!r}")
    if path_builder == 'walk' and (obs_pos is None or obs_faces is None):
        raise ValueError("path_builder='walk' needs obs_pos/obs_faces: the walk crosses "
                         "actual triangles, not centroid/normal summaries.")
    if path_normals == 'multi':
        if arc_cone != 'on':
            raise ValueError("path_normals='multi' requires arc_cone='on': the chord-factor "
                             "'off' law has no seat for per-segment loads.")
        if any(v is None for v in (h, m_sub, drive_gi, gi_proj, lam_primary)):
            raise ValueError("path_normals='multi' needs mu/h/m_sub/drive_gi/gi_proj/lam_primary.")
    # 1) fallback / baseline completion — also the output for guarded vertices.
    v13_out, v13_stats = normal_completion_plane_gap(
        x_pre, x_t, cap, face_centroids, face_normals, mode,
        iters=iters, slack=slack, K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos,
        resid_tol=resid_tol)
    # 2) build the slide path ('multi' additionally keeps the per-segment normals).
    wk_stats = None
    if path_builder == 'walk':
        P, seg, L, chord, amb_march, n_switch, g_hat, g_pre, n_seg, wk_stats = triangle_walk_path(
            x_hat, x_pre, obs_pos, obs_faces, face_centroids, face_normals,
            max_cross=walk_max_cross, K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos,
            fast=walk_fast)
    elif path_normals == 'multi':
        P, seg, L, chord, amb_march, n_switch, g_hat, g_pre, n_seg = surface_march(
            x_hat, x_pre, face_centroids, face_normals, substeps=substeps,
            K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos, return_normals=True)
    else:
        P, seg, L, chord, amb_march, n_switch, g_hat, g_pre = surface_march(
            x_hat, x_pre, face_centroids, face_normals, substeps=substeps,
            K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos)
    # 3) retained arc length: 'off' reuses the chord factor; 'on' is the arc-native cone
    #    (removes exactly the Coulomb budget b of TRUE slide distance).
    mn = None                                                       # stats bundle
    if arc_cone == 'on':
        reg_L = torch.sqrt(L * L + float(band) * float(band))
        if path_normals == 'multi':
            # per-segment loads via DELTA-form re-projection of the exact
            # gravity+inertia drive (the gated elastic channel rides along frozen at the primary
            # frame, identical in every λ_k). Where n_k equals the primary normal the delta is
            # exactly zero, so λ_k — and the budget b_k below, built with the caller's own
            # expression — is bitwise the single-normal value (the uniform-limit gate).
            lam_seg = torch.clamp(
                lam_primary + (drive_gi.unsqueeze(1) * n_seg).sum(dim=-1) - gi_proj, min=0.0)  # (n,S)
            wrap_step = torch.acos(
                (n_seg[:, 1:] * n_seg[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0))                   # (n,S-1)
            wrap_cum = torch.cat([seg.new_zeros(seg.shape[0], 1),
                                  torch.cumsum(wrap_step, dim=1)], dim=1)                      # (n,S)
            b_seg = float(mu) * lam_seg * (h * h) / m_sub                                      # (n,S)
            gain = None
            if path_capstan == 'on':
                # discrete belt/capstan: budget arresting slide that crossed junctions is
                # amplified by the wrap accumulated over the slid portion [0, s*].
                gain = torch.exp(float(mu) * wrap_cum)
                b_seg = b_seg * gain
            cand = torch.clamp(1.0 - b_seg / reg_L, min=0.0) * L                               # (n,S)
            s_star, land_k, junction, exhausted = multi_normal_arc_landing(seg, cand)
            f_arc_single = torch.clamp(1.0 - b / reg_L, min=0.0)      # reference (stats only)
            mn = dict(s_single=f_arc_single * L, lam_seg=lam_seg, wrap_cum=wrap_cum,
                      gain=gain, s_star=s_star, junction=junction, exhausted=exhausted)
        else:
            f_arc = torch.clamp(1.0 - b / reg_L, min=0.0)
            s_star = f_arc * L
    elif arc_cone == 'off':
        s_star = factor * L
    else:
        raise ValueError(f"geometry_aware_transport: unknown arc_cone {arc_cone!r}")
    if drag_k:
        # over-budget drag: multiplicatively shrink the RETAINED slide
        # arc by exp(-drag_k*mu). Rate-type, NOT Coulomb -- a ceiling probe for per-vertex slowing,
        # never a ship path. Acts only on the marched set (mu=0 never reaches here: the ne set is
        # empty). Stick (s_star=0) is unchanged; sliders stay ON the marched path, just land earlier.
        s_star = s_star * float(np.exp(-float(drag_k) * float(mu)))
    # 4) walk back to q, then re-apply the endpoint restore there (near-sheet correspondence).
    q = _polyline_point_at_arc(P, seg, L, s_star)
    surf_out, _ = normal_completion_plane_gap(
        x_pre, q, cap, face_centroids, face_normals, mode,
        iters=iters, slack=slack, K=K, ambig_ratio=ambig_ratio, ambig_cos=ambig_cos,
        resid_tol=resid_tol)
    # 5) guard: revert to the chord result on ambiguous march or arc blow-up.
    arc_cap_hit = (L > float(arc_cap) * chord).reshape(-1)
    revert = amb_march | arc_cap_hit
    out = torch.where(revert.unsqueeze(-1), v13_out, surf_out)

    marched = ~revert
    dn = (out - x_t).norm(dim=-1)
    infl = L.reshape(-1) / chord.reshape(-1).clamp(min=1e-12)
    stats = dict(v13_stats)                                           # ne_* = baseline completion
    stats.update({
        'sm_candidates': int(x_hat.shape[0]),
        'sm_marched': int(marched.sum()),
        'sm_fallback_ambig': int(amb_march.sum()),
        'sm_fallback_cap': int((arc_cap_hit & ~amb_march).sum()),
        'sm_face_switch': int(n_switch.sum()),
        'sm_arc_infl_sum': float(infl[marched].sum()) if int(marched.sum()) else 0.0,
        'sm_arc_infl_max': float(infl.max()) if infl.numel() else 0.0,
        'sm_dn_sum': float(dn.sum()) if dn.numel() else 0.0,
        'sm_dn_max': float(dn.max()) if dn.numel() else 0.0,
    })
    if wk_stats is not None:
        stats.update(wk_stats)                              # per-class walk counters
    if mn is not None:
        # counters, restricted to the marched (non-reverted) set the multi law actually acts on.
        mm = marched
        n_mm = int(mm.sum())
        moved = ((mn['s_star'] - mn['s_single']).abs().reshape(-1) > 1e-12) & mm
        lamp = lam_primary.clamp(min=0.0).reshape(-1)
        sp_mask = mm & (lamp > 0)
        spread = (mn['lam_seg'].max(dim=1).values.reshape(-1)[sp_mask]
                  / lamp[sp_mask]) if int(sp_mask.sum()) else lamp.new_zeros(0)
        stats.update({
            'mn_active': n_mm,
            'mn_landing_moved': int(moved.sum()),
            'mn_junction_land': int((mn['junction'] & mm).sum()),
            'mn_fullstick': int(((mn['s_star'].reshape(-1) <= 0) & mm).sum()),
            'mn_exhausted': int((mn['exhausted'] & mm).sum()),
            'mn_lambda_spread_max': float(spread.max()) if spread.numel() else 0.0,
            'mn_wrap_sum': float(mn['wrap_cum'][mm, -1].sum()) if n_mm else 0.0,
            'mn_capstan_gain_max': (float(mn['gain'][mm].max()) if (mn['gain'] is not None and n_mm)
                                    else 1.0),
        })
    if return_tangential:
        # the pure tangential outcome, matching `out`'s per-vertex marched/reverted split
        return out, torch.where(revert.unsqueeze(-1), x_t, q), stats
    return out, stats


def vertex_areas(pos, faces):
    """diagnostic: per-vertex barycentric area (V,1) -- one third of each incident face's area.

    Summing over vertices recovers the mesh area exactly, so this is a partition, the same
    construction edge_membrane_stiffness uses for edges. Computed on REST positions when available:
    the weight is a property of the material's own discretisation, not of the current deformation,
    so it stays constant through a rollout and can be cached.
    """
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    a = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 3.0          # (F,)
    out = torch.zeros((pos.shape[0], 1), dtype=pos.dtype, device=pos.device)
    for k in range(3):
        out.index_add_(0, faces[:, k], a.unsqueeze(-1))
    return out


def ring_spread(corr_T, donor_m, pinned, faces, edges=None, vw=None, stats=None):
    """diagnostic: spread donor corrections to their 1-ring, uniform mean.

    The crude precursor of the C1 compliant patch solve. Pure function of (corrections, donor set,
    topology) — no physics inside, which is what makes it testable by exact arithmetic.

    :param corr_T:  (V,3) per-vertex TANGENTIAL correction (only donor rows are read)
    :param donor_m: (V,) bool, vertices donating their correction (the moved contacts)
    :param pinned:  (V,) bool, HANDLE vertices — never receive (they are hard-driven anyway)
    :param faces:   (F,3) long cloth faces — the 1-ring topology
    :param edges:   optional precomputed (E,2) directed edge list (hoisted out of the C2 hop loop)
    :param vw:      optional (V,1) per-source WEIGHT. None (default) = uniform, i.e. the plain
                    arithmetic mean over distinct neighbours, byte-identical to the original law.
                    diagnostic passes per-vertex AREA here: a uniform mean weights by mesh connectivity
                    rather than geometry, which on a 3x-non-uniform triangulation biases the reach
                    toward densely-meshed regions.
    :param stats:   optional dict; when given, fills diagnostic counters (source multiplicity and
                    the cancellation ratio |SUM w d| / SUM w|d| over multi-source receivers).
                    Pure observation -- it cannot change the returned field.
    :return: (dx, recv_m) with dx (V,3) the per-receiver mean donor correction (zero elsewhere,
             UNSCALED — the caller applies spread_w) and recv_m (V,) the receiver set.

    Semantics: a receiver is a vertex that is NOT itself a donor, is NOT pinned, and has >= 1 donor
    among its DISTINCT 1-ring neighbors; it receives the arithmetic mean of those donors'
    corrections. Distinct matters: an interior edge belongs to two faces, so the raw face-derived
    edge list double-counts it and would weight interior donors 2x in the "uniform" mean.
    """
    if edges is None:
        e = torch.cat([faces[:, [0, 1]], faces[:, [1, 0]], faces[:, [1, 2]],
                       faces[:, [2, 1]], faces[:, [0, 2]], faces[:, [2, 0]]], 0)
        e = torch.unique(e, dim=0)                                      # (E,2) distinct src->dst
    else:
        e = edges                       # diagnostic: hoisted out of the multi-hop loop (same list)
    donor_f = donor_m.unsqueeze(-1).to(corr_T.dtype)                    # (V,1)
    if vw is not None:
        donor_f = donor_f * vw          # diagnostic: weight each source by its own area
    num = torch.zeros_like(corr_T)
    num.index_add_(0, e[:, 1], (corr_T * donor_f)[e[:, 0]])
    den = torch.zeros_like(donor_f)
    den.index_add_(0, e[:, 1], donor_f[e[:, 0]])
    recv_m = (~donor_m) & (~pinned) & (den.reshape(-1) > 0)
    # the clamp is only a divide-by-zero guard -- for a receiver den is >= 1 under uniform weights
    # (a count) but ~1e-4 under area weights (a sum of m^2), so the floor has to follow the units.
    den_safe = den.clamp(min=1.0) if vw is None else den.clamp(min=1e-30)
    dx = torch.where(recv_m.unsqueeze(-1), num / den_safe, torch.zeros_like(num))
    if stats is not None:
        with torch.no_grad():
            # CANCELLATION: |SUM w d| / SUM w|d| per receiver. 1 = the sources agree; 0 = they null
            # each other out. Averaging is what destroyed the capstan signal, so this
            # is the term that says how much of the field a mean is throwing away.
            mag = torch.zeros_like(den)
            mag.index_add_(0, e[:, 1],
                           (corr_T.norm(dim=-1, keepdim=True) * donor_f)[e[:, 0]])
            cnt = torch.zeros_like(den)
            cnt.index_add_(0, e[:, 1], donor_m.unsqueeze(-1).to(den.dtype)[e[:, 0]])
            multi = recv_m & (cnt.reshape(-1) >= 2)
            stats['sp_recv_n'] = stats.get('sp_recv_n', 0) + int(recv_m.sum())
            stats['sp_src_sum'] = stats.get('sp_src_sum', 0.0) + float(cnt[recv_m].sum())
            if int(multi.sum()):
                r = num[multi].norm(dim=-1) / mag[multi].reshape(-1).clamp(min=1e-30)
                stats['sp_multi'] = stats.get('sp_multi', 0) + int(multi.sum())
                stats['sp_cancel_sum'] = stats.get('sp_cancel_sum', 0.0) + float(r.sum())
    return dx, recv_m


def mesh_edges_undirected(faces):
    """diagnostic helper: the mesh's DISTINCT undirected edges, canonicalised i<j.

    :param faces: (F,3) long
    :return: (uni (Eu,2) long with uni[:,0] < uni[:,1], inv (3F,) index of each face-edge into uni,
             in the face-edge order [e01 (F rows) ; e12 (F rows) ; e02 (F rows)])

    The inverse index is what lets a per-FACE quantity (rest area) be accumulated onto edges
    without an explicit edge->face adjacency structure.
    """
    ef = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)   # (3F,2)
    ef = torch.stack([ef.min(dim=1).values, ef.max(dim=1).values], dim=1)       # canonical i<j
    return torch.unique(ef, dim=0, return_inverse=True)


def ring_dilate(mask, uni, depth=1):
    """diagnostic helper: grow a vertex mask by `depth` 1-ring hops. Pure topology, symmetric.

    :param mask:  (V,) bool seed set
    :param uni:   (Eu,2) undirected edges (from mesh_edges_undirected)
    :param depth: number of hops (0 => mask returned unchanged)
    """
    out = mask
    for _ in range(int(depth)):
        acc = torch.zeros(out.shape[0], dtype=torch.int32, device=out.device)
        acc.index_add_(0, uni[:, 1], out[uni[:, 0]].to(torch.int32))
        acc.index_add_(0, uni[:, 0], out[uni[:, 1]].to(torch.int32))
        out = out | (acc > 0)
    return out


def edge_membrane_stiffness(rest_pos, faces, uni, inv, lame_mu, lame_lambda):
    """diagnostic: per-edge axial stiffness k_j [N/m] from the garment's OWN material.

    The standard membrane -> edge-spring equivalence, so C1's compliance carries no fitted constant:

        Y_2D = 4 mu (mu + lambda) / (2 mu + lambda)      [plane-stress 2D Young's modulus, N/m]
        Abar_j = (1/3) * sum of the REST areas of the faces incident to edge j      [m^2]
        k_j = Y_2D * Abar_j / L_j^2                                                 [N/m]

    (units check: [N/m][m^2]/[m^2] = N/m). The 1/3 splits each face's area equally over its three
    edges, so summing Abar over all edges recovers the garment's rest area exactly.

    :param rest_pos: (V,3) UNDEFORMED vertex positions (cloth.rest_pos)
    :param faces:    (F,3) long
    :param uni, inv: from mesh_edges_undirected(faces)
    :param lame_mu, lame_lambda: scalars (the sample's per-garment Lame parameters)
    :return: (Eu,1) k_j
    """
    v0, v1, v2 = rest_pos[faces[:, 0]], rest_pos[faces[:, 1]], rest_pos[faces[:, 2]]
    area = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)             # (F,) rest area
    a3 = (area / 3.0).repeat(3).unsqueeze(-1)                                   # (3F,1), matches `inv`
    abar = torch.zeros((uni.shape[0], 1), dtype=rest_pos.dtype, device=rest_pos.device)
    abar.index_add_(0, inv, a3)                                                 # (Eu,1)
    lr = (rest_pos[uni[:, 0]] - rest_pos[uni[:, 1]]).norm(dim=-1, keepdim=True)  # (Eu,1)
    lm, ll = float(lame_mu), float(lame_lambda)
    y2d = 4.0 * lm * (lm + ll) / max(2.0 * lm + ll, 1e-30)
    return y2d * abar / lr.clamp(min=1e-12).pow(2)


def patch_relax(x, x0, edges, w_inv, alpha_t, iters, k_edge=None, feas_tol=1e-6):
    """diagnostic: XPBD compliant patch relaxation — Macklin/Muller/Chentanez Eq. 18.

    Per constraint j on edge (a,b), one Gauss-Seidel update is

        dlambda_j = (-C_j - alpha~_j lambda_j) / (grad C_j M^-1 grad C_j^T + alpha~_j)
        dx_a = +w_a n dlambda_j ,  dx_b = -w_b n dlambda_j ,  n = (x_a - x_b)/|x_a - x_b|

    with grad C M^-1 grad C^T = w_a + w_b for a distance constraint. We sweep JACOBI (all
    constraints from the same x, then apply, dividing each vertex's accumulated correction by the
    number of constraints touching it — the standard GPU mass-splitting of Macklin et al.); the
    per-constraint multiplier lambda_j is kept, so the compliance semantics of XPBD survive the
    change of sweep order. This is a documented deviation from Eq. 18's sequential order, taken for
    vectorisation; it converges more slowly, which is exactly what `iters` and the logged residual
    measure.

    TARGET: L_j = |x0_a - x0_b|, the
    CURRENT length under the frictionless prediction x0 — NOT the rest length. Our garments sit at
    ~8% strain, so projecting to rest would contract the patch every frame: a drape intervention
    that would flatter the stretch guardrail for a reason unrelated to coupling. With the current
    length as target, C_j == 0 on every edge whose endpoints are both free, so the solve is driven
    PURELY by the anchors' displacement — it is a coupling operator and nothing else.

    ANCHORING is encoded entirely in `w_inv`: a vertex with w_inv == 0 receives zero correction by
    construction (dx = w * ... ), so no masking is needed and the invariant is directly testable.

    :param x:       (V,3) positions to relax (anchors already carry the operator's correction)
    :param x0:      (V,3) metric reference (the frictionless prediction) — supplies the targets
    :param edges:   (E,2) long, distinct undirected edges with >= 1 free endpoint
    :param w_inv:   (V,1) inverse mass, ZERO on anchors / pinned / outside-patch vertices
    :param alpha_t: (E,1) alpha~_j = alpha_j / h^2 = 1 / (k_j h^2). Zero => stiff PBD projection.
    :param iters:   K sweeps
    :param k_edge:  (E,1) axial stiffness, for the injected-elastic-energy diagnostic only
    :param feas_tol: absolute feasibility tolerance [m] for the iterations-to-feasibility counter
    :return: (x_new (V,3), stats dict of ABSOLUTE floats — requires absolute, not relative)
    """
    a, b = edges[:, 0], edges[:, 1]
    tgt = (x0[a] - x0[b]).norm(dim=-1, keepdim=True)                            # (E,1) current-length
    wa, wb = w_inv[a], w_inv[b]                                                 # (E,1)
    wsum = wa + wb
    lam = torch.zeros_like(tgt)
    cnt = torch.zeros_like(w_inv)                                               # (V,1) constraints/vertex
    ones = torch.ones_like(tgt)
    cnt.index_add_(0, a, ones)
    cnt.index_add_(0, b, ones)
    cnt = cnt.clamp(min=1.0)
    x_in, x = x, x.clone()
    c_pre = None
    it_feas = float(int(iters) + 1)                                             # sentinel: never reached
    for k in range(int(iters)):
        d = x[a] - x[b]
        l = d.norm(dim=-1, keepdim=True)
        c = l - tgt                                                             # (E,1) residual
        if c_pre is None:
            c_pre = c.detach().abs()
        if float(c.detach().abs().max()) < feas_tol:
            it_feas = float(k)
            break
        n = d / l.clamp(min=1e-12)
        dlam = (-c - alpha_t * lam) / (wsum + alpha_t).clamp(min=1e-30)
        lam = lam + dlam
        acc = torch.zeros_like(x)
        acc.index_add_(0, a, (wa * dlam) * n)
        acc.index_add_(0, b, -(wb * dlam) * n)
        x = x + acc / cnt
    with torch.no_grad():
        d = x[a] - x[b]
        c_post = (d.norm(dim=-1, keepdim=True) - tgt).abs()
        if c_pre is None:                                                       # iters == 0
            c_pre = c_post
        if float(c_post.max()) < feas_tol and it_feas > int(iters):
            it_feas = float(int(iters))
        dxn = (x - x_in).norm(dim=-1)
        # alpha~ / (grad C M^-1 grad C^T + alpha~): 0 => stiff regime, 1 => fully compliant.
        # Logged because the material value is expected to be ~1e-6 here and that
        # claim must be measured on the real garments, not asserted from a back-of-envelope.
        ratio = float((alpha_t / (wsum + alpha_t).clamp(min=1e-30)).mean()) if wsum.numel() else 0.0
        stats = {
            'pc_edges': int(edges.shape[0]),
            'pc_C_pre_sum': float(c_pre.sum()), 'pc_C_pre_max': float(c_pre.max()),
            'pc_C_post_sum': float(c_post.sum()), 'pc_C_post_max': float(c_post.max()),
            'pc_iters_feas': it_feas,
            'pc_alpha_ratio': ratio,
            'pc_dx_sum': float(dxn.sum()), 'pc_dx_max': float(dxn.max()),
            'pc_energy_inj': (float((0.5 * k_edge * c_post.pow(2)).sum())
                              if k_edge is not None else 0.0),
        }
    return x, stats


def ring_spread_multi(corr_T, donor_m, pinned, faces, hops, gamma, vw=None, stats=None):
    """diagnostic: MULTI-HOP damped propagation of the correction field.

    C0 stops at the 1-ring. This walks the correction outward as a BFS wavefront -- a receiver at
    hop k becomes a source at hop k+1 -- attenuating by `gamma` per hop, i.e. a screened harmonic
    extension of the contact corrections over the mesh graph.

    THE DESIGN CHOICE THAT MAKES THE LADDER READABLE: hop 1 keeps EXACTLY C0's displacement, and
    gamma governs only hops >= 2. So gamma = 0 is byte-identically C0 at any hop count, every rung
    is a strict superset of C0, and the whole difference between a C2 arm and the frozen C0
    measurement is ADDED REACH -- never a re-scaled near field.

        hop 1 : mean over distinct donor neighbours                  (== C0)
        hop k : gamma^(k-1) * mean over assigned hop-(k-1) neighbours

    Each vertex is assigned exactly ONCE, at its hop distance from the donor set, so the hops are
    disjoint and the accumulated field has no double counting. PINNED vertices neither receive nor
    CONDUCT: they are hard-driven, so their displacement is not ours to propagate, and the front
    routes around them.

    Physical reading: gamma sets a screening length. The unscreened limit (gamma=1) is what an
    implicit solver does -- in-plane this fabric is near-inextensible (alpha_ratio ~1e-6, and the
    in-plane wave crosses metres per timestep), so the true support of a friction drag is wide and
    C0's 1-ring is a severe truncation. What bounds the real correlation length is out-of-plane
    buckling, not the wave speed, which is why gamma is SWEPT rather than derived.

    :param hops:  maximum hop count (numerical, like path_substeps -- large enough that gamma is
                  the binding knob). The walk stops early when the front dies.
    :param gamma: per-hop attenuation in [0, 1]. 0 => exactly C0.
    :return: (dx (V,3) accumulated UNSCALED field, recv_m (V,) all receivers, hops_used int)
    """
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 0]], faces[:, [1, 2]],
                   faces[:, [2, 1]], faces[:, [0, 2]], faces[:, [2, 0]]], 0)
    e = torch.unique(e, dim=0)                                          # hoisted: reused per hop
    dx, recv = ring_spread(corr_T, donor_m, pinned, faces, e, vw, stats)  # hop 1 == exactly C0
    assigned = donor_m | recv
    prev_recv, prev_dx, used = recv, dx, 1
    for _ in range(2, int(hops) + 1):
        if not int(prev_recv.sum()):
            break
        # sources = the previous hop's receivers; everything already assigned (and everything
        # pinned) is blocked from receiving again, which is what keeps the hops disjoint.
        dxk, recvk = ring_spread(prev_dx, prev_recv, pinned | assigned, faces, e, vw)
        if not int(recvk.sum()):
            break
        dxk = dxk * float(gamma)                    # prev_dx already carries gamma^(k-2)
        dx = dx + dxk
        assigned = assigned | recvk
        prev_recv, prev_dx, used = recvk, dxk, used + 1
    return dx, assigned & (~donor_m), used


def bolt_on_friction_projection(sample, mu, eps=2e-3, stiffness=1.0, eps_v=1e-2,
                                g=9.81, z_axis=2, return_diag=False, return_force=False,
                                lambda_n_source='barrier', fb_use_gravity=True,
                                fb_use_inertia=True, elastic_grad=None,
                                fb_elastic_conform=False, fb_elastic_pen_gate=False,
                                fb_elastic_smooth_iters=0, fb_elastic_pen_dilate=False,
                                fb_elastic_approach_gate=False, fb_elastic_noise_floor_q=0.0,
                                fb_elastic_deflate_profile=False, lambda_smooth_iters=0,
                                lambda_smooth_window=0.0, lambda_smooth_conserve=True,
                                lambda_smooth_render_only=False,
                                normal_exact='off', normal_exact_iters=2,
                                normal_exact_slack=1e-4, normal_exact_max_step=0.0,
                                normal_exact_dust=1e-8, normal_exact_ambig_k=4,
                                normal_exact_ambig_cos=0.0, normal_exact_ambig_ratio=1.5,
                                normal_exact_control=False,
                                tangential_path='chord', path_substeps=8,
                                path_arc_cone='off', path_arc_cap=3.0,
                                path_normals='single', path_capstan='off',
                                path_walk_max_cross=16, path_walk_fast=False, drag_k=0.0,
                                spread_w=0.0, spread_project='off',
                                spread_hops=1, spread_gamma=0.0, spread_weight='uniform',
                                patch_iters=0, patch_depth=1, patch_alpha_scale=1.0,
                                patch_signorini='preserve', patch_feas_tol=1e-6,
                                wave_enabled=False, wave_visco=0.0, wave_tau_frames=0.5,
                                wave_strain_cap=0.0,
                                return_metric_log=False, shadow=False,
                                detach_lambda_n=False, return_action=False):
    """bolt-on operator (cubic-barrier proposal): an INFERENCE-TIME, parameter-free Coulomb
    friction layer applied as a block-soft-threshold velocity projection on top of a frictionless
    prediction. No training, no learnable weights -- the friction operator's only inputs are
    physical (mu, lambda_n, m, h, slip). This is the "operator view" of the friction dissipation
    potential; for the trained arm (BR) the SAME physics enters the loss via friction_dissipative.

    Per cloth vertex (nearest body FACE, mirroring collision_penalty; no reliance on the GNN world
    edges, so it works wherever the barrier sees contact):

      lambda_n = 3 * stiffness * max(eps - d, 0)^2 # barrier normal force (proposal)
      u_T^new  = u_T * max(0, 1 - mu*lambda_n*h / (m*||u_T||))  # cone-bounded tangential update

    UNITS (critical): HOOD/ContourCraft integrate in *displacement* units -- pred_pos = pos + velocity
    + accel, where stored `velocity` is the per-step displacement (= physical v * h). We therefore
    work on displacements Delta (= v*h). The cone factor is max(0, 1 - mu*lambda_n*h^2 / (m*||Delta_T||))
    -- note the h^2 (NOT h). The static band eps_v regularizes the norm so near-zero slip is treated
    as sticking without a hard 1/||.|| kink (IPC).

    The normal direction and the body's own motion are preserved exactly: only the tangential RELATIVE
    (cloth-minus-body) displacement is shrunk. For non-contact vertices lambda_n = 0 => factor = 1 =>
    no change (and we additionally mask to keep them, and pinned vertices, byte-identical).

    :param sample: a pyg batch AFTER the model forward (needs cloth.pos, cloth.pred_pos, cloth.v_mass,
                   cloth.vertex_type, cloth.timestep; obstacle.pos, obstacle.target_pos, obstacle faces).
    :param mu: Coulomb coefficient. float (global mu) or a (B,) tensor (per-sample mu).
    :param eps: barrier offset (match collision_penalty.eps; default 2e-3).
    :param stiffness: barrier stiffness w feeding lambda_n = 3*w*interp^2 (calibration knob).
    :param eps_v: static-velocity band (proposal / IPC).
    :param g, z_axis: only for the rest-load calibration probe in the diagnostics (lambda_n vs m*g*cos).
    :param return_diag: also return a dict of magnitudes (for the required logging / mu-monotonicity check).
    :param return_force: also return the per-vertex normal-force magnitude (Vc_total,), contact-gated (0 off
             the contact set), in the SAME node order as pred_pos. For the --render-force colormap ONLY; does
             not affect the projection. Returns append: (pred,force) | (pred,diag,force) when combined.
    :param lambda_n_source: 'barrier' (default; byte-identical) => lambda_n = 3*w*max(eps-d,0)^2;
             'force_balance' (=> lambda_n = max(0, n_hat. grad(E_drive)), the
             inward projection of the driving-energy gradients (Signorini-clamped). Orientation-, tightness-
             and dynamics-aware, and calibration-free (real Newtons; `stiffness` is ignored). The barrier
             `interp>0` is still used only as the active-set GATE `d<eps` in both modes.
    :param fb_use_gravity: include the gravity gradient  grad(E_grav)  = m*g*e_up            (force_balance).
    :param fb_use_inertia: include the inertia gradient  grad(E_inertia)= (m/h^2)*(x_pred - (x + v))  (force_balance).
    :param elastic_grad: optional (Vc_total,3) precomputed grad(E_stretch+E_bend) w.r.t. pred_pos (physical,
             detached), in the SAME node order as sample['cloth'].pred_pos. Added to G when force_balance
             (Step 2). None => Step 1 (grav+inertia only). Computed by the runner (needs the criterions).
    :param fb_elastic_conform: Appendix-A fix for the Step-2 elastic misfire. When False (default), the
             elastic force is projected on the BODY normal (raw membrane projection => conflates in-plane
             fold tension with contact load on non-conforming cloth => 184x mg*cos). When True, the elastic
             contribution is the CLOTH-normal Laplace pressure (n_c . grad(E_elastic), which excludes the
             in-plane membrane force by construction) GATED by the cloth-vs-body conformance |n_c . n_b|:
             only the part genuinely pressing INTO the body counts. grav+inertia are unaffected (always
             projected on n_b). No effect unless elastic_grad is provided.
    :param fb_elastic_pen_gate: fix component: zero the ELASTIC contribution on
             penetrating vertices (d<0). Evidence: mean|n.grad(E_str)| is ~14x larger on penetrating than
             non-penetrating contacts and carries ~46% of the elastic lambda mass; a penetrating vertex's
             correspondence/strain state reflects constraint violation (barrier-fighting deformation), not
             transmitted contact pressure, and feeding it to the cone locks the tangential escape path that
             would let the barrier push it out (the observed runaway trap). Contact traction is only
             defined on the feasible boundary. grav+inertia unaffected.
    :param fb_elastic_smooth_iters: N Jacobi steps of 1-ring mesh averaging of the scalar elastic pressure.
              (sparse spikes); on mature checkpoints: the surviving
             elastic pressure is COHERENT (trained-penalty reaction), so smoothing spreads rather than
             cancels it -- prefer 0 with the gates below. Kept for ablation.
    :param fb_elastic_pen_dilate: strengthen the pen gate: zero the elastic pressure also on the 1-ring
             NEIGHBORS of penetrating vertices. A penetrating vertex's strain field extends to
             its ring; the dilated mask removes the trap halo (offline: stick_dn 66->61, and in-loop it
             breaks the strain->lambda->stick feedback earlier).
    :param fb_elastic_approach_gate: count elastic pressure only where the relative normal displacement is
             non-separating (rel_n <= 0.5*eps_v*h). A vertex moving AWAY from the body is not transmitting
             elastic load; resting/pressing contact passes the band.
    :param fb_elastic_noise_floor_q: 0 = off. Else (e.g. 0.90): per step, evaluate the SAME conform elastic
             pressure on all NON-contact cloth vertices -- where the true contact pressure is exactly zero
             -- take the q-quantile of its positive part as the state-noise floor, and one-sided
             soft-threshold the contact pressure by it. Self-calibrating: the floor is measured
             from the estimator's own zero-truth population each step, not tuned.
    :param lambda_smooth_iters:: K masked-Jacobi steps mollifying the ASSEMBLED lambda_n
             field over the contact set (any lambda source). The per-vertex lambda is a noisy point sample
             of a continuous contact traction; K MPNN-style mean-aggregation rounds (receptive field = K
             rings) redistribute it among CONTACT vertices only -- no force is bled onto non-contact
             vertices, isolated contacts are left unchanged. 0 = off (/b/c byte-identical).
    :param lambda_smooth_window: alternative way to set K, in METERS: K = round(window / mean rest edge
             length) per garment (needs cloth.rest_pos). Used only when lambda_smooth_iters == 0.
    :param lambda_smooth_conserve: rescale so the mass(~area)-weighted total normal load over the contact
             set is unchanged by smoothing -- redistribute the resultant, do not create/destroy it.
    :param lambda_smooth_render_only: apply the mollified field ONLY to the returned/rendered force and
             the diagnostics; the friction cone keeps the un-smoothed point samples (dynamics byte-
             identical to K=0). Zero-risk reviewer-facing mode; see for when to prefer it.
    :param normal_exact:: normal completion of the friction projection.
             'off' (default, byte-identical) | 'preserve' (-T: restore the network's own refreshed
             nearest-face plane-gap proxy at the friction-shifted position) | 'preserve_clamp'
             (-TC: additionally Signorini-clamp the target at normal_exact_slack). Applied ONLY to
             friction-altered vertices (|friction_dx| > normal_exact_dust), so the µ=0 rollout stays
             byte-identical. Local per-face guarantee only; see normal_completion_plane_gap.
    :param normal_exact_iters: correspondence-refresh iteration budget (no convergence guarantee).
    :param normal_exact_slack: TC clamp slack δ_s in meters.
    :param normal_exact_max_step: per-vertex cap on the total correction norm; 0 => auto
             max(2*eps, |friction_dx|). Cap exceeded => vertex reverted (skip, not clip) + counted.
    :param normal_exact_dust: altered-set predicate threshold; the delta reassembly leaves ~1e-9 fp
             residue on factor==1 contacts.
    :param normal_exact_ambig_k / _ambig_cos / _ambig_ratio: K-candidate ambiguity guard — near-equal
             candidate faces (centroid-dist ratio <= ambig_ratio) with normal dot < ambig_cos flag the
             vertex ambiguous => correction skipped + counted (thin parts / wedges / opposing surfaces).
    :param normal_exact_control: TC fairness control — candidate set becomes the matched
             ACTIVE-CONTACT set (in_contact & lambda_n>0) regardless of friction_dx, so the clamp also
             fires at µ=0 (where x_t == x_pre and the preserve part is a no-op). Requires
             normal_exact='preserve_clamp'. NEVER use in a headline arm.
    :param tangential_path:: how the friction shrink traverses from v̂ to v_tentative.
             'chord' (default, byte-identical : straight ambient chord + normal_exact endpoint
             restore) | 'surface' (: geometry_aware_transport — a surface-marched offset path so
             the slide follows the body instead of cutting through creases) | 'walk' (: the
             parameter-free triangle walk — the crossed body faces ARE the segments, path_substeps
             does not exist; same downstream landing/completion/fallback code as 'surface').
             'surface' and 'walk' REQUIRE normal_exact in {'preserve','preserve_clamp'} (=> /
             ) and per-vertex fall back to the exact 'chord' result on ambiguity / arc blow-up
             / walk-guard trips (floor =).
    :param path_substeps: march resolution K_march (surface_march substeps; default 8). On a flat
             face any value gives path==chord; higher only sharpens crease rounding. IGNORED under
             tangential_path='walk' (the mesh topology sets the resolution).
    :param path_walk_fast:, 'walk' only — SCHEDULING fast path (merged seating + CUDA-graph
             replay of both walk loops). Bitwise-identical output by construction; declines to the
             shipped eager loop whenever capture is not available. Default False = verbatim.
    :param path_walk_max_cross:, 'walk' only — SAFETY cap on edge crossings per pair
             (never-binds class, sibling of path_arc_cap; exceed => per-vertex chord fallback,
             counted wk_fb_cross). NOT a resolution knob; a non-default value on any other path
             refuses (unreachable-key discipline).
    :param path_arc_cone: friction slip measure on the marched path. 'off' (default) reuses the
              chord cone factor applied to arc length (removes b·L/c ≥ b of arc — the path-isolating
             arm, cone identical to). 'on' is the arc-native Coulomb cone (removes exactly the
             budget b of TRUE slide distance; shifts the µ-response — benchmarked separately).
    :param path_arc_cap: guard — revert a vertex to the chord result when marched arc length
             exceeds this multiple of the straight chord (default 3.0; correspondence-thrash guard).
    :param path_normals:. 'single' (default, byte-identical) uses ONE λ_n and
             one budget for the whole marched arc. 'multi' re-projects the exact gravity+inertia
             drive vector onto each march segment's own normal (the gated elastic channel stays a
             frozen primary-frame scalar — re-projecting it would re-litigate the in-plane
             conflation frame-by-frame) and lands via the far-end-first landing-segment prox
             (multi_normal_arc_landing). REQUIRES tangential_path='surface', path_arc_cone='on',
             lambda_n_source='force_balance', and the mollifier OFF (the multi decomposition
             reconstructs the UNSMOOTHED λ; a smoothed cone λ would silently mismatch at the
             primary frame). Uniform-normal limit is bitwise 'single'.
    :param spread_w: diagnostic DIAGNOSTIC correction spreading, default 0.0 = off
             (branch skipped, byte-identical). Post-pass on the operator's output: every moved
             contact donates its TANGENTIAL correction to its 1-ring; each receiver (not itself a
             donor, not pinned) takes the uniform mean over its DISTINCT donor neighbors, applied
             scaled by spread_w. Tests whether patch coherence survives the backbone's ~75%/frame
             re-drive. Path-agnostic by design (composes with 'chord' => the
             march-vs-chord 2x2). Structural no-op at mu=0. Never a ship path; C1 replaces the
             mean with a compliant (XPBD-style) constraint solve.
    :param spread_project: diagnostic. 'off' (default, byte-identical C0) applies the
             received mean RAW; 'own' first projects it onto the RECEIVER's own tangent plane
             (I - n_r n_r^T, n_r = that vertex's nearest-face normal) instead of leaving it in the
             DONOR's frame. Isolates how much of C0's penetration defect is merely the wrong FRAME
             versus the missing SOLVE — the cheap half of C1, and the decomposition
             asked for. Requires spread_w > 0.
    :param spread_hops: diagnostic MULTI-HOP reach for the spread, default 1 = C0's
             single ring. A receiver at hop k becomes a source at hop k+1 (BFS wavefront), each
             vertex assigned exactly once at its hop distance; pinned vertices neither receive nor
             conduct. Numerical parameter -- set it large enough that spread_gamma is the binding
             knob; the walk stops early when the front dies.
    :param spread_gamma: per-hop attenuation in [0, 1], default 0.0 = byte-identically C0. Hop 1
             ALWAYS keeps exactly C0's displacement and gamma governs only hops >= 2, so every rung
             of the ladder is a strict superset of C0 and any difference from the frozen C0
             measurement is ADDED REACH rather than a re-scaled near field. Physically a screening
             length: gamma=1 is the unscreened (implicit-solver-like) limit. Swept, not derived --
             the real correlation length is set by out-of-plane buckling, not by the in-plane wave
             speed, which at this stiffness would imply near-global support.
    :param spread_weight: diagnostic how a receiver aggregates MULTIPLE sources.
             'uniform' (default, byte-identical) = plain arithmetic mean over distinct neighbours.
             'area' = weight each source by its own barycentric REST area. Motivation is measured,
             not aesthetic: 66% of hop-1 receivers aggregate 2+ sources (mean 2.11) and triangle
             areas vary 2.98x across a real garment, so the uniform mean weights by MESH
             CONNECTIVITY rather than geometry and biases the reach toward densely-meshed regions.
             Cotangent (Laplace-Beltrami) weights would be the exact correction, but 23.1% of these
             triangles are obtuse and would carry NEGATIVE weights; area weights capture the same
             density correction with no overshoot risk. Do not describe 'area' as the exact
             Laplace-Beltrami operator -- it is not.
    :param patch_iters: diagnostic COMPLIANT PATCH RELAXATION, default 0 = off (branch
             skipped, byte-identical). K XPBD sweeps (Eq. 18) over the distance constraints of the
             patch around each engaged contact: the operator-corrected contacts are ANCHORS
             (w = 0), their `patch_depth`-ring is free, and every edge with a free endpoint carries
             a constraint whose target is its length under the FRICTIONLESS prediction. So the
             patch follows the anchors subject to its own metric, rather than being translated
             along a neighbour's tangent plane.
             Structural no-op at mu=0 (no anchor moves => every C_j == 0 => zero dlambda).
             Mutually exclusive with spread_w: they are alternative couplings, and running both
             would make the attribution unreadable.
    :param patch_depth: ring depth of the free set (default 1 = the matched comparison with C0).
    :param patch_alpha_scale: multiplier on the material-derived compliance alpha (default 1.0 =
             the garment's own material, no fitted constant). Ablation-only knob: the measured
             alpha~/(grad C M^-1 grad C^T + alpha~) is ~1e-6 on these fabrics (logged as
             pc_alpha_ratio), i.e. real cloth is effectively inextensible at the sub-mm scale of a
             friction correction, so the solve is stiff-limited and the reach is governed by
             patch_iters / patch_depth — numerical parameters of the same kind as path_substeps.
             Scaling alpha up is how that claim is DEMONSTRATED rather than asserted.
    :param patch_signorini: normal handling on the free set after the sweeps. 'preserve' (default)
             is strictly NON-WORSENING: no free vertex is allowed to end with a smaller nearest-face
             plane gap than it had under the frictionless prediction, pushed back along its OWN
             refreshed normal. Deliberately not an improving clamp — pushing penetrating vertices
             out to a slack would improve pen for a reason unrelated to coupling and confound the
             primary read. Ambiguous queries are skipped (refreshed_plane_gap's contract: a
             wrong-side push is worse than no push). 'off' is the isolation arm (solve only).
    :param patch_feas_tol: absolute feasibility tolerance [m] for the iterations-to-feasibility
             counter.
    :param path_capstan:. 'on' adds the discrete belt/capstan junction
             surcharge — each segment's budget amplified by e^{µ·Σ Δθ_j} over the wrap accumulated
             along the slid portion (corners resist sliding through changing contact normals, in
             addition to tangential friction). Requires path_normals='multi'. 'off' (default) is
             the isolation arm; benchmark-second knob, same off/on discipline as path_arc_cone.
    :param return_metric_log: fmetrics logging hook. When True, additionally return a
             dict of flat per-contact tensors over the operator's ACTIVE SET (in_contact & ~pinned,
             INCLUDING lambda_n==0 contacts), concatenated across the batch in pred_pos node order:
               vertex (N,) int32   — cloth node index (batch-offset, matches pred_pos rows)
               face   (N,) int32   — convention-A nearest face id (current-frame centroid KNN)
               lambda_n (N,) f32   — the cone-driving load exactly as consumed [N]
               delta_T (N,3) f32   — PRE-friction tangential relative displacement [m]
               factor (N,) f32     — applied shrink factor in [0,1]
               x_pre  (N,3) f32    — trial (pre-friction) position [m]
             Everything downstream (v*, v+, f_t, j, E_diss) derives from these; conversions live in
             utils/fmetrics/io.py ONLY. Read-only capture (detached) — zero effect on dynamics.
    :param shadow: fmetrics CSA arm: compute the full lambda stack, cone factor and
             metric log, but return pred_pos UNCHANGED (the trajectory stays the backbone's).
             Used to reconstruct operator quantities for models that have no runtime operator
             (HOOD-loss-only ablation). Combine with normal_exact='off' to avoid dead compute.
    :param detach_lambda_n:. Read the WHOLE lambda_n stack and the active-set gate
             from a DETACHED alias of cloth.pred_pos. Forward is bitwise unchanged (.detach() shares
             storage), so every existing caller is byte-identical either way; the flag matters only
             when the operator is differentiated (the composite action objective), where lambda_n
             must be the lagged/frozen cone coefficient and NOT a decision variable. This is the one
             necessary stop-gradient (T3: lambda_n live inside the dissipation term is a real
             exploit — the optimizer lifts off the surface to drive lambda_n -> 0 and dodge the
             friction price). Under no_grad (every inference call site) it is a strict no-op.
    :param return_action:. Additionally return the DIFFERENTIABLE frictional
             dissipation actually paid by this projection, summed over the batch:

                 D = sum_{active contacts}  mu * sg[lambda_n] * ( f0(||Delta_T(x^+)||) - eps_v*h )

             with f0(s) = sqrt(s^2 + (eps_v*h)^2) — the SAME pseudo-Huber mollifier whose proximal
             operator this projection is (`coulomb_block_soft_threshold`'s `reg`), so adding D to a
             loss evaluated at x^+ reconstructs the frictional incremental potential of the SHIPPED
             frame. The band offset eps_v*h is subtracted so a fully-stuck contact pays exactly 0;
             that is a per-contact CONSTANT (lambda_n and the active set are detached/boolean) and
             therefore changes no gradient (test T-C7). Units: Joules, matching the backbone's
             incremental potential (postcvpr/inertia.py divides by h^2) — no extra h^2 factor, the
             HOOD / friction_dissipative convention. NOT divided by the batch size: the caller
             applies the /B and the weight, mirroring the criterion convention. Requires
             detach_lambda_n=True (validated) and is incompatible with shadow=True.
    :return: pred_pos_new (Vc_total, 3) in the SAME node order as sample['cloth'].pred_pos;
             trailing elements appended per flags, in the fixed order
             (pred[, diag][, force][, metric_log][, action]).
    """
    from pytorch3d.ops import knn_points
    from utils.cloth_and_material import FaceNormals, VertexNormals
    from utils.common import gather, NodeType

    device = sample['cloth'].pred_pos.device
    f_normals_f = FaceNormals().to(device)
    v_normals_f = VertexNormals().to(device)  # area-weighted per-cloth-vertex normal (force_balance conform)
    B = sample.num_graphs

    if lambda_n_source not in ('barrier', 'force_balance'):
        raise ValueError(f"Unknown lambda_n_source: {lambda_n_source!r} "
                         "(expected 'barrier' or 'force_balance').")
    if normal_exact not in ('off', 'preserve', 'preserve_clamp'):
        raise ValueError(f"Unknown normal_exact: {normal_exact!r} "
                         "(expected 'off', 'preserve', or 'preserve_clamp').")
    if normal_exact_control and normal_exact != 'preserve_clamp':
        raise ValueError("normal_exact_control=True is the TC fairness control and "
                         "requires normal_exact='preserve_clamp'.")
    # YAML 1.1 parses the barewords off/on (and yes/no) as booleans, so an unquoted
    # `path_arc_cone: off` in a config arrives here as Python False. Coerce defensively so the
    # footgun can never reach the string check below (quoting in the config is the source fix).
    if isinstance(path_arc_cone, bool):
        path_arc_cone = 'on' if path_arc_cone else 'off'
    if tangential_path not in ('chord', 'surface', 'walk'):
        raise ValueError(f"Unknown tangential_path: {tangential_path!r} "
                         "(expected 'chord', 'surface' or 'walk').")
    if path_arc_cone not in ('off', 'on'):
        raise ValueError(f"Unknown path_arc_cone: {path_arc_cone!r} (expected 'off' or 'on').")
    if tangential_path in ('surface', 'walk') and normal_exact == 'off':
        raise ValueError(f"tangential_path={tangential_path!r} requires normal_exact in "
                         "{'preserve','preserve_clamp'}: the geometric slide path composes with "
                         "the endpoint restore (=> /).")
    # triangle walk: the crossing cap is a SAFETY guard of the walk alone. A non-default
    # value on any other path is a silent no-op => refuse (same discipline as drag_k / spread_*).
    path_walk_max_cross = int(path_walk_max_cross)
    if path_walk_max_cross < 1:
        raise ValueError(f"bolt_on.path_walk_max_cross must be >= 1, got {path_walk_max_cross}")
    if path_walk_max_cross != 16 and tangential_path != 'walk':
        raise ValueError("bolt_on.path_walk_max_cross is a triangle-walk guard and "
                         "requires tangential_path='walk'; on any other path it is unreachable "
                         "and setting it would be a fake_icontour-class dead key.")
    # the walk fast path is scheduling only (bitwise output), so it is NOT an arm of the
    # physics -- but it is still a dead key on any other builder. Same discipline.
    path_walk_fast = bool(path_walk_fast)
    if path_walk_fast and tangential_path != 'walk':
        raise ValueError("bolt_on.path_walk_fast is the triangle-walk scheduling fast path "
                         " and requires tangential_path='walk'; on any other path it is "
                         "unreachable and setting it would be a fake_icontour-class dead key.")
    # CUDA-graph capture cannot record autograd, so the fast path DETACHES the two walk
    # loops. Measured inconsequential for inference (no backward traverses the walk's outputs: 0
    # firings / 2220 hooks over a full rollout) -- but a train-through arm prices its action
    # THROUGH the transport, and there the detach would silently zero a gradient rather than fail.
    # Refuse loudly instead. The eager walk remains available to that arm at full accuracy.
    if path_walk_fast and return_action:
        raise ValueError("bolt_on.path_walk_fast is inference-only: CUDA-graph capture "
                         "requires a detached path builder, and return_action=True (composite "
                         "action) differentiates through it. Drop path_walk_fast for that arm.")
    # same YAML-1.1 bareword coercion for path_capstan as for path_arc_cone.
    if isinstance(path_capstan, bool):
        path_capstan = 'on' if path_capstan else 'off'
    if path_normals not in ('single', 'multi'):
        raise ValueError(f"Unknown path_normals: {path_normals!r} (expected 'single' or 'multi').")
    if path_capstan not in ('off', 'on'):
        raise ValueError(f"Unknown path_capstan: {path_capstan!r} (expected 'off' or 'on').")
    if path_capstan == 'on' and path_normals != 'multi':
        raise ValueError("path_capstan='on' requires path_normals='multi': the junction "
                         "surcharge lives on the per-segment budgets.")
    # diagnostic over-budget drag. Default 0.0 = the knob does not exist (the
    # guarded branch in geometry_aware_transport is skipped => byte-identical off path).
    drag_k = float(drag_k)
    if drag_k < 0:
        raise ValueError(f"bolt_on.drag_k must be >= 0, got {drag_k}")
    if drag_k > 0 and (tangential_path != 'surface' or path_normals != 'multi'):
        raise ValueError("bolt_on.drag_k > 0 (R2 probe) requires tangential_path='surface' AND "
                         "path_normals='multi': the drag scales the retained slide arc s_star of "
                         "the marched walk-back. Refusing to no-op silently on other paths.")
    # correction spreading. Default 0.0 = branch skipped (byte-identical off
    # path). Deliberately path-AGNOSTIC: it is a post-pass on the operator's output, so it composes
    # with 'chord' too -- that is exactly the march-vs-chord 2x2 arm. No path gate, by design.
    spread_w = float(spread_w)
    if spread_w < 0:
        raise ValueError(f"bolt_on.spread_w must be >= 0, got {spread_w}")
    if spread_w and shadow:
        raise ValueError("bolt_on.spread_w > 0 with shadow=True is meaningless: shadow returns the "
                         "untouched backbone trajectory, discarding every correction to spread.")
    if spread_project not in ('off', 'own'):
        raise ValueError(f"Unknown spread_project: {spread_project!r} (expected 'off' or 'own').")
    if spread_project == 'own' and not spread_w:
        raise ValueError("bolt_on.spread_project='own' (diagnostic) requires spread_w > 0: it "
                         "re-frames the spread displacement, it does not create one.")
    # multi-hop reach. spread_gamma=0 is byte-identically C0 at ANY hop count --
    # it is the ladder's ANCHOR rung, not a misconfiguration, so it is deliberately allowed. A
    # gamma with hops<2 IS a misconfiguration (there is no hop 2 for it to act on): refuse rather
    # than no-op silently, same discipline as drag_k on the wrong path.
    spread_hops = int(spread_hops)
    spread_gamma = float(spread_gamma)
    if spread_hops < 1:
        raise ValueError(f"bolt_on.spread_hops must be >= 1, got {spread_hops}")
    if not 0.0 <= spread_gamma <= 1.0:
        raise ValueError(f"bolt_on.spread_gamma must be in [0, 1], got {spread_gamma} "
                         "(> 1 would AMPLIFY the correction with distance from the contact)")
    if spread_gamma > 0 and spread_hops < 2:
        raise ValueError("bolt_on.spread_gamma > 0 with spread_hops < 2 is a silent no-op: gamma "
                         "governs hops >= 2 only (hop 1 is exactly C0 by design). Raise "
                         "spread_hops, or drop spread_gamma to state the C0 rung explicitly.")
    if spread_weight not in ('uniform', 'area'):
        raise ValueError(f"Unknown spread_weight: {spread_weight!r} (expected 'uniform' or 'area').")
    if spread_weight != 'uniform' and not spread_w:
        raise ValueError("bolt_on.spread_weight (diagnostic) requires spread_w > 0: it re-weights how "
                         "a spread is aggregated, it does not create one.")
    if (spread_hops > 1 or spread_gamma > 0) and not spread_w:
        raise ValueError("bolt_on.spread_hops / spread_gamma (diagnostic) require spread_w > 0: they "
                         "extend the REACH of the spread, they do not create one.")
    # compliant patch relaxation. Default 0 = branch skipped (byte-identical
    # off path), same discipline as spread_w / drag_k. Path-agnostic by design (the chord arm is
    # the march-rescue test), so no path gate here either.
    patch_iters = int(patch_iters)
    if patch_iters < 0:
        raise ValueError(f"bolt_on.patch_iters must be >= 0, got {patch_iters}")
    if patch_iters and int(patch_depth) < 1:
        raise ValueError(f"bolt_on.patch_depth must be >= 1, got {patch_depth}")
    if patch_signorini not in ('off', 'preserve'):
        raise ValueError(f"Unknown patch_signorini: {patch_signorini!r} "
                         "(expected 'off' or 'preserve').")
    if patch_iters and spread_w:
        raise ValueError("bolt_on.patch_iters > 0 with spread_w > 0: C0 spreading and the C1 patch "
                         "solve are ALTERNATIVE couplings of the same correction field. Running "
                         "both stacks two reach operators and makes the attribution unreadable — "
                         "pick one (C1 is the principled form; C0/C0.5 are the probes).")
    # the wave stage is the THIRD mutually-exclusive op_patch coupling: same
    # correction field, material-wave reach law (no depth, no gamma, no iteration budget).
    # Default False = branch skipped AND `want_tang` reduces to return_action (byte-identical
    # off path, same discipline as spread_w / patch_iters).
    wave_enabled = bool(wave_enabled)
    wave_visco = float(wave_visco)
    wave_tau_frames = float(wave_tau_frames)
    if wave_visco < 0:
        raise ValueError(f"bolt_on.wave_visco must be >= 0, got {wave_visco}")
    if wave_tau_frames <= 0:
        raise ValueError(f"bolt_on.wave_tau_frames must be > 0, got {wave_tau_frames}")
    if wave_visco and not wave_enabled:
        raise ValueError("bolt_on.wave_visco > 0 requires wave_enabled=True: the SLS branch "
                         "lives inside the wave stage, it does not create one.")
    # per-edge strain budget on the wave's composed output. 0.0 (default) =
    # branch skipped, byte-identical to the uncapped wave stage.
    wave_strain_cap = float(wave_strain_cap)
    if wave_strain_cap < 0:
        raise ValueError(f"bolt_on.wave_strain_cap must be >= 0, got {wave_strain_cap}")
    if wave_strain_cap and not wave_enabled:
        raise ValueError("bolt_on.wave_strain_cap > 0 requires wave_enabled=True: the cap "
                         "clips the wave stage's own output, it does not create one.")
    if wave_enabled and (patch_iters or spread_w):
        raise ValueError("bolt_on.wave_enabled with patch_iters/spread_w: the membrane-wave, the C1 "
                         "patch solve and C0 spreading are ALTERNATIVE couplings of the same "
                         "correction field — running two stacks two reach operators. Pick one.")
    if wave_enabled and shadow:
        raise ValueError("bolt_on.wave_enabled with shadow=True is meaningless: shadow returns "
                         "the untouched backbone trajectory, discarding the corrections the wave "
                         "would transport.")
    if patch_iters and shadow:
        raise ValueError("bolt_on.patch_iters > 0 with shadow=True is meaningless: shadow returns "
                         "the untouched backbone trajectory, so there are no anchors to relax "
                         "around.")
    if path_normals == 'multi':
        if tangential_path not in ('surface', 'walk'):
            raise ValueError("path_normals='multi' requires tangential_path='surface' or "
                             "'walk': the per-segment frames come from the geometric slide path.")
        if path_arc_cone != 'on':
            raise ValueError("path_normals='multi' requires path_arc_cone='on': the "
                             "chord-factor 'off' law has no seat for per-segment loads.")
        if lambda_n_source != 'force_balance':
            raise ValueError("path_normals='multi' requires lambda_n_source='force_balance': "
                             "the multi law re-projects the force-balance drive VECTOR; the barrier "
                             "scalar has no frame to re-project.")
        if int(lambda_smooth_iters) > 0 or float(lambda_smooth_window) > 0:
            raise ValueError("path_normals='multi' requires the lambda mollifier OFF "
                             "(lambda_smooth_iters=0, lambda_smooth_window=0): the per-segment "
                             "decomposition reconstructs the UNSMOOTHED lambda and would silently "
                             "mismatch a smoothed cone lambda at the primary frame.")
    if return_action and not detach_lambda_n:
        raise ValueError(
            "return_action=True (composite action objective) requires detach_lambda_n=True: a "
            "live lambda_n inside the dissipation term is the one exploitable channel (T3 "
            "— the optimizer separates from the surface to drive lambda_n -> 0 instead of paying "
            "the friction price).")
    if return_action and shadow:
        raise ValueError(
            "return_action=True with shadow=True is meaningless: shadow returns the UNPROJECTED "
            "backbone trajectory, so there is no shipped frame whose dissipation could be priced.")
    # Which of the four operator stages this configuration actually runs. Printed ONCE per
    # process (utils/trace.py), so an eval lane gains one block, not one per rollout step.
    from utils import trace
    if trace.once('operator_stages'):
        mu0 = float(mu) if not torch.is_tensor(mu) else float(mu.reshape(-1)[0])
        trav = ('triangle walk' if tangential_path == 'walk' else
                f'surface march S={path_substeps}' if tangential_path == 'surface' else
                'ambient chord')
        prop = ('membrane wave' if wave_enabled else
                f'patch relaxation, {patch_iters} iters' if patch_iters else
                f'ring spread w={spread_w}' if spread_w else None)
        stages = [
            ('normal load + Coulomb correction', True,
             f'{lambda_n_source}, eps_v {eps_v:g}, mu {mu0:g}'),
            ('contact traversal + normal completion', tangential_path != 'chord',
             f'{trav}, completion {normal_exact}'),
            ('friction-induced motion propagation', prop is not None, prop or ''),
            ('strain limiting', bool(wave_strain_cap), f'per-edge cap {wave_strain_cap:g}'),
        ]
        trace.at('utils/contact.py', 'bolt_on_friction_projection',
                 'applying the contact operator:')
        for i, (name, on, detail) in enumerate(stages, 1):
            print(f'    stage {i}  {name:<38} {"ON " if on else "off"}  {detail if on else ""}'.rstrip())

    ne_totals = {}  # normal-completion counters, accumulated across batch examples
    action_total = None       # differentiable sum of the per-example dissipation
    action_slip_sum = 0.0     # diagnostics (detached floats)
    action_n_active = 0

    pred_new_chunks = []
    contact_chunks = []
    force_chunks = []  # per-vertex lambda_n (contact-gated), for --render-force visualization only
    mlog_chunks = []   # fmetrics per-contact capture (return_metric_log)
    off = 0  # running cloth-node offset, for slicing elastic_grad per example (force_balance Step 2)
    # diagnostics accumulators (concatenated across the batch, restricted to contact later)
    diag_lambda_n, diag_slip_pre, diag_slip_post = [], [], []
    diag_factor, diag_mg_cos, diag_n_up = [], [], []
    diag_spike = []   # per-example (p95 ring-ratio, max ring-ratio, TV, n_contact)

    for i in range(B):
        ex = sample.get_example(i)
        obstacle_next = ex['obstacle'].target_pos          # (Vo,3) body next pose
        obstacle_curr = ex['obstacle'].pos                 # (Vo,3) body current pose
        obstacle_faces = ex['obstacle'].faces_batch.T      # (Fo,3)
        cloth_curr = ex['cloth'].pos                       # (Vc,3) x^t
        cloth_next = ex['cloth'].pred_pos                  # (Vc,3) frictionless tentative x^{t+1}
        m = ex['cloth'].v_mass                             # (Vc,1)
        h = float(ex['cloth'].timestep.reshape(-1)[0])     # scalar timestep for this example
        pinned = (ex['cloth'].vertex_type == NodeType.HANDLE).reshape(-1)  # (Vc,)
        mu_i = float(mu) if not torch.is_tensor(mu) else float(mu.reshape(-1)[i])
        # the lambda_n stack AND the active-set gate below read this alias. With
        # detach_lambda_n it is the detached view of the prediction — bitwise identical in forward
        # (shared storage), so every legacy call site is unaffected, but the cone coefficient stays
        # a frozen (lagged) constant when the operator is differentiated. Everything AFTER the gate
        # (the tangential shrink, the normal completion, the surface march) keeps using the LIVE
        # cloth_next: that is the channel the composite action objective learns through.
        cloth_next_lam = cloth_next.detach() if detach_lambda_n else cloth_next

        # Time-profiling markers. PROF is disabled by default; every begin/end is then a
        # single `if not self.enabled: return`, so the operator is byte-identical off the profiling
        # path. begin/end rather than `with` deliberately: wrapping would re-indent ~130 lines of
        # the one module whose off-path byte-identity is a standing working agreement.
        PROF.begin('op_correspond')
        # nearest obstacle face per cloth vertex (current-frame correspondence, as in collision_penalty)
        face_curr_centroid = gather(obstacle_curr, obstacle_faces, 0, 1, 1).mean(dim=-2)  # (Fo,3)
        _, nn_idx, _ = knn_points(cloth_curr.unsqueeze(0), face_curr_centroid.unsqueeze(0),
                                  return_nn=True)
        nn_idx = nn_idx[0]                                                                # (Vc,1)

        # face geometry at the next pose: centroid (for body motion) and outward normal n_hat
        face_next_centroid = gather(obstacle_next, obstacle_faces, 0, 1, 1).mean(dim=-2)  # (Fo,3)
        obstacle_fn = f_normals_f(obstacle_next.unsqueeze(0), obstacle_faces.unsqueeze(0))[0]  # (Fo,3)
        nn_cen_next = gather(face_next_centroid, nn_idx, 0, 1, 1)[:, 0]                    # (Vc,3)
        nn_cen_curr = gather(face_curr_centroid, nn_idx, 0, 1, 1)[:, 0]                    # (Vc,3)
        n_hat = gather(obstacle_fn, nn_idx, 0, 1, 1)[:, 0]                                 # (Vc,3) unit

        # signed gap d and the barrier active-set gate interp>0 (d<eps), used by BOTH sources.
        d = ((cloth_next_lam - nn_cen_next) * n_hat).sum(dim=-1, keepdim=True)             # (Vc,1) signed gap
        interp = torch.clamp(eps - d, min=0.0)                                            # (Vc,1)
        PROF.end('op_correspond')

        PROF.begin('op_lambda')
        if lambda_n_source == 'barrier':
            # barrier normal force = 3*w*max(eps-d,0)^2 (gap-only; orientation-blind).
            lambda_n = stiffness * 3.0 * interp.pow(2)                                     # (Vc,1) >= 0
        else:
            # (lambda_n = max(0, n_hat. grad(E_drive)) -- the inward projection
            # of the driving-energy gradients, Signorini-clamped. Coincides with the barrier reaction at a
            # frictionless equilibrium, but vanishes on unloaded near-contact (loose drape / down-facing).
            G = torch.zeros_like(cloth_next_lam)                                           # (Vc,3)
            if fb_use_gravity:
                up = torch.zeros(3, device=device, dtype=cloth_next_lam.dtype)
                up[z_axis] = 1.0
                G = G + (m * g) * up                                                       # grad(E_grav) = m*g*e_up
            if fb_use_inertia:
                if 'velocity' not in ex['cloth']:
                    raise KeyError("force_balance+inertia needs cloth.velocity (inertial predictor "
                                   "x_hat = pos + velocity); not found on the sample.")
                x_hat = cloth_curr + ex['cloth'].velocity                                 # (Vc,3) pos + per-step disp
                G = G + (m / (h * h)) * (cloth_next_lam - x_hat)                          # grad(E_inertia)
            # keep the exact gravity+inertia drive VECTOR for per-segment
            # re-projection. Captured BEFORE any raw elastic add, so the (fragile, gated) elastic
            # channel stays a primary-frame scalar in the multi decomposition below.
            drive_gi = G
            # Elastic term. Three regimes, in increasing fidelity:
            # * RAW (Step 2 /): accumulate grad(E_elastic) into G, ONE dot product on n_b --
            # byte-identical to the pre-fix code path (and reproduce bit-exactly).
            #  * CONFORM (Appendix-A): project the elastic part on the CLOTH normal instead -- the membrane
            #    force lies in the cloth tangent plane, so its body-normal projection leaks in-plane fold
            #    tension on non-conforming cloth (184x mg*cos on longskirt). n_c.grad(E) is the Laplace/
            #    curvature pressure; gated by conformance |n_c.n_b|.
            # * PEN-GATE / SMOOTH: additionally zero the elastic pressure on
            #    penetrating vertices (invalid correspondence; the runaway-trap source) and 1-ring-average
            #    the scalar pressure (the lambda mass sits in sparse single-vertex spikes of the stiff
            #    stretch gradient -- point samples of a per-area continuum field).
            fix_path = (fb_elastic_conform or fb_elastic_pen_gate or fb_elastic_smooth_iters > 0
                        or fb_elastic_pen_dilate or fb_elastic_approach_gate
                        or fb_elastic_noise_floor_q > 0)
            if elastic_grad is not None and not fix_path:
                G = G + elastic_grad[off:off + cloth_next.shape[0]]                        # grad(E_stretch+E_bend)
            lam = (G * n_hat).sum(dim=-1, keepdim=True)                                    # (Vc,1) driving grad on n_b
            if elastic_grad is not None and fix_path:
                eg = elastic_grad[off:off + cloth_next.shape[0]]                           # grad(E_stretch+E_bend)
                if fb_elastic_conform:
                    cloth_faces = ex['cloth'].faces_batch.T                                # (Fc,3)
                    n_c = v_normals_f(cloth_next_lam.unsqueeze(0), cloth_faces.unsqueeze(0))[0]  # (Vc,3) unit
                    align = (n_c * n_hat).sum(dim=-1, keepdim=True)                        # signed cos(n_c, n_b)
                    n_c_s = n_c * torch.sign(align)                                        # orient to n_b hemisphere
                    p_el = align.abs() * (eg * n_c_s).sum(dim=-1, keepdim=True)            # conform-weighted pressure
                else:
                    p_el = (eg * n_hat).sum(dim=-1, keepdim=True)                          # raw body-normal pressure
                pen_free = (d >= 0).to(p_el.dtype)                                         # (Vc,1) feasible-side gate
                if fb_elastic_pen_dilate:
                    # #4: dilate the penetration mask 1 ring -- a penetrating vertex's strain field
                    # contaminates its neighbors (the trap halo). Gate = no pen in the closed 1-ring.
                    cf_d = ex['cloth'].faces_batch.T
                    e_d = torch.cat([cf_d[:, [0, 1]], cf_d[:, [1, 0]], cf_d[:, [1, 2]],
                                     cf_d[:, [2, 1]], cf_d[:, [0, 2]], cf_d[:, [2, 0]]], 0)
                    pen_f = 1.0 - pen_free
                    acc_d = torch.zeros_like(pen_f)
                    acc_d.index_add_(0, e_d[:, 1], pen_f[e_d[:, 0]])
                    p_el = p_el * ((pen_f + acc_d) <= 0).to(p_el.dtype)
                elif fb_elastic_pen_gate:
                    p_el = p_el * pen_free
                if fb_elastic_approach_gate:
                    # #4: only non-separating contacts transmit elastic load.
                    rel_n = (((cloth_next_lam - cloth_curr) - (nn_cen_next - nn_cen_curr))
                             * n_hat).sum(dim=-1, keepdim=True)
                    p_el = p_el * (rel_n <= 0.5 * eps_v * h).to(p_el.dtype)
                if fb_elastic_deflate_profile:
                    # #5b: barrier-profile deflation. The coherent residual on mature backbones is
                    # the TRAINED collision-penalty reaction stored in the membrane, a function of the gap
                    # (p ~ a*(eps-d)^2, loss units). Robust-fit `a` per step over the active set (median of
                    # ratios) and keep only the EXCESS over the fitted proximity profile -- real tightness/
                    # dynamic load is a spatial anomaly w.r.t. proximity, the plateau is not.
                    xq = interp.pow(2)                                                     # (Vc,1)
                    fit_m = ((interp.reshape(-1) > 0) & (~pinned)
                             & (p_el.reshape(-1) > 0) & (xq.reshape(-1) > 1e-12))
                    if int(fit_m.sum()) > 20:
                        a_fit = torch.median(p_el[fit_m] / xq[fit_m])
                        p_el = torch.clamp(p_el - a_fit * xq, min=0.0)
                if fb_elastic_noise_floor_q > 0:
                    # #5: self-calibrating noise floor -- the same estimator on OFF-contact verts
                    # (true contact pressure exactly zero there) gives the state-noise scale; one-sided
                    # soft-threshold the contact pressure by its q-quantile.
                    offc = (interp.reshape(-1) <= 0) & (~pinned)
                    if fb_elastic_conform:
                        p_ref = align.abs() * (eg * n_c_s).sum(dim=-1, keepdim=True)
                    else:
                        p_ref = (eg * n_hat).sum(dim=-1, keepdim=True)
                    pop = torch.clamp(p_ref[offc], min=0.0)
                    if pop.numel() > 100:
                        p_el = torch.clamp(p_el - torch.quantile(pop, float(fb_elastic_noise_floor_q)),
                                           min=0.0)
                if fb_elastic_smooth_iters > 0:
                    cf = ex['cloth'].faces_batch.T
                    e2 = torch.cat([cf[:, [0, 1]], cf[:, [1, 0]], cf[:, [1, 2]],
                                    cf[:, [2, 1]], cf[:, [0, 2]], cf[:, [2, 0]]], 0)
                    deg = torch.zeros(p_el.shape[0], 1, device=p_el.device, dtype=p_el.dtype)
                    deg.index_add_(0, e2[:, 1], torch.ones(e2.shape[0], 1, device=deg.device,
                                                           dtype=deg.dtype))
                    for _ in range(int(fb_elastic_smooth_iters)):
                        acc = torch.zeros_like(p_el)
                        acc.index_add_(0, e2[:, 1], p_el[e2[:, 0]])
                        p_el = (acc + p_el) / (deg + 1.0)                                  # 1-ring Jacobi avg
                    if fb_elastic_pen_gate:
                        p_el = p_el * pen_free                                             # re-zero pen after avg
                lam = lam + p_el
            lambda_n = torch.clamp(lam, min=0.0)                                           # (Vc,1) >= 0
            if path_normals == 'multi':
                # primary-frame pieces for the transport's delta-form per-segment λ.
                # e_prim is implicit as (lam_pre_mn - gi_proj) — the frozen elastic scalar.
                gi_proj = (drive_gi * n_hat).sum(dim=-1, keepdim=True)                     # (Vc,1)
                lam_pre_mn = lam                                                           # pre-clamp

        in_contact = (interp.reshape(-1) > 0) & (~pinned)                                  # (Vc,)

        # mollify the assembled normal-force field. K masked-Jacobi steps over the mesh
        # 1-ring, support restricted to the CONTACT set (non-contact stays untouched; isolated contact
        # verts have no donors and stay unchanged), optionally conserving the mass(~area)-weighted total
        # normal load. Applies to ANY lambda source; affects the cone AND the --render-force field.
        K_sm = int(lambda_smooth_iters)
        if K_sm <= 0 and lambda_smooth_window > 0 and 'rest_pos' in ex['cloth']:
            cf_s = ex['cloth'].faces_batch.T
            e_s = torch.cat([cf_s[:, [0, 1]], cf_s[:, [1, 2]], cf_s[:, [0, 2]]], 0)
            rp = ex['cloth'].rest_pos
            l_bar = (rp[e_s[:, 0]] - rp[e_s[:, 1]]).norm(dim=-1).mean()
            K_sm = max(1, int(round(float(lambda_smooth_window / l_bar))))
        if K_sm > 0:
            cf_s = ex['cloth'].faces_batch.T
            e_s = torch.cat([cf_s[:, [0, 1]], cf_s[:, [1, 0]], cf_s[:, [1, 2]],
                             cf_s[:, [2, 1]], cf_s[:, [0, 2]], cf_s[:, [2, 0]]], 0)
            cmask = in_contact.to(lambda_n.dtype).unsqueeze(-1)                            # (Vc,1)
            lam_s = lambda_n * cmask
            tot0 = (m * lam_s).sum()
            for _ in range(K_sm):
                num = torch.zeros_like(lam_s)
                num.index_add_(0, e_s[:, 1], lam_s[e_s[:, 0]])
                den = torch.zeros_like(lam_s)
                den.index_add_(0, e_s[:, 1], cmask[e_s[:, 0]])
                lam_s = ((lam_s + num) / (1.0 + den)) * cmask                              # masked Jacobi
            if lambda_smooth_conserve:
                lam_s = lam_s * (tot0 / (m * lam_s).sum().clamp(min=1e-30))
            lambda_render = torch.where(cmask > 0, lam_s, lambda_n)
            if not lambda_smooth_render_only:
                lambda_n = lambda_render                          # smoothed field also drives the cone
        else:
            lambda_render = lambda_n
        PROF.end('op_lambda')
        PROF.count('n_contact', in_contact.sum())

        PROF.begin('op_cone')
        # displacement-space block soft-threshold on the tangential RELATIVE motion (shared core)
        delta_c = cloth_next - cloth_curr                                                 # (Vc,3) cloth disp
        delta_b = nn_cen_next - nn_cen_curr                                               # (Vc,3) body face disp
        delta_c_new, factor, slip, slip_post = coulomb_block_soft_threshold(
            delta_c, delta_b, n_hat, lambda_n, m, mu_i, h, eps_v)
        # Apply ONLY the friction correction (tangential, = delta_T_new - delta_T) to the original
        # prediction, instead of reassembling from cloth_curr. This is ~0 wherever friction does not act
        # (factor=1: mu=0, lambda_n=0, or full slack), so non-acting / non-contact / pinned vertices stay
        # byte-identical to baseline (no reconstruction round-off).
        friction_dx = delta_c_new - delta_c                                               # (Vc,3)
        friction_dx = torch.where(in_contact.unsqueeze(-1), friction_dx,
                                  torch.zeros_like(friction_dx))
        cloth_next_proj = cloth_next + friction_dx
        # the pure TANGENTIAL-transport outcome, i.e. the shipped frame minus the endpoint
        # restore. The restore moves each vertex along its LOCAL (refreshed) surface normal, and
        # motion along the local normal does no frictional work — so this, not cloth_next_proj, is
        # where the dissipation must be priced. Pricing post-restore charges the restore's component
        # inside the n_hat-tangent plane as slip, which on ANY curved geometry puts a mu-scaled
        # gradient on fully-arrested contacts (measured: 24% of a genuine sliding gradient) — an
        # arrest pull on the mu-blind net, exactly the property the objective exists to avoid.
        # On flat geometry the two coincide exactly (n_c == n_hat ⇒ the restore is projected out).
        cloth_next_tang = cloth_next_proj
        PROF.end('op_cone')

        PROF.begin('op_normal')
        # --- normal completion — restore the refreshed nearest-face plane-gap
        # proxy on the friction-ALTERED subset only (µ=0 => empty set => byte-identical). Uses the
        # SAME next-pose face geometry as the gate above. Ambiguous / over-cap verts revert. -----
        if normal_exact != 'off':
            fdx_norm = friction_dx.norm(dim=-1)                                           # (Vc,)
            if normal_exact_control:
                ne_cand = in_contact & (lambda_n.reshape(-1) > 0) & (~pinned)             # matched set
            else:
                ne_cand = in_contact & (fdx_norm > normal_exact_dust) & (~pinned)
            ne_idx = ne_cand.nonzero().reshape(-1)
            PROF.count('n_altered', ne_idx.numel())   # host-side: .nonzero() has already synced
            if ne_idx.numel():
                fdx_n = fdx_norm[ne_idx].unsqueeze(-1)                                    # (n,1)
                if normal_exact_max_step > 0:
                    cap = torch.full_like(fdx_n, float(normal_exact_max_step))
                else:
                    cap = torch.maximum(torch.full_like(fdx_n, 2.0 * eps), fdx_n)
                x_t_sub = cloth_next_proj[ne_idx]
                if tangential_path in ('surface', 'walk'):
                    # walk the friction shrink back along a geometric slide path
                    # from v̂ = cloth_curr + delta_b, then re-apply the endpoint restore at the
                    # landing point. Falls back per-vertex to the 'chord' result on ambiguity / arc
                    # blow-up (floor =). arc_cone='off' reuses the chord factor; 'on' is arc-native.
                    # 'surface' builds the path with the S-substep march; 'walk' with the
                    # parameter-free triangle walk -- everything downstream of the builder is shared.
                    x_hat_sub = (cloth_curr + delta_b)[ne_idx]                            # (n,3) v̂
                    b_sub = (mu_i * lambda_n * (h * h) / m)[ne_idx]                       # (n,1) budget
                    mn_kwargs = {}
                    if path_normals == 'multi':
                        # per-segment re-projection inputs. Validation above
                        # guarantees force_balance (so drive_gi/gi_proj/lam_pre_mn exist) and no
                        # mollifier (so lambda_n == clamp(lam_pre_mn, 0) bitwise).
                        mn_kwargs = dict(path_normals='multi', path_capstan=path_capstan,
                                         mu=mu_i, h=h, m_sub=m[ne_idx],
                                         drive_gi=drive_gi[ne_idx], gi_proj=gi_proj[ne_idx],
                                         lam_primary=lam_pre_mn[ne_idx])
                    if tangential_path == 'walk':
                        mn_kwargs.update(path_builder='walk', obs_pos=obstacle_next,
                                         obs_faces=obstacle_faces,
                                         walk_max_cross=path_walk_max_cross,
                                         walk_fast=path_walk_fast)
                    tang_out = geometry_aware_transport(
                        x_hat_sub, cloth_next[ne_idx], x_t_sub, cap,
                        factor[ne_idx], b_sub, eps_v * h,
                        face_next_centroid, obstacle_fn, mode=normal_exact,
                        drag_k=drag_k,
                        arc_cone=path_arc_cone, substeps=path_substeps, arc_cap=path_arc_cap,
                        iters=normal_exact_iters, slack=normal_exact_slack,
                        K=normal_exact_ambig_k, ambig_ratio=normal_exact_ambig_ratio,
                        ambig_cos=normal_exact_ambig_cos,
                        return_tangential=(return_action or wave_enabled), **mn_kwargs)
                    if return_action or wave_enabled:
                        # [membrane wave] the wave's source is the marched tangential landing, so the wave
                        # arm needs q_eff even when the action loss is off (Eq. (2)).
                        x_out, q_eff, ne_stats = tang_out
                        # the marched walk-back landing point IS the tangential outcome here
                        cloth_next_tang = cloth_next_tang.clone()
                        cloth_next_tang[ne_idx] = q_eff
                    else:
                        x_out, ne_stats = tang_out
                else:
                    x_out, ne_stats = normal_completion_plane_gap(
                        cloth_next[ne_idx], x_t_sub, cap,
                        face_next_centroid, obstacle_fn, mode=normal_exact,
                        iters=normal_exact_iters, slack=normal_exact_slack,
                        K=normal_exact_ambig_k, ambig_ratio=normal_exact_ambig_ratio,
                        ambig_cos=normal_exact_ambig_cos)
                if NE_DEBUG_SINK is not None:
                    NE_DEBUG_SINK.append({'idx': ne_idx.detach().cpu(),
                                          'x_t': x_t_sub.detach().cpu(),
                                          'x_out': x_out.detach().cpu()})
                cloth_next_proj = cloth_next_proj.clone()
                cloth_next_proj[ne_idx] = x_out
                for k_ne, v_ne in ne_stats.items():
                    if k_ne.endswith('_max'):
                        ne_totals[k_ne] = max(ne_totals.get(k_ne, 0.0), v_ne)
                    else:
                        ne_totals[k_ne] = ne_totals.get(k_ne, 0) + v_ne
        PROF.end('op_normal')

        # `op_normal` ENDS HERE, not after the patch. It must keep the exact extent it had
        # in (" restore / surface march / multi-normal transport") or the 77/23
        # march-vs-clamp split measured in stops being comparable to the number it split.
        # The R3 patch coupling is its own substage. C0 (spread) and C1 (patch relaxation) are
        # MUTUALLY EXCLUSIVE by the guard above, so one stage can never contain both and `op_patch`
        # is unambiguous within any single run; wrapping both is what keeps a coupling cost from
        # falling silently into the operator residual when a C0 arm is timed.
        PROF.begin('op_patch')
        # --- diagnostic: CORRECTION SPREADING — the crude precursor of the C1 patch
        # solve. Each moved contact donates its TANGENTIAL correction to its 1-ring; receivers take
        # the uniform mean over their donor neighbors, scaled by spread_w. Motivation:
        # a lone slowed vertex is re-driven by the network (~75%/frame) and pays stretch/pen because
        # its ring does not come along; carrying the patch is the mechanism under test.
        #   * TANGENTIAL only: the normal channel is contact-specific (each contact has its own
        #     face/gap) — spreading it would push neighbors through geometry they never touched.
        #   * donors = in_contact & |corr| > dust; receivers = NOT donors & NOT pinned & has >=1
        #     donor neighbor. A contact that the operator did not move can receive.
        #   * mu=0 => factor==1 => no corrections => STRUCTURAL no-op (free twin floor).
        # * placed AFTER all Coulomb accounting (mn_*/ne_*, the action, the metric log all
        #     read pre-spread quantities) => attribution stays clean; `shadow` still wins below.
        #   * DIAGNOSTIC: spread_w=0.0 skips the branch entirely (byte-identical off path). C0 is a
        #     probe, never a ship path — C1 replaces this mean with a compliant constraint solve.
        if spread_w:
            corr = cloth_next_proj - cloth_next                                         # (Vc,3)
            donor_m = in_contact & (corr.norm(dim=-1) > normal_exact_dust)               # (Vc,)
            if int(donor_m.sum()):
                corr_T = corr - (corr * n_hat).sum(dim=-1, keepdim=True) * n_hat        # (Vc,3)
                cf_sp = ex['cloth'].faces_batch.T
                vw_sp = None
                if spread_weight == 'area':
                    # rest positions when the sample carries them (a material property, constant
                    # through the rollout and cacheable); current pose is the documented fallback.
                    rp_sp = ex['cloth'].rest_pos if 'rest_pos' in ex['cloth'] else cloth_curr
                    vw_sp = vertex_areas(rp_sp, cf_sp)                                  # (Vc,1)
                sp_stats = {}
                if spread_hops > 1 and spread_gamma > 0:
                    # diagnostic: walk the field outward, attenuating per hop (hop 1 is unchanged)
                    dx_s, recv_m, hops_used = ring_spread_multi(
                        corr_T, donor_m, pinned, cf_sp,
                        spread_hops, spread_gamma, vw_sp, sp_stats)
                else:
                    dx_s, recv_m = ring_spread(corr_T, donor_m, pinned, cf_sp,
                                               None, vw_sp, sp_stats)                   # (Vc,3),(Vc,)
                    hops_used = 1
                for k_sp, v_sp in sp_stats.items():
                    ne_totals[k_sp] = ne_totals.get(k_sp, 0) + v_sp
                if spread_project == 'own':
                    # diagnostic: re-frame the received mean into the RECEIVER's own tangent plane.
                    # C0 leaves it in the donor's frame, which is exactly why a receiver over a
                    # body curving away beneath it drives inward. Uses the
                    # already-computed per-vertex n_hat — no extra KNN, and receivers move sub-mm,
                    # far below the scale at which a refreshed query would differ.
                    dxp = dx_s - (dx_s * n_hat).sum(dim=-1, keepdim=True) * n_hat
                    with torch.no_grad():
                        ne_totals['sp_proj_sum'] = (ne_totals.get('sp_proj_sum', 0.0)
                                                    + float((dxp - dx_s).norm(dim=-1).sum()))
                    dx_s = dxp
                cloth_next_proj = cloth_next_proj + float(spread_w) * dx_s
                with torch.no_grad():
                    dxn_s = dx_s.norm(dim=-1)[recv_m] * float(spread_w)
                    ne_totals['sp_donors'] = ne_totals.get('sp_donors', 0) + int(donor_m.sum())
                    ne_totals['sp_receivers'] = ne_totals.get('sp_receivers', 0) + int(recv_m.sum())
                    ne_totals['sp_dx_sum'] = (ne_totals.get('sp_dx_sum', 0.0)
                                              + (float(dxn_s.sum()) if dxn_s.numel() else 0.0))
                    ne_totals['sp_dx_max'] = max(ne_totals.get('sp_dx_max', 0.0),
                                                 float(dxn_s.max()) if dxn_s.numel() else 0.0)
                    ne_totals['sp_hops_used'] = max(ne_totals.get('sp_hops_used', 0), hops_used)
        # --- diagnostic: COMPLIANT PATCH RELAXATION. C0 proved patch coupling is worth
        # having (slowing deeper AND cheaper) and paid for it in penetration, because it slides a
        # free vertex along someone ELSE's tangent plane. C1 keeps the coupling and removes that
        # cause: the corrected contacts become anchors, their ring is relaxed by an XPBD solve that
        # answers to its own edge lengths, and a non-worsening Signorini step guarantees no free
        # vertex ends deeper in the body than the backbone had it.
        #   * anchors = the same donor predicate as C0 (in_contact & |corr| > dust) => the two
        #     couplings are driven by an IDENTICAL correction field; only the reach law differs.
        #   * targets = edge lengths under cloth_next (the frictionless prediction), NOT rest
        #     lengths — see patch_relax's docstring for why rest would be a drape intervention.
        #   * mu=0 => no anchor moves => every C_j == 0 => zero dlambda => STRUCTURAL no-op, so
        #     each arm's mu0 rung is a free exact twin (same argument as C0 / the capstan).
        #   * placed AFTER all Coulomb accounting => friction attribution stays clean, as in C0.
        if patch_iters:
            corr_pc = cloth_next_proj - cloth_next                                      # (Vc,3)
            anchor_m = in_contact & (corr_pc.norm(dim=-1) > normal_exact_dust)          # (Vc,)
            if int(anchor_m.sum()):
                cf_pc = ex['cloth'].faces_batch.T
                uni_pc, inv_pc = mesh_edges_undirected(cf_pc)
                free_m = (ring_dilate(anchor_m, uni_pc, int(patch_depth))
                          & (~anchor_m) & (~pinned))                                    # (Vc,)
                if int(free_m.sum()):
                    # every edge with a free endpoint: the ones reaching OUTSIDE the patch are kept
                    # on purpose — they are what anchors the patch to the un-moved surroundings, and
                    # their residual IS the injected boundary stretch we report.
                    sel_pc = free_m[uni_pc[:, 0]] | free_m[uni_pc[:, 1]]
                    ed_pc = uni_pc[sel_pc]                                              # (E,2)
                    if PROF.enabled:
                        # The patch solve's cost is problem-size driven — K sweeps over E
                        # edges — so a time without these is not interpretable.
                        # Masks are passed as TENSORS (stashed, read at end-of-sequence: no sync);
                        # `ed_pc.shape` is a host-side shape and free. Gated so the off path does
                        # not even launch the extra reductions.
                        PROF.count('n_patch_anchors', anchor_m.sum())
                        PROF.count('n_patch_free', free_m.sum())
                        PROF.count('n_patch_edges', ed_pc.shape[0])
                    w_inv = torch.where(free_m.unsqueeze(-1), 1.0 / m.clamp(min=1e-30),
                                        torch.zeros_like(m))                            # (Vc,1)
                    k_pc = None
                    if ('rest_pos' in ex['cloth'] and 'lame_mu' in ex['cloth']
                            and 'lame_lambda' in ex['cloth']):
                        k_pc = edge_membrane_stiffness(
                            ex['cloth'].rest_pos, cf_pc, uni_pc, inv_pc,
                            ex['cloth'].lame_mu.reshape(-1)[0],
                            ex['cloth'].lame_lambda.reshape(-1)[0])[sel_pc]              # (E,1)
                        # alpha = 1/k (scaled), alpha~ = alpha/h^2. scale=0 => stiff PBD exactly.
                        alpha_t = float(patch_alpha_scale) / (k_pc * (h * h)).clamp(min=1e-30)
                    else:
                        # no material on the sample => stiff PBD (alpha = 0). Counted, never silent.
                        alpha_t = torch.zeros((ed_pc.shape[0], 1), dtype=m.dtype, device=device)
                        ne_totals['pc_nomat'] = ne_totals.get('pc_nomat', 0) + 1
                    x_pc, pc_stats = patch_relax(cloth_next_proj, cloth_next, ed_pc, w_inv,
                                                 alpha_t, patch_iters, k_edge=k_pc,
                                                 feas_tol=float(patch_feas_tol))
                    if patch_signorini != 'off':
                        # NON-WORSENING only: restore each free vertex's own pre-solve plane gap if
                        # the relaxation reduced it. Never an improving clamp (that would buy `pen`
                        # for a reason unrelated to coupling and confound the primary read), and
                        # never applied where the nearest-face query is ambiguous.
                        f_idx = free_m.nonzero().reshape(-1)
                        gq = dict(K=normal_exact_ambig_k, ambig_ratio=normal_exact_ambig_ratio,
                                  ambig_cos=normal_exact_ambig_cos)
                        g_pre_pc, _, _, amb_a = refreshed_plane_gap(
                            cloth_next_proj[f_idx], face_next_centroid, obstacle_fn, **gq)
                        g_post_pc, n_post_pc, _, amb_b = refreshed_plane_gap(
                            x_pc[f_idx], face_next_centroid, obstacle_fn, **gq)
                        ok_pc = (~amb_a) & (~amb_b)
                        push = torch.clamp(g_pre_pc - g_post_pc, min=0.0) * ok_pc.unsqueeze(-1)
                        x_pc = x_pc.clone()
                        x_pc[f_idx] = x_pc[f_idx] + push * n_post_pc
                        with torch.no_grad():
                            pc_stats['pc_sig_n'] = int((push.reshape(-1) > 0).sum())
                            pc_stats['pc_sig_sum'] = float(push.sum())
                            pc_stats['pc_sig_max'] = float(push.max()) if push.numel() else 0.0
                            pc_stats['pc_sig_ambig'] = int((~ok_pc).sum())
                    cloth_next_proj = x_pc
                    with torch.no_grad():
                        pc_stats['pc_frames'] = 1
                        pc_stats['pc_anchors'] = int(anchor_m.sum())
                        pc_stats['pc_free'] = int(free_m.sum())
                        for k_pcs, v_pcs in pc_stats.items():
                            if k_pcs.endswith('_max'):
                                ne_totals[k_pcs] = max(ne_totals.get(k_pcs, 0.0), v_pcs)
                            else:
                                ne_totals[k_pcs] = ne_totals.get(k_pcs, 0) + v_pcs
        # --- membrane wave: incremental tangent-membrane WAVE — the material-transport
        # alternative to C0/C1 (mutually exclusive by the guard above). The tangential landing
        # c_T = cloth_next_tang - cloth_next is carried through non-contact cloth as a beta=0
        # mass-lumped membrane wave over exactly one frame; resolved contacts and pins are
        # zero-Dirichlet boundaries and KEEP their complete Coulomb result (Eq. (16)).
        #   * no depth / gamma / K: reach = material speed x h; substeps certified per frame
        #     (Gershgorin, 0.8 safety) and bucket-laddered for one CUDA graph per garment.
        #   * mu=0 => c_T all-dust => A empty => STRUCTURAL no-op (free twin, as C0/C1).
        #   * same non-worsening Signorini guard as C1, counted under wv_sig_*.
        if wave_enabled:
            from utils.itmw_wave import wave_transport                 # lazy: no import cycle
            gq_wv = dict(K=normal_exact_ambig_k, ambig_ratio=normal_exact_ambig_ratio,
                         ambig_cos=normal_exact_ambig_cos)
            cloth_next_proj = wave_transport(
                ex['cloth'], cloth_next, cloth_next_tang, cloth_next_proj,
                in_contact, pinned, m, h,
                lambda xs: refreshed_plane_gap(xs, face_next_centroid, obstacle_fn, **gq_wv),
                float(normal_exact_dust), ne_totals,
                visco_g=wave_visco, visco_tau_frames=wave_tau_frames,
                strain_cap=wave_strain_cap)
        PROF.end('op_patch')
        # --- the frictional dissipation this projection just paid.
        # D = sum_active mu * sg[lambda_n] * (f0(||Delta_T||) - band), f0 the same pseudo-Huber whose
        # prox the cone above IS, band = eps_v*h (so a stuck contact pays 0 — a per-contact constant,
        # gradient-neutral). Measured on cloth_next_TANG (the tangential transport, BEFORE the
        # endpoint restore — see its definition above for why), with the slip origin the body-carried
        # previous position (delta_b), identical to the cone's own.
        if return_action:
            delta_c_post = cloth_next_tang - cloth_curr                                    # (Vc,3)
            delta_rel_post = delta_c_post - delta_b                                        # (Vc,3)
            dT_post = (delta_rel_post
                       - (delta_rel_post * n_hat).sum(dim=-1, keepdim=True) * n_hat)       # (Vc,3)
            band_a = eps_v * h
            f0_post = torch.sqrt((dT_post * dT_post).sum(dim=-1, keepdim=True) + band_a ** 2)
            act_mask = in_contact.unsqueeze(-1).to(f0_post.dtype)                          # (Vc,1)
            D_ex = (mu_i * lambda_n * (f0_post - band_a) * act_mask).sum()
            action_total = D_ex if action_total is None else (action_total + D_ex)
            with torch.no_grad():
                slip_post_act = torch.linalg.vector_norm(dT_post, dim=-1, keepdim=True)
                action_slip_sum += float((slip_post_act * act_mask).sum())
                action_n_active += int(in_contact.sum())
        # --- fmetrics logging hook: capture the operator's ACTIVE SET exactly as
        # consumed (post gates/smoothing lambda_n, pre-friction delta_T, applied factor). Pure
        # read-only capture of already-computed tensors; placed after the completion so all
        # values are final. Units: delta_T/x_pre in meters (displacement form, P3). -------------
        if return_metric_log:
            ml_idx = in_contact.nonzero().reshape(-1)
            delta_rel_ml = delta_c - delta_b
            rel_n_ml = (delta_rel_ml * n_hat).sum(dim=-1, keepdim=True) * n_hat
            delta_T_full = delta_rel_ml - rel_n_ml
            mlog_chunks.append({
                'vertex': (ml_idx + off).to(torch.int32),
                'face': nn_idx[ml_idx, 0].to(torch.int32),
                'lambda_n': lambda_n[ml_idx].reshape(-1).detach(),
                'delta_T': delta_T_full[ml_idx].detach(),
                'factor': factor[ml_idx].reshape(-1).detach(),
                'x_pre': cloth_next[ml_idx].detach(),
            })
        # --- fmetrics shadow mode: the trajectory stays the untouched backbone; the
        # lambda stack / cone factor / metric log above were still computed (CSA reconstruction
        # for operator-less ablations). ---------------------------------------------------------
        if shadow:
            cloth_next_proj = cloth_next
        pred_new_chunks.append(cloth_next_proj)
        contact_chunks.append(in_contact)
        if return_force:
            # per-vertex normal-force magnitude (mollified if lambda_smooth on — even in render_only
            # mode), zero off the contact set (render only)
            force_chunks.append(lambda_render.reshape(-1) * in_contact.to(lambda_n.dtype))

        if return_diag:
            cm = in_contact
            diag_lambda_n.append(lambda_n[cm].reshape(-1))
            diag_slip_pre.append(slip[cm].reshape(-1))
            diag_slip_post.append(slip_post[cm].reshape(-1))
            diag_factor.append(factor[cm].reshape(-1))
            cos = n_hat[:, z_axis].abs().unsqueeze(-1)                                     # |n_hat . up|
            diag_mg_cos.append((m * g * cos)[cm].reshape(-1))
            diag_n_up.append(n_hat[:, z_axis][cm].reshape(-1))                             # signed n_up (falsify)
            # spikiness instrumentation: how point-spiky is the (rendered) lambda field?
            if int(cm.sum()) > 2:
                cf_q = ex['cloth'].faces_batch.T
                e_q = torch.cat([cf_q[:, [0, 1]], cf_q[:, [1, 0]], cf_q[:, [1, 2]],
                                 cf_q[:, [2, 1]], cf_q[:, [0, 2]], cf_q[:, [2, 0]]], 0)
                cmf = cm.to(lambda_n.dtype).unsqueeze(-1)
                lamc = lambda_render * cmf
                num_q = torch.zeros_like(lamc); num_q.index_add_(0, e_q[:, 1], lamc[e_q[:, 0]])
                den_q = torch.zeros_like(lamc); den_q.index_add_(0, e_q[:, 1], cmf[e_q[:, 0]])
                ring = num_q / den_q.clamp(min=1.0)                                        # contact-ring mean
                has_nbr = cm & (den_q.reshape(-1) > 0)
                eps_r = 0.05 * float(lamc[cm].mean()) + 1e-30
                r_spk = (lamc[has_nbr].reshape(-1) / (ring[has_nbr].reshape(-1) + eps_r))
                both = cm[e_q[:, 0]] & cm[e_q[:, 1]]
                tv = float((lamc[e_q[:, 0]][both] - lamc[e_q[:, 1]][both]).abs().sum()
                           / (2.0 * lamc[cm].sum().clamp(min=1e-30)))
                diag_spike.append((float(torch.quantile(r_spk, 0.95)) if r_spk.numel() > 10 else 0.0,
                                   float(r_spk.max()) if r_spk.numel() else 0.0, tv, int(cm.sum())))

        off += cloth_next.shape[0]  # advance for the next example's elastic_grad slice

    pred_pos_new = torch.cat(pred_new_chunks, dim=0)
    force_full = None
    if return_force:
        force_full = (torch.cat(force_chunks, dim=0) if force_chunks
                      else pred_pos_new.new_zeros(pred_pos_new.shape[0]))
    mlog_full = None
    if return_metric_log:
        if mlog_chunks:
            mlog_full = {k: torch.cat([c[k] for c in mlog_chunks], dim=0) for k in mlog_chunks[0]}
        else:
            dev = pred_pos_new.device
            mlog_full = {'vertex': torch.zeros(0, dtype=torch.int32, device=dev),
                         'face': torch.zeros(0, dtype=torch.int32, device=dev),
                         'lambda_n': torch.zeros(0, device=dev),
                         'delta_T': torch.zeros(0, 3, device=dev),
                         'factor': torch.zeros(0, device=dev),
                         'x_pre': torch.zeros(0, 3, device=dev)}
    action_full = None
    if return_action:
        # No contact anywhere in the batch => a genuine constant zero (adding it to a loss sum is
        # harmless; the other terms carry the graph).
        action_full = (action_total if action_total is not None
                       else pred_pos_new.new_zeros(()))
    if not return_diag:
        ret = [pred_pos_new]
        if return_force:
            ret.append(force_full)
        if return_metric_log:
            ret.append(mlog_full)
        if return_action:
            ret.append(action_full)
        return tuple(ret) if len(ret) > 1 else pred_pos_new

    contact = torch.cat(contact_chunks, dim=0)
    n_contact = int(contact.sum().item())

    def _stat(chunks, fn):
        if n_contact == 0:
            return 0.0
        v = torch.cat(chunks, dim=0)
        return float(fn(v).item()) if v.numel() else 0.0

    ln = torch.cat(diag_lambda_n, dim=0) if diag_lambda_n else pred_pos_new.new_zeros(0)
    fac = torch.cat(diag_factor, dim=0) if diag_factor else pred_pos_new.new_zeros(0)
    nup = torch.cat(diag_n_up, dim=0) if diag_n_up else pred_pos_new.new_zeros(0)
    diag = {
        'n_contact': n_contact,
        'n_total': int(contact.numel()),
        'contact_frac': (n_contact / max(1, contact.numel())),
        'lambda_n_mean': _stat(diag_lambda_n, torch.mean),
        'lambda_n_median': _stat(diag_lambda_n, torch.median),
        'lambda_n_max': float(ln.max().item()) if ln.numel() else 0.0,
        'mg_cos_mean': _stat(diag_mg_cos, torch.mean),        # rest-load reference for calibration
        'slip_pre_mean': _stat(diag_slip_pre, torch.mean),    # ||Delta_T|| before projection
        'slip_post_mean': _stat(diag_slip_post, torch.mean),  # ||Delta_T|| after projection
        # stick_frac = fraction fully arrested (factor==0). factor_mean = mean retained tangential
        # fraction over contact verts (1=no friction, 0=all stuck) -- the cone strength, mu-controlled.
        # slip_redux = aggregate tangential slip removed by friction (the headline effect size).
        'stick_frac': (float((fac <= 1e-12).float().mean().item()) if fac.numel() else 0.0),
        'factor_mean': (float(fac.mean().item()) if fac.numel() else 1.0),
    }
    pre, post = diag['slip_pre_mean'], diag['slip_post_mean']
    diag['slip_redux'] = (1.0 - post / pre) if pre > 0 else 0.0
    # normal-completion counters (zeros when normal_exact='off' or no candidates this step)
    # + surface-transport counters (zeros unless tangential_path='surface').
    for k_ne in ('ne_candidates', 'ne_corrected', 'ne_skip_ambig', 'ne_skip_cap',
                 'ne_residual', 'ne_face_switch', 'ne_dn_sum', 'ne_dn_max',
                 'sm_candidates', 'sm_marched', 'sm_fallback_ambig', 'sm_fallback_cap',
                 'sm_face_switch', 'sm_arc_infl_sum', 'sm_arc_infl_max',
                 'sm_dn_sum', 'sm_dn_max',
                 # triangle walk (zeros unless tangential_path='walk'). Listed HERE at
                 # birth -- the strain cap allowlist trap below is exactly what the G5 walk-active
                 # witness exists to catch, so these keys must never be the ones it catches.
                 'wk_pairs', 'wk_same_face', 'wk_seat_edge', 'wk_walked', 'wk_fb_seat',
                 'wk_fb_cross', 'wk_fb_geom', 'wk_cross_sum', 'wk_cross_max',
                 'mn_active', 'mn_landing_moved', 'mn_junction_land', 'mn_fullstick',
                 'mn_exhausted', 'mn_lambda_spread_max', 'mn_wrap_sum', 'mn_capstan_gain_max',
                 'sp_donors', 'sp_receivers', 'sp_dx_sum', 'sp_dx_max', 'sp_proj_sum',
                 'sp_hops_used', 'sp_recv_n', 'sp_src_sum', 'sp_multi', 'sp_cancel_sum',
                 # diagnostic patch relaxation. Sums accumulate over frames+examples; divide by
                 # pc_frames for a per-frame mean (pc_iters_feas and pc_alpha_ratio especially).
                 'pc_frames', 'pc_anchors', 'pc_free', 'pc_edges', 'pc_nomat',
                 'pc_C_pre_sum', 'pc_C_pre_max', 'pc_C_post_sum', 'pc_C_post_max',
                 'pc_iters_feas', 'pc_alpha_ratio', 'pc_energy_inj',
                 'pc_dx_sum', 'pc_dx_max',
                 'pc_sig_n', 'pc_sig_sum', 'pc_sig_max', 'pc_sig_ambig',
                 # membrane-wave stage. wv_frames counts stage entries; wv_empty the
                 # structural no-ops; wv_ladder_overflow / wv_degenerate the identity tripwires.
                 'wv_frames', 'wv_empty', 'wv_active', 'wv_reached', 'wv_n_sub', 'wv_umax',
                 'wv_ladder_overflow', 'wv_degenerate',
                 'wv_sig_n', 'wv_sig_sum', 'wv_sig_max',
                 # strain cap. NOTE the trap this line fixes: this tuple is an explicit
                 # allowlist — a stats key not listed here silently prints as 0 in the runner
                 # diag while the mechanism fires.
                 'wv_cap_edges0', 'wv_cap_resid', 'wv_cap_dxmax'):
        diag[k_ne] = ne_totals.get(k_ne, 0)
    # spikiness (contact-weighted across examples): ring-ratio p95/max + total variation per unit load
    if diag_spike:
        wts = np.array([s[3] for s in diag_spike], dtype=float); wts /= max(1e-30, wts.sum())
        diag['spike_p95'] = float(np.sum(wts * np.array([s[0] for s in diag_spike])))
        diag['spike_max'] = float(max(s[1] for s in diag_spike))
        diag['lam_tv'] = float(np.sum(wts * np.array([s[2] for s in diag_spike])))
    else:
        diag['spike_p95'] = diag['spike_max'] = diag['lam_tv'] = 0.0

    # --- falsification metrics: does sticking depend on contact orientation? -------------------
    # Splits the contact set by the SIGN of n_up (n_hat . e_up): down/under-facing (n_up<0) vs up-facing
    # (n_up>=0). For the barrier source these rates should be ~equal (orientation-blind => the observed
    # upside-down sticking); force_balance should drive stick_rate_downface -> 0 (gravity unloads there).
    if fac.numel():
        stuck = (fac <= 1e-12)
        down = (nup < 0.0)
        n_stuck = int(stuck.sum().item())
        n_down = int(down.sum().item())
        n_up_c = int((~down).sum().item())
        n_stuck_down = int((stuck & down).sum().item())
        n_stuck_up = int((stuck & ~down).sum().item())
        diag['n_up_signed_mean'] = float(nup.mean().item()) if nup.numel() else 0.0
        diag['contact_frac_downface'] = float(down.float().mean().item())
        diag['stick_rate_downface'] = (n_stuck_down / n_down) if n_down else 0.0
        diag['stick_rate_upface'] = (n_stuck_up / n_up_c) if n_up_c else 0.0
        diag['stuck_frac_downface'] = (n_stuck_down / n_stuck) if n_stuck else 0.0
        # raw counts for exact cross-step aggregation in falsification tooling
        diag['n_contact_downface'] = n_down
        diag['n_contact_upface'] = n_up_c
        diag['n_stuck_downface'] = n_stuck_down
        diag['n_stuck_upface'] = n_stuck_up
    else:
        for k in ('n_up_signed_mean', 'contact_frac_downface', 'stick_rate_downface',
                  'stick_rate_upface', 'stuck_frac_downface', 'n_contact_downface',
                  'n_contact_upface', 'n_stuck_downface', 'n_stuck_upface'):
            diag[k] = 0.0
    # action diagnostics (detached; the tensor itself rides the `action` return slot).
    if return_action:
        diag['action_D'] = float(action_full.detach())
        diag['action_n_active'] = action_n_active
        diag['action_slip_post_mean'] = (action_slip_sum / action_n_active) if action_n_active else 0.0
    ret = [pred_pos_new, diag]
    if return_force:
        ret.append(force_full)
    if return_metric_log:
        ret.append(mlog_full)
    if return_action:
        ret.append(action_full)
    return tuple(ret)
