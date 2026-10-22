"""membrane wave A1 reference membrane operator.

Standalone, matrix-free implementation of the frozen-base incremental triangle StVK tangent
K_0 and its mass-normalized symmetric form A = M^{-1/2} K_0 M^{-1/2}, exactly as frozen by the
A0 contract:

    D_s(x_f) = [x_0 - x_2  |  x_1 - x_2]                      (repository get_shape_matrix order)
    F^0_f    = D_s(x^0_f) Dm_inv_f                            (frozen per call)
    dF_f(z)  = D_s(z_f)  Dm_inv_f
    eta_f(z) = 0.5 * ((F^0)^T dF + dF^T F^0)                  (incremental Green strain, F x 2 x 2)
    S_f(z)   = lambda tr(eta) I + 2 mu_s eta                  (incremental 2nd PK stress, Pa)
    V_w(z)   = sum_f A_f t [ lambda/2 tr(eta)^2 + mu_s tr(eta^2) ]      [J]
    (K_0 z)  = scatter of P_f = A_f t * F^0 S_f(z) Dm_inv^T   (columns -> verts 0,1; vert 2 = -sum)

Deliberately EXCLUDES positional geometric/initial-stress stiffness (A0 honesty clause) and
carries no body-force or world-up branch (A0 axis courtesy). Units: positions m, Lame Pa (volumetric),
thickness m, f_area m^2, v_mass kg, energy J, K_0 z N when z is m.

No imports from utils/, runners/, criterions/ or models/: the Dm_inv construction is a small
*copy* of the production path (utils/cloth_and_material.py::make_Dm_inv) made dtype-preserving
(the production edges_3d_to_2d allocates float32 unconditionally); the test suite proves the copy
matches production bitwise on float32 inputs (test A05/U01 companions in an internal tool).
"""
import math

import numpy as np
import torch


# ── geometry construction (dtype-preserving copies of the production recipe) ──────────────────

def gather_tri(x, faces):
    """(V,3) positions + (F,3) long faces -> (F,3,3) per-face vertex stack."""
    return x[faces]


def shape_matrix(tri):
    """(F,3,3) triangle verts -> (F,3,2) D_s with columns [x0-x2, x1-x2] (repo convention)."""
    return torch.stack([tri[:, 0] - tri[:, 2], tri[:, 1] - tri[:, 2]], dim=-1)


def edges_3d_to_2d(edges):
    """Dtype/device-preserving copy of utils.cloth_and_material.edges_3d_to_2d.

    :param edges: (F,2,3) rows are the two D_s columns in 3D
    :return: (F,2,2) same edges expressed in the triangle's intrinsic orthonormal basis
    """
    edges0 = edges[:, 0]
    edges1 = edges[:, 1]
    basis2d_0 = edges0 / torch.norm(edges0, dim=-1).unsqueeze(-1)
    n = torch.cross(basis2d_0, edges1, dim=-1)
    basis2d_1 = torch.cross(n, edges0, dim=-1)
    basis2d_1 = basis2d_1 / torch.norm(basis2d_1, dim=-1).unsqueeze(-1)
    edges2d = torch.zeros((edges.shape[0], edges.shape[1], 2),
                          dtype=edges.dtype, device=edges.device)
    edges2d[:, 0, 0] = (edges0 * basis2d_0).sum(-1)
    edges2d[:, 0, 1] = (edges0 * basis2d_1).sum(-1)
    edges2d[:, 1, 0] = (edges1 * basis2d_0).sum(-1)
    edges2d[:, 1, 1] = (edges1 * basis2d_1).sum(-1)
    return edges2d


def make_dm_inv(rest_pos, faces):
    """(V,3),(F,3) -> (F,2,2) inverse rest shape matrix, production recipe, dtype-preserving.

    Mirrors utils.cloth_and_material.Cloth.make_Dm_inv: D_s -> rows -> intrinsic 2D -> transpose
    back -> inverse. Raises on singular rest triangles instead of clamping (plan A1.1).
    """
    ds = shape_matrix(gather_tri(rest_pos, faces))         # (F,3,2)
    edges = ds.permute(0, 2, 1)                            # (F,2,3) rows = D_s columns
    edges_2d = edges_3d_to_2d(edges).permute(0, 2, 1)      # (F,2,2)
    det = (edges_2d[:, 0, 0] * edges_2d[:, 1, 1] - edges_2d[:, 0, 1] * edges_2d[:, 1, 0])
    if not torch.isfinite(edges_2d).all():
        bad = (~torch.isfinite(edges_2d).reshape(edges_2d.shape[0], -1).all(dim=1)).nonzero()
        raise ValueError(f'nonfinite rest shape matrix at faces {bad.reshape(-1).tolist()[:10]}')
    if (det.abs() <= 0).any():
        bad = (det.abs() <= 0).nonzero().reshape(-1)
        raise ValueError(f'singular rest triangle at faces {bad.tolist()[:10]}')
    return torch.inverse(edges_2d)


def face_areas(rest_pos, faces):
    """(V,3),(F,3) -> (F,) rest areas in m^2 (0.5 |cross|, same formula as production)."""
    v0, v1, v2 = rest_pos[faces[:, 0]], rest_pos[faces[:, 1]], rest_pos[faces[:, 2]]
    return 0.5 * torch.cross(v2 - v0, v1 - v0, dim=-1).norm(dim=-1)


def vertex_mass(rest_pos, faces, density):
    """(V,3),(F,3),areal-density kg/m^2 -> (V,) lumped mass in kg (production: area/3 per vert).

    Density is AREAL (kg/m^2) and is never multiplied by thickness (A0).
    """
    a = face_areas(rest_pos, faces) * (density / 3.0)      # (F,)
    m = torch.zeros(rest_pos.shape[0], dtype=rest_pos.dtype, device=rest_pos.device)
    for i in range(3):
        m.index_add_(0, faces[:, i], a)
    return m


def wave_speeds(lame_mu, lame_lambda, thickness, density):
    """Flat small-strain membrane diagnostic speeds (m/s) — NOT spectral certification.

    c_s = sqrt(mu_s t / rho_A), c_p = sqrt((lambda + 2 mu_s) t / rho_A).
    """
    c_s = math.sqrt(lame_mu * thickness / density)
    c_p = math.sqrt((lame_lambda + 2.0 * lame_mu) * thickness / density)
    return c_s, c_p


# ── the operator ───────────────────────────────────────────────────────────────────────────────

class MembraneOperator:
    """Matrix-free K_0 / A application with a frozen base F^0.

    All tensors share one dtype/device. lame_mu/lame_lambda are Pa scalars (float or 0-dim
    tensor); thickness m; f_area (F,) m^2; v_mass (V,) kg. No autograd is used on the apply path.
    """

    def __init__(self, faces, dm_inv, f_area, v_mass, lame_mu, lame_lambda, thickness):
        if (v_mass <= 0).any():
            bad = (v_mass <= 0).nonzero().reshape(-1)
            raise ValueError(f'nonpositive vertex mass at vertices {bad.tolist()[:10]}')
        if not torch.isfinite(dm_inv).all():
            raise ValueError('nonfinite Dm_inv')
        self.faces = faces
        self.dm_inv = dm_inv                                  # (F,2,2)
        self.f_area = f_area.reshape(-1)                      # (F,)
        self.v_mass = v_mass.reshape(-1)                      # (V,)
        self.lame_mu = float(lame_mu)
        self.lame_lambda = float(lame_lambda)
        self.thickness = float(thickness)
        self.n_verts = self.v_mass.shape[0]
        self.f0 = None                                        # (F,3,2) frozen base
        self._at = (self.f_area * self.thickness)             # (F,) A_f * t
        self._inv_sqrt_m = (1.0 / torch.sqrt(self.v_mass)).unsqueeze(-1)   # (V,1)

    def set_base(self, x0):
        """Freeze F^0 = D_s(x0) Dm_inv for this operator call."""
        self.f0 = shape_matrix(gather_tri(x0, self.faces)) @ self.dm_inv
        return self

    # -- kernels ------------------------------------------------------------------------------

    def delta_f(self, z, face_idx=None):
        """(V,3) -> (F,3,2) incremental deformation gradient dF(z) (optionally on a face subset)."""
        faces = self.faces if face_idx is None else self.faces[face_idx]
        dm = self.dm_inv if face_idx is None else self.dm_inv[face_idx]
        return shape_matrix(gather_tri(z, faces)) @ dm

    def eta(self, z, face_idx=None):
        """(V,3) -> (F,2,2) incremental Green-strain linearization around F^0."""
        f0 = self.f0 if face_idx is None else self.f0[face_idx]
        df = self.delta_f(z, face_idx)
        ftdf = f0.transpose(1, 2) @ df                        # (F,2,2)
        return 0.5 * (ftdf + ftdf.transpose(1, 2))

    def energy(self, u):
        """V_w(u) in J via the explicit Eq. (7) sum (test oracle; equals 0.5 u.K u)."""
        e = self.eta(u)
        tr = e[:, 0, 0] + e[:, 1, 1]
        tr_e2 = (e * e).sum(dim=(1, 2))                       # tr(eta^T eta) == tr(eta^2), sym
        dens = 0.5 * self.lame_lambda * tr * tr + self.lame_mu * tr_e2
        return (self._at * dens).sum()

    def apply_K(self, z, face_idx=None, out=None):
        """Analytic K_0 z (N when z is m). Face-local stress + scatter; no autograd.

        :param face_idx: optional (Fa,) long subset of active faces (static-mask timing mode)
        :param out: optional preallocated (V,3) output, zeroed here (timing path avoids realloc)
        """
        faces = self.faces if face_idx is None else self.faces[face_idx]
        f0 = self.f0 if face_idx is None else self.f0[face_idx]
        dm = self.dm_inv if face_idx is None else self.dm_inv[face_idx]
        at = self._at if face_idx is None else self._at[face_idx]

        df = shape_matrix(gather_tri(z, faces)) @ dm          # (F,3,2)
        ftdf = f0.transpose(1, 2) @ df
        e = 0.5 * (ftdf + ftdf.transpose(1, 2))               # (F,2,2)
        tr = (e[:, 0, 0] + e[:, 1, 1]).unsqueeze(-1).unsqueeze(-1)
        s = self.lame_lambda * tr * torch.eye(2, dtype=z.dtype, device=z.device) \
            + 2.0 * self.lame_mu * e                          # (F,2,2)
        p = at.unsqueeze(-1).unsqueeze(-1) * (f0 @ s @ dm.transpose(1, 2))   # (F,3,2)

        if out is None:
            out = torch.zeros(self.n_verts, 3, dtype=z.dtype, device=z.device)
        else:
            out.zero_()
        out.index_add_(0, faces[:, 0], p[:, :, 0])
        out.index_add_(0, faces[:, 1], p[:, :, 1])
        out.index_add_(0, faces[:, 2], -(p[:, :, 0] + p[:, :, 1]))
        return out

    def apply_A(self, z):
        """Symmetric mass-normalized operator A z = M^{-1/2} K_0 (M^{-1/2} z). Units 1/s^2."""
        return self._inv_sqrt_m * self.apply_K(self._inv_sqrt_m * z)

    # -- spectrum -----------------------------------------------------------------------------

    def element_matrices(self):
        """(F,9,9) local element stiffness; rows/cols ordered (vertex 0..2) x (coord x,y,z).

        Used only for the conservative Gershgorin-style bound and tiny dense checks —
        never on the timed path.
        """
        f = self.faces.shape[0]
        dt, dev = self.dm_inv.dtype, self.dm_inv.device
        r0 = self.dm_inv[:, 0, :]                             # (F,2)
        r1 = self.dm_inv[:, 1, :]
        w = torch.stack([r0, r1, -(r0 + r1)], dim=1)          # (F,3,2) rows per source vertex j
        # dF basis for (j,a): row a of dF equals w_j -> (F,9,3,2)
        df = torch.zeros(f, 9, 3, 2, dtype=dt, device=dev)
        for j in range(3):
            for a in range(3):
                df[:, j * 3 + a, a, :] = w[:, j, :]
        ftdf = torch.einsum('fip,fkiq->fkpq', self.f0, df)    # (F,9,2,2)
        e = 0.5 * (ftdf + ftdf.transpose(2, 3))
        tr = (e[:, :, 0, 0] + e[:, :, 1, 1])[:, :, None, None]
        eye = torch.eye(2, dtype=dt, device=dev)
        s = self.lame_lambda * tr * eye + 2.0 * self.lame_mu * e
        p = torch.einsum('fip,fkpq,frq->fkir', self.f0, s, self.dm_inv)
        p = self._at[:, None, None, None] * p                 # (F,9,3,2)
        force = torch.stack([p[..., 0], p[..., 1], -(p[..., 0] + p[..., 1])], dim=2)  # (F,9,3v,3c)
        ke = force.reshape(f, 9, 9).transpose(1, 2)           # rows = (vert,coord), cols = basis
        return ke

    def gershgorin_upper(self):
        """Conservative upper bound on lambda_max(A) via element-wise absolute row sums.

        |A_ij| <= sum_e |A^e_ij| and Gershgorin: lambda_max <= max_i sum_j |A_ij|.
        """
        ke = self.element_matrices()
        mf = self.v_mass[self.faces]                          # (F,3)
        s9 = (1.0 / torch.sqrt(mf)).repeat_interleave(3, dim=1)   # (F,9)
        ae = ke * s9[:, :, None] * s9[:, None, :]
        rs = ae.abs().sum(dim=2)                              # (F,9) per-element abs row sums
        acc = torch.zeros(self.n_verts, 3, dtype=ke.dtype, device=ke.device)
        for i in range(3):
            acc.index_add_(0, self.faces[:, i], rs[:, i * 3:(i + 1) * 3])
        return float(acc.max())

    def power_lambda_max(self, n_iter=5000, tol=1e-10, seed=0, free_mask=None):
        """Deterministic power iteration on symmetric PSD A. Returns dict with the Rayleigh
        estimate omega_max^2 (1/s^2), normalized residual ||Az - lam z||/||z||/lam, iterations,
        and convergence flag. Rigid modes (eigenvalue 0) cannot contaminate the MAX estimate.

        :param free_mask: optional (V,) bool — True for FREE vertices. When given, the iteration
            runs on the principal subproblem of the free set (the frozen C u P degrees of freedom
            are zeroed on input and output every iteration, per), so the estimate is
            the constrained lambda_max, which can never exceed the full one (Cauchy interlacing).
        """
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(self.n_verts, 3, generator=g, dtype=torch.float64)
        z = z.to(dtype=self.dm_inv.dtype, device=self.dm_inv.device)
        frozen = None
        if free_mask is not None:
            frozen = ~free_mask.to(device=z.device)
            z[frozen] = 0.0
            if float(z.norm()) == 0.0:
                return dict(lam=0.0, resid=0.0, iters=0, converged=True)
        z = z / z.norm()
        lam_prev, lam, resid = 0.0, 0.0, float('inf')
        it = 0
        for it in range(1, n_iter + 1):
            w = self.apply_A(z)
            if frozen is not None:
                w[frozen] = 0.0
            lam = float((z * w).sum())
            nw = float(w.norm())
            if nw == 0.0:
                return dict(lam=0.0, resid=0.0, iters=it, converged=True)
            resid = float((w - lam * z).norm()) / max(abs(lam), 1e-300)
            z = w / nw
            if it > 8 and abs(lam - lam_prev) <= tol * max(abs(lam), 1e-300):
                break
            lam_prev = lam
        return dict(lam=lam, resid=resid, iters=it, converged=(it < n_iter))


# ── explicit stability (modal 2x2 amplification, numpy float64) ────────────────────────────────

def amplification_rho(dt, omega2, beta):
    """Spectral radius of the symplectic-Euler amplification matrix for one mode (plan).

        v+ = (1 - dt*beta*w2) v - dt*w2 q
        q+ = (1 - dt^2*w2)  q + dt (1 - dt*beta*w2) v

    Vectorized over omega2 (array). Returns max over the array.

    Numerically robust form: with s = dt*beta*w2 and e = dt^2*w2 the matrix identities
    det = 1 - s (exact) and disc = tr^2 - 4 det = (s+e)^2 - 4e (exact) hold; the naive
    tr^2 - 4 det evaluation cancels catastrophically for near-critically-damped modes
    (true disc ~ 1e-20 against ~4e-16 rounding of tr^2 ~ 4), which once inflated rho to
    1 + 1e-8 at every dt and collapsed the stability bisection by 16 orders of magnitude.
    """
    w2 = np.asarray(omega2, dtype=np.float64)
    s = dt * beta * w2
    e = (dt * dt) * w2
    det = 1.0 - s
    tr = 2.0 - s - e
    disc = (s + e) ** 2 - 4.0 * e
    sq = np.sqrt(np.maximum(disc, 0.0))
    rho = np.where(disc >= 0.0,
                   0.5 * (np.abs(tr) + sq),                   # real pair: max|lam|
                   np.sqrt(np.maximum(det, 0.0)))             # complex pair: |lam| = sqrt(det)
    return float(rho.max())


def stable_dt_closed_form(omega2_max, beta):
    """Exact largest stable dt for the scheme above: the Jury conditions for the
    2x2 amplification matrix reduce, for beta >= 0, to  w2 dt^2 + 2 beta w2 dt <= 4, whose
    positive root is  dt_max = sqrt(beta^2 + 4/w2) - beta.  The binding mode is always w2_max
    (the left side is monotone in w2 at fixed dt). beta=0 recovers dt*omega = 2 exactly.
    Cross-checked against the numeric bracket+bisection `stable_dt` by test R01."""
    if omega2_max <= 0:
        return float('inf')
    return math.sqrt(beta * beta + 4.0 / omega2_max) - beta


def omega2_grid(omega2_max, n_grid=512):
    """Modal test grid: 0 plus a log sweep up to omega2_max (the certified upper bound)."""
    if omega2_max <= 0:
        return np.array([0.0])
    return np.concatenate([[0.0], omega2_max * np.logspace(-8, 0, n_grid)])


def stable_dt(omega2_max, beta, n_grid=512, rho_tol=1.0 + 1e-12, iters=200):
    """Largest dt with spectral radius <= 1 over [0, omega2_max], bracket + bisection.

    beta=0 must recover dt*omega = 2 (test M13). Deterministic float64.
    """
    if omega2_max <= 0:
        return float('inf')
    grid = omega2_grid(omega2_max, n_grid)
    lo = 0.0
    hi = 2.0 / math.sqrt(omega2_max)                          # beta=0 elastic limit as seed
    # grow until unstable (beta>0 can only shrink the region; guard anyway)
    for _ in range(200):
        if amplification_rho(hi, grid, beta) > rho_tol:
            break
        lo = hi
        hi *= 2.0
    else:
        return float('inf')
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if amplification_rho(mid, grid, beta) <= rho_tol:
            lo = mid
        else:
            hi = mid
    return lo


def substep_schedule(omega2_upper, beta, h, safety=0.8):
    """(dt_max, dt_use, N) for one frame of length h from the certified upper omega^2.

    The 0.8 safety factor is a declared numerical margin, never a reach parameter (plan).
    """
    dt_max = stable_dt(omega2_upper, beta)
    if not math.isfinite(dt_max) or dt_max <= 0:
        return dt_max, float('nan'), -1
    dt_use = safety * dt_max
    n = int(math.ceil(h / dt_use))
    return dt_max, dt_use, n


# ── the executable uniform loop (timing target; no host syncs inside) ──────────────────────────

def run_uniform_loop(op, u, v, dt, beta, n_substeps, face_idx=None, out=None, free=None):
    """Symplectic-Euler reference loop with the fused damping form K(u + beta v).

    Mutates/returns (u, v). Contains no host-side tensor conversions and no host convergence
    checks — test G03 scans this function's source for the forbidden token spellings, so name
    them nowhere here. Source injection is not part of the A1 feasibility loop.

    :param free: optional (V,1) multiplicative mask, 1.0 on free vertices and 0.0 on the frozen
        C u P set. Frozen velocities are forced to exactly 0 every
        substep, so frozen displacements never move off their initial value (start them at 0).
    """
    inv_m = (1.0 / op.v_mass).unsqueeze(-1)
    if out is None:
        out = torch.zeros_like(u)
    for _ in range(n_substeps):
        if beta != 0.0:
            f = op.apply_K(u + beta * v, face_idx=face_idx, out=out)
        else:
            f = op.apply_K(u, face_idx=face_idx, out=out)
        v = v - dt * (f * inv_m)
        if free is not None:
            v = v * free
        u = u + dt * v
    return u, v
