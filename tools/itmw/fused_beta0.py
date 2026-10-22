"""membrane wave A1R fused conservative (beta = 0) substep prototype.

One genuinely fused GPU realization of the conservative incremental-membrane substep

    f  = K_0 u                       (face-local stress + atomic scatter, ONE kernel)
    v <- (v - dt f / m) * free       (per-dof update, frozen C u P forced to exactly 0)
    u <- u + dt v                    (same kernel; also zeroes f for the next substep)

as two Triton kernels per substep, replayed under a single CUDA graph per frame so the
per-substep cost is kernel EXECUTION, not Python/launch overhead. The math is byte-for-byte
the reference recipe from tools/itmw/reference_membrane.py (same D_s column convention
[x0-x2, x1-x2], same incremental stress, same scatter with vertex 2 = -(col0+col1)); the only
liberty is float32 atomic accumulation order, which test R08 bounds against the float64 oracle.

Strictly beta = 0: no damping term of any form exists in this module (forbids
positive-beta rescue work during A1R). Not imported by any production path.
"""
import math

import torch

from tools.itmw.reference_membrane import gather_tri, shape_matrix

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                      # CPU-only box: tests skip
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _face_force_kernel(u_ptr, f_ptr, faces_ptr, dm_ptr, f0_ptr, at_ptr,
                           lame_mu, lame_lambda, n_faces, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_faces
        i0 = tl.load(faces_ptr + offs * 3 + 0, mask=m, other=0)
        i1 = tl.load(faces_ptr + offs * 3 + 1, mask=m, other=0)
        i2 = tl.load(faces_ptr + offs * 3 + 2, mask=m, other=0)
        # gather u at the three vertices (spatial components x,y,z)
        u0x = tl.load(u_ptr + i0 * 3 + 0, mask=m, other=0.0)
        u0y = tl.load(u_ptr + i0 * 3 + 1, mask=m, other=0.0)
        u0z = tl.load(u_ptr + i0 * 3 + 2, mask=m, other=0.0)
        u1x = tl.load(u_ptr + i1 * 3 + 0, mask=m, other=0.0)
        u1y = tl.load(u_ptr + i1 * 3 + 1, mask=m, other=0.0)
        u1z = tl.load(u_ptr + i1 * 3 + 2, mask=m, other=0.0)
        u2x = tl.load(u_ptr + i2 * 3 + 0, mask=m, other=0.0)
        u2y = tl.load(u_ptr + i2 * 3 + 1, mask=m, other=0.0)
        u2z = tl.load(u_ptr + i2 * 3 + 2, mask=m, other=0.0)
        # D_s(z) columns [u0-u2, u1-u2]  (repository convention)
        d0x, d0y, d0z = u0x - u2x, u0y - u2y, u0z - u2z
        d1x, d1y, d1z = u1x - u2x, u1y - u2y, u1z - u2z
        # Dm_inv (F,2,2) row-major
        dm00 = tl.load(dm_ptr + offs * 4 + 0, mask=m, other=0.0)
        dm01 = tl.load(dm_ptr + offs * 4 + 1, mask=m, other=0.0)
        dm10 = tl.load(dm_ptr + offs * 4 + 2, mask=m, other=0.0)
        dm11 = tl.load(dm_ptr + offs * 4 + 3, mask=m, other=0.0)
        # dF = D_s(z) @ Dm_inv : dF[k,b] = d0[k] dm[0,b] + d1[k] dm[1,b]
        dfax = d0x * dm00 + d1x * dm10
        dfay = d0y * dm00 + d1y * dm10
        dfaz = d0z * dm00 + d1z * dm10
        dfbx = d0x * dm01 + d1x * dm11
        dfby = d0y * dm01 + d1y * dm11
        dfbz = d0z * dm01 + d1z * dm11
        # frozen base F0 (F,3,2) row-major: F0[k,a] at offs*6 + k*2 + a
        f0x0 = tl.load(f0_ptr + offs * 6 + 0, mask=m, other=0.0)
        f0x1 = tl.load(f0_ptr + offs * 6 + 1, mask=m, other=0.0)
        f0y0 = tl.load(f0_ptr + offs * 6 + 2, mask=m, other=0.0)
        f0y1 = tl.load(f0_ptr + offs * 6 + 3, mask=m, other=0.0)
        f0z0 = tl.load(f0_ptr + offs * 6 + 4, mask=m, other=0.0)
        f0z1 = tl.load(f0_ptr + offs * 6 + 5, mask=m, other=0.0)
        # ftdf[a,b] = sum_k F0[k,a] dF[k,b]
        t00 = f0x0 * dfax + f0y0 * dfay + f0z0 * dfaz
        t01 = f0x0 * dfbx + f0y0 * dfby + f0z0 * dfbz
        t10 = f0x1 * dfax + f0y1 * dfay + f0z1 * dfaz
        t11 = f0x1 * dfbx + f0y1 * dfby + f0z1 * dfbz
        # eta = sym(ftdf); S = lambda tr(eta) I + 2 mu eta
        e01 = 0.5 * (t01 + t10)
        trc = t00 + t11
        s00 = lame_lambda * trc + 2.0 * lame_mu * t00
        s11 = lame_lambda * trc + 2.0 * lame_mu * t11
        s01 = 2.0 * lame_mu * e01
        # q = S @ Dm_inv^T : q[a,b] = S[a,0] dm[b,0] + S[a,1] dm[b,1]
        q00 = s00 * dm00 + s01 * dm01
        q01 = s00 * dm10 + s01 * dm11
        q10 = s01 * dm00 + s11 * dm01
        q11 = s01 * dm10 + s11 * dm11
        at = tl.load(at_ptr + offs, mask=m, other=0.0)
        # P[k,b] = A_f t (F0[k,0] q[0,b] + F0[k,1] q[1,b])
        p0x = at * (f0x0 * q00 + f0x1 * q10)
        p0y = at * (f0y0 * q00 + f0y1 * q10)
        p0z = at * (f0z0 * q00 + f0z1 * q10)
        p1x = at * (f0x0 * q01 + f0x1 * q11)
        p1y = at * (f0y0 * q01 + f0y1 * q11)
        p1z = at * (f0z0 * q01 + f0z1 * q11)
        # scatter: vert0 += P[:,0], vert1 += P[:,1], vert2 += -(P[:,0]+P[:,1])
        tl.atomic_add(f_ptr + i0 * 3 + 0, p0x, mask=m)
        tl.atomic_add(f_ptr + i0 * 3 + 1, p0y, mask=m)
        tl.atomic_add(f_ptr + i0 * 3 + 2, p0z, mask=m)
        tl.atomic_add(f_ptr + i1 * 3 + 0, p1x, mask=m)
        tl.atomic_add(f_ptr + i1 * 3 + 1, p1y, mask=m)
        tl.atomic_add(f_ptr + i1 * 3 + 2, p1z, mask=m)
        tl.atomic_add(f_ptr + i2 * 3 + 0, -(p0x + p1x), mask=m)
        tl.atomic_add(f_ptr + i2 * 3 + 1, -(p0y + p1y), mask=m)
        tl.atomic_add(f_ptr + i2 * 3 + 2, -(p0z + p1z), mask=m)

    @triton.jit
    def _update_kernel(u_ptr, v_ptr, f_ptr, invm_ptr, free_ptr, dt, n_dof,
                       BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_dof
        fo = tl.load(f_ptr + offs, mask=m, other=0.0)
        vv = tl.load(v_ptr + offs, mask=m, other=0.0)
        uu = tl.load(u_ptr + offs, mask=m, other=0.0)
        im = tl.load(invm_ptr + offs, mask=m, other=0.0)
        fr = tl.load(free_ptr + offs, mask=m, other=0.0)
        vv = (vv - dt * fo * im) * fr        # frozen C u P dofs forced to exactly 0
        uu = uu + dt * vv                    # frozen u never moves off its start value
        tl.store(v_ptr + offs, vv, mask=m)
        tl.store(u_ptr + offs, uu, mask=m)
        tl.store(f_ptr + offs, 0.0, mask=m)  # zero the accumulator for the next substep


class FusedBeta0Loop:
    """Owns the device buffers and the captured CUDA graph for one topology.

    Contract: `set_base(x0)` freezes F^0 for the frame (eager, tiny — counted in setup time);
    `capture(n_substeps)` records the graph once per (topology, N); `replay()` runs the whole
    N-substep frame as one graph launch. Buffers u/v/f are graph-static; `reset_state()`
    zeroes them (production would seed u/v from the contact source instead — NOT built here).
    """
    BLOCK_FACE = 128
    BLOCK_DOF = 256

    def __init__(self, op32, free_vert_mask, device):
        if not HAVE_TRITON:
            raise RuntimeError('triton unavailable — fused prototype cannot run')
        assert op32.dm_inv.dtype == torch.float32, 'fused prototype is float32-only'
        self.device = device
        self.n_verts = op32.n_verts
        self.n_faces = int(op32.faces.shape[0])
        self.n_dof = self.n_verts * 3
        self.lame_mu = float(op32.lame_mu)
        self.lame_lambda = float(op32.lame_lambda)
        self.faces_long = op32.faces.to(device)
        self.faces = self.faces_long.to(torch.int32).contiguous()
        self.dm = op32.dm_inv.to(device).contiguous()
        self.dm_flat = self.dm.reshape(-1)
        self.at = op32._at.to(device).contiguous()
        inv_m = (1.0 / op32.v_mass.to(device)).reshape(-1, 1).expand(-1, 3)
        self.inv_m_dof = inv_m.contiguous().reshape(-1)
        free = free_vert_mask.to(device=device).reshape(-1, 1).to(torch.float32)
        self.free_dof = free.expand(-1, 3).contiguous().reshape(-1)
        self.f0 = torch.zeros(self.n_faces, 3, 2, dtype=torch.float32, device=device)
        self.f0_flat = self.f0.reshape(-1)
        self.u = torch.zeros(self.n_verts, 3, dtype=torch.float32, device=device)
        self.v = torch.zeros_like(self.u)
        self.f = torch.zeros_like(self.u)
        self.graph = None
        self.graph_n = 0
        self._grid_face = (math.ceil(self.n_faces / self.BLOCK_FACE),)
        self._grid_dof = (math.ceil(self.n_dof / self.BLOCK_DOF),)

    def set_base(self, x0):
        """Per-frame F^0 freeze (eager torch; included in the timed frame as setup)."""
        f0 = shape_matrix(gather_tri(x0, self.faces_long)) @ self.dm
        self.f0.copy_(f0.reshape(self.n_faces, 3, 2))

    def reset_state(self):
        self.u.zero_()
        self.v.zero_()
        self.f.zero_()

    def _launch_substep(self, dt):
        _face_force_kernel[self._grid_face](
            self.u.reshape(-1), self.f.reshape(-1), self.faces.reshape(-1),
            self.dm_flat, self.f0_flat, self.at,
            self.lame_mu, self.lame_lambda, self.n_faces, BLOCK=self.BLOCK_FACE)
        _update_kernel[self._grid_dof](
            self.u.reshape(-1), self.v.reshape(-1), self.f.reshape(-1),
            self.inv_m_dof, self.free_dof, dt, self.n_dof, BLOCK=self.BLOCK_DOF)

    def apply_K_once(self, z):
        """Single fused K_0 z for the oracle-parity witness (R08). Returns a fresh (V,3)."""
        self.f.zero_()
        u_save = self.u.clone()
        self.u.copy_(z)
        _face_force_kernel[self._grid_face](
            self.u.reshape(-1), self.f.reshape(-1), self.faces.reshape(-1),
            self.dm_flat, self.f0_flat, self.at,
            self.lame_mu, self.lame_lambda, self.n_faces, BLOCK=self.BLOCK_FACE)
        out = self.f.clone()
        self.u.copy_(u_save)
        self.f.zero_()
        return out

    def step_eager(self, dt, n_substeps):
        """Uncaptured launches — warmup/compile path and the graph-free parity loop."""
        for _ in range(n_substeps):
            self._launch_substep(dt)

    def capture(self, dt, n_substeps, warmup_iters=3):
        """Record the N-substep frame into one CUDA graph (once per topology + N)."""
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            self.step_eager(dt, warmup_iters)             # compile + allocator warmup
        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(self.device)
        self.reset_state()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step_eager(dt, n_substeps)
        self.graph_n = n_substeps
        return self

    def replay(self):
        self.graph.replay()
