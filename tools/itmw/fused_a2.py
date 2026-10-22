"""membrane wave A2C-lean fused SOURCE-AWARE conservative loop.

Extends the adjudicated A1R substrate (tools/itmw/fused_beta0.py — arithmetic untouched,
kernels reused by import) with the one thing A1R deliberately lacked: a Dirichlet source.
Per substep, inside ONE captured CUDA graph per (topology, N):

    face kernel      f = K_0 u                       (A1R `_face_force_kernel`, unchanged)
    update kernel    v <- (v - dt f/m) * free;  u <- u + dt v;  f <- 0   (A1R, unchanged;
                     free = 0 on A u Z u P, so the symplectic step never touches actuators)
    boundary kernel  u_A <- s_n c_T,  v_A <- (sdot_n / h) c_T            (NEW, this module)

The per-substep ramp scalars are baked per graph node: substep n's boundary launch holds a
pointer to element n of a precomputed schedule buffer, so the cubic index advances INSIDE the
captured frame with zero host work at replay. Z = (C \\ A) u P stays exactly 0
through the A1R free-mask mechanism. beta = 0; no damping term exists here.
"""
import math
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.itmw.fused_beta0 import (HAVE_TRITON, FusedBeta0Loop, _face_force_kernel,
                                    _update_kernel)
from tools.itmw.reference_membrane import MembraneOperator, substep_schedule

if HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _face_force_visco_kernel(u_ptr, f_ptr, hh_ptr, faces_ptr, dm_ptr, f0_ptr, at_ptr,
                                 lame_mu, lame_lambda, visco_g, decay, n_faces,
                                 BLOCK: tl.constexpr):
        """A1R face kernel + one SLS Maxwell branch:
        S_total = S(eta) + g * S(eta - hh);  hh <- eta + (hh - eta) * decay  (per face,
        stored back — no cross-face race). g = 0 with hh = 0 reproduces the elastic kernel
        arithmetic exactly; the separate A1R `_face_force_kernel` remains byte-untouched."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_faces
        i0 = tl.load(faces_ptr + offs * 3 + 0, mask=m, other=0)
        i1 = tl.load(faces_ptr + offs * 3 + 1, mask=m, other=0)
        i2 = tl.load(faces_ptr + offs * 3 + 2, mask=m, other=0)
        u0x = tl.load(u_ptr + i0 * 3 + 0, mask=m, other=0.0)
        u0y = tl.load(u_ptr + i0 * 3 + 1, mask=m, other=0.0)
        u0z = tl.load(u_ptr + i0 * 3 + 2, mask=m, other=0.0)
        u1x = tl.load(u_ptr + i1 * 3 + 0, mask=m, other=0.0)
        u1y = tl.load(u_ptr + i1 * 3 + 1, mask=m, other=0.0)
        u1z = tl.load(u_ptr + i1 * 3 + 2, mask=m, other=0.0)
        u2x = tl.load(u_ptr + i2 * 3 + 0, mask=m, other=0.0)
        u2y = tl.load(u_ptr + i2 * 3 + 1, mask=m, other=0.0)
        u2z = tl.load(u_ptr + i2 * 3 + 2, mask=m, other=0.0)
        d0x, d0y, d0z = u0x - u2x, u0y - u2y, u0z - u2z
        d1x, d1y, d1z = u1x - u2x, u1y - u2y, u1z - u2z
        dm00 = tl.load(dm_ptr + offs * 4 + 0, mask=m, other=0.0)
        dm01 = tl.load(dm_ptr + offs * 4 + 1, mask=m, other=0.0)
        dm10 = tl.load(dm_ptr + offs * 4 + 2, mask=m, other=0.0)
        dm11 = tl.load(dm_ptr + offs * 4 + 3, mask=m, other=0.0)
        dfax = d0x * dm00 + d1x * dm10
        dfay = d0y * dm00 + d1y * dm10
        dfaz = d0z * dm00 + d1z * dm10
        dfbx = d0x * dm01 + d1x * dm11
        dfby = d0y * dm01 + d1y * dm11
        dfbz = d0z * dm01 + d1z * dm11
        f0x0 = tl.load(f0_ptr + offs * 6 + 0, mask=m, other=0.0)
        f0x1 = tl.load(f0_ptr + offs * 6 + 1, mask=m, other=0.0)
        f0y0 = tl.load(f0_ptr + offs * 6 + 2, mask=m, other=0.0)
        f0y1 = tl.load(f0_ptr + offs * 6 + 3, mask=m, other=0.0)
        f0z0 = tl.load(f0_ptr + offs * 6 + 4, mask=m, other=0.0)
        f0z1 = tl.load(f0_ptr + offs * 6 + 5, mask=m, other=0.0)
        t00 = f0x0 * dfax + f0y0 * dfay + f0z0 * dfaz
        t01 = f0x0 * dfbx + f0y0 * dfby + f0z0 * dfbz
        t10 = f0x1 * dfax + f0y1 * dfay + f0z1 * dfaz
        t11 = f0x1 * dfbx + f0y1 * dfby + f0z1 * dfbz
        e01 = 0.5 * (t01 + t10)
        # SLS internal state (symmetric 2x2 stored as 3 floats per face)
        h00 = tl.load(hh_ptr + offs * 3 + 0, mask=m, other=0.0)
        h01 = tl.load(hh_ptr + offs * 3 + 1, mask=m, other=0.0)
        h11 = tl.load(hh_ptr + offs * 3 + 2, mask=m, other=0.0)
        # total stress: elastic on eta plus g x elastic-map on (eta - hh)
        g00 = t00 + visco_g * (t00 - h00)
        g01 = e01 + visco_g * (e01 - h01)
        g11 = t11 + visco_g * (t11 - h11)
        trc = g00 + g11
        s00 = lame_lambda * trc + 2.0 * lame_mu * g00
        s11 = lame_lambda * trc + 2.0 * lame_mu * g11
        s01 = 2.0 * lame_mu * g01
        q00 = s00 * dm00 + s01 * dm01
        q01 = s00 * dm10 + s01 * dm11
        q10 = s01 * dm00 + s11 * dm01
        q11 = s01 * dm10 + s11 * dm11
        at = tl.load(at_ptr + offs, mask=m, other=0.0)
        p0x = at * (f0x0 * q00 + f0x1 * q10)
        p0y = at * (f0y0 * q00 + f0y1 * q10)
        p0z = at * (f0z0 * q00 + f0z1 * q10)
        p1x = at * (f0x0 * q01 + f0x1 * q11)
        p1y = at * (f0y0 * q01 + f0y1 * q11)
        p1z = at * (f0z0 * q01 + f0z1 * q11)
        tl.atomic_add(f_ptr + i0 * 3 + 0, p0x, mask=m)
        tl.atomic_add(f_ptr + i0 * 3 + 1, p0y, mask=m)
        tl.atomic_add(f_ptr + i0 * 3 + 2, p0z, mask=m)
        tl.atomic_add(f_ptr + i1 * 3 + 0, p1x, mask=m)
        tl.atomic_add(f_ptr + i1 * 3 + 1, p1y, mask=m)
        tl.atomic_add(f_ptr + i1 * 3 + 2, p1z, mask=m)
        tl.atomic_add(f_ptr + i2 * 3 + 0, -(p0x + p1x), mask=m)
        tl.atomic_add(f_ptr + i2 * 3 + 1, -(p0y + p1y), mask=m)
        tl.atomic_add(f_ptr + i2 * 3 + 2, -(p0z + p1z), mask=m)
        # relax AFTER the force (beginning-of-substep hh feeds the stress above)
        tl.store(hh_ptr + offs * 3 + 0, t00 + (h00 - t00) * decay, mask=m)
        tl.store(hh_ptr + offs * 3 + 1, e01 + (h01 - e01) * decay, mask=m)
        tl.store(hh_ptr + offs * 3 + 2, t11 + (h11 - t11) * decay, mask=m)

    @triton.jit
    def _boundary_kernel(u_ptr, v_ptr, a_ptr, ct_ptr, s_ptr, sdh_ptr, n_a,
                         BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_a
        s = tl.load(s_ptr)                       # this substep's ramp value (baked pointer)
        sdh = tl.load(sdh_ptr)                   # this substep's sdot / h
        vi = tl.load(a_ptr + offs, mask=m, other=0)
        ctx = tl.load(ct_ptr + offs * 3 + 0, mask=m, other=0.0)
        cty = tl.load(ct_ptr + offs * 3 + 1, mask=m, other=0.0)
        ctz = tl.load(ct_ptr + offs * 3 + 2, mask=m, other=0.0)
        tl.store(u_ptr + vi * 3 + 0, s * ctx, mask=m)
        tl.store(u_ptr + vi * 3 + 1, s * cty, mask=m)
        tl.store(u_ptr + vi * 3 + 2, s * ctz, mask=m)
        tl.store(v_ptr + vi * 3 + 0, sdh * ctx, mask=m)
        tl.store(v_ptr + vi * 3 + 1, sdh * cty, mask=m)
        tl.store(v_ptr + vi * 3 + 2, sdh * ctz, mask=m)


def _cubic_schedules(n, dt, h, device):
    """Per-substep boundary values s(r_{n+1}) and sdot(r_{n+1})/h, r clipped at 1."""
    r = torch.clamp(torch.arange(1, n + 1, dtype=torch.float64, device=device) * dt / h,
                    max=1.0)
    s = 3.0 * r * r - 2.0 * r * r * r
    sdh = 6.0 * r * (1.0 - r) / h
    return s.to(torch.float32).contiguous(), sdh.to(torch.float32).contiguous()


class FusedSourceLoop(FusedBeta0Loop):
    """A1R buffers + graph mechanics, with the source boundary added per captured substep.

    `free` handed to the A1R update kernel is the TRUE free set (complement of A u Z u pins);
    A dofs are then overwritten by `_boundary_kernel`. Empty A degenerates exactly to the
    A1R zero-state loop (G25 / L01).
    """
    BLOCK_A = 128

    def __init__(self, op32, free_vert_mask, a_idx, z_idx, c_t32, dt, h, device,
                 visco_g=0.0, visco_tau_s=None):
        super().__init__(op32, free_vert_mask, device)
        self.dt = float(dt)
        self.h = float(h)
        self.n_a = int(a_idx.numel())
        self.a_idx32 = a_idx.to(device=device, dtype=torch.int32).contiguous()
        self.ct = c_t32.to(device=device, dtype=torch.float32).contiguous()
        self._grid_a = (max(1, math.ceil(self.n_a / self.BLOCK_A)),)
        self.s_sched = None
        self.sdh_sched = None
        self.schedule_len = 0
        self.captured = False
        self.visco_g = float(visco_g)
        self.visco_tau_s = (0.5 * float(h)) if visco_tau_s is None else float(visco_tau_s)
        self.hh = torch.zeros(self.n_faces, 3, dtype=torch.float32, device=device)

    def reset_state(self):
        super().reset_state()
        self.hh.zero_()

    # -- capture ------------------------------------------------------------------------------

    def _capture_body(self, n_substeps):
        """The exact per-frame work recorded into the graph: N x (force, update, boundary).
        No host-side scalar conversion, transfer, or synchronization may appear here; the
        boundary scalars come from per-substep slices of the device schedule buffers.
        visco_g = 0 uses the byte-untouched A1R elastic kernel; g > 0 the SLS variant."""
        decay = math.exp(-self.dt / self.visco_tau_s)
        for n in range(n_substeps):
            if self.visco_g == 0.0:
                _face_force_kernel[self._grid_face](
                    self.u.reshape(-1), self.f.reshape(-1), self.faces.reshape(-1),
                    self.dm_flat, self.f0_flat, self.at,
                    self.lame_mu, self.lame_lambda, self.n_faces, BLOCK=self.BLOCK_FACE)
            else:
                _face_force_visco_kernel[self._grid_face](
                    self.u.reshape(-1), self.f.reshape(-1), self.hh.reshape(-1),
                    self.faces.reshape(-1), self.dm_flat, self.f0_flat, self.at,
                    self.lame_mu, self.lame_lambda, self.visco_g, decay,
                    self.n_faces, BLOCK=self.BLOCK_FACE)
            _update_kernel[self._grid_dof](
                self.u.reshape(-1), self.v.reshape(-1), self.f.reshape(-1),
                self.inv_m_dof, self.free_dof, self.dt, self.n_dof, BLOCK=self.BLOCK_DOF)
            if self.n_a > 0:
                _boundary_kernel[self._grid_a](
                    self.u.reshape(-1), self.v.reshape(-1), self.a_idx32, self.ct.reshape(-1),
                    self.s_sched[n:n + 1], self.sdh_sched[n:n + 1], self.n_a,
                    BLOCK=self.BLOCK_A)

    def capture_source(self, n_substeps, warmup_iters=3):
        self.s_sched, self.sdh_sched = _cubic_schedules(n_substeps, self.dt, self.h,
                                                        self.device)
        self.schedule_len = n_substeps
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):                       # compile + allocator warmup
            self._capture_body(min(warmup_iters, n_substeps))
        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(self.device)
        self.reset_state()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._capture_body(n_substeps)
        self.graph_n = n_substeps
        self.captured = True
        return self

    def run_frame(self):
        """One complete frame: state reset + graph replay (set_base already holds F^0)."""
        self.reset_state()
        self.graph.replay()
        torch.cuda.synchronize(self.device)

    # -- construction / oracle hooks ----------------------------------------------------------

    @classmethod
    def from_reference(cls, op64, a_idx, z_idx, c_t, h, n_substeps, device='cuda:0',
                       visco_g=0.0, visco_tau_s=None):
        """Build the float32 twin of a float64 reference operator + masks, capture, return."""
        op32 = MembraneOperator(op64.faces, op64.dm_inv.to(torch.float32),
                                op64.f_area.to(torch.float32), op64.v_mass.to(torch.float32),
                                op64.lame_mu, op64.lame_lambda, op64.thickness)
        free = torch.ones(op64.n_verts, dtype=torch.bool)
        free[a_idx.reshape(-1).to(torch.long)] = False
        free[z_idx.reshape(-1).to(torch.long)] = False
        self = cls(op32, free, a_idx.reshape(-1), z_idx.reshape(-1),
                   c_t.to(torch.float32), h / n_substeps, h, device,
                   visco_g=visco_g, visco_tau_s=visco_tau_s)
        self._op64 = op64
        self._free_cpu = free
        if op64.f0 is None:
            raise ValueError('reference operator has no frozen base — call set_base first')
        # take F^0 from the reference's own frozen base (bitwise same convention)
        self.f0.copy_(op64.f0.to(device=device, dtype=torch.float32))
        self.capture_source(n_substeps)
        return self

    def energy_cpu(self):
        """E = 0.5 v_F M v_F + V_w(u), computed float64 on CPU from downloaded state."""
        u = self.u.detach().to('cpu', torch.float64)
        v = self.v.detach().to('cpu', torch.float64)
        vf = v[self._free_cpu]
        ke = 0.5 * float((self._op64.v_mass[self._free_cpu].unsqueeze(-1) * vf * vf).sum())
        return ke + float(self._op64.energy(u))


# ── the single worst-state source-aware timing cell ────────────

def time_worst_state(states_dir='ccraft_data/eval/itmw_a1_states', out_json=None,
                     repeats=100, device='cuda:0'):
    """A single worst-state timing cell for the fused kernels.

    NOTE: this benchmark reads a cached state tree and two preflight helpers that are not part of
    this distribution, so calling it raises ImportError. It is kept for reference; nothing on the
    documented rollout path imports it.

    dress/01_01: actual topology, actual C u P, factor-proxy A,
    deterministic synthetic tangent-dominant c_T (timing only — never rollout physics).
    Frame = state reset + F^0 set_base + mask/c_T refresh + graph replay. BLOCKED on a
    contended device; PASS/FAIL is carried by p95 vs the A1R gates."""
    import json

    from tools.itmw.a1_preflight import load_state_cells, other_gpu_processes
    from tools.itmw.a1r_conservative import free_vertex_mask

    procs = other_gpu_processes(device)
    if procs:
        print(f'[fused_a2] BLOCKED_TIMING: contended device ({procs})', flush=True)
        return dict(verdict='BLOCKED_TIMING', other_processes=procs)

    cells = load_state_cells(Path(states_dir).resolve()
                             if Path(states_dir).is_absolute()
                             else (_REPO_ROOT / states_dir), 40)
    cell = next(c for c in cells if c['garment'] == 'dress' and c['sequence'] == '01_01')
    mat = dict(lame_mu_pa=23600.0, lame_lambda_pa=44400.0, density_kg_m2=0.20022)
    from tools.itmw.a1_preflight import build_operator
    op64 = build_operator(cell, mat, 4.7e-4)
    h = float(cell['timestep_s'])
    ub = op64.gershgorin_upper()
    _, _, n_cert = substep_schedule(ub, 0.0, h, safety=0.8)

    free = free_vertex_mask(cell)
    frozen_idx = (~free).nonzero().reshape(-1)
    cf = cell['contact_factor'].reshape(-1)
    active_local = (cf < 1.0 - 1e-7).nonzero().reshape(-1)
    a_idx = cell['contact_vertex'].reshape(-1).to(torch.long)[active_local]
    pins = (cell['vertex_type'].reshape(-1) == 3).nonzero().reshape(-1)
    a_idx = a_idx[~torch.isin(a_idx, pins)]
    z_mask = torch.zeros(cell['n_vertices'], dtype=torch.bool)
    z_mask[frozen_idx] = True
    z_mask[a_idx] = False
    z_idx = z_mask.nonzero().reshape(-1)

    # deterministic tangent-dominant synthetic source: a fixed direction minus its component
    # along the (area-weighted) vertex normal of the frozen base
    x0 = cell['x0']
    faces = cell['faces']
    fn = torch.cross(x0[faces[:, 1]] - x0[faces[:, 0]],
                     x0[faces[:, 2]] - x0[faces[:, 0]], dim=-1)
    vn = torch.zeros_like(x0)
    for k in range(3):
        vn.index_add_(0, faces[:, k], fn)
    vn = vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    d = torch.tensor([1.0, 0.0, 0.3], dtype=x0.dtype) / math.sqrt(1.09)
    tang = d - vn[a_idx] * (vn[a_idx] * d).sum(-1, keepdim=True)
    c_t = 1e-4 * tang / tang.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    loop = FusedSourceLoop.from_reference(op64, a_idx, z_idx, c_t, h, n_cert, device=device)
    x0_dev = x0.to(device=device, dtype=torch.float32)
    ct_dev = c_t.to(device=device, dtype=torch.float32)

    def one_frame():
        loop.reset_state()
        loop.set_base(x0_dev)                    # per-frame F^0 (eager, counted)
        loop.ct.copy_(ct_dev)                    # per-frame source refresh (counted)
        loop.graph.replay()

    passes = []
    for _ in range(2):
        for _ in range(5):
            one_frame()
        torch.cuda.synchronize(device)
        times = []
        for _ in range(repeats):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            one_frame()
            t1.record()
            torch.cuda.synchronize(device)
            times.append(t0.elapsed_time(t1))
        times.sort()
        passes.append(dict(p50=times[len(times) // 2], p95=times[int(0.95 * len(times))]))
    procs_after = other_gpu_processes(device)
    if procs_after:
        return dict(verdict='BLOCKED_TIMING', other_processes=procs_after, passes=passes)

    worst = max(p['p95'] for p in passes)
    res = dict(verdict=('PASS' if worst <= 25.816 else
                        'AMBER' if worst <= 28.684 else 'FAIL'),
               cell='dress/01_01', n_substeps=n_cert, n_active=int(a_idx.numel()),
               n_frozen=int(frozen_idx.numel()), n_verts=cell['n_vertices'],
               passes=passes, worst_p95_ms=worst,
               green_ms=25.816, hard_ms=28.684, repeats=repeats,
               free_response_nonzero=bool(float(loop.u.abs().max()) > 0))
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1))
    print(f'[fused_a2] worst source-aware frame p95 = {worst:.3f} ms '
          f'(N={n_cert}, A={int(a_idx.numel())}) -> {res["verdict"]}', flush=True)
    return res


if __name__ == '__main__':
    time_worst_state(out_json=str(_REPO_ROOT / 'itmw_a2_timing.json'))
