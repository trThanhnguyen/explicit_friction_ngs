"""membrane wave production wave stage — incremental tangent-membrane wave, beta = 0.

The third mutually-exclusive `op_patch` coupling (beside diagnostic spread and diagnostic patch relax):
the contact stage's TANGENTIAL correction c_T = cloth_next_tang - cloth_next is transported
through non-contact cloth as a conservative mass-lumped membrane wave over exactly one frame
interval h (Eqs. (2)-(17); carrier certified by A1R + the A2C-lean kill screen).

Zero physical hyperparameters at inference:
  * reach     = material speed x frame time (no depth d, no per-hop gamma, no iteration K);
  * substeps  = certified per frame from the Gershgorin bound of the frozen-base operator
                (0.8 safety, the A0 declared numerical margin), rounded UP a fixed bucket
                ladder so one CUDA graph per (garment, bucket) is replayed;
  * source    = the already-computed Coulomb tangential landing, injected as a Dirichlet
                cubic ramp s(r) = 3r^2 - 2r^3 over the frame (fixed, parameter-free);
  * absorption: NONE (beta = 0). Resolved contacts and pins are zero-Dirichlet boundaries —
                the friction sink in its modular limit; material sets the speed and hence the
                per-frame reach. The absorption decision tree is, owner-gated.

Semantics (Eq. (16)): contact vertices keep the complete Coulomb + normal-completion
result x^c; pins keep their incumbent value; ONLY free vertices receive x^0 + u(h). mu = 0 or
an all-dust correction field means A = empty and the stage returns its input untouched — the
structural-twin guarantee shared with C0/C1.

Engineering: reuses the A1R-adjudicated Triton kernels (tools/itmw/fused_beta0) plus the A2C
boundary kernel (tools/itmw/fused_a2) under one CUDA graph per (garment, bucket). Buffers are
capacity-V and graph-static; a ghost vertex row (index V) absorbs source-slot padding so the
same graph serves any per-frame contact/source set. Thickness has ONE owner: the stretching
criterion's default (criterions/postcvpr/mataug/stretching_energy.Config.thickness).
"""
import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:                      # tools.* is a repo-root namespace pkg
    sys.path.insert(0, str(_REPO_ROOT))

# fixed engineering constants — numerics, not physics knobs
BUCKETS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
SAFETY = 0.8                                             # A0 declared margin, never a reach law
_CACHE = {}                                              # (garment key) -> _GarmentWave


def _thickness():
    """Single-owner cloth thickness: the stretching criterion's canonical default."""
    from criterions.postcvpr.mataug.stretching_energy import Config as _SC
    return float(_SC().thickness)


def _garment_key(cloth):
    """Cheap stable key: sizes + a rest-geometry checksum (one small host sync per frame)."""
    rest = cloth.rest_pos
    faces = cloth.faces_batch
    return (int(rest.shape[0]), int(faces.shape[1] if faces.shape[0] == 3 else faces.shape[0]),
            round(float(rest.sum()), 6))


class _GarmentWave:
    """Per-garment persistent state: geometry caches, capacity buffers, graph-per-bucket."""

    def __init__(self, cloth, m, device, visco_g=0.0, visco_tau_s=None):
        from tools.itmw.fused_a2 import HAVE_TRITON
        from tools.itmw.reference_membrane import MembraneOperator, face_areas, make_dm_inv
        if not HAVE_TRITON:
            raise RuntimeError('membrane-wave stage requires triton (fused kernels)')
        faces = cloth.faces_batch.T.to(torch.long)                       # (F,3)
        rest = cloth.rest_pos.to(torch.float32)
        if not ('lame_mu' in cloth and 'lame_lambda' in cloth):
            raise ValueError('membrane-wave stage requires lame_mu/lame_lambda on the sample')
        lam = float(cloth.lame_lambda.reshape(-1)[0])
        mu_s = float(cloth.lame_mu.reshape(-1)[0])
        t = _thickness()
        self.device = device
        self.n_verts = int(rest.shape[0])
        self.n_faces = int(faces.shape[0])
        self.faces = faces.to(device)
        # float32 GPU operator: set_base + Gershgorin certification + energy diagnostics
        self.op = MembraneOperator(self.faces, make_dm_inv(rest, faces.cpu()).to(device),
                                   face_areas(rest, faces.cpu()).to(device),
                                   m.reshape(-1).to(device=device, dtype=torch.float32),
                                   mu_s, lam, t)
        # Gershgorin structure precomputed once (dF basis is F^0-independent)
        dm = self.op.dm_inv
        r0, r1 = dm[:, 0, :], dm[:, 1, :]
        w = torch.stack([r0, r1, -(r0 + r1)], dim=1)                     # (F,3,2)
        df = torch.zeros(self.n_faces, 9, 3, 2, dtype=torch.float32, device=device)
        for j in range(3):
            for a in range(3):
                df[:, j * 3 + a, a, :] = w[:, j, :]
        self._df_basis = df
        mf = self.op.v_mass[self.faces]                                  # (F,3)
        self._s9 = (1.0 / torch.sqrt(mf)).repeat_interleave(3, dim=1)    # (F,9)
        # capacity buffers (+1 ghost row absorbing padded source slots)
        v1 = self.n_verts + 1
        self.u = torch.zeros(v1, 3, dtype=torch.float32, device=device)
        self.v = torch.zeros_like(self.u)
        self.f = torch.zeros_like(self.u)
        self.f0 = torch.zeros(self.n_faces, 3, 2, dtype=torch.float32, device=device)
        inv_m = torch.zeros(v1, dtype=torch.float32, device=device)
        inv_m[:self.n_verts] = 1.0 / self.op.v_mass
        self.inv_m_dof = inv_m.reshape(-1, 1).expand(-1, 3).contiguous().reshape(-1)
        self.free_dof = torch.zeros(v1 * 3, dtype=torch.float32, device=device)
        self.a_idx = torch.full((self.n_verts,), self.n_verts, dtype=torch.int32,
                                device=device)                           # padded with ghost
        self.ct = torch.zeros(self.n_verts, 3, dtype=torch.float32, device=device)
        self._faces32 = self.faces.to(torch.int32).contiguous().reshape(-1)
        # strain cap: undirected mesh edges for the per-edge strain budget on the output
        ed_c = torch.cat([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]], 0)
        self.edges = torch.unique(torch.sort(ed_c, dim=1).values, dim=0)   # (E,2) long
        # SLS Maxwell branch: per-face internal strain state.
        # g = 0 selects the byte-untouched A1R elastic kernel and this buffer stays unused.
        self.visco_g = float(visco_g)
        self.visco_tau_s = visco_tau_s
        self.hh = torch.zeros(self.n_faces, 3, dtype=torch.float32, device=device)
        self.h = None
        self.graphs = {}                                                 # bucket -> CUDAGraph
        self.scheds = {}                                                 # bucket -> (s, sdh)

    # -- per-frame pieces ---------------------------------------------------------------------

    def set_base(self, x0):
        from tools.itmw.reference_membrane import gather_tri, shape_matrix
        self.f0.copy_(shape_matrix(gather_tri(x0, self.faces)) @ self.op.dm_inv)

    def gershgorin(self):
        """Exact per-frame Gershgorin bound on lambda_max(M^-1/2 K0 M^-1/2), from the frozen
        base — same math as MembraneOperator.gershgorin_upper, loop-free per frame."""
        f0 = self.f0
        ftdf = torch.einsum('fip,fkiq->fkpq', f0, self._df_basis)        # (F,9,2,2)
        e = 0.5 * (ftdf + ftdf.transpose(2, 3))
        tr = (e[:, :, 0, 0] + e[:, :, 1, 1])[:, :, None, None]
        eye = torch.eye(2, dtype=torch.float32, device=self.device)
        s = self.op.lame_lambda * tr * eye + 2.0 * self.op.lame_mu * e
        p = torch.einsum('fip,fkpq,frq->fkir', f0, s, self.op.dm_inv)
        p = self.op._at[:, None, None, None] * p                         # (F,9,3,2)
        force = torch.stack([p[..., 0], p[..., 1], -(p[..., 0] + p[..., 1])], dim=2)
        ke = force.reshape(self.n_faces, 9, 9).transpose(1, 2)
        ae = ke * self._s9[:, :, None] * self._s9[:, None, :]
        rs = ae.abs().sum(dim=2)                                         # (F,9)
        acc = torch.zeros(self.n_verts, 3, dtype=torch.float32, device=self.device)
        for i in range(3):
            acc.index_add_(0, self.faces[:, i], rs[:, i * 3:(i + 1) * 3])
        return float(acc.max())

    def _capture(self, bucket):
        from tools.itmw.fused_a2 import (_boundary_kernel, _cubic_schedules,
                                         _face_force_visco_kernel)
        from tools.itmw.fused_beta0 import _face_force_kernel, _update_kernel
        dt = self.h / bucket
        tau_s = (0.5 * self.h) if self.visco_tau_s is None else float(self.visco_tau_s)
        decay = math.exp(-dt / tau_s)
        s_sched, sdh_sched = _cubic_schedules(bucket, dt, self.h, self.device)
        self.scheds[bucket] = (s_sched, sdh_sched)
        n_dof = (self.n_verts + 1) * 3
        grid_face = (math.ceil(self.n_faces / 128),)
        grid_dof = (math.ceil(n_dof / 256),)
        grid_a = (math.ceil(self.n_verts / 128),)

        def body(n_sub):
            for n in range(n_sub):
                if self.visco_g == 0.0:
                    _face_force_kernel[grid_face](
                        self.u.reshape(-1), self.f.reshape(-1), self._faces32,
                        self.op.dm_inv.reshape(-1), self.f0.reshape(-1), self.op._at,
                        self.op.lame_mu, self.op.lame_lambda, self.n_faces, BLOCK=128)
                else:
                    _face_force_visco_kernel[grid_face](
                        self.u.reshape(-1), self.f.reshape(-1), self.hh.reshape(-1),
                        self._faces32,
                        self.op.dm_inv.reshape(-1), self.f0.reshape(-1), self.op._at,
                        self.op.lame_mu, self.op.lame_lambda, self.visco_g, decay,
                        self.n_faces, BLOCK=128)
                _update_kernel[grid_dof](
                    self.u.reshape(-1), self.v.reshape(-1), self.f.reshape(-1),
                    self.inv_m_dof, self.free_dof, dt, n_dof, BLOCK=256)
                _boundary_kernel[grid_a](
                    self.u.reshape(-1), self.v.reshape(-1), self.a_idx, self.ct.reshape(-1),
                    s_sched[n:n + 1], sdh_sched[n:n + 1], self.n_verts, BLOCK=128)

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            body(3)                                       # compile + allocator warmup
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.u.zero_(), self.v.zero_(), self.f.zero_(), self.hh.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body(bucket)
        self.graphs[bucket] = g
        return g


def _get_state(cloth, m, device, visco_g, visco_tau_s):
    key = _garment_key(cloth) + (round(float(visco_g), 9),)
    st = _CACHE.get(key)
    if st is None:
        st = _GarmentWave(cloth, m, device, visco_g=visco_g, visco_tau_s=visco_tau_s)
        _CACHE[key] = st
    return st


@torch.no_grad()
def _cap_added_strain(out, base, free_v, edges, cap, stats, m=None, sweeps=24):
    """strain cap: clip the added field's per-edge strain |dw|/l0 to `cap` in place.

    Degree-averaged, mobility-masked Jacobi sweeps — plain Jacobi DIVERGES here (shared-edge
    over-correction, housekeeping). Frozen rows (mobility 0) never move: C u P keep
    their complete Coulomb result (Eq. (16) inviolate). Fixed sweep count; the residual is
    REPORTED (wv_cap_resid), never iterated to tolerance — a diag quantity, not a solve.

    Endpoint split: with `m` (V,1) lumped masses, each flagged
    edge's correction splits by INVERSE MASS — the light (short-edge) vertex, the one actually
    flung, takes the larger share. m=None => uniform 50/50 split (exactly equal to inverse-mass
    on a uniform mesh). Frozen rows carry weight 0 either way.
    """
    e0, e1 = edges[:, 0], edges[:, 1]
    l0 = (base[e0] - base[e1]).norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cap_l0 = cap * l0
    w = out - base
    w0 = w.clone()
    mob = free_v.to(w.dtype).unsqueeze(-1)
    if m is not None:
        mob = mob / m.reshape(-1, 1).to(w.dtype).clamp_min(1e-12)
    n_over0 = 0
    for it in range(sweeps):
        du = w[e0] - w[e1]
        dn = du.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        over = (dn - cap_l0).clamp(min=0.0)
        if it == 0:
            n_over0 = int((over > 0).sum())
            if n_over0 == 0:
                break
        exc = du * (over / dn)
        wi, wj = mob[e0], mob[e1]
        wsum = (wi + wj).clamp_min(1e-12)
        upd = torch.zeros_like(w)
        upd.index_add_(0, e0, -exc * (wi / wsum))
        upd.index_add_(0, e1, exc * (wj / wsum))
        onebad = (over > 0).to(w.dtype)
        cnt = torch.zeros(w.shape[0], 1, dtype=w.dtype, device=w.device)
        cnt.index_add_(0, e0, onebad)
        cnt.index_add_(0, e1, onebad)
        w = w + upd / cnt.clamp_min(1.0)
    stats['wv_cap_edges0'] = stats.get('wv_cap_edges0', 0) + n_over0
    if not n_over0:
        return
    du = w[e0] - w[e1]
    resid = int(((du.norm(dim=-1, keepdim=True) - cap_l0) > 1e-9).sum())
    out[free_v] = base[free_v] + w[free_v]
    stats['wv_cap_resid'] = stats.get('wv_cap_resid', 0) + resid
    stats['wv_cap_dxmax'] = max(stats.get('wv_cap_dxmax', 0.0),
                                float((w - w0)[free_v].norm(dim=-1).max()))


@torch.no_grad()
def wave_transport(cloth, cloth_next, cloth_next_tang, cloth_next_proj,
                   in_contact, pinned, m, h, plane_gap_fn, dust, stats,
                   visco_g=0.0, visco_tau_frames=0.5, strain_cap=0.0):
    """The op_patch wave branch. Returns the composed positions (Eq. (16)); mutates `stats`.

    :param cloth: ex['cloth'] (needs faces_batch, rest_pos, lame_mu, lame_lambda)
    :param cloth_next: (V,3) frozen raw proposal x^0
    :param cloth_next_tang: (V,3) tangential landing x^T (chord or surface-marched)
    :param cloth_next_proj: (V,3) complete contact result x^c (returned untouched on C u P)
    :param plane_gap_fn: callable(x_sub) -> (gap, normal, ambiguous) for the non-worsening
        Signorini guard — passed in from contact.py to avoid an import cycle
    :param dust: source dust threshold in m (the shared normal_exact_dust)
    """
    device = cloth_next.device
    c_t_full = cloth_next_tang - cloth_next                              # (V,3)
    src_m = in_contact & (c_t_full.norm(dim=-1) > dust) & (~pinned)      # A
    n_a = int(src_m.sum())
    stats['wv_frames'] = stats.get('wv_frames', 0) + 1
    if n_a == 0:
        stats['wv_empty'] = stats.get('wv_empty', 0) + 1                 # structural no-op
        return cloth_next_proj

    st = _get_state(cloth, m, device, float(visco_g), float(visco_tau_frames) * float(h))
    st.h = float(h) if st.h is None else st.h
    x0_32 = cloth_next.to(torch.float32)
    st.set_base(x0_32)

    # certified bucket for this frame
    lam_g = st.gershgorin() * (1.0 + st.visco_g)     # SLS instantaneous modulus = (1+g) K0
    if lam_g <= 0:
        stats['wv_degenerate'] = stats.get('wv_degenerate', 0) + 1
        return cloth_next_proj
    dt_max = 2.0 / math.sqrt(lam_g)
    n_req = int(math.ceil(float(h) / (SAFETY * dt_max)))
    bucket = next((b for b in BUCKETS if b >= n_req), None)
    if bucket is None:                                                   # tripwire, never a law
        stats['wv_ladder_overflow'] = stats.get('wv_ladder_overflow', 0) + 1
        return cloth_next_proj

    # refresh graph-static buffers: masks, source list (+ ghost padding), c_T
    frozen = in_contact | pinned
    free_v = ~frozen
    fd = st.free_dof.reshape(-1, 3)
    fd.zero_()
    fd[:st.n_verts][free_v] = 1.0
    a_list = src_m.nonzero().reshape(-1).to(torch.int32)
    st.a_idx.fill_(st.n_verts)                                           # ghost
    st.a_idx[:n_a] = a_list
    st.ct.zero_()
    st.ct[:n_a] = c_t_full[a_list.to(torch.long)].to(torch.float32)

    if bucket not in st.graphs:
        st._capture(bucket)
    st.u.zero_(), st.v.zero_(), st.f.zero_(), st.hh.zero_()
    st.graphs[bucket].replay()

    u_free = st.u[:st.n_verts]
    out = cloth_next_proj.clone()
    out[free_v] = cloth_next[free_v] + u_free[free_v].to(cloth_next.dtype)

    # strain cap: per-edge strain budget on the composed added field. The wave's one
    # measured failure channel is |du|/l0 on source-adjacent short edges; the cap
    # clips that tail (~1% of edges on the screen state) and leaves the mid/far payload
    # untouched. cap == 0 => branch skipped, byte-identical uncapped stage
    # (including the guard's change-set below, which then reads the raw u as before).
    if strain_cap > 0:
        _cap_added_strain(out, cloth_next, free_v, st.edges, float(strain_cap), stats, m=m)
        ch_norm = (out - cloth_next).norm(dim=-1)
    else:
        ch_norm = u_free.norm(dim=-1)

    # non-worsening Signorini guard, once, on changed free vertices only (counted, C1 semantics)
    ch_idx = (free_v & (ch_norm > dust)).nonzero().reshape(-1)
    if ch_idx.numel():
        g_pre, _, _, amb_a = plane_gap_fn(cloth_next_proj[ch_idx])
        g_post, n_post, _, amb_b = plane_gap_fn(out[ch_idx])
        ok = (~amb_a) & (~amb_b)
        push = torch.clamp(g_pre - g_post, min=0.0) * ok.unsqueeze(-1)
        out[ch_idx] = out[ch_idx] + push * n_post
        stats['wv_sig_n'] = stats.get('wv_sig_n', 0) + int((push.reshape(-1) > 0).sum())
        stats['wv_sig_sum'] = stats.get('wv_sig_sum', 0.0) + float(push.sum())
        stats['wv_sig_max'] = max(stats.get('wv_sig_max', 0.0),
                                  float(push.max()) if push.numel() else 0.0)

    # bookkeeping
    stats['wv_active'] = stats.get('wv_active', 0) + n_a
    stats['wv_n_sub'] = max(stats.get('wv_n_sub', 0), bucket)
    stats['wv_reached'] = stats.get('wv_reached', 0) + int(ch_idx.numel())
    stats['wv_umax'] = max(stats.get('wv_umax', 0.0), float(u_free.norm(dim=-1).max()))
    return out
