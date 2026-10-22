"""trained-friction arm Phase 1 — Coulomb friction as a dissipation potential in the TRAINING loss.

Implements the single new term:

    D = [h^2] * weight * sum_{i in C} mu_i * sg[lambda_n^i] * f0(||delta_f^i||)

  - C           : active contact set — cloth verts with signed gap d < eps to the nearest body face
                  (same nearest-face-by-centroid correspondence as ccraft.collision_penalty and the
                  bolt-on operator projection in utils/contact.py), pinned verts excluded.
  - lambda_n^i: force-balance normal load (grav + inertia ONLY in Phase 1), closed form:
                      lambda_n = relu( m*g*n_up + (m/h^2) * n_hat . (x_pred - x_hat) ),
                  with x_hat = pos + velocity (the SAME inertial predictor as postcvpr.inertia — the
                  stored `velocity` is the per-step displacement) and n_up = n_hat[z_axis] (Y-up here,
                  matching postcvpr.gravitational_energy z_axis: 1). Signorini-clamped, evaluated on the
                  CURRENT prediction (: lagged, but not to the accepted state), and DETACHED —
                  no gradient ever flows through lambda_n in Phase 1.
  - delta_f^i   : tangential part of the relative slip displacement
                      delta = (x_pred - x_t) - h*v_body,   h*v_body = nearest-face centroid motion
                  Gradients DO flow through delta_f (through x_pred) — this is the learning channel.
  - f0(s): C1 mollified slip magnitude, gated by `mollifier`:
                    'sqrt' (default) : sqrt(s^2 + (eps_v*h)^2) — pseudo-Huber; approaches the Coulomb
                                       cone only ASYMPTOTICALLY (f0' < 1 at every finite slip). Kept
                                       as the default so every pre- config is bit-identical.
                    'ipc'            : r + relu(a-r)^3/(3a^2), a = eps_v*h — EXACTLY IPC's C1
                                       mollifier (Li 2020 Eq. 13 / supplement C1 f0), branch-free;
                                       f0' = 1 exactly for every slip >= a (full cone at finite slip).
  - mu_i        : per-sample friction coefficient, read from cloth.friction_mu (sampled by the runner's
                  arm_br block; the SAME value feeds the model's per-body-edge mu feature).

UNITS / the h^2 factor (box vs force bound): the backbone loss is the incremental
potential in Joules (inertia divides by h^2; stretch/bend/gravity are raw energies), so the gradient of
a bare D = mu*lambda_n*f0 is already a force of magnitude <= mu*lambda_n Newtons — exactly 's
cone bound. An explicit h^2 (~1.1e-3) would starve friction by 3 orders of magnitude relative to the
driving forces (the F5 failure). Following the HOOD precedent (criterions/postcvpr/
friction_dissipative.py: "h2 is ABSORBED into `weight` by default"), `use_h2` defaults to False and is
exposed only for completeness.

Phase 1 scope guards:
  - use_elastic_lambda is scaffolded OFF; enabling it raises NotImplementedError (Phase 2).
  - detach_lambda_n must stay True (a valid dissipation potential requires the lagged cone).
  - No inference-time projection anywhere: this criterion is loss-only; the BR test step is a plain
    forward pass (runner contact_arm.mode='none').

Gates (loss forced to zero, metrics still computed and logged):
  - iter < friction_warmup_steps (global training iteration, read from cloth.iter);
  - is_init macro-steps when skip_init_steps (the LBS-init transient runs at 10x the timestep;
    the initial garment state's relaxation must not feed the friction cone).
"""
from dataclasses import dataclass

import torch
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import NodeType, gather


# ----------------------------------------------------------------------------------------------- #
# Pure math core (no pyg dependency) — unit-tested in an internal tool #
# ----------------------------------------------------------------------------------------------- #

def grav_inertia_lambda_n(n_hat, m, x_pred, x_hat, h, g, up_axis):
    """ force-balance normal load, closed form, in Newtons.

        lambda_n = relu( m*g*n_up + (m/h^2) * n_hat . (x_pred - x_hat) )

    :param n_hat:  (V,3) unit outward BODY normal at the contact
    :param m:      (V,1) nodal mass (kg)
    :param x_pred: (V,3) predicted next position (caller detaches; no grad through lambda_n)
    :param x_hat:  (V,3) inertial predictor pos + velocity (velocity = per-step displacement)
    :param h:      timestep (s);  g: gravity magnitude;  up_axis: index of the up axis (1 = Y-up)
    :return: (V,1) lambda_n >= 0 (Signorini: compression only, no adhesion)
    """
    n_up = n_hat[:, up_axis:up_axis + 1]                                   # (V,1) SIGNED n_hat . e_up
    lam_grav = m * g * n_up                                                # (V,1) mg*n_up, signed
    lam_inertia = (m / (h * h)) * ((x_pred - x_hat) * n_hat).sum(dim=-1, keepdim=True)
    return torch.clamp(lam_grav + lam_inertia, min=0.0)


def tangential_slip(delta, n_hat):
    """Project the relative slip displacement onto the contact tangent plane.
    delta_f = delta - (delta . n_hat) n_hat. Gradients flow through `delta`."""
    return delta - (delta * n_hat).sum(dim=-1, keepdim=True) * n_hat


def mollified_slip_norm(delta_f, eps_v, h):
    """'sqrt' mollifier (legacy default): f0(||delta_f||) = sqrt(||delta_f||^2 + (eps_v*h)^2).
    Pseudo-Huber, NOT IPC's piecewise form: f0' = s/sqrt(s^2+a^2) < 1 everywhere, so the friction
    force reaches the cone mu*lambda_n only asymptotically (71% at s=a, 95% at s=3a).
    eps_v is a VELOCITY band (m/s); eps_v*h converts it to the displacement band. Epsilon-safe:
    the argument of sqrt is >= (eps_v*h)^2 > 0, so the gradient is smooth including delta_f = 0."""
    return torch.sqrt((delta_f * delta_f).sum(dim=-1, keepdim=True) + (eps_v * h) ** 2)


def ipc_mollified_slip_norm(delta_f, eps_v, h):
    """'ipc' mollifier: IPC's C1 f0 (Li 2020 Eq. 13 / supplement C1 form), branch-free:

        f0(r) = r + relu(a - r)^3 / (3 a^2),   r = ||delta_f||,  a = eps_v * h

    Identical to the piecewise  -r^3/(3a^2) + r^2/a + a/3  on [0,a)  /  r  on [a,inf): the friction
    force factor f0' = f1 hits 1 EXACTLY at r = a (full Coulomb cone at finite slip; the sqrt form
    never gets there). Gradient at delta_f = 0 is exactly 0: vector_norm's subgradient at 0 is 0
    (verified) and f1(0) = 0 kills the direction singularity."""
    a = eps_v * h
    r = torch.linalg.vector_norm(delta_f, dim=-1, keepdim=True)
    return r + torch.relu(a - r) ** 3 / (3.0 * a * a)


def ipc_f1(slip, eps_v, h):
    """f0' of the 'ipc' mollifier: f1(r) = 1 - (relu(a-r)/a)^2 = 2r/a - r^2/a^2 on [0,a), 1 beyond.
    In [0,1], with equality 1 attained for every r >= a (Coulomb saturation). Diagnostic-only here:
    the loss differentiates f0 by autograd; this closed form feeds f1_mean/implied_force_mean."""
    a = eps_v * h
    return 1.0 - (torch.relu(a - slip) / a) ** 2


# ----------------------------------------------------------------------------------------------- #
# Criterion                                                                                        #
# ----------------------------------------------------------------------------------------------- #

@dataclass
class Config:
    weight: float = 1.0               # loss weight on D (absorbs IPC's h^2 by default; see use_h2)
    eps: float = 2e-3                 # active-set gate d < eps (match ccraft.collision_penalty.eps)
    eps_v: float = 1e-2               # static-velocity band (m/s); displacement band = eps_v * h
    mollifier: str = 'sqrt'           # 'sqrt' (legacy pseudo-Huber, default: pre- configs are
                                      # bit-identical) | 'ipc'
    z_axis: int = 1                   # up-axis index (match postcvpr.gravitational_energy z_axis)
    g: float = 9.81                   # gravity magnitude for the force-balance gravity term
    use_h2: bool = False              # multiply D by h^2 (boxed form). Default False:
                                      # absorbed into `weight` (HOOD precedent; see module doc)
    detach_lambda_n: bool = True      # Phase 1 invariant: must stay True (lagged IPC cone)
    use_elastic_lambda: bool = False  # Phase 2 scaffold — True raises NotImplementedError
    friction_warmup_steps: int = 0    # global iters with the loss forced to 0 (metrics still logged)
    friction_ramp_steps: int = 0      # (BR attempt-2 (d)): linear weight ramp 0->1 over this
                                      #   many global iters AFTER friction_warmup_steps; 0 = hard gate
                                      #   at warmup (legacy behavior, bit-identical for old configs)
    skip_init_steps: bool = True      # zero the loss on is_init macro-steps (LBS transient, 10x ts)


def create(mcfg):
    return Criterion(mcfg)


_METRIC_KEYS = ('D_raw', 'D_wtd', 'n_active', 'active_frac', 'mu_mean', 'lam_mean', 'lam_max',
                'slip_mean', 'static_band_frac', 'f1_mean', 'implied_force_mean', 'gate_on')


class Criterion(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.name = 'friction_dissipative'
        self.f_normals_f = FaceNormals()
        # Phase 1 invariants — fail at construction, not mid-run (phasing).
        if getattr(mcfg, 'use_elastic_lambda', False):
            raise NotImplementedError(
                "friction_dissipative: use_elastic_lambda=True is the Phase 2 elastic/tightness "
                "channel and is not implemented yet. Phase 1 uses grav+inertia lambda_n "
                "only — set use_elastic_lambda: false.")
        if not getattr(mcfg, 'detach_lambda_n', True):
            raise ValueError(
                "friction_dissipative: detach_lambda_n=False is invalid in Phase 1 — the lagged "
                "(detached) lambda_n is what makes D a genuine dissipation potential.")
        # resolved once so a config missing the key falls back to the legacy form
        self.mollifier = getattr(mcfg, 'mollifier', 'sqrt') or 'sqrt'
        if self.mollifier not in ('sqrt', 'ipc'):
            raise ValueError(
                f"friction_dissipative: mollifier={self.mollifier!r} is not one of ('sqrt', 'ipc'). "
                "'sqrt' = legacy pseudo-Huber f0; 'ipc' = exact IPC C1 f0 "
                ".")

    # ------------------------------------------------------------------------------------------ #

    def _resolve_mu(self, sample):
        """(B,) per-sample friction coefficient. Fails loudly if the runner did not attach it."""
        if 'friction_mu' not in sample['cloth']:
            raise KeyError(
                "friction_dissipative needs cloth.friction_mu but it is not on the sample. Enable "
                "runner.ccraft.arm_br.enabled=True (the runner samples mu and attaches it) or set "
                "runner.ccraft.material.friction_mu_override.")
        return sample['cloth'].friction_mu.reshape(-1)

    def _contact_geometry_and_lambda_n(self, ex):
        """No-grad nearest-body-face geometry + grav+inertia lambda_n.

        Extracted out of _example_terms so this ONE implementation is shared by the training loss
        (below) and render-time force capture (runners/ccraft.py's trained-friction arm path, via
        compute_render_force) -- the rendered force and the loss's contact set/lambda_n can never
        drift apart. Pure geometry: no mu, no slip, no loss weighting.

        Returns (n_hat, nn_cen_next, nn_cen_curr, lam, active), all (Vc, ...), all detached.
        `lam` is NOT yet zeroed outside `active` -- callers that want a contact-only force field
        (compute_render_force) mask it themselves; _example_terms needs the raw values plus
        `active` separately (it also uses nn_cen_* with a live gradient path via cloth_next).
        """
        m = self.mcfg
        obstacle_next = ex['obstacle'].target_pos            # (Vo,3) body next pose
        obstacle_curr = ex['obstacle'].pos                   # (Vo,3) body current pose
        obstacle_faces = ex['obstacle'].faces_batch.T        # (Fo,3)
        cloth_curr = ex['cloth'].pos                         # (Vc,3) x^t
        cloth_next = ex['cloth'].pred_pos                    # (Vc,3) predicted x^{t+1}
        mass = ex['cloth'].v_mass                            # (Vc,1)
        h = float(ex['cloth'].timestep.reshape(-1)[0])
        pinned = (ex['cloth'].vertex_type == NodeType.HANDLE).reshape(-1)

        from pytorch3d.ops import knn_points

        with torch.no_grad():
            # nearest body face by CURRENT-frame centroid (mirrors collision_penalty / bolt-on operator)
            face_curr_centroid = gather(obstacle_curr, obstacle_faces, 0, 1, 1).mean(dim=-2)
            _, nn_idx, _ = knn_points(cloth_curr.unsqueeze(0), face_curr_centroid.unsqueeze(0),
                                      return_nn=True)
            nn_idx = nn_idx[0]                                                       # (Vc,1)
            face_next_centroid = gather(obstacle_next, obstacle_faces, 0, 1, 1).mean(dim=-2)
            obstacle_fn = self.f_normals_f(obstacle_next.unsqueeze(0),
                                           obstacle_faces.unsqueeze(0))[0]           # (Fo,3) unit
            nn_cen_next = gather(face_next_centroid, nn_idx, 0, 1, 1)[:, 0]          # (Vc,3)
            nn_cen_curr = gather(face_curr_centroid, nn_idx, 0, 1, 1)[:, 0]          # (Vc,3)
            n_hat = gather(obstacle_fn, nn_idx, 0, 1, 1)[:, 0]                       # (Vc,3)

            # active set: signed gap of the PREDICTION vs the next body pose (BO/barrier convention)
            d = ((cloth_next.detach() - nn_cen_next) * n_hat).sum(dim=-1)            # (Vc,)
            active = (d < m.eps) & (~pinned)                                          # (Vc,)

            # force-balance lambda_n on the CURRENT prediction, DETACHED (whole block is no_grad)
            x_hat = cloth_curr + ex['cloth'].velocity                                # inertial predictor
            lam = grav_inertia_lambda_n(n_hat, mass, cloth_next.detach(), x_hat, h,
                                        m.g, m.z_axis)                               # (Vc,1) >= 0

        return n_hat, nn_cen_next, nn_cen_curr, lam, active

    def compute_render_force(self, ex):
        """(Vc,) contact-gated grav+inertia normal load for visualization (render_test_set.py /
        inference.py --render-force, trained-friction arm path: contact_arm.mode='none', so there is no
        inference-time projection to read a force from -- this is the read-only equivalent for a
        loss-only friction arm). Zero outside the active contact set, same convention as bolt-on operator's
        rendered force (utils/contact.py::bolt_on_friction_projection's return_force). Detached;
        does not affect the rollout in any way, purely a render-time query."""
        _, _, _, lam, active = self._contact_geometry_and_lambda_n(ex)
        return (lam.squeeze(-1) * active.float()).detach()

    def _example_terms(self, ex, mu_i):
        """Per-example D contribution and detached diagnostics. Returns (D_sum, diag dict)."""
        m = self.mcfg
        cloth_curr = ex['cloth'].pos                         # (Vc,3) x^t
        cloth_next = ex['cloth'].pred_pos                    # (Vc,3) predicted x^{t+1} (grads flow)
        h = float(ex['cloth'].timestep.reshape(-1)[0])

        n_hat, nn_cen_next, nn_cen_curr, lam, active = self._contact_geometry_and_lambda_n(ex)

        # slip: gradients flow through cloth_next ONLY (body motion + n_hat are detached constants)
        delta = (cloth_next - cloth_curr) - (nn_cen_next - nn_cen_curr)              # (Vc,3)
        delta_f = tangential_slip(delta, n_hat)                                      # (Vc,3)
        if self.mollifier == 'ipc':
            f0 = ipc_mollified_slip_norm(delta_f, m.eps_v, h)                        # (Vc,1)
        else:
            f0 = mollified_slip_norm(delta_f, m.eps_v, h)                            # (Vc,1)

        D_pv = mu_i * lam * f0                                                       # (Vc,1)
        act = active.unsqueeze(-1)
        D_sum = (D_pv * act).sum()

        with torch.no_grad():
            slip = torch.norm(delta_f, dim=-1, keepdim=True)                         # (Vc,1)
            band = m.eps_v * h
            if self.mollifier == 'ipc':
                f1 = ipc_f1(slip, m.eps_v, h)                                        # f0' in [0,1]
            else:
                f1 = slip / f0                                                       # f0' in [0,1)
            n_act = int(active.sum())
            diag = dict(
                n_active=n_act,
                n_total=int(active.numel()),
                lam_sum=float((lam * act).sum()),
                lam_max=float(lam[active].max()) if n_act else 0.0,
                slip_sum=float((slip * act).sum()),
                band_count=int(((slip < band) & act).sum()),
                f1_sum=float((f1 * act).sum()),
                implied_sum=float((mu_i * lam * f1 * act).sum()),   # mu*lambda_n*f0' (N)
                h=h,
            )
        return D_sum, diag

    def forward(self, sample):
        # No-body training sample (nobody_freq > 0): friction is undefined without a body.
        # Mirrors the guard in ccraft.collision_penalty.
        if 'obstacle' not in sample.node_types:
            return dict()

        m = self.mcfg
        B = sample.num_graphs
        mu = self._resolve_mu(sample)                                                # (B,)
        ref = sample['cloth'].pred_pos

        iter_num = int(sample['cloth'].iter[0].item()) if 'iter' in sample['cloth'] else 0
        warmup = int(m.friction_warmup_steps)
        ramp = int(getattr(m, 'friction_ramp_steps', 0) or 0)
        if iter_num < warmup:
            warmup_gate = 0.0
        elif ramp > 0:
            # linear 0->1 over `ramp` iters after warmup (BR attempt-2 (d): weight ramp, no re-peak)
            warmup_gate = min(1.0, (iter_num - warmup + 1) / ramp)
        else:
            warmup_gate = 1.0

        D_total = ref.new_zeros(())          # gated (enters the loss)
        D_raw_total = ref.new_zeros(())      # ungated (diagnostic)
        agg = dict(n_active=0, n_total=0, lam_sum=0.0, lam_max=0.0, slip_sum=0.0,
                   band_count=0, f1_sum=0.0, implied_sum=0.0)
        h2_mean = 0.0
        gate_frac = 0.0

        for i in range(B):
            ex = sample.get_example(i)
            is_init = ('is_init' in ex['cloth']
                       and float(ex['cloth'].is_init.reshape(-1)[0]) > 0)
            D_sum, diag = self._example_terms(ex, mu[i])
            h2 = (diag['h'] ** 2) if m.use_h2 else 1.0
            h2_mean += h2 / B

            gate = warmup_gate * (0.0 if (is_init and m.skip_init_steps) else 1.0)
            gate_frac += gate / B
            D_total = D_total + gate * h2 * D_sum
            D_raw_total = D_raw_total + D_sum.detach()

            agg['lam_max'] = max(agg['lam_max'], diag['lam_max'])
            for k in ('n_active', 'n_total', 'lam_sum', 'slip_sum', 'band_count',
                      'f1_sum', 'implied_sum'):
                agg[k] += diag[k]

        loss = D_total / B * m.weight

        n_act = max(agg['n_active'], 1)
        def _t(v):
            return ref.new_tensor(float(v))
        out = dict(
            loss=loss,
            weight=m.weight,
            # diagnostics (detached scalars; keys must avoid 'loss'/'weight'/'gradient' substrings)
            D_raw=(D_raw_total / B),                             # ungated, unweighted sum(D)/B
            D_wtd=loss.detach(),                                 # what actually enters the loss
            n_active=_t(agg['n_active']),
            active_frac=_t(agg['n_active'] / max(agg['n_total'], 1)),
            mu_mean=mu.mean().detach(),
            lam_mean=_t(agg['lam_sum'] / n_act),
            lam_max=_t(agg['lam_max']),
            slip_mean=_t(agg['slip_sum'] / n_act),
            static_band_frac=_t(agg['band_count'] / n_act),
            f1_mean=_t(agg['f1_sum'] / n_act),
            implied_force_mean=_t(agg['implied_sum'] / n_act),
            gate_on=_t(gate_frac),
        )
        return out
