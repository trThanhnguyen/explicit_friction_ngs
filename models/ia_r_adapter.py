"""W4-residual adapter: the µ-gated residual adapter.

    z = z0 + ρ(µ) · A_φ(s, µ, q),        ρ(µ) = min(µ / rho_mu_max, rho_cap),   ρ(0) = 0

with the backbone θ0 FROZEN. The point is not extra capacity — it is **structural attribution**:

* **The µ=0 anchor cannot drift.** ρ(0)=0 and θ0 is frozen, so at µ=0 the network's output is
  bitwise the backbone's, *independently of training*. Every wave so far had to MEASURE that
  anchor (T2) and /3 AO broke it; here it is a property of the architecture. The µ=0 row of
  an residual adapter results table simply IS the frozen backbone's published row.
* **All µ-response is attributable to φ.** gave the network µ as *information* (a body-edge
  feature on shared weights); this gives it a *separate parameter set* that is switched off at
  µ=0. That is why the backbone must be the µ-BLIND trained-through endpoint endpoint: a µ-aware backbone would leave
  two µ paths and muddy exactly the attribution this arm exists to make.

The adapter never learns the Coulomb law — the bolt-on operator still owns stick/slide entirely.
It only learns the smooth deformation adjustment *around* contacts, which is why its inputs are
the operator's own detached diagnostics `q` rather than raw geometry.

FEATURES (per cloth vertex, all DETACHED, assembled by `Runner._ia_r_features`; the layout is
pinned by test R2 so a checkpoint's adapter cannot silently change meaning):

    idx  feature                       why
    0-2  c0 = P_µ(z0) − z0             the operator's own first-pass correction: direction AND
                                       magnitude of the friction displacement this vertex is about
                                       to receive. The single richest diagnostic, and free — the
                                       impulse-aware hook already computes exactly this quantity.
    3    ‖c0‖ / max_disp               its magnitude on the adapter's own output scale
    4    contact (λ_n > 0)             the active set, as a hard 0/1
    5    λ_n / (m·g)                    normal load in rest-weight units (the diagnostics' own
                                       calibration ratio, so the number is O(1) on real contacts)
    6 ρ(µ) the adapter's µ input — the SINGLE-MU principle
                                       applies: it is derived from the µ the operator resolves for
                                       ITSELF, never sampled a second time
    7-9  velocity (per-step displacement)   the vertex's own motion, for context

ZERO-INIT HEAD: the last Linear is zeroed at construction, so A_φ ≡ 0 and **residual adapter at init is
bitwise the frozen backbone at every µ** (test R3). Training can only move away from a known-good
model, and a failed adapter degrades toward it rather than toward noise.

TANH BOUND: the output is `max_disp · tanh(·)`, so a single adapter step can never displace a
vertex further than `max_disp` (default 2 cm) regardless of φ. The adapter is a perturbation of a
trained simulator, not a second simulator.
"""
import torch
from torch import nn

from models.core import networks

# feature layout — index: name. Pinned by the selftests; the runner assembles in this order.
FEATURE_NAMES = ('c0_x', 'c0_y', 'c0_z', 'c0_norm', 'contact', 'lambda_n_rel', 'rho',
                 'vel_x', 'vel_y', 'vel_z')
N_FEATURES = len(FEATURE_NAMES)


class ResidualAdapter(nn.Module):
    """Per-vertex MLP: (V, N_FEATURES) -> (V, 3) bounded displacement.

    Deliberately per-vertex (no message passing): the backbone already carries all the spatial
    coupling, and a local correction around contacts is what asks . The residual risk of the
    choice — no spatial smoothing of the adapter's own field — is disclosed in the build report.
    """

    def __init__(self, hidden: int = 128, layers: int = 3, max_disp: float = 0.02,
                 n_features: int = N_FEATURES):
        super().__init__()
        self.n_features = n_features
        self.max_disp = float(max_disp)
        widths = [n_features] + [hidden] * layers + [3]
        self.mlp = networks.MLP(widths, activate_final=None)
        head = self.mlp.layers[-1]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def forward(self, features):
        if features.shape[-1] != self.n_features:
            raise ValueError(f'residual adapter adapter expects{self.n_features} features, '
                             f'got {features.shape[-1]}')
        return self.max_disp * torch.tanh(self.mlp(features))
