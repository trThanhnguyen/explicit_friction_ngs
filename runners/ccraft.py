# wbody2 == mimp_cut_wbody_omw_ombp
import os
from pathlib import Path
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from huepy import yellow
from omegaconf import II, OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData, Batch
from tqdm import tqdm

from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.impulses import CollisionSolver
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug
from utils.common import move2device, save_checkpoint, add_field_to_pyg_batch, copy_pyg_batch, TorchTimer, NodeType
import utils.contact as C
from utils.timeprof import PROF


@dataclass
class MaterialConfig:
    density_min: float = 0.20022
    density_max: float = 0.20022
    lame_mu_min: float = 23600.0
    lame_mu_max: float = 23600.0
    lame_lambda_min: float = 44400
    lame_lambda_max: float = 44400
    bending_coeff_min: float = 3.9625778333333325e-05
    bending_coeff_max: float = 3.9625778333333325e-05
    bending_multiplier: float = 1.

    density_override: Optional[float] = None
    lame_mu_override: Optional[float] = None
    lame_lambda_override: Optional[float] = None
    bending_coeff_override: Optional[float] = None
    friction_mu_override: Optional[float] = None   # bolt-on operator: --mu sweep; wins over contact_arm.bolt_on.mu


@dataclass
class OptimConfig:
    lr: float = 1e-4
    decay_rate: float = 1e-1
    decay_min: float = 0
    decay_steps: int = 5_000_000
    step_start: int = 0
    # --- trained-friction arm LR-schedule knobs. Defaults keep behavior byte-identical. -------------
    # Linear LR warm-up over the first `warmup_steps` SCHEDULER steps (scheduler.step() runs once per
    # autoregressive roll step, so these are INFLATED units — ~5x global steps in the short phase at
    # roll_max, and up to roll_max_long per long sample). 0 = off.
    # ⚠️ Unrelated to runner.warmup_steps and to the two bolt_on warm-ups — this one delays the LR
    #    ramp and nothing else. See the NAME COLLISION table on BoltOnConfig.warmup_steps.
    warmup_steps: int = 0
    # LR-rewarm route: start the LR schedule from THIS step instead of the checkpoint's global step.
    # train.py propagates config.step_start (=restart.step_start) into optimizer.step_start, which
    # would resume the bottomed-out schedule; setting step_start_override: 0 gives the friction phase
    # a fresh schedule while restart.step_start keeps the loss ramps / roll schedule mature.
    step_start_override: Optional[int] = None


@dataclass
class SafecheckConfig:
    n_impulse_iters: int = 10
    max_riz_size: int = 30
    n_rest2pos_steps: int = 10
    riz_max_mp_steps: int = 10
    riz_max_steps_total: int = 100
    pinned_mass: float = 1e9
    riz_epsilon: float = 1e-10
    max_ncoll: int = -1
    max_impulse_norm: Optional[float] = None
    double_precision_impulse: bool = True
    double_precision_riz: bool = True
    device: str = II('device')


@dataclass
class BoltOnConfig:
    """bolt-on operator sub-config. Active when ContactArmConfig.mode='bolt_on'."""
    eps: float = 2e-3              # barrier offset (match postcvpr.collision_penalty.eps) / force_balance gate
    stiffness: float = 1.0         # λ_n = 3·w·max(eps-d,0)² (barrier CALIBRATION KNOB; see calib_bolt_on_stiffness.py)
    eps_v: float = 1e-2            # static-velocity band (IPC)
    mu: Optional[float] = None     # fallback friction coeff; material.friction_mu_override wins
    z_axis: int = 1                # up-axis (Y-up = index 1); mg·cos probe AND force_balance gravity axis
    log_every: int = 0             # log projection diagnostics every N steps (0=off)
    # --- (normal-force source for the friction cone ---
    lambda_n_source: str = 'barrier'   # 'barrier' (default => byte-identical) | 'force_balance'
    fb_use_gravity: bool = True        # force_balance: include grad(E_grav)   = m*g*e_up
    fb_use_inertia: bool = True        # force_balance: include grad(E_inertia) = (m/h²)*(x_pred-(x+v))
    fb_use_elastic: bool = False       # force_balance: include grad(E_stretch+E_bend). Step1=False, Step2=True
    fb_elastic_conform: bool = False   # Step2 Appendix-A fix: cloth-normal Laplace pressure gated by
                                       # cloth·body conformance (excludes in-plane fold tension). See
    fb_elastic_pen_gate: bool = False  # fix: zero elastic pressure on penetrating verts (d<0) —
                                       # invalid correspondence + the runaway trap
    fb_elastic_smooth_iters: int = 0   # DEPRECATED (spreads the
                                       #   coherent trained-penalty reaction). Kept for ablation only.
    fb_elastic_pen_dilate: bool = False    # #4: pen gate dilated 1 ring (trap halo removal)
    fb_elastic_approach_gate: bool = False # #4: elastic only on non-separating contacts
    fb_elastic_noise_floor_q: float = 0.0  # #5: soft-threshold by q-quantile of off-contact
                                           #   pressure (self-calibrating state-noise floor). 0=off.
    fb_elastic_deflate_profile: bool = False  # #5b: subtract per-step robust fit a*(eps-d)^2 —
                                           #   removes the trained-penalty proximity plateau, keeps excess
    lambda_smooth_iters: int = 0       # K masked-Jacobi (MPNN-style) rounds mollifying the
                                       #   assembled lambda_n over the CONTACT set; 0 = off
    lambda_smooth_window: float = 0.0  # physical window (m); K = round(window / mean rest edge len)
                                       #   (used only when lambda_smooth_iters == 0)
    lambda_smooth_conserve: bool = True  # conserve mass(~area)-weighted total normal load
    lambda_smooth_render_only: bool = False  # mollify only the rendered/diagnosed field; cone
                                             #   keeps point samples (dynamics identical to K=0)
    # ⚠️ NAME COLLISION — there are FOUR `warmup*` knobs in this file and they are unrelated. This is
    # the one people get wrong. Full disambiguation, authoritative:
    #
    #   | key                                  | unit / scope                  | what it delays        |
    #   |--------------------------------------|-------------------------------|-----------------------|
    #   | runner.ccraft.warmup_steps (=100)    | GLOBAL steps                  | optimizer updates —   |
    #   |                                      |                               | rolls run but do not  |
    #   |                                      |                               | train (step_short)    |
    #   | runner.ccraft.optimizer.warmup_steps | INFLATED scheduler steps      | the LR ramp           |
    #   |                                      | (~65.8 per global step!)      |                       |
    #   | bolt_on.warmup_steps  (THIS ONE)     | rollout steps, INFERENCE path | the friction OPERATOR |
    #   | bolt_on.warmup_regular_steps         | regular steps per TRAINING    | the friction OPERATOR |
    #   |                                      | roll                          |                       |
    #
    # The last two are the same mechanism on the two different code paths, and THIS one additionally
    # means two different things depending on which config family it appears in:
    #
    # * inference configs (configs/contourcraft_armBO_*.yaml): the ELASTIC BLINDFOLD. 30 there
    #     exists only because the LBS over-stretch transient reaches lambda_n through the elastic
    # channel. -lineage arms set 0 and need no blindfold — that is the whole point of
    # /. The value is a property of the LAMBDA_N SOURCE.
    #   * training configs (configs/bo_tt_*.yaml): the inference-path twin of warmup_regular_steps,
    # held EQUAL to it (: both 2) purely so a valid_rollout run under a training config
    #     gates the operator the same way training did. The value is a property of the TRAINING
    #     TRANSIENT, unrelated to the elastic channel.
    #
    # So do NOT copy this value between the two config families. Copying `0` from an inference
    # config into a bo_tt_* config silently changes the train/eval gating contract; copying `2`
    # the other way silently reintroduces a blindfold that arm does not want.
    warmup_steps: int = 0              # skip the WHOLE bolt-on for the first N rollout steps: the LBS
                                       #   initial garment state is not energetically optimal (HOOD supp)
                                       #   and its relaxation transient must not feed the friction cone.
                                       #   INFERENCE path only (_apply_bolt_on_friction's step_idx).
    fb_g: float = 9.81                 # gravity magnitude for the force_balance gravity term
    # --- normal completion of the friction projection (ported for train-in-loop D5:
    # the same-step gap erosion of the tangential edit must be repaired ANALYTICALLY — training on
    # states would adapt the net to an artifact the shipping operator removes) ---------------
    normal_exact: str = 'off'          # 'off' (byte-identical) | 'preserve' (-T: restore the
                                       #   refreshed nearest-face plane-gap proxy of the pre-friction
                                       # prediction) | 'preserve_clamp' (-TC: + Signorini clamp
    normal_exact_iters: int = 2        # correspondence-refresh budget (NO convergence guarantee)
    normal_exact_slack: float = 1e-4   # TC clamp slack δ_s (m); ablated at {0, 1e-4}
    normal_exact_max_step: float = 0.0 # total-correction cap; 0 => auto max(2*eps, |friction_dx|);
                                       #   over-cap => vertex reverted (skip, not clip) + counted
    normal_exact_dust: float = 1e-8    # altered-set predicate: |friction_dx| > dust (fp reassembly
                                       # residue is ~1e-9 on factor==1 contacts
    normal_exact_ambig_k: int = 4      # K candidate faces for the ambiguity guard
    normal_exact_ambig_cos: float = 0.0    # near-equal candidates with normal dot < this => skip+count
    normal_exact_ambig_ratio: float = 1.5  # "near-equal" = centroid dist within this ratio of nearest
    normal_exact_control: bool = False # TC fairness control: candidate set = matched
                                       #   active-contact set (fires at µ=0). NEVER a headline arm.
    # --- (geometry-aware (surface-marched) friction path. Defaults = the
    # 'chord' behavior, so any config not setting these is byte-identical to v13. --------------
    tangential_path: str = 'chord'     # 'chord' (byte-identical) | 'surface' (: surface-
                                       # marched slide path + endpoint restore; requires
                                       # normal_exact in {'preserve','preserve_clamp'} => /
                                       # | 'walk' (: parameter-free triangle walk -- the
                                       #   crossed body faces ARE the segments; same normal_exact
                                       #   requirement and downstream code as 'surface')
    path_substeps: int = 8             # march resolution (surface_march substeps); flat => path==chord
                                       #   IGNORED under 'walk' (mesh topology sets the resolution);
                                       #   walk configs DROP the key (unreachable-key discipline)
    path_arc_cone: str = 'off'         # 'off' (reuse the chord factor on arc — path-isolating arm) |
                                       #   'on' (arc-native Coulomb cone: removes exactly b of TRUE slide)
    path_arc_cap: float = 3.0          # revert to the chord result when marched arc > this x chord
    # --- (multi-normal walk-back. Defaults = the 'single' behavior,
    # so any config not setting these is byte-identical to v17. ------------------------------------
    path_normals: str = 'single'       # 'single' (byte-identical) | 'multi' (: per-segment λ
                                       #   re-projection of the grav+inertia drive + landing-segment
                                       #   prox; requires surface + arc_cone 'on' + force_balance +
                                       # mollifier off => /
    path_capstan: str = 'off'          # junction wrap surcharge (discrete belt e^{µ·ΣΔθ} on the
                                       #   per-segment budgets); requires path_normals 'multi';
                                       #   benchmark-second knob (same off/on discipline as arc_cone)
    path_walk_max_cross: int = 16      # triangle-walk SAFETY cap on edge crossings per
                                       #   pair (never-binds class, sibling of path_arc_cap;
                                       #   exceed => per-vertex chord fallback, counted
                                       #   wk_fb_cross). NOT a resolution knob. Non-default with
                                       #   tangential_path != 'walk' refuses (dead-key discipline).
    path_walk_fast: bool = False       # triangle-walk SCHEDULING fast path: merged
                                       #   seating + CUDA-graph replay of both walk loops. Output
                                       #   is BITWISE the default path (no arithmetic changes), so
                                       #   this is a cost knob, not an arm; requires 'walk'.
    spread_w: float = 0.0              # DIAGNOSTIC correction spreading: each
                                       #   moved contact donates its TANGENTIAL correction to its
                                       #   1-ring; receivers take the uniform mean x spread_w.
                                       #   Path-agnostic post-pass (composes with chord => the 2x2).
                                       #   0.0 (default) = branch skipped, byte-identical off path.
                                       #   Probe only; C1 replaces the mean with an XPBD-style solve.
    spread_project: str = 'off'        # 'own' re-frames the received mean
                                       #   into the RECEIVER's own tangent plane before applying it
                                       #   (C0 leaves it in the DONOR's frame — the measured cause
                                       #   of C0's penetration). Isolates frame-vs-solve. Needs
                                       #   spread_w > 0. 'off' (default) = byte-identical C0.
    spread_hops: int = 1               # MULTI-HOP reach: a receiver at hop k
                                       #   becomes a source at hop k+1 (BFS wavefront, each vertex
                                       #   assigned once at its hop distance). 1 (default) = C0's
                                       #   single ring. Numerical -- set it large enough that
                                       #   spread_gamma is the binding knob.
    spread_gamma: float = 0.0          #   per-hop attenuation in [0,1]. 0.0 (default) = BYTE-
                                       #   IDENTICALLY C0 at any hop count. Hop 1 always keeps
                                       #   exactly C0's displacement, so every rung is a strict
                                       #   superset of C0 and the ladder measures ADDED REACH only.
                                       #   A screening length, swept not derived.
    spread_weight: str = 'uniform'     # how a receiver aggregates MULTIPLE
                                       #   sources. 'uniform' (default) = plain mean over distinct
                                       #   neighbours, byte-identical. 'area' = weight each source
                                       #   by its own barycentric rest area, removing the
                                       #   mesh-density bias of umbrella weights (areas vary 2.98x
                                       #   on a real garment; 66% of receivers see 2+ sources).
                                       #   NOT the exact Laplace-Beltrami operator -- cotangent
                                       #   weights would be, but 23.1% of triangles are obtuse.
    patch_iters: int = 0               # COMPLIANT PATCH RELAXATION: K XPBD
                                       #   (Eq. 18) sweeps over the patch around each engaged
                                       #   contact — corrected contacts are anchors (w=0), their
                                       #   patch_depth-ring is free, edge targets are the lengths
                                       #   under the FRICTIONLESS prediction (not rest: our garments
                                       #   sit at ~8% strain, so rest targets would be a drape
                                       #   intervention). The principled form of the coupling C0
                                       #   probed. 0 (default) = branch skipped, byte-identical.
                                       #   MUTUALLY EXCLUSIVE with spread_w (alternative couplings).
    patch_depth: int = 1               #   ring depth of the free set (1 = matched with C0).
    patch_alpha_scale: float = 1.0     #   multiplier on the MATERIAL-derived compliance. 1.0 = the
                                       #   garment's own Lamé params, no fitted constant. Measured
                                       #   α̃/(∇CM⁻¹∇Cᵀ+α̃) ≈ 1e-6 on these fabrics (logged as
                                       #   pc_alpha_ratio) ⇒ stiff-limited; scaling it up is how
                                       #   that claim gets demonstrated instead of asserted.
    patch_signorini: str = 'preserve'  #   'preserve' (default) = strictly NON-WORSENING normal
                                       #   handling on the free set (never let a free vertex end
                                       #   with a smaller plane gap than the backbone gave it;
                                       #   ambiguous queries skipped). NOT an improving clamp — that
                                       #   would buy `pen` for a non-coupling reason. 'off' isolates
                                       #   the solve.
    patch_feas_tol: float = 1e-6       #   absolute feasibility tol [m] for iterations-to-feasibility
                                       # (: read the composability axis on ABSOLUTE
                                       #   quantities, never relative deltas).
    wave_enabled: bool = False         # incremental tangent-membrane WAVE: the
                                       #   material-transport alternative to C0/C1 (mutually
                                       #   exclusive with spread_w / patch_iters). No reach
                                       #   knobs: reach = material speed x h, substeps certified
                                       #   per frame, source = the Coulomb tangential landing on a
                                       #   fixed cubic ramp. mu=0 => exact structural no-op.
                                       #   False (default) = byte-identical.
    wave_visco: float = 0.0            # [membrane-wave absorption] SLS/Maxwell branch
                                       #   fraction g: overstress
                                       #   g*S(eta - hh) with hh relaxing toward eta at tau_r.
                                       #   Loss peaks at omega*tau_r = 1 and VANISHES at the mesh
                                       #   tail (explicit-compatible; the S3 one-beta CFL collapse
                                       #   cannot occur). Strain-based => rigid-motion-null ("good
                                       #   damping"). 0.0 = the beta=0 conservative carrier,
                                       #   byte-identical to the pre-visco arm.
    wave_tau_frames: float = 0.5       #   tau_r in frame units (tau_r = value * h). 0.5 puts peak
                                       #   loss in the intra-frame band that rings at beta=0
                                       #   A disclosed material parameter (SLS), not a reach law.
    wave_strain_cap: float = 0.0       # per-edge strain budget |du|/l0 on the
                                       #   wave's composed output (Provot-lineage strain limiting;
                                       #   degree-averaged mobility-masked Jacobi clip). 0.2 = the
                                       #   backbone's own p95 edge-strain regime
                                       #   (self-calibrated), certified to remove the source-
                                       #   adjacent short-edge tail at byte-identical mid/far
                                       # payload. 0.0 (default) = byte-identical
                                       #   uncapped wave stage. Endpoint split: inverse-mass
                                       # (Bridson '02 momentum-conserving impulses; adopted
                                       # outcome-neutral vs 50/50).
    drag_k: float = 0.0                # DIAGNOSTIC over-budget drag: retained slide
                                       #   arc s_star *= exp(-drag_k*µ). Rate-type, NOT Coulomb —
                                       #   ceiling probe for per-vertex slowing, never a ship knob.
                                       #   0.0 (default) = branch skipped, byte-identical off path.
                                       #   Requires surface + path_normals 'multi'.
    # --- train-in-loop (BO-in-the-loop training. ALL defaults are no-ops. -------------
    train_in_loop: bool = False        # apply the operator to the DETACHED post-loss state inside
                                       #   training rolls, so the NEXT step trains on the frictional
                                       #   state distribution. The per-step loss stays on the raw
                                       # proposal; no gradient flows through the operator.
    skip_init_steps: bool = False      # never run the operator on an is_init (initial_ts macro) step:
                                       #   the cone bound R = mu*lambda_n*h^2/m scales with h^2, so the
                                       #   10x settle step inflates R ~100x and would freeze every gated
                                       # contact. Gates on the step's is_init field,
                                       #   NOT the step index (1-step rolls flip a coin for initial_ts).
    warmup_regular_steps: int = 0      # training warm-up (T0's N_diag): skip the operator for the first
                                       #   N REGULAR (non-init) steps of every training roll — the
                                       #   LBS-init relaxation transient must not feed the cone.
                                       #   TRAINING path only; the inference-path twin is
                                       #   `warmup_steps` above — see its NAME COLLISION table.
    mu_sample: bool = False            # per-roll mu sampler: mu=0 with prob mu_sample_p_zero,
                                       #   else LogUniform[mu_sample_lo, mu_sample_hi]. Draws come from a
                                       #   DEDICATED numpy Generator so the global RNG streams (data
                                       #   order) stay in lockstep between the OP and CTRL twin runs.
    mu_sample_p_zero: float = 0.2
    mu_sample_lo: float = 0.05
    mu_sample_hi: float = 0.7
    mu_sample_seed: int = 57
    expect_exposure_after: int = 0     # >0: hard-fail if the operator has ALTERED zero vertices after
                                       # this many training rolls (guard 4 exposure assert;
                                       #   0 = off). Altered = |dx| > train_altered_dust on >=1 vertex.
    train_altered_dust: float = 1e-7   # meters; below = float32 reassembly round-off at factor==1
                                       #   (mu=0 / no-friction contacts), above = real friction dx
                                       #   (typical correction mu*g*h^2 ~ 3e-3 m).
    # --- (the composite ACTION objective. Default 'stop_grad' reproduces
    # train-in-loop byte-for-byte, so bo_tt_probe_op/ctrl.yaml are untouched by this feature. -------
    train_loss_mode: str = 'stop_grad'  # 'stop_grad' (: loss on the RAW proposal, operator
                                        # applied after loss+detach) | 'action' (: the
                                        #   operator runs BEFORE the loss and IN the graph, the
                                        #   backbone loss is evaluated on the shipped frame x^+, and
                                        #   the dissipation the operator paid is added as a term =>
                                        #   L = L_CC(x^+) + mu*sg[lambda_n]*||Delta_T(x^+)||_moll.
                                        #   One operator application serves BOTH the loss and the
                                        #   state transition, so the state distribution is identical
                                        # to 'stop_grad' and only the loss differs.
                                        # | 'impulse_aware': straight-through
                                        #   operated-state loss. The operator runs under no_grad;
                                        #   c = sg[P_mu(z) - z]; every criterion EXCEPT inertia is
                                        #   evaluated on x~ = z + c (numerically P_mu(z), identity
                                        # Jacobian), inertia on the raw z (== the inertia-
                                        #   reference-shift identity). NO dissipation term. The
                                        #   transition applies the Signorini TC completion to the
                                        # raw proposal at mu>0 only (next state = (z)); at
                                        #   mu=0 everything is bitwise the 'stop_grad' path.
    action_weight: float = 1.0          # weight on the dissipation term D. 1.0 = the physically
                                        #   correct pricing (D is already in Joules, the same units
                                        #   as the backbone incremental potential); exposed only so
                                        #   the term can be ablated to 0 (=> naive post-operator
                                        #   loss, the REJECTED design — never a shipping value).
    action_raw_probe_every: int = 10    # every N operator applications, additionally evaluate the
                                        #   collision/stretch criterions on the RAW proposal x^ (no
                                        #   grad) alongside the shipped x^+. This is the TC-leak
                                        #   canary: 'action' scores the repaired frame, so a net
                                        #   could in principle learn to let the clamp fix its dives.
                                        #   Divergence of ao/raw_collision from ao/post_collision
                                        #   over training is that signature. 0 = off.


@dataclass
class TrainedConfig:
    """trained-friction arm scaffolding — trained in-loop with friction as a dissipation term (P3).
    NOT yet implemented; see and for the activation checklist. These
    fields describe the intended BR knobs so a BR config validates and the design is explicit. BR reuses
    the SAME force_balance λ_n as bolt-on operator, applied inside a differentiable friction dissipation loss
    term (criterions/ccraft/friction_dissipative.py, to be created) rather than a post-hoc projection."""
    normal_source: str = 'force_balance'  # BR keys the cone to the force_balance λ_n (not the barrier)
    eps: float = 2e-3                      # active-set gate (match collision_penalty.eps)
    eps_v: float = 1e-2                    # IPC static-velocity band for the C1 mollifier f0
    mu: float = 0.5                        # fallback friction coeff (constant-μ training)
    weight: float = 1.0                    # weight on the dissipation potential D in the minimized energy
    fb_use_gravity: bool = True            # force_balance λ_n terms (mirror BoltOnConfig.fb_*)
    fb_use_inertia: bool = True
    fb_use_elastic: bool = True            # BR: tightness matters for trained garments -> default on
    fb_g: float = 9.81
    detach_lambda_n: bool = True           # lagged IPC: freeze λ_n so friction is a genuine potential


@dataclass
class ArmBRConfig:
    """trained-friction arm (trained Coulomb friction. Phase 1 = friction as a dissipation
    LOSS term (criterions/ccraft/friction_dissipative.py) with grav+inertia lambda_n, plus the
    per-body-edge mu feature (model.ccraft.enable_friction_mu_feature). NO inference operator: the BR
    test step is a plain forward pass with contact_arm.mode='none'.

    This block owns MU SAMPLING (the loss's mu and the model's mu feature read the same
    cloth.friction_mu / cloth.friction_mu_input fields — one path for train-sampled and
    test-requested mu). Loss-term knobs (weight, eps_v, warmup) live on the criterion node.
    Resolution priority at any time: material.friction_mu_override (evaluate.py --mu) > mu_mode.

    : mu_min/mu_max are read as the FEATURE NORMALIZATION SPAN even when
    enabled=False. The span is a property of the checkpoint's input convention, not of which arm
    samples mu, so a bolt-on run and a BR run share one definition repo-wide and their
    mu features are directly comparable. Everything else in this block stays BR-only."""
    enabled: bool = False
    phase: int = 1                          # bookkeeping; Phase 1 = grav+inertia lambda_n only
    lambda_source: str = 'grav_inertia'     # 'grav_inertia' (Phase 1) | 'grav_inertia_elastic' (Phase 2,
                                            #   NOT implemented — validated at init)
    mu_min: float = 0.0                     # training mu range [mu_min, mu_max] (ARGUS: [0, 0.6]);
    mu_max: float = 0.6                     #   also the feature normalization span (linear, unclamped)
    mu_mode: str = 'uniform'                # 'fixed' | 'uniform' | 'curriculum'
    mu_fixed: float = 0.0                   # mu_mode='fixed' value
    # (BR attempt-2 (c), H-MEAN postmortem): spike-and-slab mu=0 anchor — with this
    # probability a 'uniform' draw is replaced by EXACTLY mu=0, anchoring the frictionless end
    # of the conditioning axis. 0.0 = pure uniform (legacy, bit-identical for old configs).
    mu_zero_prob: float = 0.0
    # mu_mode='curriculum': iter < warmup => mu = 0; then |N(0, sigma)| truncated to
    # [mu_min, mu_max] with sigma ramping 0 -> mu_max over ramp_steps; after warmup+ramp => uniform.
    # Steps are ABSOLUTE global training iterations (cloth.iter) — for the rewarm route count from the
    # checkpoint's step_start, not from 0. Keep this ramp LAGGING the collision-weight ramp (F3).
    mu_curriculum_warmup_steps: int = 0
    mu_curriculum_ramp_steps: int = 1
    # Sanity bookkeeping for the three routes: 'scratch' | 'lr_rewarm' |
    # 'terminal_lr_negative_control'. Informational (routes differ via restart/optimizer config).
    train_start: str = 'scratch'
    # Double-ablation escape hatch: allow arm_br without the model mu feature (loss-only
    # ablation arm). Default False => init-time error on a mismatched config.
    allow_feature_off: bool = False


def sample_arm_br_mu(abr, B, iter_num, device):
    """(B) raw mu per sample, per arm_br.mu_mode. Torch RNG (seeded in train.py).

    'curriculum': mu = 0 while iter < mu_curriculum_warmup_steps; then a one-sided truncated Gaussian
    |N(0, sigma)| clamped to [mu_min, mu_max] with sigma ramping 0 -> mu_max over
    mu_curriculum_ramp_steps; uniform [mu_min, mu_max] afterwards. Steps are ABSOLUTE global iters."""
    if abr.mu_mode == 'fixed':
        return torch.full((B,), float(abr.mu_fixed), device=device)
    if abr.mu_mode == 'uniform':
        mu = torch.rand(B, device=device) * (abr.mu_max - abr.mu_min) + abr.mu_min
        zp = float(getattr(abr, 'mu_zero_prob', 0.0) or 0.0)
        if zp > 0.0:
            mu = mu * (torch.rand(B, device=device) >= zp).float()   # spike-and-slab mu=0 anchor
        return mu
    warm = int(abr.mu_curriculum_warmup_steps)
    ramp = max(int(abr.mu_curriculum_ramp_steps), 1)
    if iter_num < warm:
        return torch.zeros(B, device=device)
    progress = (iter_num - warm) / ramp
    if progress >= 1.0:
        return torch.rand(B, device=device) * (abr.mu_max - abr.mu_min) + abr.mu_min
    sigma = progress * abr.mu_max
    mu = (torch.randn(B, device=device) * sigma).abs() + abr.mu_min
    return mu.clamp(max=abr.mu_max)


@dataclass
class ContactArmConfig:
    """Post-model friction arm selector. mode='none' => byte-identical baseline (default).

    Arms:
      none     — no friction projection; rollout byte-identical to baseline ContourCraft.
      bolt_on — bolt-on operator: inference-time block-soft-threshold Coulomb friction.
                 Works with any checkpoint; calibrate stiffness via calib_bolt_on_stiffness.py.
      trained — trained-friction arm: same friction layer trained in-loop (not yet implemented).
    """
    mode: str = 'none'                    # 'none' | 'bolt_on' | 'trained'
    bolt_on: BoltOnConfig = BoltOnConfig()
    trained: TrainedConfig = TrainedConfig()


@dataclass
class Config:
    optimizer: OptimConfig = OptimConfig()
    material: MaterialConfig = MaterialConfig()
    safecheck: SafecheckConfig = SafecheckConfig()
    contact_arm: ContactArmConfig = ContactArmConfig()   # friction arm (none=baseline by default)
    arm_br: ArmBRConfig = ArmBRConfig()                  # trained-friction arm (disabled=baseline by default)
    # ⚠️ GLOBAL steps before the optimizer is allowed to update at all (step_short/step_long pass
    #    optimizer=None below this). Unrelated to optimizer.warmup_steps (LR ramp) and to the two
    #    bolt_on warm-ups (operator gating) — see the NAME COLLISION table on BoltOnConfig.warmup_steps.
    warmup_steps: int = 100
    increase_roll_every: int = 5000
    roll_max: int = 5
    roll_max_long: int = 300
    push_eps: float = 2e-3
    grad_clip: Optional[float] = 1.
    # explosion circuit breaker: if a single roll step's summed loss exceeds this cap,
    # skip the optimizer update and abort the rollout (grad_clip bounds each update's magnitude
    # but not how many garbage updates one exploding rollout delivers — up to roll_max_long).
    # None = disabled: every pre- config behaves bit-identically.
    loss_explosion_cap: Optional[float] = None
    overwrite_pos_every_step: bool = False
    nocollect_after: int = 1
    no_world_edges_every: int = -1
    always_fake_icontour: bool = False
    use_safecheck: bool = False

    enable_attractions: bool = True

    # Re-initialise the rollout from the Ando reference's drape state instead of
    # the LBS-skinned frame 0 (which penetrates the body, so the reference cannot
    # start there; the two states differ by ~35 mm rms). None = OFF and
    # byte-identical to prior behaviour. See tools/reference/gnn_init.py -- and note
    # the parity warning there: every method in a comparison must get the same
    # initial condition or the comparison is scoring the initial condition.
    init_state_file: Optional[str] = None

    initial_ts: float = II("experiment.initial_ts")
    regular_ts: float = II("experiment.regular_ts")


class Runner(nn.Module):
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module],
                 mcfg: DictConfig):
        super().__init__()

        self.model = model
        self.criterion_dict = criterion_dict['ccraft']
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
        self.normals_f = FaceNormals()

        self.sample_collector = SampleCollector(mcfg, obstacle=True)
        self.random_material = RandomMaterial(mcfg.material)
        self.safecheck_solver = CollisionSolver(mcfg.safecheck)
        self.collision_solver = CollisionPreprocessor(mcfg)


        self.short_steps = 0
        self.long_steps = 0

        # train-in-loop train-in-loop state. _bo_train_mu is the per-roll sampled mu (None outside
        # an active training roll); _bo_prev_altered feeds the guard-5 conditional loss split.
        self._bo_train_mu = None
        self._bo_prev_altered = False
        self._bo_mu_rng = None
        self._bo_train_roll = None
        self._bo_exposure_rolls = 0
        self._bo_exposure_altered_steps = 0
        # the pre-loss hook's detached raw proposal, consumed by the TC
        # transition on the same step. None outside an operator-active impulse_aware step.
        self._bo_ia_raw_z = None
        # residual adapter: count of steps whose update was structurally empty (mu=0 on a frozen
        # backbone). Monotone; surfaced as iar/nograd_updates_total.
        self._iar_nograd_updates = 0

        self._validate_arm_br()
        self._validate_train_in_loop()
        self._validate_mu_feature()
        self._validate_ia_r()

    # --------------------------------------------------------------------------------------------- #
    # trained-friction arm: mu sampling + config-consistency guards (friction itself lives in the LOSS — #
    # criterions/ccraft/friction_dissipative.py — and in the model's body-edge mu feature).          #
    # --------------------------------------------------------------------------------------------- #

    def _arm_br_cfg(self):
        return getattr(self.mcfg, 'arm_br', None)

    def _model_mu_feature_on(self) -> bool:
        """True iff the model appends the friction-mu channel to its body-edge features
        (model.ccraft.enable_friction_mu_feature). That flag is what makes cloth.friction_mu_input
        MANDATORY on every sample handed to the model, so it — not arm_br — is the real question
        'does something have to carry mu into the network?'.

        `self.model` is fetched with a default because this is now read on the OPERATOR HOT PATH
        (_assert_single_mu), not just at __init__: nn.Module.__getattr__ RAISES for a missing
        submodule rather than returning None, so a partially-constructed Runner — every selftest
        fixture built via Runner.__new__ — would blow up inside the operator. A Runner with no
        model has no mu feature by definition, so False is the right answer."""
        return bool(getattr(getattr(getattr(self, 'model', None), 'mcfg', None),
                            'enable_friction_mu_feature', False))

    def _mu_feature_span(self):
        """(mu_min, mu_max) — the linear, UNCLAMPED normalization span of the model's mu feature.

        Read from arm_br REGARDLESS of arm_br.enabled (see ArmBRConfig): the span is the
        checkpoint's input convention, so there is exactly one definition of it in the repo and a
        BR checkpoint and a bolt-on checkpoint encode mu identically. Defaults (0.0, 0.6)
        are the ARGUS range every mu-feature run so far has used."""
        abr = self._arm_br_cfg()
        if abr is None:
            return 0.0, 0.6
        return float(abr.mu_min), float(abr.mu_max)

    def _validate_arm_br(self):
        """Fail at construction on inconsistent BR configs (the silent versions are exactly the F5/F6
        failure modes: mu sampled but invisible, or friction loss with no mu)."""
        abr = self._arm_br_cfg()
        enabled = abr is not None and getattr(abr, 'enabled', False)
        has_criterion = 'friction_dissipative' in self.criterion_dict
        model_mu_feature = self._model_mu_feature_on()
        if not enabled:
            if has_criterion:
                raise ValueError(
                    "criterions include ccraft.friction_dissipative but runner.ccraft.arm_br.enabled "
                    "is False — the runner would never attach cloth.friction_mu. Enable arm_br or "
                    "remove the criterion.")
            # the mu feature is no longer arm_br's exclusive property. bolt-on operator
            # resolves a mu of its own (per-roll sampler / --mu override / bolt_on.mu), and
            # _add_friction_mu_from_operator now carries THAT SAME mu into the feature. So the
            # feature is legal without arm_br as long as the bolt-on operator is the one owning mu.
            # Nothing else about this clause is relaxed: with neither carrier the model would still
            # crash on the missing cloth.friction_mu_input, so that case still fails here, at init.
            if model_mu_feature and self.contact_arm_mode() != 'bolt_on':
                raise ValueError(
                    "model.ccraft.enable_friction_mu_feature=True but neither carrier is active: "
                    "runner.ccraft.arm_br.enabled is False AND runner.ccraft.contact_arm.mode is "
                    f"{self.contact_arm_mode()!r} (not 'bolt_on') — the mu feature would crash on "
                    "the missing cloth.friction_mu_input. Enable arm_br (BR: mu carried by the "
                    "sampler + dissipation loss), or set contact_arm.mode='bolt_on' (: mu "
                    "carried by the operator's own resolved mu), or disable the model flag.")
            return
        # enabled:
        if self.contact_arm_mode() == 'bolt_on':
            raise ValueError(
                "arm_br.enabled=True together with contact_arm.mode='bolt_on' — BR forbids any "
                "inference-time friction projection (/). Set contact_arm.mode='none'.")
        if abr.lambda_source != 'grav_inertia':
            raise NotImplementedError(
                f"arm_br.lambda_source={abr.lambda_source!r}: Phase 1 implements 'grav_inertia' only "
                "(the elastic/tightness channel is Phase 2).")
        if not has_criterion:
            raise ValueError(
                "arm_br.enabled=True but ccraft.friction_dissipative is not in the criterion set. "
                "Add it (use weight: 0.0 for the loss-ablation arm rather than removing it).")
        if not model_mu_feature and not getattr(abr, 'allow_feature_off', False):
            raise ValueError(
                "arm_br.enabled=True but model.ccraft.enable_friction_mu_feature=False — mu would "
                "have no test-time carrier (failure F5). Enable the model flag, or set "
                "arm_br.allow_feature_off=True if this is the deliberate feature-ablation arm.")
        if not (abr.mu_max > abr.mu_min):
            raise ValueError(f"arm_br: mu_max ({abr.mu_max}) must exceed mu_min ({abr.mu_min}).")
        if abr.mu_mode not in ('fixed', 'uniform', 'curriculum'):
            raise ValueError(f"arm_br.mu_mode={abr.mu_mode!r}: expected fixed|uniform|curriculum.")

    def _validate_ia_r(self):
        """ residual adapter: guards for the mu-gated residual adapter. Returns immediately
        for every config that does not enable it."""
        mcfg_m = getattr(self.model, 'mcfg', None)
        if not getattr(mcfg_m, 'enable_residual_adapter', False):
            return
        if self.contact_arm_mode() != 'bolt_on':
            raise ValueError(
                "model.ccraft.enable_residual_adapter=True requires contact_arm.mode='bolt_on': the "
                "adapter is conditioned on the operator's own detached diagnostics q, and rho(mu) "
                "reads the mu the OPERATOR resolves.")
        abr = self._arm_br_cfg()
        if abr is not None and getattr(abr, 'enabled', False):
            raise ValueError(
                "residual adapter with arm_br.enabled=True: two friction owners. BR puts friction in the LOSS; "
                "residual adapter keeps the bolt-on operator as the sole owner of the Coulomb law.")
        if self._model_mu_feature_on():
            raise ValueError(
                "residual adapter with model.ccraft.enable_friction_mu_feature=True: that gives the network TWO "
                "mu paths (the body-edge feature on shared weights, and the adapter), which destroys "
                "the attribution this arm exists to make — with a mu-BLIND frozen backbone, every "
                "bit of mu response is the adapter's. Use the 13-in trained-through endpoint endpoint.")
        span = float(getattr(mcfg_m, 'adapter_rho_mu_max', 0.6))
        if not span > 0:
            raise ValueError(f"adapter_rho_mu_max must be > 0 (got {span}): rho(mu) = mu/span, and "
                             f"rho(0)=0 is the arm's structural mu=0 anchor.")
        if float(getattr(mcfg_m, 'adapter_max_disp', 0.02)) <= 0:
            raise ValueError("adapter_max_disp must be > 0: it is the tanh bound on one adapter step.")
        bo = self._contact_arm_cfg().bolt_on
        override = getattr(self.mcfg.material, 'friction_mu_override', None)
        if not (getattr(bo, 'mu_sample', False) or override is not None or bo.mu is not None):
            raise ValueError(
                "residual adapter is on but no mu source: set bolt_on.mu_sample=true (training), "
                "material.friction_mu_override / --mu (eval), or bolt_on.mu. rho(mu) has no value "
                "without one, and the adapter would have no gate.")
        if not getattr(mcfg_m, 'freeze_backbone', True):
            print('[residual-adapter] WARNING: freeze_backbone=False — this is the ABLATION, not the arm. '
                  'The mu=0 anchor is no longer structural; score it with T2 like any other arm.')

    def _validate_mu_feature(self):
        """: guards for the mu feature carried by the BOLT-ON operator.

        Only fires when the model's mu feature is on, arm_br is off and the operator is bolt_on —
        i.e. exactly the combination. Every other config reaches this and returns."""
        if not self._model_mu_feature_on():
            return
        abr = self._arm_br_cfg()
        if abr is not None and getattr(abr, 'enabled', False):
            return                                   # BR path — _validate_arm_br already covered it
        if self.contact_arm_mode() != 'bolt_on':
            return                                   # already rejected in _validate_arm_br
        bo = self._contact_arm_cfg().bolt_on
        mu_min, mu_max = self._mu_feature_span()
        if not (mu_max > mu_min):
            raise ValueError(
                f"mu-feature span from arm_br: mu_max ({mu_max}) must exceed mu_min ({mu_min}). "
                f"The span normalizes the feature and is read even with arm_br.enabled=False.")
        # The encoder width is derived (models/core/ccraft.py: n_edgefeatures_body + 1 when the flag
        # is on). A config that ALSO bumps n_edgefeatures_body to 14 silently makes the body-edge
        # normalizer 13-wide against 12 raw channels and crashes on the first forward with a shape
        # error far from its cause. Refuse it here instead.
        n_body = getattr(getattr(self.model, 'mcfg', None), 'n_edgefeatures_body', 13)
        if int(n_body) != 13:
            raise ValueError(
                f"model.ccraft.n_edgefeatures_body={n_body} with enable_friction_mu_feature=True. "
                f"Leave it at 13: the +1 mu channel is added by the model itself "
                f"(models/core/ccraft.py), and the body-edge normalizer is sized "
                f"n_edgefeatures_body-1 against 12 raw channels.")
        # A mu source must exist, or _resolve_bolt_on_mu raises mid-rollout instead of at init.
        override = getattr(self.mcfg.material, 'friction_mu_override', None)
        if not (getattr(bo, 'mu_sample', False) or override is not None or bo.mu is not None):
            raise ValueError(
                "enable_friction_mu_feature=True with contact_arm.mode='bolt_on' but no mu source: "
                "set bolt_on.mu_sample=true (training), material.friction_mu_override / --mu "
                "(eval), or bolt_on.mu (fallback). The feature and the operator both read that ONE "
                "resolved mu, so with no source neither has a value.")
        src = ('per-roll sampler' if getattr(bo, 'mu_sample', False)
               else f'friction_mu_override={override}' if override is not None
               else f'bolt_on.mu={bo.mu}')
        print(f"[mu-feature] ON via the BOLT-ON operator — single-mu source: {src}; "
              f"feature span [{mu_min}, {mu_max}] (linear, unclamped); arm_br disabled.", flush=True)

    def _add_friction_mu(self, sample):
        """Attach cloth.friction_mu (raw, for the loss) and cloth.friction_mu_input (normalized, for
        the model's body-edge feature). Sampled ONCE per sequence (skipped if already present, so mu
        stays constant across the autoregressive rollout); evaluate.py --mu / material.
        friction_mu_override wins over sampling — one feature path for train and test.

        Two carriers, mutually exclusive by construction (arm_br forbids contact_arm='bolt_on'):
          * arm_br.enabled  — the BR path below, unchanged.
          * the bolt-on operator —, _add_friction_mu_from_operator."""
        abr = self._arm_br_cfg()
        if abr is None or not getattr(abr, 'enabled', False):
            if self._model_mu_feature_on():
                return self._add_friction_mu_from_operator(sample)
            return sample
        cloth = sample['cloth']
        if 'friction_mu' in cloth:
            return sample
        B = sample.num_graphs
        device = cloth.pos.device
        override = getattr(self.mcfg.material, 'friction_mu_override', None)
        if override is not None:
            mu = torch.full((B,), float(override), device=device)
        else:
            iter_num = int(cloth.iter[0].item()) if 'iter' in cloth else 0
            mu = sample_arm_br_mu(abr, B, iter_num, device)
        # Linear rescale to the training range, UNCLAMPED: 's saturating clamp made distinct
        # mu values byte-identical (a real past bug); out-of-range test mu extrapolates instead.
        mu_input = (mu - abr.mu_min) / (abr.mu_max - abr.mu_min)
        add_field_to_pyg_batch(sample, 'friction_mu', mu, 'cloth', reference_key=None,
                               one_per_sample=True)
        add_field_to_pyg_batch(sample, 'friction_mu_input', mu_input, 'cloth', reference_key=None,
                               one_per_sample=True)
        return sample

    def _add_friction_mu_from_operator(self, sample):
        """ — the SINGLE-MU carrier.

        Writes cloth.friction_mu_input from the mu the BOLT-ON OPERATOR will apply, obtained from
        _resolve_bolt_on_mu: literally the same function, on the same state, that
        _apply_bolt_on_friction calls a few lines later. There is no second sampler, so the network
        cannot be shown one mu while the operator applies another — the silent failure that would
        have looked exactly like "the repair didn't work" after 30 h of training.

        Deliberately does NOT write cloth.friction_mu. That field is read by _resolve_bolt_on_mu
        itself (branch 2 of its priority ladder) and by the BR dissipation criterion; writing it
        here would feed the resolver its own output and turn the operator's mu from a float into a
        (B,) tensor at eval — a real code-path change in the SHIPPING operator for no benefit. The
        feature is normalized, and the assert in _apply_bolt_on_friction de-normalizes it to check
        the two agree, so nothing needs the raw copy.

        Rewritten on EVERY call rather than skipped-if-present (unlike the BR path, which samples
        once per sequence): mu here changes per ROLL, and add_field_to_pyg_batch replaces the
        mapping entry, so an unconditional write is idempotent within a roll and cannot go stale
        across one. That holds regardless of whether the caller hands us a fresh sample."""
        bo = self._contact_arm_cfg().bolt_on
        mu = self._resolve_bolt_on_mu(sample, bo)          # THE mu — one resolution, one source
        cloth = sample['cloth']
        B = sample.num_graphs
        mu_t = (mu.reshape(-1).to(cloth.pos.device).float() if torch.is_tensor(mu)
                else torch.full((B,), float(mu), device=cloth.pos.device))
        mu_min, mu_max = self._mu_feature_span()
        # Same linear, UNCLAMPED rescale as the BR path ('s saturating clamp made distinct
        # mu byte-identical — a real past bug), so both carriers encode mu identically.
        mu_input = (mu_t - mu_min) / (mu_max - mu_min)
        add_field_to_pyg_batch(sample, 'friction_mu_input', mu_input, 'cloth', reference_key=None,
                               one_per_sample=True)
        return sample

    def _assert_single_mu(self, state, mu):
        """ invariant I1, checked LIVE on every operator application.

        De-normalizes cloth.friction_mu_input and requires it to equal the mu about to be applied.
        This is the tripwire for B2 in its general form: any future path that attaches the feature
        from a different source than the operator's resolved mu dies here, loudly, on the first
        step — instead of training a network conditioned on a lie for 30 h. Costs one small compare
        per operator call (the projection already syncs far more than this in its diagnostics).

        Inert unless the model's mu feature is on, so every earlier config is untouched."""
        if not self._model_mu_feature_on():
            return
        cloth = state['cloth']
        if 'friction_mu_input' not in cloth:
            raise RuntimeError(
                " single-mu invariant (I1): the model's mu feature is on but "
                "cloth.friction_mu_input is missing at the operator call — _add_friction_mu did not "
                "run on this sample (is _add_cloth_obj called before the model on this path?).")
        mu_min, mu_max = self._mu_feature_span()
        carried = cloth.friction_mu_input.reshape(-1) * (mu_max - mu_min) + mu_min
        applied = (mu.reshape(-1).to(carried.device) if torch.is_tensor(mu)
                   else torch.full_like(carried, float(mu)))
        if not torch.allclose(carried, applied, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                f" single-mu invariant VIOLATED: the network is being shown"
                f"mu={carried.tolist()} (from cloth.friction_mu_input, de-normalized over "
                f"[{mu_min}, {mu_max}]) while the operator is applying mu={applied.tolist()}. Two "
                f"independent mu sources are live; the run would silently train a mu-conditioned "
                f"network on the wrong conditioning. Refusing to continue.")

    def valid_rollout(self, sequence, n_steps=-1, bare=False, record_time=False, safecheck=True):

        record_time = True

        # train-in-loop a stale per-roll training mu must never leak into an inference rollout. Cleared
        # BEFORE _add_cloth_obj, because since that call resolves the bolt-on mu to
        # write the model's mu feature — with the reset after it, a stale _bo_train_mu would be the
        # value the network is conditioned on for the whole sequence.
        self._bo_train_mu = None

        PROF.begin_sequence()
        with PROF.stage('seq_init'):
            sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
            sequence = self._add_cloth_obj(sequence)

        is_obstacle = 'obstacle' in sequence.node_types
        if is_obstacle:
            n_samples = sequence['obstacle'].lookup.shape[1]
        else:
            n_samples = sequence['cloth'].lookup.shape[1]

        if n_steps >= 0:
            n_samples = min(n_samples, n_steps)

        from utils import trace
        trace.at('runners/ccraft.py', 'valid_rollout',
                 f'rolling {n_samples} steps, one GNN forward + contact operator each')

        trajectories_dicts = defaultdict(list)

        # Per-sequence reset: _last_bolt_on_force is a render-only side channel (see
        # _capture_step_force). It must NOT carry over from a previous sequence -- otherwise this
        # sequence's warmup frames (when _apply_bolt_on_friction early-returns without updating it)
        # would record the previous sequence's force tensor, which has a different vertex count and
        # breaks the torch.stack below. Reset touches only the recorded 'force' trajectory, never the
        # dynamics (friction is applied from the local pred_pos_new in _apply_bolt_on_friction).
        self._last_bolt_on_force = None

        if record_time:
            st_time = time.time()

        st = 0
        self.model.eval()
        # Problem-size counters, the analogue of ppf's num_contact / max_nnz_row: cost is meaningless
        # without them, because our sequences differ in mesh resolution as well as length.
        PROF.count('n_cloth_verts', sequence['cloth'].pos.shape[0])
        PROF.count('n_cloth_faces', sequence['cloth'].faces_batch.shape[1])
        if is_obstacle:
            PROF.count('n_body_verts', sequence['obstacle'].pos.shape[0])
            PROF.count('n_body_faces', sequence['obstacle'].faces_batch.shape[1])
        with PROF.stage('run_total'):
            trajectories, metrics_dict = self._rollout(sequence, st, n_samples - st,
                                                        progressbar=True, bare=bare, safecheck=safecheck)

        if record_time:
            total_time = time.time() - st_time
            metrics_dict['time'] = total_time

        tp = PROF.end_sequence()
        if tp is not None:
            metrics_dict['timeprof'] = tp

        # trajectories_dicts['pred'] = trajectory

        cloth_faces = sequence['cloth'].faces_batch.T.cpu().numpy()

        trajectories_dicts.update(trajectories)
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = cloth_faces

        if 'garment_id' in sequence['cloth']:
            trajectories_dicts['garment_id'] = sequence['cloth'].garment_id.cpu().numpy()

        trajectories_dicts['vertex_type'] = sequence['cloth'].vertex_type.cpu().numpy()

        if 'uv_coords' in sequence['cloth']:
            uv_faces = sequence['cloth'].uv_faces_batch.T.cpu().numpy()
            trajectories_dicts['uv_coords'] = sequence['cloth'].uv_coords.cpu().numpy()
            trajectories_dicts['uv_faces'] = uv_faces

        if is_obstacle:
            trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        for s in ['pred', 'obstacle', 'force']:
            if s in trajectories_dicts:
                trajectories_dicts[s] = torch.stack(trajectories_dicts[s], dim=0).cpu().numpy()
        return trajectories_dicts

    def _rollout(self, sample, start_idx, n_steps, progressbar=False, bare=False, safecheck=True):
        trajectories = defaultdict(list)
        is_obstacle = 'obstacle' in sample.node_types

        metrics_dict = defaultdict(list)

        if n_steps == 0:
            pbar = range(start_idx, start_idx + 1)
        else:
            pbar = range(start_idx, start_idx + n_steps)
        if progressbar:
            pbar = tqdm(pbar)

        prev_out_sample = None
        fail_reason = 'SUCCESS'

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with PROF.stage('seq_init'):
            sample = self.prepare_sample(sample)

        for i in pbar:
            with PROF.stage('collect_sample'):
                sample_step = self.collect_sample(sample, i, prev_out_sample)
            if i == 0:
                sample_step, sample = self.update_sample_1st_step(sample_step, sample)
                # Gated: unset => untouched. Applied after the runner's own
                # first-step setup so it wins over whatever the dataset placed
                # there, and before the first trajectory record below so the
                # written trajectory starts from the state actually simulated.
                init_file = getattr(self.mcfg, 'init_state_file', None)
                if init_file:
                    from tools.reference.gnn_init import apply_drape_init
                    sample_step = apply_drape_init(sample_step, init_file)

            if i == 0:
                trajectories['pred'].append(sample_step['cloth'].prev_pos)
                trajectories['pred'].append(sample_step['cloth'].pos)

                ncoll = self.safecheck_solver.calc_tritri_collisions2(sample_step, verts_key='pos')
                metrics_dict['ncoll'].append(ncoll)

                if is_obstacle:
                    trajectories['obstacle'].append(sample_step['obstacle'].prev_pos)
                    trajectories['obstacle'].append(sample_step['obstacle'].pos)

                if getattr(self, '_capture_force', False):
                    # two leading frames (prev_pos, pos) have no projection force -> zeros (frame-align)
                    zf = sample_step['cloth'].pos.new_zeros(sample_step['cloth'].pos.shape[0])
                    trajectories['force'].append(zf)
                    trajectories['force'].append(zf)

            if n_steps == 0:
                break

            with PROF.stage('model_fwd'):
                with TorchTimer(metrics_dict, 'hood_time', start=start, end=end):
                    sample_step = self.model(sample_step)
            # residual adapter: the mu-gated residual adapter is part of the NETWORK, so it applies on the
            # inference path too. Inert (byte-identical) unless enable_residual_adapter, and at mu=0
            # always. Must run BEFORE the operator: the operator acts on the adapted proposal.
            with PROF.stage('ia_r'):
                self._apply_ia_r(sample_step, metrics_dict)
            # bolt-on operator: bolt-on Coulomb friction projection on the frictionless tentative motion.
            # No-op (byte-identical to baseline) unless contact_arm.mode='bolt_on'. Runs BEFORE
            # safecheck so the collision detector and recorded trajectory see frictional positions.
            with PROF.stage('operator'):
                self._apply_bolt_on_friction(sample_step, step_idx=i)

            pred_pos = sample_step['cloth'].pred_pos
            if not torch.isfinite(pred_pos).all():
                # Rollout blew up to NaN/Inf (e.g. an untrained/short-only checkpoint driven past
                # its trained horizon) -- abort the sequence instead of feeding garbage geometry
                # into find_close_faces.
                fail_reason = f'DIVERGED@step{i}'
                break

            # EVAL INSTRUMENTATION, not the method: `ncoll` is a diagnostic counter, never fed back
            # into the dynamics. Timed as its own stage so it can be excluded from `method_total`
            # when quoting our cost against ppf's `solver_total` (whose `check_intersection` IS part
            # of the solve, and so is not the analogue of this).
            try:
                with PROF.stage('safecheck_ncoll'):
                    ncoll = self.safecheck_solver.calc_tritri_collisions2(sample_step, verts_key='pred_pos')
            except (MemoryError, torch.cuda.OutOfMemoryError) as e:
                # A divergent rollout can also crumple into a self-intersecting tangle dense
                # enough that find_close_faces' BVH broad phase OOMs, even with finite/bounded
                # positions (observed on short-only-phase checkpoints run past their horizon).
                # Abort this sequence instead of crashing the whole eval run.
                torch.cuda.empty_cache()
                fail_reason = f'OOM@step{i}: {str(e)[:200]}'
                break
            metrics_dict['ncoll'].append(ncoll)

            with PROF.stage('record'):
                prev_out_sample = sample_step.detach()

                trajectories['pred'].append(prev_out_sample['cloth'].pred_pos)
                if is_obstacle:
                    trajectories['obstacle'].append(prev_out_sample['obstacle'].target_pos)

            if getattr(self, '_capture_force', False):
                f = self._capture_step_force(prev_out_sample)
                if f is None:  # neither bolt-on operator nor trained-friction arm applies -> zeros
                    f = prev_out_sample['cloth'].pred_pos.new_zeros(prev_out_sample['cloth'].pred_pos.shape[0])
                trajectories['force'].append(f)

            if not bare:
                loss_dict_hood, loss_weight_dict_hood, _, _ = self.criterion_pass(sample_step, self.criterion_dict)
                metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood, None, None,
                                                metrics_dict)

        metrics_dict['fail_reason'] = fail_reason

        return trajectories, metrics_dict

    # --------------------------------------------------------------------------------------------- #
    # bolt-on operator: inference-time bolt-on Coulomb friction (cubic-barrier proposal). #
    # --------------------------------------------------------------------------------------------- #

    def _contact_arm_cfg(self):
        return getattr(self.mcfg, 'contact_arm', None)

    def contact_arm_mode(self):
        ca = self._contact_arm_cfg()
        if ca is None:
            return 'none'
        return getattr(ca, 'mode', 'none')

    def has_force_source(self) -> bool:
        """True if this runner/config can produce a real (non-zero) per-vertex contact force during
        rollout, from either friction arm:
          - bolt-on operator (contact_arm.mode='bolt_on'): the inference-time friction-projection force.
          - trained-friction arm (arm_br.enabled): loss-only grav+inertia lambda_n -- contact_arm.mode stays
            'none' (no inference-time operator), but the force is still capturable read-only via the
            attached friction_dissipative criterion (see _capture_step_force).
        Callers use this to decide whether to set _capture_force, and to warn when --render-force was
        requested but neither arm applies (nothing to render but zeros)."""
        if self.contact_arm_mode() == 'bolt_on':
            return True
        abr = getattr(self.mcfg, 'arm_br', None)
        return abr is not None and abr.enabled

    def _capture_step_force(self, state):
        """(V,) per-vertex contact-force magnitude for this step's --render-force capture, or None
        if there is nothing to report (caller falls back to zeros). Checks both friction arms:
          - bolt-on operator: reads _last_bolt_on_force, set by _apply_bolt_on_friction just before this runs.
          - trained-friction arm: contact_arm.mode='none' so no projection ran -- instead ask the attached
            friction_dissipative criterion for its read-only grav+inertia lambda_n (same contact
            geometry the loss uses; see criterions/ccraft/friction_dissipative.py::compute_render_force
            and has_force_source's docstring)."""
        f = getattr(self, '_last_bolt_on_force', None)
        # Shape guard: only trust _last_bolt_on_force if it matches the current cloth vertex count.
        # During BO warmup (_apply_bolt_on_friction early-returns without updating it) or across a
        # sequence boundary, a stale tensor of a different length would otherwise be recorded and
        # break the torch.stack of the 'force' trajectory. A mismatch/None -> fall through to zeros
        # (correct: no friction projection ran this step, so the render force is zero).
        if f is not None and f.shape[0] == state['cloth'].pred_pos.shape[0]:
            return f
        abr = getattr(self.mcfg, 'arm_br', None)
        if abr is not None and abr.enabled:
            crit = self.criterion_dict.get('friction_dissipative')
            if crit is not None:
                return crit.compute_render_force(state)
        return None

    def _resolve_bolt_on_mu(self, state, bo):
        """mu priority: train-in-loop per-roll sample > per-sample cloth.friction_mu >
        material.friction_mu_override > bo.mu fallback."""
        # train-in-loop set only inside an active training roll by _bo_train_roll_start,
        # reset to None at roll end and defensively in valid_rollout — inference is untouched.
        if self._bo_train_mu is not None:
            return float(self._bo_train_mu)
        if 'friction_mu' in state['cloth']:
            return state['cloth'].friction_mu  # (B,) per-sample (varying-mu models)
        override = getattr(self.mcfg.material, 'friction_mu_override', None)
        if override is not None:
            return float(override)
        if bo.mu is not None:
            return float(bo.mu)
        raise ValueError(
            "bolt-on operator is enabled but no friction coefficient is set. Pass "
            "runner.ccraft.material.friction_mu_override=<value> on the CLI or set "
            "runner.ccraft.contact_arm.bolt_on.mu in the config.")

    def _apply_bolt_on_friction(self, state, step_idx=None, want_action=False):
        """Overwrite cloth.pred_pos and pred_velocity with the friction-projected prediction.
        Contact vertices only; non-contact and pinned are left byte-identical. Returns diag or None.

        : with want_action=True the projection additionally returns the differentiable
        dissipation D it just paid (and freezes lambda_n); the caller then gets (diag, action).
        Everything else about the call is IDENTICAL, which is what keeps the 'action' arm's
        transition byte-parity with the 'stop_grad' arm and with inference (guard 2): there is
        exactly ONE operator call site in this runner."""
        from utils import trace
        if trace.once('bo_hop'):
            trace.at('runners/ccraft.py', '_apply_bolt_on_friction',
                     'handing each frictionless prediction to '
                     'utils/contact.py (bolt_on_friction_projection)')
        ca = self._contact_arm_cfg()
        mode = self.contact_arm_mode()
        if mode == 'none':
            return None
        if mode == 'trained':
            raise ValueError(
                "contact_arm.mode='trained' is not a valid inference mode: trained-friction arm has NO "
                "inference-time friction operator — friction lives only in the training loss "
                "(criterions/ccraft/friction_dissipative.py) and mu is carried by the model's "
                "body-edge feature. For a BR checkpoint use contact_arm.mode='none' (plain forward "
                "pass) with arm_br.enabled=True and --mu / material.friction_mu_override.")
        if mode != 'bolt_on':
            raise ValueError(f"Unknown contact_arm.mode: {mode!r}. Expected 'none', 'bolt_on', or 'trained'.")

        bo = ca.bolt_on
        # LBS-init warm-up: the initial garment state is far from membrane equilibrium; let the
        # first N steps relax friction-free rather than feed the transient into the cone. 0 = off.
        if step_idx is not None and step_idx < int(getattr(bo, 'warmup_steps', 0) or 0):
            return None
        mu = self._resolve_bolt_on_mu(state, bo)
        # the network's mu feature must show exactly this mu. Inert unless the
        # model's mu feature is on, so earlier arms are byte-identical.
        self._assert_single_mu(state, mu)
        do_print = bool(bo.log_every) and (step_idx is None or step_idx % bo.log_every == 0)
        collect = getattr(self, '_bolt_on_diag_log', None) is not None
        want_diag = do_print or collect
        # choose the friction cone's λ_n source. 'barrier' (default) is byte-identical to v10.
        source = getattr(bo, 'lambda_n_source', 'barrier')
        elastic_grad = None
        if source == 'force_balance' and getattr(bo, 'fb_use_elastic', False):
            elastic_grad = self._force_balance_elastic_grad(state)   # ∇(E_stretch+E_bend), detached (Step 2)
        want_force = bool(getattr(self, '_capture_force', False))  # --render-force capture (read-only)
        out = C.bolt_on_friction_projection(
            state, mu, eps=bo.eps, stiffness=bo.stiffness, eps_v=bo.eps_v, z_axis=bo.z_axis,
            g=getattr(bo, 'fb_g', 9.81), return_diag=want_diag, return_force=want_force,
            lambda_n_source=source, fb_use_gravity=getattr(bo, 'fb_use_gravity', True),
            fb_use_inertia=getattr(bo, 'fb_use_inertia', True), elastic_grad=elastic_grad,
            fb_elastic_conform=getattr(bo, 'fb_elastic_conform', False),
            fb_elastic_pen_gate=getattr(bo, 'fb_elastic_pen_gate', False),
            fb_elastic_smooth_iters=getattr(bo, 'fb_elastic_smooth_iters', 0),
            fb_elastic_pen_dilate=getattr(bo, 'fb_elastic_pen_dilate', False),
            fb_elastic_approach_gate=getattr(bo, 'fb_elastic_approach_gate', False),
            fb_elastic_noise_floor_q=getattr(bo, 'fb_elastic_noise_floor_q', 0.0),
            fb_elastic_deflate_profile=getattr(bo, 'fb_elastic_deflate_profile', False),
            lambda_smooth_iters=getattr(bo, 'lambda_smooth_iters', 0),
            lambda_smooth_window=getattr(bo, 'lambda_smooth_window', 0.0),
            lambda_smooth_conserve=getattr(bo, 'lambda_smooth_conserve', True),
            lambda_smooth_render_only=getattr(bo, 'lambda_smooth_render_only', False),
            normal_exact=getattr(bo, 'normal_exact', 'off'),
            normal_exact_iters=getattr(bo, 'normal_exact_iters', 2),
            normal_exact_slack=getattr(bo, 'normal_exact_slack', 1e-4),
            normal_exact_max_step=getattr(bo, 'normal_exact_max_step', 0.0),
            normal_exact_dust=getattr(bo, 'normal_exact_dust', 1e-8),
            normal_exact_ambig_k=getattr(bo, 'normal_exact_ambig_k', 4),
            normal_exact_ambig_cos=getattr(bo, 'normal_exact_ambig_cos', 0.0),
            normal_exact_ambig_ratio=getattr(bo, 'normal_exact_ambig_ratio', 1.5),
            normal_exact_control=getattr(bo, 'normal_exact_control', False),
            tangential_path=getattr(bo, 'tangential_path', 'chord'),
            path_substeps=getattr(bo, 'path_substeps', 8),
            path_arc_cone=getattr(bo, 'path_arc_cone', 'off'),
            path_arc_cap=getattr(bo, 'path_arc_cap', 3.0),
            path_normals=getattr(bo, 'path_normals', 'single'),
            path_capstan=getattr(bo, 'path_capstan', 'off'),
            path_walk_max_cross=getattr(bo, 'path_walk_max_cross', 16),
            path_walk_fast=getattr(bo, 'path_walk_fast', False),
            drag_k=getattr(bo, 'drag_k', 0.0),
            spread_w=getattr(bo, 'spread_w', 0.0),
            spread_project=getattr(bo, 'spread_project', 'off'),
            spread_hops=getattr(bo, 'spread_hops', 1),
            spread_gamma=getattr(bo, 'spread_gamma', 0.0),
            spread_weight=getattr(bo, 'spread_weight', 'uniform'),
            patch_iters=getattr(bo, 'patch_iters', 0),
            patch_depth=getattr(bo, 'patch_depth', 1),
            patch_alpha_scale=getattr(bo, 'patch_alpha_scale', 1.0),
            patch_signorini=getattr(bo, 'patch_signorini', 'preserve'),
            patch_feas_tol=getattr(bo, 'patch_feas_tol', 1e-6),
            wave_enabled=getattr(bo, 'wave_enabled', False),
            wave_visco=getattr(bo, 'wave_visco', 0.0),
            wave_tau_frames=getattr(bo, 'wave_tau_frames', 0.5),
            wave_strain_cap=getattr(bo, 'wave_strain_cap', 0.0),
            detach_lambda_n=bool(want_action), return_action=bool(want_action))
        # unpack (pred[, diag][, force][, metric_log][, action]) — the callee's fixed order.
        # (return_metric_log is never requested from this call site.)
        outs = list(out) if isinstance(out, tuple) else [out]
        pred_pos_new = outs.pop(0)
        diag = outs.pop(0) if want_diag else None
        force = outs.pop(0) if want_force else None
        action = outs.pop(0) if want_action else None
        if want_force:
            self._last_bolt_on_force = force.detach()
        cloth = state['cloth']
        # pred_velocity must stay consistent with pred_pos so the next step's velocity (read by
        # SampleCollector.add_velocity from pred_velocity) reflects the frictional position.
        cloth.pred_velocity = pred_pos_new - cloth.pos
        cloth.pred_pos = pred_pos_new
        if diag is not None:
            if collect:
                self._bolt_on_diag_log.append(diag)
            if do_print:
                self._log_bolt_on_diag(diag, mu, step_idx)
        if want_action:
            return diag, action
        return diag

    # --------------------------------------------------------------------------------------------- #
    # train-in-loop (BO-in-the-loop on-policy fine-tuning with stop-gradient operator #
    # transitions. The per-step loss stays on the RAW proposal; the operator is applied to the       #
    # DETACHED post-loss state so the next step trains on the frictional state distribution.         #
    # Training and inference execute the same composite transition P_mu . N_theta; no gradient       #
    # flows through the operator. #
    # --------------------------------------------------------------------------------------------- #

    def _bo_train_cfg(self):
        """The bolt_on sub-config iff BO-in-the-loop training is active, else None."""
        ca = self._contact_arm_cfg()
        if ca is None or getattr(ca, 'mode', 'none') != 'bolt_on':
            return None
        bo = ca.bolt_on
        return bo if getattr(bo, 'train_in_loop', False) else None

    def _validate_train_in_loop(self):
        """Fail at construction on configs that would corrupt the train-in-loop probe (—
        operator ownership — plus mu-source sanity). Mirrors _validate_arm_br."""
        bo = self._bo_train_cfg()
        if bo is None:
            return
        if not getattr(bo, 'skip_init_steps', False):
            raise ValueError(
                "train-in-loop (train_in_loop) requires skip_init_steps=true — the initial_ts macro-step "
                "inflates the cone bound R ∝ h² by ~100x and would freeze every gated contact "
                "(verify finding: without the flag, init steps also bypass "
                "the warm-up counter entirely).")
        if 'friction_dissipative' in self.criterion_dict:
            raise ValueError(
                "train-in-loop (train_in_loop) forbids ccraft.friction_dissipative in the criterion set — "
                "the operator owns the friction law exclusively.")
        abr = self._arm_br_cfg()
        if abr is not None and getattr(abr, 'enabled', False):
            raise ValueError("train-in-loop (train_in_loop) with arm_br.enabled=True — remove one.")
        # criterion_dict is REKEYED to criterion.name by create_criterion_dicts
        # (utils/arguments.py:209-210), NOT the dotted config key — so the key is 'friction_energy'
        # (criterions/postcvpr/friction_energy.py:36), not 'postcvpr.friction_energy'. (A bug
        # once found in review: the dotted lookup silently returned None and this whole
        # guard was dead; test T16 pins the runtime key against the real config path.)
        fe = self.criterion_dict.get('friction_energy')
        if fe is not None:
            fe_mu = getattr(fe, 'mu', None)                      # Criterion stores mu directly
            if fe_mu is None:
                fe_mu = getattr(getattr(fe, 'mcfg', None), 'mu', None)
            if fe_mu is None:
                raise ValueError(
                    "train-in-loop cannot read friction_energy's mu to verify it is 0.0 "
                    "(criterion layout changed?) — refusing to run unverified.")
            if float(fe_mu) != 0.0:
                raise ValueError(
                    f"train-in-loop (train_in_loop) requires friction_energy mu==0.0 (dead term, as in the"
                    f"seed recipe) but got mu={float(fe_mu)} — a live friction loss would "
                    f"double-teach the law.")
        override = getattr(self.mcfg.material, 'friction_mu_override', None)
        if getattr(bo, 'mu_sample', False):
            if override is not None:
                raise ValueError(
                    "train-in-loop material.friction_mu_override is set but bolt_on.mu_sample=True — the "
                    "per-roll sampler wins (_resolve_bolt_on_mu checks _bo_train_mu FIRST) so the "
                    "override would be silently IGNORED, not honored. Ambiguous intent — unset one.")
            if not (0.0 <= bo.mu_sample_p_zero <= 1.0):
                raise ValueError(f"train-in-loop mu_sample_p_zero={bo.mu_sample_p_zero} outside [0,1].")
            if not (0.0 < bo.mu_sample_lo < bo.mu_sample_hi):
                raise ValueError(
                    f"train-in-loop need 0 < mu_sample_lo < mu_sample_hi, got"
                    f"[{bo.mu_sample_lo}, {bo.mu_sample_hi}].")
        elif override is None and bo.mu is None:
            raise ValueError(
                "train-in-loop train_in_loop with mu_sample=False needs a mu source — set bolt_on.mu or "
                "material.friction_mu_override (or enable mu_sample).")
        # --- composite action objective -------------------------------------------
        mode = self._bo_train_loss_mode(bo)
        if mode not in ('stop_grad', 'action', 'impulse_aware'):
            raise ValueError(
                f"train-in-loop unknown contact_arm.bolt_on.train_loss_mode={mode!r} (expected 'stop_grad' "
                f"= loss-on-raw-proposal, 'action' = composite action objective, or"
                f"'impulse_aware' = straight-through operated-state loss).")
        if mode == 'impulse_aware' and 'inertia' not in self.criterion_dict:
            raise ValueError(
                " (train_loss_mode='impulse_aware') requires the 'inertia' criterion: the "
                " inertia-reference-shift identity is implemented as an inertia re-pass on the "
                "raw proposal, so a recipe without inertia would silently train a different "
                "objective than the one specified.")
        if mode == 'action':
            w = float(getattr(bo, 'action_weight', 1.0))
            if w < 0:
                raise ValueError(f"train-in-loop action_weight={w} must be >= 0.")
            if w == 0.0:
                raise ValueError(
                    "train-in-loop train_loss_mode='action' with action_weight=0 is exactly the NAIVE "
                    "post-operator loss rejected by construction in (affinely invertible "
                    "slide branch => cancellation directly rewarded; identically flat stick basin). "
                    "Set action_weight=1.0 (the physically correct pricing) or use "
                    "train_loss_mode='stop_grad'.")
            if getattr(bo, 'lambda_smooth_render_only', False) is False and \
                    int(getattr(bo, 'lambda_smooth_iters', 0) or 0) > 0:
                # The mollified field drives the cone; it is assembled from detached inputs either
                # way, so this is a note rather than a hazard — but it changes what sg[lambda_n]
                # means, so make the operator's own telemetry the record.
                print("[train-in-loop] note: lambda_smooth_iters>0 — the dissipation term prices the "
                      "SMOOTHED lambda_n (same field the cone uses).", flush=True)

    def _sample_train_mu(self, bo):
        """Per-roll mu from the D1 mixture: p(mu=0)=p_zero, else LogUniform[lo, hi]. Uses a DEDICATED
        numpy Generator (never the global stream) so the OP and CTRL twin runs consume identical
        global-RNG sequences — data order and material sampling stay in lockstep (guard 3)."""
        if self._bo_mu_rng is None:
            self._bo_mu_rng = np.random.default_rng(int(getattr(bo, 'mu_sample_seed', 57)))
        rng = self._bo_mu_rng
        if rng.random() < float(bo.mu_sample_p_zero):
            return 0.0
        lo, hi = float(bo.mu_sample_lo), float(bo.mu_sample_hi)
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    @staticmethod
    def _bo_mu_bin(mu):
        """Fixed mu-bin tag for per-bin canary trend reads (mu is constant within a roll)."""
        if mu == 0.0:
            return 'mu0'
        if mu <= 0.2:
            return 'mulo'
        if mu <= 0.45:
            return 'mumid'
        return 'muhi'

    def _bo_train_roll_start(self, roll_type):
        """Arm the per-roll train-in-loop state: sample this roll's mu (D1), reset counters and the
        guard-5 conditional flag. No-op (state=None) when train-in-loop is off."""
        bo = self._bo_train_cfg()
        self._bo_prev_altered = False
        self._bo_train_mu = None
        self._bo_ia_raw_z = None
        if bo is None:
            self._bo_train_roll = None
            return
        mu = self._sample_train_mu(bo) if getattr(bo, 'mu_sample', False) else None
        if mu is not None:
            self._bo_train_mu = mu
        mu_eff = mu if mu is not None else self._resolve_bolt_on_mu_fallback(bo)
        self._bo_train_roll = {
            'roll_type': roll_type, 'mu': mu_eff, 'bin': self._bo_mu_bin(mu_eff),
            'applied': 0, 'altered_steps': 0, 'skipped_init': 0, 'skipped_warmup': 0,
            'skipped_terminal': 0, 'regular_seen': 0, 'skipped_nonfinite': 0,
        }

    def _resolve_bolt_on_mu_fallback(self, bo):
        """Roll-level mu when the sampler is off (override > bo.mu), for binning/logging only."""
        override = getattr(self.mcfg.material, 'friction_mu_override', None)
        if override is not None:
            return float(override)
        return float(bo.mu) if bo.mu is not None else 0.0

    @staticmethod
    def _bo_train_loss_mode(bo):
        """'stop_grad' (train-in-loop the default everywhere), 'action', or
        'impulse_aware'."""
        if bo is None:
            return 'stop_grad'
        return str(getattr(bo, 'train_loss_mode', 'stop_grad') or 'stop_grad')

    def _bo_train_step_gate(self, cloth, st, bo, is_last, terminal_skip):
        """Shared operator-gating predicate for BOTH train-in-loop loss modes.
        True => the operator runs on this step. Otherwise the skip is recorded on the per-roll state,
        the guard-5 'previous step was altered' flag is cleared, and False is returned.

        `terminal_skip` is the ONLY difference between the modes, and it follows from where the
        operator sits relative to the loss:
          * 'stop_grad' (True): the operator runs AFTER the loss, so its output is read only by the
            NEXT step — on a roll's last step nothing consumes it (verify finding: dead
            compute that would inflate the exposure counters without any training effect).
          * 'action' (False): the operator runs BEFORE the loss, so its output IS consumed on every
            step it runs — by the loss itself. Skipping the terminal step would silently discard one
            training signal per roll.
        Everything else (init skip, warm-up counting) is shared verbatim, so the two arms visit the
        same operator-altered state distribution step for step; only the loss differs. Note the
        terminal step never increments `regular_seen` under 'stop_grad' because the terminal check
        precedes the warm-up counter — 'action' increments it there instead, which cannot change any
        gating decision (the warm-up window has always closed by then)."""
        if is_last and terminal_skip:
            st['skipped_terminal'] += 1
            self._bo_prev_altered = False
            return False
        is_init = bool('is_init' in cloth and float(cloth.is_init.reshape(-1)[0].item()) > 0)
        if getattr(bo, 'skip_init_steps', False) and is_init:
            st['skipped_init'] += 1
            self._bo_prev_altered = False
            return False
        if not is_init:
            reg_idx = st['regular_seen']
            st['regular_seen'] += 1
            if reg_idx < int(getattr(bo, 'warmup_regular_steps', 0) or 0):
                st['skipped_warmup'] += 1
                self._bo_prev_altered = False
                return False
        return True

    def _bo_record_alteration(self, st, bo, raw_pred, cloth, diag, metrics_dict):
        """Exposure/canary bookkeeping shared by both loss modes, so the arms' wandb keys are
        directly comparable. Detached throughout."""
        dust = float(getattr(bo, 'train_altered_dust', 1e-7))
        with torch.no_grad():
            dx = (cloth.pred_pos - raw_pred).norm(dim=-1)
            altered_mask = dx > dust
            n_altered = int(altered_mask.sum().item())
            altered_frac = n_altered / max(1, dx.shape[0])
            dx_mean = float(dx[altered_mask].mean().item()) if n_altered else 0.0
        if n_altered:
            st['altered_steps'] += 1
        self._bo_prev_altered = n_altered > 0
        b = st['bin']
        metrics_dict['bo_tt/applied'].append(1.0)
        metrics_dict['bo_tt/altered_frac'].append(float(altered_frac))
        metrics_dict['bo_tt/altered_dx_mean'].append(dx_mean)
        metrics_dict[f'bo_tt/{b}/altered_frac'].append(float(altered_frac))
        if diag is not None:
            for k in ('contact_frac', 'stick_frac', 'factor_mean', 'slip_pre_mean',
                      'slip_post_mean', 'slip_redux', 'lambda_n_mean'):
                metrics_dict[f'bo_tt/{k}'].append(float(diag[k]))
            for k in ('stick_frac', 'factor_mean', 'slip_pre_mean'):
                metrics_dict[f'bo_tt/{b}/{k}'].append(float(diag[k]))
            # -TC observability (D5): completion counters, so training-time skip/revert behavior
            # is visible in wandb (ambiguity/over-cap skips leave gap erosion UNREPAIRED there).
            # adds sm_* (surface-march marched/fallback/arc-inflation) for the same reason.
            # adds mn_* (multi-normal landing moved/junction/spread/wrap) for the same reason.
            for k in ('ne_candidates', 'ne_corrected', 'ne_skip_ambig', 'ne_skip_cap',
                      'ne_residual', 'ne_face_switch',
                      'sm_marched', 'sm_fallback_ambig', 'sm_fallback_cap', 'sm_arc_infl_max',
                      'mn_active', 'mn_landing_moved', 'mn_junction_land', 'mn_fullstick',
                      'mn_lambda_spread_max', 'mn_capstan_gain_max'):
                if k in diag:
                    metrics_dict[f'bo_tt/{k}'].append(float(diag[k]))

    # --------------------------------------------------------------------------------------------- #
    # (the composite ACTION objective. The operator runs BEFORE the loss and #
    # INSIDE the graph; the backbone loss is evaluated on the shipped frame x^+ and the dissipation  #
    # the operator paid is added as a term. One application serves both the loss and the transition, #
    # so the state distribution matches 'stop_grad' exactly and only the loss changes.               #
    # --------------------------------------------------------------------------------------------- #

    def _bo_probe_losses(self, sample_step):
        """collision/stretch losses of whatever currently sits in cloth.pred_pos (no grad)."""
        out = {}
        sample_step.cloth_obj = self.cloth_obj
        with torch.no_grad():
            for name in ('collision_penalty', 'stretching_energy'):
                crit = self.criterion_dict.get(name)
                if crit is None:
                    continue
                ld = crit(sample_step)
                v = ld.get('loss') if isinstance(ld, dict) else None
                if v is not None:
                    out[name] = float(v.detach())
        return out

    def _bo_action_pre_loss(self, sample_step, metrics_dict, is_last=False):
        """Apply the operator in-graph ahead of the criterion pass and return the differentiable
        dissipation term D (already /B and weighted), or None when this step is not operator-active
        or the arm is not in 'action' mode. cloth.pred_pos/pred_velocity are left holding x^+, so the
        criterion pass that follows scores the SHIPPED frame and `sample_step.detach()` downstream
        already carries the operated state — which is why the 'action' arm must NOT also run
        _bolt_on_train_transition (it early-returns on this mode)."""
        st = self._bo_train_roll
        if st is None:
            return None
        bo = self._bo_train_cfg()
        if self._bo_train_loss_mode(bo) != 'action':
            return None
        cloth = sample_step['cloth']
        if not torch.isfinite(cloth.pred_pos).all():
            # The proposal has already diverged; the explosion guard aborts this roll a few lines
            # later, on the loss. Running the operator on non-finite geometry first would only risk
            # a hard crash inside its KNN / quantile calls instead of a clean, counted guard trip.
            # ('stop_grad' never sees such a state either — there the guard breaks the roll BEFORE
            # the post-loss transition runs, so this keeps the two modes' failure behaviour matched.)
            st['skipped_nonfinite'] += 1
            self._bo_prev_altered = False
            return None
        if not self._bo_train_step_gate(cloth, st, bo, is_last, terminal_skip=False):
            return None
        every = int(getattr(bo, 'action_raw_probe_every', 0) or 0)
        # Counter must be GLOBAL, not the per-roll st['applied'] — that one is reset to 0 by
        # _bo_train_roll_start, so `% every == 0` would be true on the FIRST operator-active step of
        # every roll regardless of `every`. That both overruns the intended duty cycle ~10x and,
        # worse, samples only one step-position per roll: a TC leak that develops on later in-roll
        # applications (where the operator's repairs have compounded over frames it already shipped
        # — precisely where "the net learns to dive and let the clamp fix it" would show) would be
        # structurally invisible to its own canary.
        self._bo_action_applied_total = getattr(self, '_bo_action_applied_total', 0)
        probe_now = every > 0 and (self._bo_action_applied_total % every) == 0
        raw_probe = self._bo_probe_losses(sample_step) if probe_now else None
        raw_pred = cloth.pred_pos          # x^ — reassigned (never mutated) by the call below
        prev_sink = getattr(self, '_bolt_on_diag_log', None)
        self._bolt_on_diag_log = []        # force want_diag inside _apply_bolt_on_friction
        try:
            diag, action = self._apply_bolt_on_friction(sample_step, step_idx=None, want_action=True)
        finally:
            self._bolt_on_diag_log = prev_sink
        st['applied'] += 1
        self._bo_action_applied_total += 1
        self._bo_record_alteration(st, bo, raw_pred, cloth, diag, metrics_dict)
        B = sample_step.num_graphs
        weight = float(getattr(bo, 'action_weight', 1.0))
        D = action / B * weight
        metrics_dict['ao/D'].append(float(D.detach()))
        if diag is not None:
            metrics_dict['ao/D_raw'].append(float(diag.get('action_D', 0.0)))
            metrics_dict['ao/slip_post_mean'].append(float(diag.get('action_slip_post_mean', 0.0)))
            metrics_dict['ao/n_active'].append(float(diag.get('action_n_active', 0)))
            metrics_dict[f"ao/{st['bin']}/D"].append(float(D.detach()))
        if raw_probe is not None:
            post_probe = self._bo_probe_losses(sample_step)
            for key, short in (('collision_penalty', 'collision'), ('stretching_energy', 'stretch')):
                if key in raw_probe:
                    metrics_dict[f'ao/raw_{short}'].append(raw_probe[key])
                if key in post_probe:
                    metrics_dict[f'ao/post_{short}'].append(post_probe[key])
        return D

    # --------------------------------------------------------------------------------------------- #
    # the impulse-aware straight-through objective. The operator #
    # runs under no_grad; the criteria see the operated configuration through a stop-gradient      #
    # correction (identity Jacobian); inertia keeps reading the raw proposal (the reference- #
    # shift identity); NO dissipation term; the transition ships the Signorini-completed #
    # state at mu>0 (train-what-you-deploy) while mu=0 stays bitwise the 'stop_grad' path.         #
    # --------------------------------------------------------------------------------------------- #

    def _bo_impulse_pre_loss(self, sample_step, metrics_dict, is_last=False):
        """ pre-loss hook. When active, replaces cloth.pred_pos with the straight-through
        state x~ = z + sg[P_mu(z) - z] (numerically the operated state, identity Jacobian) and
        returns the in-graph RAW proposal z so the caller can re-evaluate inertia on it. Returns
        None when the arm is not in 'impulse_aware' mode or this step is not operator-active.
        Gating is _bo_train_step_gate with terminal_skip=False: like 'action', the operator output
        is consumed by the LOSS on every step it runs, including the roll's last. The detached raw
        proposal is stashed for the TC transition (consumed there; reset here so a skipped step can
        never ship a stale one)."""
        self._bo_ia_raw_z = None
        st = self._bo_train_roll
        if st is None:
            return None
        bo = self._bo_train_cfg()
        if self._bo_train_loss_mode(bo) != 'impulse_aware':
            return None
        cloth = sample_step['cloth']
        if not torch.isfinite(cloth.pred_pos).all():
            # mirror _bo_action_pre_loss: the explosion guard aborts on the loss a few lines later
            st['skipped_nonfinite'] += 1
            self._bo_prev_altered = False
            return None
        if not self._bo_train_step_gate(cloth, st, bo, is_last, terminal_skip=False):
            return None
        raw_pred = cloth.pred_pos                    # z, in the graph (reassigned below, not mutated)
        prev_sink = getattr(self, '_bolt_on_diag_log', None)
        self._bolt_on_diag_log = []                  # force want_diag inside _apply_bolt_on_friction
        try:
            with torch.no_grad():
                diag = self._apply_bolt_on_friction(sample_step, step_idx=None)
        finally:
            self._bolt_on_diag_log = prev_sink
        x_T = cloth.pred_pos                         # P_mu(z), detached (no_grad application)
        c = x_T - raw_pred.detach()                  # sg[P_mu(z) - z]
        x_tilde = raw_pred + c                       # numerically x_T; d(x_tilde)/dz = I
        cloth.pred_pos = x_tilde
        cloth.pred_velocity = x_tilde - cloth.pos
        st['applied'] += 1
        self._bo_record_alteration(st, bo, raw_pred.detach(), cloth, diag, metrics_dict)
        self._bo_ia_raw_z = raw_pred.detach()        # for the TC transition (mu>0)
        with torch.no_grad():
            cn = c.norm(dim=-1)
            metrics_dict['ia/c_mean'].append(float(cn.mean()))
            metrics_dict['ia/c_max'].append(float(cn.max()))
            metrics_dict[f"ia/{st['bin']}/c_mean"].append(float(cn.mean()))
        return raw_pred

    def _bo_impulse_fix_inertia(self, sample_step, raw_pred, loss_dict_hood):
        """Replace the inertia entry computed on x~ with inertia on the RAW z — exactly the
        inertia-reference-shift identity L_inertia(x~, x_hat + c) == L_inertia(z, x_hat), realized
        with ZERO criterion edits by re-running the unchanged criterion on the other state. The loss
        state is restored afterwards so downstream consumers (ncoll telemetry, detach) see x~."""
        crit = self.criterion_dict.get('inertia')
        if crit is None:                             # refused at init; belt-and-braces
            return loss_dict_hood
        cloth = sample_step['cloth']
        x_tilde = cloth.pred_pos
        cloth.pred_pos = raw_pred
        try:
            ld = crit(sample_step)
        finally:
            cloth.pred_pos = x_tilde
        for k, v in ld.items():                      # mirror criterion_pass keying exactly
            if 'loss' in k:
                loss_dict_hood[f'inertia_{k}'] = v
        return loss_dict_hood

    def _bo_impulse_tc_transition(self, prev_out_sample, metrics_dict, is_last=False):
        """ transition: the next rollout state is the Signorini-COMPLETED
        operator output (z) — 'preserve_clamp' run on the same detached raw proposal the
        pre-loss operator consumed — so the training transition matches the deployed inference
        operator while the LOSS stayed on the pre-clamp x_T (the clamp must never hide the net's
        penetration from the collision loss). At mu=0 the state is left as-is: x~ is bitwise z
        there, which keeps the mu=0 anchor and the B-series regression bitwise against 'stop_grad'.
        The stash-driven trigger means the step gate is never consulted twice (its counters and the
        guard-5 flag mutate on read)."""
        st = self._bo_train_roll
        raw_z = getattr(self, '_bo_ia_raw_z', None)
        self._bo_ia_raw_z = None
        if st is None or raw_z is None:
            return
        if is_last:
            # the terminal state is never consumed — the TC pass would be dead compute
            metrics_dict['ia/tc_skipped_terminal'].append(1.0)
            return
        if float(st['mu']) == 0.0:
            return                                   # bitwise stop_grad parity at the anchor
        bo = self._bo_train_cfg()
        cloth = prev_out_sample['cloth']
        x_T = cloth.pred_pos                         # detached x~ (== P_mu(z))
        cloth.pred_pos = raw_z
        cloth.pred_velocity = raw_z - cloth.pos
        saved_ne = bo.normal_exact
        try:
            bo.normal_exact = 'preserve_clamp'       # -> for this one application
            with torch.no_grad():
                self._apply_bolt_on_friction(prev_out_sample, step_idx=None)
        finally:
            bo.normal_exact = saved_ne
        with torch.no_grad():
            tc_delta = (cloth.pred_pos - x_T).norm(dim=-1)
            metrics_dict['ia/tc_applied'].append(1.0)
            metrics_dict['ia/tc_delta_max'].append(float(tc_delta.max()))
            metrics_dict['ia/tc_moved_frac'].append(
                float((tc_delta > 1e-9).float().mean()))

    # --------------------------------------------------------------------------------------------- #
    # residual adapter: the mu-gated residual adapter on a FROZEN backbone. #
    #   z = z0 + rho(mu) * A_phi(s, mu, q),  rho(0) = 0,  q = DETACHED operator pre-pass diagnostics #
    # Applied by the runner (not inside Model.forward) because q needs the operator, and the runner  #
    # owns the single operator call site. Called from forward_short / forward_long / _rollout right  #
    # after the model forward, so the adapter is part of the NETWORK at train and test alike.        #
    # --------------------------------------------------------------------------------------------- #

    def _ia_r_adapter(self):
        """The adapter module, or None when this arm is off (every pre- config)."""
        model = getattr(self, 'model', None)
        return getattr(model, 'residual_adapter', None) if model is not None else None

    def _ia_r_rho(self, mu, sample_step):
        """rho(mu) = min(mu / rho_mu_max, rho_cap) as a (V,1) tensor, so a per-sample (B,) mu works
        the same way a global float does. rho(0) = 0 EXACTLY — that identity is the arm's structural
        mu=0 anchor, so it is computed here and nowhere else."""
        mcfg = self.model.mcfg
        cloth = sample_step['cloth']
        span = float(getattr(mcfg, 'adapter_rho_mu_max', 0.6))
        cap = float(getattr(mcfg, 'adapter_rho_cap', 2.0))
        if torch.is_tensor(mu):
            if mu.numel() == 1:
                rho_v = cloth.pred_pos.new_full((cloth.pred_pos.shape[0], 1), float(mu) / span)
            else:
                batch = getattr(cloth, 'batch', None)
                if batch is None:
                    raise ValueError(
                        'residual adapter: per-sample mu given but cloth has no `batch` index to expand it '
                        'per vertex. Use a global mu (material.friction_mu_override / bolt_on.mu) '
                        'or the per-roll sampler.')
                rho_v = (mu.to(cloth.pred_pos.device).reshape(-1)[batch] / span).unsqueeze(-1)
        else:
            rho_v = cloth.pred_pos.new_full((cloth.pred_pos.shape[0], 1), float(mu) / span)
        return rho_v.clamp(min=0.0, max=cap)

    def _ia_r_prepass(self, sample_step):
        """Run the operator ONCE on the raw backbone proposal z0, under no_grad, purely to read its
        diagnostics; then restore z0. Returns (c0, lambda_n), both detached.

        c0 = P_mu(z0) - z0 is the correction the operator WOULD apply to the un-adapted proposal —
        the adapter's job is to pre-empt it. step_idx=None deliberately bypasses the inference
        warm-up gate: this call never ships state, it only measures, and a diagnostic that silently
        returned zeros for the first 30 steps of every rollout would teach the adapter that contacts
        do not exist there."""
        cloth = sample_step['cloth']
        z0 = cloth.pred_pos
        # pred_velocity always exists after Model._get_position on the real path; the getattr keeps
        # the restore honest on a bare fixture (and never invents a field that was not there).
        v0 = cloth.pred_velocity if 'pred_velocity' in cloth else None
        saved_cap = getattr(self, '_capture_force', False)
        saved_force = getattr(self, '_last_bolt_on_force', None)
        self._capture_force = True                     # ask the projection for per-vertex lambda_n
        try:
            with torch.no_grad():
                self._apply_bolt_on_friction(sample_step, step_idx=None)
            c0 = (cloth.pred_pos - z0).detach()
            lam = getattr(self, '_last_bolt_on_force', None)
            lam = lam.detach() if lam is not None else None
        finally:
            cloth.pred_pos = z0
            if v0 is not None:
                cloth.pred_velocity = v0
            self._capture_force = saved_cap
            self._last_bolt_on_force = saved_force     # never let the pre-pass poison --render-force
        return c0, lam

    def _ia_r_features(self, sample_step, c0, lam, rho_v):
        """(V, N_FEATURES) detached adapter input; layout pinned by models/ia_r_adapter.FEATURE_NAMES
        and by selftest R2. Lengths are expressed in units of the adapter's own output bound so every
        channel is O(1) on real contacts."""
        cloth = sample_step['cloth']
        scale = float(getattr(self.model.mcfg, 'adapter_max_disp', 0.02))
        c0n = c0.norm(dim=-1, keepdim=True)
        if lam is not None and lam.shape[0] == c0.shape[0]:
            lam_v = lam.reshape(-1, 1)
        else:                                          # projection returned no force (e.g. mode off)
            lam_v = c0n.new_zeros(c0n.shape)
        m = cloth.v_mass.reshape(-1, 1)
        # normal load in rest-weight units — the diagnostics' own calibration ratio. Clamped because
        # lambda_n is in Newtons only under lambda_n_source='force_balance'; under 'barrier' it is in
        # calibration units and an unclamped ratio would swamp the MLP's other channels.
        lam_rel = (lam_v / (m * 9.81).clamp(min=1e-12)).clamp(min=0.0, max=50.0)
        contact = (lam_v > 0).to(c0.dtype)
        vel = cloth.velocity
        feats = torch.cat([c0 / scale, c0n / scale, contact, lam_rel, rho_v, vel / scale], dim=-1)
        return feats.detach()

    def _apply_ia_r(self, sample_step, metrics_dict=None):
        """z <- z0 + rho(mu) * A_phi(q). Returns the applied displacement or None when inert.

        Inert (and byte-identical to the frozen backbone) when: the arm is off, or rho == 0 — which
        includes EVERY mu=0 batch. That is the whole point of the arm: the mu=0 anchor is a property
        of the architecture, not a measurement.

        Fires on operator-free steps too (roll warm-up, is_init, inference warm-up), matching the
         mu-feature choice: the adapter is conditioned on the mu REGIME of the
        sequence, and blanking it on those steps would make a mu=0.6 warm-up indistinguishable from a
        genuine mu=0 sequence — while inference would still apply it, breaking train/test agreement."""
        adapter = self._ia_r_adapter()
        if adapter is None:
            return None
        if self.contact_arm_mode() != 'bolt_on':
            raise ValueError("residual adapter (model.ccraft.enable_residual_adapter) requires "
                             "contact_arm.mode='bolt_on': the adapter is conditioned on the "
                             "operator's own diagnostics, so there must be an operator.")
        bo = self._contact_arm_cfg().bolt_on
        mu = self._resolve_bolt_on_mu(sample_step, bo)   # SINGLE-MU: the operator's own resolution
        rho_v = self._ia_r_rho(mu, sample_step)
        if float(rho_v.max()) == 0.0:
            return None                                  # structural mu=0 anchor: z == z0, bitwise
        cloth = sample_step['cloth']
        z0 = cloth.pred_pos
        c0, lam = self._ia_r_prepass(sample_step)
        feats = self._ia_r_features(sample_step, c0, lam, rho_v)
        a = adapter(feats)
        # Never move what the integrator pins — mirrors Model._get_position's pinned_mask exactly
        # (HANDLE vertices, plus anything outside a cutout mask), or the adapter would fight the
        # target-position boundary condition it is not allowed to touch.
        pinned = (cloth.vertex_type == NodeType.HANDLE)
        if 'cutout_mask' in cloth:
            pinned = torch.logical_or(pinned, torch.logical_not(cloth.cutout_mask[:, None]))
        a = a * torch.logical_not(pinned)
        dz = rho_v * a
        cloth.pred_pos = z0 + dz
        cloth.pred_velocity = cloth.pred_pos - cloth.pos
        if metrics_dict is not None:
            with torch.no_grad():
                dn = dz.norm(dim=-1)
                metrics_dict['iar/a_mean'].append(float(dn.mean()))
                metrics_dict['iar/a_max'].append(float(dn.max()))
                metrics_dict['iar/rho'].append(float(rho_v.max()))
                metrics_dict['iar/c0_mean'].append(float(c0.norm(dim=-1).mean()))
                # running total of mu=0 steps whose update was structurally empty (see
                # optimizer_step); expected ~= mu_sample_p_zero of all steps, and a FLAT curve here
                # while mu=0 batches are being drawn would mean the anchor is no longer structural.
                metrics_dict['iar/nograd_updates_total'].append(
                    float(getattr(self, '_iar_nograd_updates', 0)))
        return dz

    def _bolt_on_train_transition(self, prev_out_sample, metrics_dict, is_last=False):
        """The stop-gradient operator transition: called on the DETACHED post-loss
        sample, AFTER optimizer_step. Overwrites pred_pos/pred_velocity by attribute REASSIGNMENT
        (never in-place — bolt_on_friction_projection builds a fresh tensor), so the raw tensors
        retained by this step's autograd graph are untouched (guard 1). Reuses the byte-identical
        _apply_bolt_on_friction path that _rollout uses (guard 2: transition parity by construction).
        Gating: never on an is_init (initial_ts) step; first warmup_regular_steps
        REGULAR steps of each roll are operator-free (T0's N_diag); never on the roll's LAST step
        (verify finding: a terminal application's output is discarded — the loss is already
        computed and no next step consumes the state — so it is dead compute that would inflate the
        exposure counters without any on-policy training effect; skipping it is parity-neutral
        because parity concerns the state handed to the next step, and there is none)."""
        st = self._bo_train_roll
        if st is None:
            return
        bo = self._bo_train_cfg()
        if self._bo_train_loss_mode(bo) == 'action':
            # the operator already ran pre-loss (_bo_action_pre_loss) and its output IS this
            # step's state — re-applying it here would friction the same step twice.
            return
        if self._bo_train_loss_mode(bo) == 'impulse_aware':
            # the operator already ran pre-loss; what remains is the TC completion of the
            # transition state at mu>0 (never a second friction application — it re-projects the
            # RAW proposal, not the operated state).
            self._bo_impulse_tc_transition(prev_out_sample, metrics_dict, is_last=is_last)
            return
        cloth = prev_out_sample['cloth']
        if not self._bo_train_step_gate(cloth, st, bo, is_last, terminal_skip=True):
            return
        raw_pred = cloth.pred_pos          # pre-operator tensor (reassigned by the call, not mutated)
        prev_sink = getattr(self, '_bolt_on_diag_log', None)
        self._bolt_on_diag_log = []        # force want_diag inside _apply_bolt_on_friction
        try:
            with torch.no_grad():
                diag = self._apply_bolt_on_friction(prev_out_sample, step_idx=None)
        finally:
            self._bolt_on_diag_log = prev_sink
        st['applied'] += 1
        self._bo_record_alteration(st, bo, raw_pred, cloth, diag, metrics_dict)

    def _bo_train_cond_losses(self, loss_dict_hood, metrics_dict, prev_altered=None):
        """Guard 5's direct adaptation signal: this step's RAW-proposal penetration/stretch losses,
        conditioned on whether BO altered the PREVIOUS state. Read before this step's own transition
        updates the flag (call order in forward_short/forward_long guarantees that).

        Loss keys are f'{criterion.name}_loss' (criterion_pass builds f'{criterion_name}_{k}' with
        criterion_name = the REKEYED .name, utils/arguments.py:209-210) — i.e. 'collision_penalty_loss'
        / 'stretching_energy_loss', NOT the dotted config paths. (Bug caught by adversarial
        verify: the dotted keys never matched, so this instrument was silently dead; T16 pins the
        keys against the real config path, so a criterion rename fails the test loudly.)"""
        if self._bo_train_roll is None:
            return
        # `prev_altered` must be the flag as it stood BEFORE this step's operator application.
        # Under 'stop_grad' that is simply the live attribute (the transition runs later in the
        # step). Under 'action' the operator runs BEFORE this call, so the caller passes a snapshot
        # taken ahead of the pre-loss hook — otherwise the tag would silently flip from "the
        # PREVIOUS state was frictioned" to "the operator ran on THIS step", mislabelling the first
        # operator-active step of every roll and making the two arms' prevBO/prevfree populations
        # non-comparable. Found by the (two independent lenses).
        tag = 'prevBO' if (self._bo_prev_altered if prev_altered is None else prev_altered) \
            else 'prevfree'
        for name, short in (('collision_penalty_loss', 'collision'),
                            ('stretching_energy_loss', 'stretch')):
            if name in loss_dict_hood:
                metrics_dict[f'bo_tt/cond_{tag}/{short}'].append(float(loss_dict_hood[name].item()))

    def _bo_train_roll_end(self, metrics_dict):
        """Close the per-roll train-in-loop state: emit exposure counters, run the guard-4 exposure
        assert, and clear the per-roll mu so it can never leak into inference."""
        st = self._bo_train_roll
        self._bo_train_mu = None
        self._bo_train_roll = None
        self._bo_prev_altered = False
        self._bo_ia_raw_z = None
        if st is None:
            return
        metrics_dict['bo_tt/mu'].append(float(st['mu']))
        metrics_dict['bo_tt/roll_applied_steps'].append(float(st['applied']))
        metrics_dict['bo_tt/roll_altered_steps'].append(float(st['altered_steps']))
        metrics_dict['bo_tt/roll_skipped_init'].append(float(st['skipped_init']))
        metrics_dict['bo_tt/roll_skipped_warmup'].append(float(st['skipped_warmup']))
        metrics_dict['bo_tt/roll_skipped_terminal'].append(float(st['skipped_terminal']))
        if st.get('skipped_nonfinite'):
            metrics_dict['bo_tt/roll_skipped_nonfinite'].append(float(st['skipped_nonfinite']))
        self._bo_exposure_rolls += 1
        self._bo_exposure_altered_steps += st['altered_steps']
        bo = self._bo_train_cfg()
        expect_after = int(getattr(bo, 'expect_exposure_after', 0) or 0) if bo is not None else 0
        if (expect_after > 0 and self._bo_exposure_rolls >= expect_after
                and self._bo_exposure_altered_steps == 0):
            raise RuntimeError(
                f"train-in-loop exposure assert: the operator ALTERED zero vertices across"
                f"the first {self._bo_exposure_rolls} training rolls — either warm-up "
                f"(warmup_regular_steps={getattr(bo, 'warmup_regular_steps', '?')}, "
                f"skip_init_steps={getattr(bo, 'skip_init_steps', '?')}) and the roll structure leave "
                f"no operator-active steps, or no contacts form, or every application lands in "
                f"stick/mu=0 (|dx| below train_altered_dust). The probe would silently train as "
                f"CTRL. NOTE: this guard assumes the probe regime (step_start past the roll "
                f"curriculum, rolls at roll_max) — in an early/from-scratch regime short rolls are "
                f"structurally operator-free and the assert would false-fire. Fix the config before "
                f"burning GPU-hours.")

    def _force_balance_elastic_grad(self, state):
        """ Step 2: ∇(E_stretch + E_bend) w.r.t. cloth.pred_pos, in PHYSICAL (unweighted) energy units,
        per cloth node (Vc_total, 3), detached (lagged IPC normal force). This is the elastic contribution
        to the force-balance λ_n = max(0, n̂·∇(E_grav+E_inertia+E_stretch+E_bend)); gravity+inertia are
        closed-form inside the projection, only the elastic part needs autograd through the criterions.

        The stretching/bending criterions return loss = raw_energy * weight / B, so we rescale by (B/weight)
        to recover the physical energy whose gradient is the real elastic force. Runs under enable_grad on a
        detached leaf copy of pred_pos and restores the original afterwards (no side effects on the rollout)."""
        cloth = state['cloth']
        saved_pred = cloth.pred_pos
        pp = saved_pred.detach().requires_grad_(True)
        cloth.pred_pos = pp
        state.cloth_obj = self.cloth_obj          # some criterions read this (mirrors criterion_pass)
        B = state.num_graphs
        try:
            total = None
            with torch.enable_grad():
                for name, crit in self.criterion_dict.items():
                    if ('stretching_energy' not in name) and ('bending_energy' not in name):
                        continue
                    out = crit(state)
                    if 'loss' not in out or out['loss'] is None:
                        continue
                    w = out.get('weight', 1.0)
                    w = float(w.item()) if torch.is_tensor(w) else float(w)
                    raw = out['loss'] * (B / w) if w != 0 else out['loss'] * 0.0
                    total = raw if total is None else (total + raw)
                if total is None:
                    grad = torch.zeros_like(pp)
                else:
                    grad = torch.autograd.grad(total, pp, retain_graph=False, create_graph=False)[0]
        finally:
            cloth.pred_pos = saved_pred            # restore the frictionless tentative motion
        return grad.detach()

    def _log_bolt_on_diag(self, diag, mu, step_idx):
        mu_val = float(mu) if not torch.is_tensor(mu) else float(mu.reshape(-1)[0])
        prefix = (f"[BO step={step_idx} mu={mu_val:.3g}] "
                  f"contact={diag['n_contact']}/{diag['n_total']} ({diag['contact_frac']*100:.1f}%)")
        if diag['n_contact'] == 0:
            print(prefix + " (no contact)")
            return
        mg = diag['mg_cos_mean']
        ratio_str = f"{diag['lambda_n_mean'] / mg:.2f}" if mg > 0 else "n/a"
        print(f"{prefix} lambda_n~{diag['lambda_n_mean']:.3g} mg*cos~{mg:.3g} "
              f"ratio={ratio_str} slip[pre->post]={diag['slip_pre_mean']:.3g}->"
              f"{diag['slip_post_mean']:.3g} redux={diag['slip_redux']*100:.1f}% "
              f"stick={diag['stick_frac']*100:.1f}% factor_mean={diag['factor_mean']:.2f}")
        if 'stick_rate_downface' in diag:  # orientation breakdown (falsification signal)
            print(f"    [orient] down-facing={diag['contact_frac_downface']*100:.0f}% of contact  "
                  f"stick_rate up/down={diag['stick_rate_upface']*100:.0f}%/"
                  f"{diag['stick_rate_downface']*100:.0f}%  "
                  f"stuck-on-downface={diag['stuck_frac_downface']*100:.0f}%  "
                  f"n_up_mean={diag['n_up_signed_mean']:+.2f}")
        if diag.get('wk_pairs', 0):  # triangle-walk observability (the G5 witness line)
            nw = max(int(diag['wk_walked']), 1)
            print(f"    [walk] pairs={int(diag['wk_pairs'])} "
                  f"same_face={int(diag['wk_same_face'])} "
                  f"edge_seat={int(diag.get('wk_seat_edge', 0))} "
                  f"walked={int(diag['wk_walked'])} "
                  f"cross_max={int(diag['wk_cross_max'])} "
                  f"cross_mean={diag['wk_cross_sum'] / nw:.2f} "
                  f"fb[seat/cross/geom]={int(diag['wk_fb_seat'])}/"
                  f"{int(diag['wk_fb_cross'])}/{int(diag['wk_fb_geom'])}")
        if diag.get('mn_active', 0):  # multi-normal walk-back observability
            print(f"    [multi-normal] active={diag['mn_active']} moved={diag['mn_landing_moved']} "
                  f"junction={diag['mn_junction_land']} fullstick={diag['mn_fullstick']} "
                  f"spread_max={diag['mn_lambda_spread_max']:.2f} "
                  f"wrap_sum={diag['mn_wrap_sum']:.3f} "
                  f"cap_gain_max={diag['mn_capstan_gain_max']:.2f}")
        if diag.get('pc_frames', 0):  # diagnostic patch relaxation observability
            nf = max(int(diag['pc_frames']), 1)
            # alpha_ratio and nomat are the two numbers that can silently invalidate the
            # "compliance from the garment's own material" claim in vivo -- alpha_ratio ~1e-6 is
            # the expected stiff-limited regime and nomat>0 is a stop-class anomaly
            # (the material never reached the operator, so alpha was 0 by fallback).
            print(f"    [r3-c1] anchors={int(diag['pc_anchors'])} free={int(diag['pc_free'])} "
                  f"edges={int(diag['pc_edges'])} "
                  f"|C|max {diag['pc_C_pre_max']:.2e}->{diag['pc_C_post_max']:.2e} "
                  f"iters_feas={diag['pc_iters_feas']/nf:.1f} "
                  f"E_inj={diag['pc_energy_inj']:.2e}J "
                  f"dx_max={diag['pc_dx_max']:.2e} "
                  f"sig={int(diag['pc_sig_n'])}/{diag['pc_sig_max']:.2e} "
                  f"alpha_ratio={diag['pc_alpha_ratio']/nf:.2e} "
                  f"nomat={int(diag['pc_nomat'])}")
        if diag.get('wv_frames', 0):  # membrane-wave observability
            print(f"    [itmw-wv] frames={int(diag['wv_frames'])} empty={int(diag['wv_empty'])} "
                  f"active={int(diag['wv_active'])} reached={int(diag['wv_reached'])} "
                  f"N={int(diag['wv_n_sub'])} umax={diag['wv_umax']:.2e} "
                  f"sig={int(diag['wv_sig_n'])}/{diag['wv_sig_max']:.2e} "
                  f"cap={int(diag.get('wv_cap_edges0', 0))}->{int(diag.get('wv_cap_resid', 0))}"
                  f"/{diag.get('wv_cap_dxmax', 0.0):.1e} "
                  f"overflow={int(diag['wv_ladder_overflow'])} "
                  f"degen={int(diag['wv_degenerate'])}")
        if diag.get('sp_receivers', 0):  # diagnostic / C0.5 spreading observability
            print(f"    [r3-c0] donors={int(diag['sp_donors'])} "
                  f"receivers={int(diag['sp_receivers'])} "
                  f"hops={int(diag.get('sp_hops_used', 1))} "
                  f"src/recv={diag.get('sp_src_sum',0.)/max(diag.get('sp_recv_n',1),1):.2f} "
                  f"cancel={diag.get('sp_cancel_sum',0.)/max(diag.get('sp_multi',1),1):.3f} "
                  f"dx_max={diag['sp_dx_max']:.2e} "
                  f"reframed={diag.get('sp_proj_sum', 0.0):.2e}")

    def set_random_material(self, sample):
        sample, self.cloth_obj = self.random_material.add_material(sample, self.cloth_obj)
        return sample

    def _add_cloth_obj(self, sample):
        sample = self.set_random_material(sample)
        sample = self._add_friction_mu(sample)   # trained-friction arm: no-op unless arm_br.enabled
        sample = self.cloth_obj.set_batch(sample, overwrite_pos=self.mcfg.overwrite_pos_every_step)
        sample['cloth'].cloth_obj = self.cloth_obj
        return sample

    def criterion_pass(self, sample_step, criterion_dict):
        sample_step.cloth_obj = self.cloth_obj
        loss_dict = dict()
        gradient_dict = dict()
        loss_weight_dict = dict()
        metrics_dict = dict()
        for criterion_name, criterion in criterion_dict.items():
            ld = criterion(sample_step)
            for k, v in ld.items():
                if 'loss' in k:
                    loss_dict[f"{criterion_name}_{k}"] = v
                elif 'gradient' in k:
                    gradient_dict[f"{criterion_name}_{k}"] = v
                elif 'weight' in k:
                    loss_weight_dict[f"{criterion_name}_{k}"] = v
                else:
                    metrics_dict[f"{criterion_name}_{k}"] = v




        return loss_dict, loss_weight_dict, gradient_dict, metrics_dict

    def check_nan(self, tensor):
        is_nan = (tensor != tensor).any()
        return is_nan

    def check_nan_pred(self, sample):
        return self.check_nan(sample['cloth'].pred_pos) or self.check_nan(sample['cloth'].pred_velocity)

    def check_nan_input(self, sample):
        return self.check_nan(sample['cloth'].pos) or self.check_nan(sample['cloth'].velocity)

    def collect_sample(self, sample, idx, prev_out_dict=None, random_ts=False, is_short=False):
        sample_step = copy_pyg_batch(sample)

        # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        if idx > 0:
            sample_step = self.sample_collector.lookup2target(sample_step, idx)

        is_init = False
        if is_short and idx == 0:
            if random_ts:
                is_init = np.random.rand() > 0.5
                if is_init:
                    ts = self.mcfg.initial_ts
            else:
                is_init = True
                ts = self.mcfg.initial_ts

        sample_step = self.sample_collector.add_is_init(sample_step, is_init)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        return sample_step

    def add_metrics_from_dict(self, loss_dict, loss_weight_dict, metrics_dict_step, prefix):
        for k, v in loss_dict.items():
            k_weight = k.replace('loss', 'weight')
            v = v.item()

            weight = loss_weight_dict[k_weight] if k_weight in loss_weight_dict else 1
            v = v / weight if weight != 0 else 0

            k_pref = prefix + k
            metrics_dict_step[k_pref] = v

        return metrics_dict_step

    def add_metrics_ratios(self, metrics_dict_step, prefix_from, prefix_to, remove_to=False):
        keys_to = [k for k in metrics_dict_step if k.startswith(prefix_to)]

        for key_to in keys_to:
            v_to = metrics_dict_step[key_to]
            key_from = key_to.replace(prefix_to, prefix_from)

            if key_from not in metrics_dict_step:
                continue

            v_from = metrics_dict_step[key_from]
            ratio = v_to / v_from

            key_ratio = key_to.replace(prefix_to, 'ratio/')
            metrics_dict_step[key_ratio] = ratio

            if remove_to:
                del metrics_dict_step[key_to]

        return metrics_dict_step

    def add_metrics(self, loss_dict_hood, loss_weight_dict_hood, loss_dict_impulse, loss_weight_dict_impulse,
                    metrics_dict):
        prefix_hood = 'hood/'
        prefix_impulse = 'impulse/'

        metrics_dict_step = {}
        metrics_dict_step = self.add_metrics_from_dict(loss_dict_hood, loss_weight_dict_hood, metrics_dict_step,
                                                       prefix=prefix_hood)

        if loss_dict_impulse is not None:
            metrics_dict_step = self.add_metrics_from_dict(loss_dict_impulse, loss_weight_dict_impulse,
                                                           metrics_dict_step,
                                                           prefix=prefix_impulse)

            metrics_dict_step = self.add_metrics_ratios(metrics_dict_step, prefix_hood, prefix_impulse, remove_to=True)

        for k, v in metrics_dict_step.items():
            metrics_dict[k].append(v)

        return metrics_dict

    def check_step(self, sample_step, prev=True, threshold=0, to_print=False):
        if self.check_nan_input(sample_step):
            print(f'NAN PRED')
            return False

        ncoll_prev = self.safecheck_solver.calc_tritri_collisions(sample_step, prev=prev, threshold=threshold)

        if to_print:
            print('ncoll_prev', ncoll_prev, ncoll_prev == 0)
        return ncoll_prev == 0

    def add_gradient(self, sample_step, optimizer, gradient, weight=1.):
        params = optimizer.param_groups[0]['params']
        params = [param for param in params if param.requires_grad]

        model_icontour_grad = torch.autograd.grad(sample_step['cloth'].pred_pos, params, grad_outputs=gradient, allow_unused=True, retain_graph=True)

        i = 0
        for param in optimizer.param_groups[0]['params']:
            if param.requires_grad:
                if model_icontour_grad[i] is not None:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad += model_icontour_grad[i]*weight
                i += 1

    def _explosion_guard(self, loss_dict, metrics_dict, sample_step, roll_i):
        """: True => this roll step's state has exploded — caller must skip the update and
        abort the rollout. Placed before the ncoll diagnostic so telemetry never runs (and can't
        OOM) on exploded geometry. Non-finite totals trip it too."""
        cap = getattr(self.mcfg, 'loss_explosion_cap', None)
        if cap is None:
            return False
        total = sum(loss_dict.values())
        if torch.isfinite(total) and total <= cap:
            return False
        iter_num = int(sample_step['cloth'].iter[0]) if 'iter' in sample_step['cloth'] else -1
        metrics_dict['explosion_skip'].append(1.0)
        terms = ', '.join(f'{k.split(".")[-1]}={float(v):.2e}' for k, v in loss_dict.items())
        print(f'[explosion-guard] iter={iter_num} roll_step={roll_i}: loss={float(total):.3e} '
              f'> cap={cap:.1e} — skipping update, aborting rollout ({terms})', flush=True)
        return True

    def optimizer_step(self, optimizer, scheduler, loss_dict, gradient_dict, sample_step):
        if optimizer is not None:
            # for group in optimizer.param_groups:
            #     print(group['lr'])
            loss = sum(loss_dict.values())
            optimizer.zero_grad()

            if not loss.requires_grad:
                # residual adapter: with a FROZEN backbone, a mu=0 batch produces a graph
                # with NO trainable leaf at all — rho(0)=0 makes the adapter a structural no-op, so
                # there is nothing to differentiate. That is the arm's mu=0 anchor showing up in the
                # optimizer, exactly as the plan predicted ("mu=0 batches carry exactly zero adapter
                # gradient"), not an error: skip the update and advance the schedule.
                # Guarded on the arm being present because for EVERY other arm this is unreachable
                # (all parameters train), and a silent skip there would hide a real defect.
                if self._ia_r_adapter() is None:
                    raise RuntimeError(
                        'training loss has no gradient path (loss.requires_grad=False) and this is '
                        'not the residual adapter frozen-backbone arm. Something detached the graph — refusing '
                        'to silently skip the update.')
                self._iar_nograd_updates = getattr(self, '_iar_nograd_updates', 0) + 1
                if scheduler is not None:
                    scheduler.step()
                return

            for gradient_name, gradient in gradient_dict.items():
                # print(gradient_name, gradient.abs().sum())
                self.add_gradient(sample_step, optimizer, gradient, weight=1.)

            loss.backward()



            if self.mcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mcfg.grad_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

    def safecheck_impulse_pass(self, sample_step, metrics_dict):
        sample_step = self.safecheck_solver.safecheck_impulses(sample_step, metrics_dict)

        if self.mcfg.safecheck.max_impulse_norm is not None and self.mcfg.safecheck.max_impulse_norm > 0:
            hc_impulse_dx = sample_step['cloth'].hc_impulse_dx
            hc_impulse_norm = torch.norm(hc_impulse_dx, dim=-1)
            hc_impulse_norm_max = hc_impulse_norm.max()
            if hc_impulse_norm_max > self.mcfg.safecheck.max_impulse_norm:
                return sample_step, metrics_dict, False

        if self.mcfg.safecheck.max_ncoll > 0 and metrics_dict['impulse_stencil_ncoll'][-1] > self.mcfg.safecheck.max_ncoll:
                return sample_step, metrics_dict, False


        if self.check_nan_pred(sample_step):
            return sample_step, metrics_dict, False
        return sample_step, metrics_dict, True

    def safecheck_riz_pass(self, sample_step, metrics_dict):
        sample_step = self.safecheck_solver.safecheck_RIZ(sample_step, metrics_dict)
        if self.check_nan_pred(sample_step):
            return sample_step, metrics_dict, False
        return sample_step, metrics_dict, True

    def prepare_sample(self, sample):
        sample = self.safecheck_solver.mark_penetrating_faces(sample, dummy=True)
        sample = self._add_cloth_obj(sample)
        return sample

    def update_sample_1st_step(self, sample_step, sample):
        if self.mcfg.use_safecheck:
            sample_step = self.safecheck_solver.mark_penetrating_faces(sample_step)

            sample = add_field_to_pyg_batch(sample, 'cutout_mask', sample_step['cloth'].cutout_mask,
                                            'cloth',
                                            reference_key='cutout_mask')
            sample = add_field_to_pyg_batch(sample, 'faces_cutout_mask_batch', sample_step['cloth'].faces_cutout_mask_batch,
                                            'cloth',
                                            reference_key='faces_cutout_mask_batch')

            # sample = add_field_to_pyg_batch(sample, 'cutout_mask', sample_step['cloth'].cutout_mask,
            #                                 'cloth')
            # sample = add_field_to_pyg_batch(sample, 'faces_cutout_mask_batch', sample_step['cloth'].faces_cutout_mask_batch,
            #                                 'cloth')
        return sample_step, sample

    def aggregate_metrics_dict(self, metrics_dict):
        metrics_dict_agg = {}
        for k, v in metrics_dict.items():
            if k == 'riz_itersmax_riz_size':
                v = np.max(v)
                v = min(v, self.mcfg.safecheck.max_riz_size)
            else:
                v = np.mean(v)

            metrics_dict_agg[k] = v
        return metrics_dict_agg

    def toggle_model(self, curr_step, roll_steps, sample_type):
        if self.mcfg.nocollect_after > 0:
            is_training = curr_step < self.mcfg.nocollect_after
        else:
            is_training = True                
        if sample_type == 'short' and curr_step == 0 and roll_steps > 1:
            is_training = False
        self.model.train(is_training)


    def forward_short(self, sample, roll_steps=1, optimizer=None, scheduler=None) -> dict:
        random_ts = (roll_steps == 1)

        # train-in-loop arm the per-roll state BEFORE prepare_sample. prepare_sample -> _add_cloth_obj ->
        # _add_friction_mu, and since that chain resolves the bolt-on mu to write
        # the model's mu feature — so this roll's mu has to exist by then. (Caught by the
        # build smoke run, which died in _resolve_bolt_on_mu on the first long roll.)
        # BYTE-IDENTICAL for every pre- arm: _bo_train_roll_start writes only _bo_* state,
        # which prepare_sample neither reads nor writes, and it draws from the DEDICATED
        # _bo_mu_rng — never the global stream prepare_sample's RandomMaterial consumes — so data
        # order and material sampling are unchanged by the move.
        self._bo_train_roll_start('short')   # train-in-loop no-op unless bolt_on.train_in_loop

        sample = self.prepare_sample(sample)

        metrics_dict = defaultdict(list)

        prev_out_sample = None
        _i = 0

        self.short_steps += 1

        use_wedges_seq = True
        if self.mcfg.no_world_edges_every > 0 and self.short_steps % self.mcfg.no_world_edges_every == 0:
            use_wedges_seq = False

        iter_num = sample['cloth'].iter[0].item()

        for i in range(roll_steps):
            sample = add_field_to_pyg_batch(sample, 'step', [i], 'cloth', reference_key=None)

            is_first_step = i == 0
            is_last_step = i == roll_steps - 1

            use_wedges = (roll_steps == 1) or (i > 0)
            use_wedges = use_wedges and use_wedges_seq
            is_safecheck = self.mcfg.use_safecheck and use_wedges

            sample = self._add_cloth_obj(sample)
            sample_step = self.collect_sample(sample, i, prev_out_sample,
                                              random_ts=random_ts, is_short=True)
            sample_step = self.safecheck_solver.mark_penetrating_faces(sample_step, object='obstacle', use_target=True) 

            if i == 0:
                sample_step = self.collision_solver.solve(sample_step)
            if i == 1:
                sample_step, sample = self.update_sample_1st_step(sample_step, sample)

            if is_safecheck and i > 0 and not self.check_step(sample_step):
                break

            self.toggle_model(i, roll_steps, 'short')

            fake_icontour = self.mcfg.always_fake_icontour or not use_wedges

            sample_step = self.model(sample_step, world_edges=use_wedges, fake_icontour=fake_icontour)
            # residual adapter: mu-gated residual adapter on the frozen backbone's proposal (inert unless
            # enable_residual_adapter; ALWAYS inert at mu=0). Runs before every operator hook below.
            self._apply_ia_r(sample_step, metrics_dict)
            # 'action' mode: the operator runs HERE — in the graph, before the loss —
            # so the criterion pass below scores the shipped frame x^+. No-op in 'stop_grad' mode
            # and whenever train-in-loop is off, so configs are byte-identical.
            # snapshot BEFORE the operator can write the flag (guard-5 ordering, see
            # _bo_train_cond_losses); a no-op under 'stop_grad', where the hook is inert.
            prev_altered_snap = self._bo_prev_altered
            action_loss = self._bo_action_pre_loss(sample_step, metrics_dict, is_last=is_last_step)
            # straight-through operated-state hook; inert unless
            # train_loss_mode='impulse_aware'. Returns the in-graph raw proposal when it fired.
            ia_raw = self._bo_impulse_pre_loss(sample_step, metrics_dict, is_last=is_last_step)
            loss_dict_hood, loss_weight_dict_hood, gradient_dict, loss_metrics_dict = self.criterion_pass(sample_step, self.criterion_dict)
            if action_loss is not None:
                loss_dict_hood['friction_action_loss'] = action_loss
                loss_weight_dict_hood['friction_action_weight'] = 1.0
            if ia_raw is not None:
                loss_dict_hood = self._bo_impulse_fix_inertia(sample_step, ia_raw, loss_dict_hood)

            if self._explosion_guard(loss_dict_hood, metrics_dict, sample_step, i):
                break

            if use_wedges_seq:
                ncoll = self.safecheck_solver.calc_tritri_collisions2(sample_step)
                metrics_dict['ncoll'].append(ncoll)

            # train-in-loop guard 5: raw-proposal losses conditioned on prev-step operator alteration.
            # Must run BEFORE this step's own transition updates _bo_prev_altered.
            self._bo_train_cond_losses(loss_dict_hood, metrics_dict,
                                       prev_altered=prev_altered_snap)

            loss_dict_impulse = None
            loss_weight_dict_impulse = None


            self.optimizer_step(optimizer, scheduler, loss_dict_hood, gradient_dict, sample_step)

            prev_out_sample = sample_step.detach()
            # train-in-loop stop-gradient operator transition — loss and backward are done,
            # the detached state becomes the next step's input. No-op unless train_in_loop.
            # is_last: the terminal step's output is never consumed — skipped, counted, never
            # credited as exposure.
            self._bolt_on_train_transition(prev_out_sample, metrics_dict, is_last=is_last_step)
            if is_safecheck and not is_first_step and not is_last_step:
                prev_out_sample, metrics_dict, is_valid = self.safecheck_impulse_pass(prev_out_sample, metrics_dict)
                if not is_valid:
                    # print('not valid impulse pass')
                    break

                prev_out_sample, metrics_dict, is_valid = self.safecheck_riz_pass(prev_out_sample, metrics_dict)
                if not is_valid:
                    # print('not valid riz pass')
                    break

            _i = i + 1



            loss_dict_hood.update(loss_metrics_dict)
            metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood, loss_dict_impulse,
                                            loss_weight_dict_impulse,
                                            metrics_dict)

        self._bo_train_roll_end(metrics_dict)   # train-in-loop exposure counters + guard-4 assert + mu reset

        metrics_dict = self.aggregate_metrics_dict(metrics_dict)


        if use_wedges_seq:
            metrics_dict['n_steps'] = _i
            metrics_dict['finished'] = int(_i == roll_steps)
        metrics_dict = {'short/' + k: v for k, v in metrics_dict.items()}

        return metrics_dict

    def forward_long(self, sample, optimizer=None, scheduler=None) -> dict:
        # train-in-loop armed before prepare_sample — see the note in forward_short.
        self._bo_train_roll_start('long')   # train-in-loop no-op unless bolt_on.train_in_loop

        sample = self.prepare_sample(sample)


        # print('pos', sample['cloth'].lookup.shape)
        roll_steps = sample['cloth'].lookup.shape[1]

        self.long_steps += 1
        use_wedges_seq = True
        if self.mcfg.no_world_edges_every > 0 and self.short_steps % self.mcfg.no_world_edges_every == 0:
            use_wedges_seq = False

        metrics_dict = defaultdict(list)
        prev_out_sample = None
        _i = 0

        roll_steps = min(roll_steps, self.mcfg.roll_max_long)

        for i in range(roll_steps):
            sample = add_field_to_pyg_batch(sample, 'step', [i], 'cloth', reference_key=None)
            is_last_step = i == roll_steps - 1

            use_wedges = True
            use_wedges = use_wedges and use_wedges_seq
            is_safecheck = self.mcfg.use_safecheck and use_wedges

            sample = self._add_cloth_obj(sample)
            sample_step = self.collect_sample(sample, i, prev_out_sample)
            sample_step = self.safecheck_solver.mark_penetrating_faces(sample_step, object='obstacle', use_target=True) 

            if i == 0:
                sample_step = self.collision_solver.solve(sample_step)
                sample_step, sample = self.update_sample_1st_step(sample_step, sample)
                # sample_step, sample = self.collect_penetrating_mask(sample_step, sample)

            if is_safecheck and i > 0 and not self.check_step(sample_step):
                break

            self.toggle_model(i, roll_steps, 'long')        
            fake_icontour = self.mcfg.always_fake_icontour or not use_wedges


            sample_step = self.model(sample_step, world_edges=use_wedges, fake_icontour=fake_icontour)
            # residual adapter: mu-gated residual adapter on the frozen backbone's proposal (inert unless
            # enable_residual_adapter; ALWAYS inert at mu=0). Runs before every operator hook below.
            self._apply_ia_r(sample_step, metrics_dict)
            # 'action' mode (see forward_short): in-graph operator ahead of the loss.
            # snapshot BEFORE the operator can write the flag (guard-5 ordering, see
            # _bo_train_cond_losses); a no-op under 'stop_grad', where the hook is inert.
            prev_altered_snap = self._bo_prev_altered
            action_loss = self._bo_action_pre_loss(sample_step, metrics_dict, is_last=is_last_step)
            # (see forward_short): straight-through operated-state hook.
            ia_raw = self._bo_impulse_pre_loss(sample_step, metrics_dict, is_last=is_last_step)
            loss_dict_hood, loss_weight_dict_hood, gradient_dict, _ = self.criterion_pass(sample_step, self.criterion_dict)
            if action_loss is not None:
                loss_dict_hood['friction_action_loss'] = action_loss
                loss_weight_dict_hood['friction_action_weight'] = 1.0
            if ia_raw is not None:
                loss_dict_hood = self._bo_impulse_fix_inertia(sample_step, ia_raw, loss_dict_hood)

            if self._explosion_guard(loss_dict_hood, metrics_dict, sample_step, i):
                break

            if use_wedges_seq:
                ncoll = self.safecheck_solver.calc_tritri_collisions2(sample_step)
                metrics_dict['ncoll'].append(ncoll)

            # train-in-loop guard 5 (see forward_short): conditioned raw losses, read pre-transition.
            self._bo_train_cond_losses(loss_dict_hood, metrics_dict,
                                       prev_altered=prev_altered_snap)

            loss_dict_impulse, loss_weight_dict_impulse = None, None
            self.optimizer_step(optimizer, scheduler, loss_dict_hood, gradient_dict, sample_step)

            prev_out_sample = sample_step.detach()
            # train-in-loop stop-gradient operator transition (see forward_short).
            self._bolt_on_train_transition(prev_out_sample, metrics_dict, is_last=is_last_step)
            if is_safecheck and not is_last_step:
                prev_out_sample, metrics_dict, is_valid = self.safecheck_impulse_pass(prev_out_sample, metrics_dict)
                if not is_valid:
                    break

                prev_out_sample, metrics_dict, is_valid = self.safecheck_riz_pass(prev_out_sample, metrics_dict)
                if not is_valid:
                    break

            _i = i + 1

            metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood, loss_dict_impulse,
                                            loss_weight_dict_impulse,
                                            metrics_dict)

        self._bo_train_roll_end(metrics_dict)   # train-in-loop exposure counters + guard-4 assert + mu reset

        metrics_dict = self.aggregate_metrics_dict(metrics_dict)
        if use_wedges_seq:
            metrics_dict['n_steps'] = _i
            metrics_dict['finished'] = int(_i == roll_steps)
        metrics_dict = {'long/' + k: v for k, v in metrics_dict.items()}
        return metrics_dict


def inflated_scheduler_step(global_step, warmup_steps, increase_roll_every, roll_max):
    """Cumulative scheduler.step() calls after `global_step` global steps (short-step phase only).
    The scheduler advances once per autoregressive roll step, so last_epoch inflates relative to
    global_step. This reconstruction gives the correct LR on resume when optimizer state is absent
    (e.g. old checkpoint). Only valid for the short-only phase; if step_start is in the long-step
    phase, load_optimizer=True and a full checkpoint are required for accuracy."""
    total = 0
    for g in range(max(1, int(warmup_steps)), int(global_step) + 1):
        total += min(1 + g // increase_roll_every, roll_max)
    return total


def create_optimizer(runner: Runner, mcfg: DictConfig):
    # residual adapter: with a frozen backbone only the adapter's parameters are trainable, so the
    # requires_grad filter keeps 8M frozen tensors out of Adam's state instead of relying on
    # grad=None.
    #
    # — THE FILTER MUST STAY GATED. It was originally applied unconditionally (5c8bfa7)
    # on the claim that it is "byte-identical for every other arm, since there every parameter has
    # requires_grad=True". That claim is false: each of the 6 normalizers holds 4
    # nn.Parameter(..., requires_grad=False) accumulators (utils/normalization.py:22-26) = 24
    # tensors that live in model.parameters() and so went into Adam on every pre-5c8bfa7 run.
    # Filtering them is numerically INERT (Adam skips grad=None), but it renumbers the param list,
    # and optimizer state is keyed BY POSITION — so an unconditional filter silently made every
    # checkpoint written before 2026-07-30 unloadable (they carry an 825-entry group; the filtered
    # build makes 801) with no faithful fallback, because inflated_scheduler_step's LR
    # reconstruction is invalid past n_steps_only_short. Gate on the same condition that actually
    # freezes anything (models/ccraft.py:104) so non-adapter arms keep the historical list.
    adapter = getattr(runner.model, 'residual_adapter', None)
    backbone_frozen = adapter is not None and getattr(
        getattr(runner.model, 'mcfg', None), 'freeze_backbone', True)
    if backbone_frozen:
        params = [p for p in runner.model.parameters() if p.requires_grad]
        n_frozen = sum(1 for p in runner.model.parameters() if not p.requires_grad)
        print(f'[optimizer] {len(params)} trainable parameter tensors, {n_frozen} FROZEN '
              f'(excluded from Adam)')
    else:
        params = list(runner.model.parameters())
        print(f'[optimizer] {len(params)} parameter tensors (unfiltered: no deliberate backbone '
              f'freeze; preserves pre-5c8bfa7 state-dict indexing)')
    optimizer = Adam(params, lr=mcfg.lr)

    warmup = int(getattr(mcfg, 'warmup_steps', 0) or 0)

    def sched_fun(step):
        decay = mcfg.decay_rate ** (step // mcfg.decay_steps)
        decay = max(decay, mcfg.decay_min)
        if warmup > 0:
            # LR-rewarm: linear warm-up over the first `warmup` scheduler (inflated) steps
            decay = decay * min(1.0, (step + 1) / warmup)
        return decay

    scheduler = LambdaLR(optimizer, sched_fun)
    rcfg = runner.mcfg
    # step_start_override decouples the LR schedule from the resume step (fresh schedule
    # for the friction phase while restart.step_start keeps loss ramps / roll schedule mature).
    override = getattr(mcfg, 'step_start_override', None)
    sched_start = int(override) if override is not None else int(mcfg.step_start)
    last = inflated_scheduler_step(sched_start, rcfg.warmup_steps,
                                   rcfg.increase_roll_every, rcfg.roll_max)
    scheduler.last_epoch = last
    for g in optimizer.param_groups:
        g['lr'] = mcfg.lr * sched_fun(last)
    print(f'[resume] step_start={mcfg.step_start} (schedule from {sched_start}) -> '
          f'scheduler.last_epoch={last}, lr={mcfg.lr * sched_fun(last):.4e}')

    return optimizer, scheduler


def compute_epoch_size(dataloader_short, dataloader_long, cfg, global_step):
    len_short = len(dataloader_short)
    len_long = len(dataloader_long)
    n_steps_only_short = cfg.experiment.n_steps_only_short
    n_steps_only_short_togo = max(n_steps_only_short - global_step, 0)

    if len_short < n_steps_only_short_togo:
        return len_short

    short_steps_alternating = len_short - n_steps_only_short_togo
    long_steps_alternating = len_long

    total_steps_alternating = min(long_steps_alternating, short_steps_alternating) * 2
    total_steps = n_steps_only_short_togo + total_steps_alternating

    return total_steps


def step_short(runner, global_step, sample, optimizer, scheduler):
    roll_steps = 1 + (global_step // runner.mcfg.increase_roll_every)
    roll_steps = min(roll_steps, runner.mcfg.roll_max)

    if global_step >= runner.mcfg.warmup_steps:
        ld_to_write = runner.forward_short(sample, roll_steps=roll_steps, optimizer=optimizer,
                                                    scheduler=scheduler)
    else:
        ld_to_write = runner.forward_short(sample, roll_steps=roll_steps, optimizer=None, scheduler=None)

    return ld_to_write


def step_long(runner, sample, optimizer, scheduler):
    ld_to_write = runner.forward_long(sample, optimizer=optimizer,
                                               scheduler=scheduler)
    return ld_to_write


def make_checkpoint(runner, aux_modules, cfg, global_step):
    if global_step < cfg.experiment.n_steps_only_short:
        to_save = global_step % cfg.experiment.save_checkpoint_every == 0
    else:
        to_save = global_step % cfg.experiment.save_checkpoint_every_wlong == 0

    if to_save:
        # run_dir is resolved once at startup (train.py -> resolve_run_dir); just read it here.
        checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
        print('Saving checkpoint to', checkpoint_path)
        save_checkpoint(runner, aux_modules, cfg, checkpoint_path)


def run_epoch(runner: Runner, aux_modules: dict, dataloaders_dict: dict, 
              cfg: DictConfig, writer=None, global_step=None):
    global_step = global_step or 0

    runner.model.train()

    optimizer = aux_modules['optimizer']
    scheduler = aux_modules['scheduler']

    dataloader_short = dataloaders_dict['short']
    dataloader_long = dataloaders_dict['long']

    n_steps = compute_epoch_size(dataloader_short, dataloader_long, cfg, global_step)
    n_steps = min(n_steps, cfg.experiment.max_iter)

    prbar = tqdm(range(n_steps), desc=cfg.config)

    short_iter = dataloader_short.__iter__()
    long_iter = dataloader_long.__iter__()

    last_step = 'short'

    # run_dir is resolved once at startup (train.py -> resolve_run_dir); just read it here.
    checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')
    print(yellow(f'>>>> run_epoch started, checkpoints will be saved in {checkpoints_dir} <<<<'))

    # save the full run config in the experiment folder, sibling to `checkpoints/` (once per run)
    os.makedirs(cfg.run_dir, exist_ok=True)
    config_save_path = os.path.join(cfg.run_dir, 'config.yaml')
    if not os.path.exists(config_save_path):
        OmegaConf.save(cfg, config_save_path)
        print(yellow(f'>>>> run config saved to {config_save_path} <<<<'))

    for i in prbar:
        global_step += 1

        if cfg.experiment.max_iter is not None and global_step > cfg.experiment.max_iter:
            break

        if last_step == 'short' and global_step > cfg.experiment.n_steps_only_short:
            curr_step = 'long'
            sample = next(long_iter)
        else:
            curr_step = 'short'
            sample = next(short_iter)

        last_step = curr_step
        sample = move2device(sample, cfg.device)

        B = sample.num_graphs
        sample = add_field_to_pyg_batch(sample, 'iter', [global_step] * B, 'cloth', reference_key=None)

        if curr_step == 'short':
            ld_to_write = step_short(runner, global_step, sample, optimizer, scheduler)
        elif curr_step == 'long':
            ld_to_write = step_long(runner, sample, optimizer, scheduler)
        else:
            raise Exception(f'Wrong step label {curr_step}')

        ld_to_write['lr'] = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.write_dict(ld_to_write, step=global_step)

        make_checkpoint(runner, aux_modules, cfg, global_step)

    return global_step
