import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import II
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug
from utils.common import copy_pyg_batch, move2device, save_checkpoint, add_field_to_pyg_batch
from utils.defaults import DEFAULTS
import utils.contact as C
# E1: the friction operator's config is SHARED with runners/ccraft.py, never copied.
# A copy would drift on the next operator revision, and the failure mode is a silently DIFFERENT
# operator between the ContourCraft arms and the HOOD arms -- the exact confound the HOOD
# comparison exists to avoid. The import costs ~0.25 s and
# pulls in cccollisions via runners.utils.impulses; that is an import-time cost only, and this
# repo cannot run without cccollisions anyway.
from runners.ccraft import BoltOnConfig


@dataclass
class MaterialConfig:
    density_min: float = 0.20022            # minimal density to sample from (used to compute nodal masses)
    density_max: float = 0.20022            # maximal density to sample from (used to compute nodal masses)
    lame_mu_min: float = 23600.0            # minimal shear modulus to sample from
    lame_mu_max: float = 23600.0            # maximal shear modulus to sample from
    lame_lambda_min: float = 44400          # minimal Lame's lambda to sample from
    lame_lambda_max: float = 44400          # maximal Lame's lambda to sample from
    bending_coeff_min: float = 3.96e-05     # minimal bending coefficient to sample from
    bending_coeff_max: float = 3.96e-05     # maximal bending coefficient to sample from
    bending_multiplier: float = 1.          # multiplier for bending coefficient

    density_override: Optional[float] = None        # if set, overrides the sampled density (used in validation)
    lame_mu_override: Optional[float] = None        # if set, overrides the sampled shear modulus (used in validation)
    lame_lambda_override: Optional[float] = None    # if set, overrides the sampled Lame's lambda (used in validation)
    bending_coeff_override: Optional[float] = None  # if set, overrides the sampled bending coefficient (used in validation)
    friction_mu_override: Optional[float] = None    # bolt-on operator: --mu sweep; wins over contact_arm.bolt_on.mu


@dataclass
class OptimConfig:
    lr: float = 1e-4                # initial learning rate
    decay_rate: float = 1e-1        # decay multiplier for the scheduler
    decay_min: float = 0            # minimal decay
    decay_steps: int = 5_000_000    # number of steps for one decay step
    step_start: int = 0             # step to start from (used to resume training)


# BoltOnConfig is imported from runners.ccraft (see the import block above). It carries the whole
# //// knob set with byte-identical-when-absent defaults ('barrier', 'off', 'chord',
# 'single', warmup_steps 0), so a pre-existing -style postcvpr config is unchanged by the import.
# Its train-in-loop / training fields (train_in_loop, mu_sample*, train_loss_mode,...) are inert on
# this path and are rejected explicitly in Runner.__init__.


@dataclass
class TrainedConfig:
    """trained-friction arm scaffolding (trained in-loop; not yet implemented)."""
    pass


@dataclass
class ContactArmConfig:
    """Post-model friction arm selector. mode='none' => byte-identical baseline (default)."""
    mode: str = 'none'
    bolt_on: BoltOnConfig = BoltOnConfig()
    trained: TrainedConfig = TrainedConfig()


@dataclass
class Config:
    optimizer: OptimConfig = OptimConfig()
    material: MaterialConfig = MaterialConfig()
    contact_arm: ContactArmConfig = ContactArmConfig()   # friction arm (none=baseline by default)
    warmup_steps: int = 100                 # number of steps to warm up the normalization statistics
    increase_roll_every: int = 5000         # we start from predicting only one step, then increase the number of steps each `increase_roll_every` steps
    roll_max: int = 5                       # maximum number of steps to predict
    push_eps: float = 2e-3                  # threshold for collision solver, we apply it once before the first step
    grad_clip: Optional[float] = 1.         # if set, clips the gradient norm to this value
    overwrite_pos_every_step: bool = False  # if true, the canonical poses of each garment are not cached

    # In the paper, the difference between the initial and regular time steps is explained with alpha coeffitient in the inertia loss term
    initial_ts: float = 1 / 3   # time between the first two steps in training, used to allow the model to faster reach static equiliblium
    regular_ts: float = 1 / 30  # time between the regular steps in training and validation

    device: str = II('device')


class Runner(nn.Module):
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module], mcfg: DictConfig):
        super().__init__()

        self.model = model
        self.criterion_dict = criterion_dict['ccraft']
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
        self.normals_f = FaceNormals()

        self.sample_collector = SampleCollector(mcfg, obstacle=True)
        self.collision_solver = CollisionPreprocessor(mcfg)
        self.random_material = RandomMaterial(mcfg.material)

        self._last_bolt_on_force = None
        self._validate_contact_arm()

    def _validate_contact_arm(self):
        """Fail at construction on operator settings this runner cannot honour.

        Both checks exist because the alternative is a SILENTLY DIFFERENT operator, not an error:
        the run would complete, the logs would look normal, and the numbers would be wrong in a
        believable way. Loud beats plausible.
        """
        ca = self._contact_arm_cfg()
        if ca is None or getattr(ca, 'mode', 'none') != 'bolt_on':
            return
        bo = ca.bolt_on
        if getattr(bo, 'train_in_loop', False):
            raise ValueError(
                "runner.postcvpr does not implement train-in-loop (contact_arm.bolt_on.train_in_loop): "
                "BO-in-the-loop training is a ccraft-only path. A postcvpr config asking for it "
                "would silently get plain inference behaviour instead. Use runners/ccraft.py.")
        if getattr(bo, 'lambda_n_source', 'barrier') == 'force_balance' and \
                getattr(bo, 'fb_use_elastic', False):
            raise NotImplementedError(
                "runner.postcvpr does not implement the force_balance ELASTIC channel "
                "(fb_use_elastic=True): it needs the ccraft-side _force_balance_elastic_grad over "
                "that runner's criterion set. Passing elastic_grad=None here would compute a "
                "different, weaker lambda_n than the ccraft arms and look entirely plausible "
                "(E2). The shipping / operator sets fb_use_elastic=False, "
                "so this is only reachable by a -lineage config.")

    def valid_rollout(self, sequence, n_steps=-1, bare=False, record_time=False):
        """
        Generates a trajectory for the full sequence, use this function for inference

        :param sequence: torch geometric batch with the sequence
        :param n_steps: number of steps to predict, if -1, predicts for the full sequence
        :param bare: if true, the loss terms are not computed, use for faster validation
        :param record_time: if true, records the time spent on the forward pass and adds it to trajectories_dicts['metrics']
        :return: trajectories_dicts: dict with the following fields:
            'pred': np.ndarray (NxVx3) predicted cloth trajectory
            'obstacle': np.ndarray (NxWx3) body trajectory
            'cloth_faces': np.ndarray (Fx3) cloth faces
            'obstacle_faces': np.ndarray (Fox3) body faces
            'metrics': dictionary with by-frame loss values
        """
        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        sequence = self.add_cloth_obj(sequence)

        is_obstacle = 'obstacle' in sequence.node_types
        if is_obstacle:
            n_samples = sequence['obstacle'].lookup.shape[1]
        else:
            n_samples = sequence['cloth'].lookup.shape[1]

        if n_steps >= 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(lambda: defaultdict(list))

        # E6: per-sequence reset of the render-only force side channel. A stale tensor from
        # the previous sequence has a different vertex count and would break the stack below. Touches
        # only the recorded 'force' trajectory, never the dynamics. (Mirrors runners/ccraft.py.)
        self._last_bolt_on_force = None

        if record_time:
            st_time = time.time()

        self.model.eval()
        with torch.no_grad():
            trajectory, obstacle_trajectory, force_trajectory, metrics_dict = self._rollout(
                sequence, n_samples, progressbar=True, bare=bare)

        if record_time:
            total_time = time.time() - st_time
            metrics_dict['time'] = total_time

        trajectories_dicts['pred'] = trajectory
        trajectories_dicts['obstacle'] = obstacle_trajectory
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = sequence['cloth'].faces_batch.T.cpu().numpy()
        trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        # E5: fmetrics REQUIRES vertex_type in the pkl (utils/fmetrics/io.py:97 raises
        # KeyError without it) and derives the HANDLE mask from it (io.py:81) -- i.e. the garment's
        # pinned vertices. Mirrors runners/ccraft.py.
        trajectories_dicts['vertex_type'] = sequence['cloth'].vertex_type.cpu().numpy()
        if 'garment_id' in sequence['cloth']:
            trajectories_dicts['garment_id'] = sequence['cloth'].garment_id.cpu().numpy()

        for s in ['pred', 'obstacle']:
            trajectories_dicts[s] = np.stack(trajectories_dicts[s], axis=0)
        if force_trajectory is not None:
            trajectories_dicts['force'] = np.stack(force_trajectory, axis=0)
        return trajectories_dicts

    def _rollout(self, sequence, n_steps, progressbar=False, bare=False):
        trajectory = []
        obstacle_trajectory = []
        want_force = bool(getattr(self, '_capture_force', False))
        force_trajectory = [] if want_force else None

        metrics_dict = defaultdict(list)

        pbar = range(0, n_steps)
        if progressbar:
            pbar = tqdm(pbar)

        prev_out_dict = None
        fail_reason = 'SUCCESS'
        is_obstacle = 'obstacle' in sequence.node_types
        # print('SEQUENCE')
        # print(sequence)
        for i in pbar:
            state = self.collect_sample_wholeseq(sequence, i, prev_out_dict)

            # if i == 0:
            #     state = self.collision_solver.solve(state)

            # print('STATE')
            # print(state); assert False

            if i == 0:
                # E3 -- FRAME ALIGNMENT, do not remove.
                # runners/ccraft.py records the two KINEMATIC SEED frames (prev_pos, pos) before the
                # first prediction, so a length-N sequence yields an N-frame trajectory. This loop
                # used to emit N-2. That is not cosmetic: the fmetrics scoring window is
                # t0 = bolt_on.warmup_steps + 2, and the '+2' IS these two frames. Without them a
                # HOOD pkl scored at t0=2 discards two genuine prediction frames and runs two frames
                # out of phase against every other arm, on all 40 cells, with no error and no
                # guardrail trip. Acceptance is numeric: pred.shape[0] == the datasplit's `length`
                # == the matching reference pkl's frame count.
                trajectory.append(state['cloth'].prev_pos.detach().cpu().numpy())
                trajectory.append(state['cloth'].pos.detach().cpu().numpy())
                if is_obstacle:
                    obstacle_trajectory.append(state['obstacle'].prev_pos.detach().cpu().numpy())
                    obstacle_trajectory.append(state['obstacle'].pos.detach().cpu().numpy())
                if want_force:
                    # the seed frames precede any projection -> zeros, to stay frame-aligned
                    zf = np.zeros(state['cloth'].pos.shape[0], dtype=np.float32)
                    force_trajectory.append(zf)
                    force_trajectory.append(zf)

            state = self.model(state)
            # bolt-on operator: bolt-on Coulomb friction. No-op unless contact_arm.mode='bolt_on'.
            self._apply_bolt_on_friction(state, step_idx=i)

            pred_pos = state['cloth'].pred_pos
            if not torch.isfinite(pred_pos).all():
                # E4: abort instead of writing garbage. Both evaluate.py and
                # utils/fmetrics/io.py:116 DEFAULT a missing fail_reason to 'SUCCESS', so without
                # this a diverged cell enters the tables as a healthy one.
                fail_reason = f'DIVERGED@step{i}'
                break

            trajectory.append(pred_pos.detach().cpu().numpy())
            obstacle_trajectory.append(state['obstacle'].target_pos.detach().cpu().numpy())
            if want_force:
                f = self._capture_step_force(state)
                if f is None:
                    f = np.zeros(pred_pos.shape[0], dtype=np.float32)
                force_trajectory.append(f)

            if not bare:
                loss_dict = self.criterion_pass(state)
                for k, v in loss_dict.items():
                    if 'loss' in k:
                        metrics_dict[k].append(v.item())
            prev_out_dict = state.clone()

        metrics_dict['fail_reason'] = fail_reason
        return trajectory, obstacle_trajectory, force_trajectory, metrics_dict

    # def collect_sample_wholeseq(self, sequence, index, prev_out_dict):

    #     """
    #     Collects a sample from the sequence, given the previous output and the index of the current step
    #     This function is only used in validation
    #     For training, see `collect_sample`

    #     :param sequence: torch geometric batch with the sequence
    #     :param index: index of the current step
    #     :param prev_out_dict: previous output of the model

    #     """
    #     sample_step = sequence.clone()

    #     # gather infor for the current step
    #     sample_step = self.sample_collector.sequence2sample(sample_step, index)

    #     # move to device
    #     sample_step = move2device(sample_step, self.mcfg.device)

    #     # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
    #     sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
    #     ts = self.mcfg.regular_ts

    #     # in the first step, the obstacle and positions of the pinned vertices are static
    #     if index == 0:
    #         sample_step = self.sample_collector.target2pos(sample_step)
    #         sample_step = self.sample_collector.pos2prev(sample_step)
    #         ts = self.mcfg.initial_ts
    #     # in the second step, we set velocities for both the cloth and the obstacle to zero
    #     elif index == 1:
    #         sample_step = self.sample_collector.pos2prev(sample_step)

    #     sample_step = self.sample_collector.add_timestep(sample_step, ts)
    #     sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
    #     return sample_step

    def collect_sample_wholeseq(self, sample, idx, prev_out_dict=None):
        sample_step = copy_pyg_batch(sample)

        # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        if idx > 0:
            sample_step = self.sample_collector.lookup2target(sample_step, idx)

        is_init = False
        sample_step = self.sample_collector.add_is_init(sample_step, is_init)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        return sample_step


    # --------------------------------------------------------------------------------------------- #
    # bolt-on operator: inference-time bolt-on Coulomb friction (cubic-barrier proposal). #
    # --------------------------------------------------------------------------------------------- #

    def _contact_arm_cfg(self):
        return getattr(self.mcfg, 'contact_arm', None)

    def contact_arm_mode(self):
        ca = self._contact_arm_cfg()
        return getattr(ca, 'mode', 'none') if ca is not None else 'none'

    def _resolve_bolt_on_mu(self, state, bo):
        if 'friction_mu' in state['cloth']:
            return state['cloth'].friction_mu
        override = getattr(self.mcfg.material, 'friction_mu_override', None)
        if override is not None:
            return float(override)
        if bo.mu is not None:
            return float(bo.mu)
        raise ValueError(
            "bolt-on operator enabled but no friction coefficient set. Pass "
            "runner.postcvpr.material.friction_mu_override=<value> or set "
            "runner.postcvpr.contact_arm.bolt_on.mu in the config.")

    def _apply_bolt_on_friction(self, state, step_idx=None):
        ca = self._contact_arm_cfg()
        mode = self.contact_arm_mode()
        if mode == 'none':
            return None
        if mode == 'trained':
            raise NotImplementedError("trained-friction arm not yet implemented. Use mode='none' or 'bolt_on'.")
        if mode != 'bolt_on':
            raise ValueError(f"Unknown contact_arm.mode: {mode!r}")
        bo = ca.bolt_on
        # E2: operator warm-up gate (LBS-init blindfold). 0 = off, which is what every
        # /-lineage config sets. INFERENCE path only. Mind the four-way `warmup*` name
        # collision documented on runners/ccraft.py::BoltOnConfig.warmup_steps.
        if step_idx is not None and step_idx < int(getattr(bo, 'warmup_steps', 0) or 0):
            return None
        mu = self._resolve_bolt_on_mu(state, bo)
        # NOTE: ccraft's _assert_single_mu is deliberately NOT ported. It guards the mu
        # FEATURE, and the HOOD/postcvpr backbone has none (enable_varying_mu=False in all five
        # checkpoints). Its absence is a decision, not an oversight.
        do_print = bool(bo.log_every) and (step_idx is None or step_idx % bo.log_every == 0)
        collect = getattr(self, '_bolt_on_diag_log', None) is not None
        want_diag = do_print or collect
        want_force = bool(getattr(self, '_capture_force', False))
        # E2: forward the FULL /// knob set, mirroring
        # runners/ccraft.py::_apply_bolt_on_friction. Every getattr default reproduces the
        # behaviour, so a config that sets none of them is byte-identical to the previous code.
        # elastic_grad stays None on purpose -- _validate_contact_arm() has already refused any
        # config that would need it (fb_use_elastic=True), so this cannot silently under-compute.
        out = C.bolt_on_friction_projection(
            state, mu, eps=bo.eps, stiffness=bo.stiffness, eps_v=bo.eps_v, z_axis=bo.z_axis,
            g=getattr(bo, 'fb_g', 9.81), return_diag=want_diag, return_force=want_force,
            lambda_n_source=getattr(bo, 'lambda_n_source', 'barrier'),
            fb_use_gravity=getattr(bo, 'fb_use_gravity', True),
            fb_use_inertia=getattr(bo, 'fb_use_inertia', True), elastic_grad=None,
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
            # the triangle-walk crossing cap. Forwarded for the same anti-drift reason as
            # every block below -- a walk config on this runner must run the walk, not silently
            # the march under a walk name. Default = OFF-path
            # identical: 16 on a non-walk path is the callee's own default and never binds.
            path_walk_max_cross=getattr(bo, 'path_walk_max_cross', 16),
            # the walk scheduling fast path. Same anti-drift reason; False is the callee's
            # default and unreachable on any non-walk path.
            path_walk_fast=getattr(bo, 'path_walk_fast', False),
            # R3: the drag / ring-spread / patch-coupling knobs. ELEVEN, not just the
            # five patch_* ones: this call also predated drag_k and spread_*, and forwarding only
            # part of the set would leave the two call sites into a 54-argument function differing
            # in six places -- the drift E1 killed the copied BoltOnConfig to prevent.
            # Every default below is the OFF value and is byte-identical to the default in
            # runners/ccraft.py and in the callee's own signature, so a config that sets none of
            # them reaches exactly the code path it reached before (: every added
            # block in utils/contact.py sits behind `if drag_k:` / `if spread_w:` / `if patch_iters:`).
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
            # membrane wave / strain cap: the wave is the THIRD coupling of the same correction
            # field, so it forwards here for the same reason the other two do -- a call site that
            # carries only part of the knob set is how the two runners drift apart.
            # All four defaults are the OFF values, so a config that sets none of them reaches
            # exactly the code path it reached before this line existed.
            wave_enabled=getattr(bo, 'wave_enabled', False),
            wave_visco=getattr(bo, 'wave_visco', 0.0),
            wave_tau_frames=getattr(bo, 'wave_tau_frames', 0.5),
            wave_strain_cap=getattr(bo, 'wave_strain_cap', 0.0))
        # NOT forwarded, on purpose: detach_lambda_n / return_action are train-in-loop training-only and
        # _validate_contact_arm() already refuses train_in_loop on this runner; return_metric_log
        # and shadow have no caller here.
        # unpack (pred[, diag][, force]) -- the callee's fixed order
        outs = list(out) if isinstance(out, tuple) else [out]
        pred_pos_new = outs.pop(0)
        diag = outs.pop(0) if want_diag else None
        force = outs.pop(0) if want_force else None
        if want_force:
            self._last_bolt_on_force = force.detach()
        cloth = state['cloth']
        cloth.pred_velocity = pred_pos_new - cloth.pos
        cloth.pred_pos = pred_pos_new
        if diag is not None:
            if collect:
                self._bolt_on_diag_log.append(diag)
            if do_print:
                self._log_bolt_on_diag(diag, mu, step_idx)
        return diag

    def has_force_source(self) -> bool:
        """True if this runner can produce a real per-vertex contact force during rollout.
        evaluate.py:568 calls this unconditionally. Only bolt-on operator applies here -- there
        is no trained-friction arm on the postcvpr path."""
        return self.contact_arm_mode() == 'bolt_on'

    def _capture_step_force(self, state):
        """(V,) per-vertex contact-force magnitude for this step, or None -> caller writes zeros.

        Shape guard mirrors runners/ccraft.py: during the operator warm-up
        (_apply_bolt_on_friction early-returns without updating it) or across a sequence boundary a
        stale tensor of a different length would be recorded and break the np.stack of the 'force'
        trajectory. Mismatch/None -> zeros, which is correct: no projection ran this step."""
        f = getattr(self, '_last_bolt_on_force', None)
        if f is not None and f.shape[0] == state['cloth'].pred_pos.shape[0]:
            return f.detach().cpu().numpy()
        return None

    def _log_bolt_on_diag(self, diag, mu, step_idx):
        import torch
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
        if diag.get('wk_pairs', 0):  # triangle-walk observability, mirroring runners/ccraft.py
            nw = max(int(diag['wk_walked']), 1)
            print(f"    [walk] pairs={int(diag['wk_pairs'])} "
                  f"same_face={int(diag['wk_same_face'])} "
                  f"edge_seat={int(diag.get('wk_seat_edge', 0))} "
                  f"walked={int(diag['wk_walked'])} "
                  f"cross_max={int(diag['wk_cross_max'])} "
                  f"cross_mean={diag['wk_cross_sum'] / nw:.2f} "
                  f"fb[seat/cross/geom]={int(diag['wk_fb_seat'])}/"
                  f"{int(diag['wk_fb_cross'])}/{int(diag['wk_fb_geom'])}")
        if diag.get('wv_frames', 0):  # membrane-wave observability, mirroring runners/ccraft.py
            print(f"    [itmw-wv] frames={int(diag['wv_frames'])} empty={int(diag['wv_empty'])} "
                  f"active={int(diag['wv_active'])} reached={int(diag['wv_reached'])} "
                  f"N={int(diag['wv_n_sub'])} umax={diag['wv_umax']:.2e} "
                  f"sig={int(diag['wv_sig_n'])}/{diag['wv_sig_max']:.2e} "
                  f"cap={int(diag.get('wv_cap_edges0', 0))}->{int(diag.get('wv_cap_resid', 0))}"
                  f"/{diag.get('wv_cap_dxmax', 0.0):.1e} "
                  f"overflow={int(diag['wv_ladder_overflow'])} "
                  f"degen={int(diag['wv_degenerate'])}")

    def set_random_material(self, sample):
        """
        Add material parameters to the cloth object and the sample
        :param sample:
        :return:
        """
        sample, self.cloth_obj = self.random_material.add_material(sample, self.cloth_obj)
        return sample

    def add_cloth_obj(self, sample):
        """
        - Updates self.cloth_obj with the cloth object in the sample
        - Samples the material properties of the cloth object and adds them to the sample
        - Adds info about the garment to the sample, which is later used by the the GNN and to compute objective terms (see utils.cloth_and_material.ClothMatAug for details)
        """
        sample = self.set_random_material(sample)
        sample = self.cloth_obj.set_batch(sample, overwrite_pos=self.mcfg.overwrite_pos_every_step)
        sample['cloth'].cloth_obj = self.cloth_obj
        return sample

    def criterion_pass(self, sample_step):
        """
        Pass the sample through all the loss terms in self.criterion_dict
        and gathers their values in a dictionary
        """
        sample_step.cloth_obj = self.cloth_obj
        loss_dict = dict()
        for criterion_name, criterion in self.criterion_dict.items():
            ld = criterion(sample_step)
            for k, v in ld.items():
                loss_dict[f"{criterion_name}_{k}"] = v

        return loss_dict

    def collect_sample(self, sample, idx, prev_out_dict=None, random_ts=False):
        """
        Collects a sample from the sequence, given the previous output and the index of the current step
        This function is only used in training
        For validation, see `collect_sample_wholeseq`

        :param sample: pytroch geometric batch from the dataloader
        :param idx: index of the current step
        :param prev_out_dict: previous output of the model
        :param random_ts: if True, the time step is randomly chosen between the initial and the regular time step
        :return: the sample for the current step
        """

        sample_step = sample.clone()

        # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        # copy positions from the lookup steps
        if idx != 0:
            sample_step = self.sample_collector.lookup2target(sample_step, idx - 1)

        # in the first step, the obstacle and positions of the pinned vertices are static
        if idx == 0:
            is_init = np.random.rand() > 0.5
            sample_step = self.sample_collector.pos2target(sample_step)
            if is_init or not random_ts:
                sample_step = self.sample_collector.pos2prev(sample_step)
                ts = self.mcfg.initial_ts
        # for the second frame, we set velocity to zero
        elif idx == 1:
            sample_step = self.sample_collector.pos2prev(sample_step)

        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        return sample_step

    def optimizer_step(self, loss_dict, optimizer=None, scheduler=None):
        if optimizer is not None:
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            if self.mcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mcfg.grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    def forward(self, sample, roll_steps=1, optimizer=None, scheduler=None) -> dict:

        # for the first 5000 steps, we randomly chose between initial and regular timesteps so that model does not overfit
        # Then, we always use initial timestep for the first frame and regular timestep for the rest of the frames
        random_ts = (roll_steps == 1)

        # add
        sample = self.add_cloth_obj(sample)

        prev_out_sample = None
        for i in range(roll_steps):
            sample_step = self.collect_sample(sample, i, prev_out_sample, random_ts=random_ts)

            if i == 0:
                sample_step = self.collision_solver.solve(sample_step)

            sample_step = self.model(sample_step)
            loss_dict = self.criterion_pass(sample_step)
            prev_out_sample = sample_step.detach()

            self.optimizer_step(loss_dict, optimizer, scheduler)

        ld_to_write = {k: v.item() for k, v in loss_dict.items()}
        return ld_to_write


def inflated_scheduler_step(global_step, warmup_steps, increase_roll_every, roll_max):
    """Cumulative scheduler.step() calls after `global_step` global steps. The scheduler advances
    once per autoregressive roll step, inflating last_epoch relative to global_step. Used to
    reconstruct the correct LR position on resume when optimizer state is absent."""
    total = 0
    for g in range(max(1, int(warmup_steps)), int(global_step) + 1):
        total += min(1 + g // increase_roll_every, roll_max)
    return total


def create_optimizer(training_module: Runner, mcfg: DictConfig):
    optimizer = Adam(training_module.parameters(), lr=mcfg.lr)

    def sched_fun(step):
        decay = mcfg.decay_rate ** (step // mcfg.decay_steps) + 1e-2
        decay = max(decay, mcfg.decay_min)
        return decay

    scheduler = LambdaLR(optimizer, sched_fun)
    rcfg = training_module.mcfg
    last = inflated_scheduler_step(int(mcfg.step_start), rcfg.warmup_steps,
                                   rcfg.increase_roll_every, rcfg.roll_max)
    scheduler.last_epoch = last
    for g in optimizer.param_groups:
        g['lr'] = mcfg.lr * sched_fun(last)
    print(f'[resume] step_start={mcfg.step_start} -> scheduler.last_epoch={last}, '
          f'lr={mcfg.lr * sched_fun(last):.4e}')

    return optimizer, scheduler


def run_epoch(training_module: Runner, aux_modules: dict, dataloader: DataLoader,
              n_epoch: int, cfg: DictConfig, global_step=None):
    global_step = global_step or len(dataloader) * n_epoch

    training_module.model.train()

    optimizer = aux_modules['optimizer']
    scheduler = aux_modules['scheduler']

    prbar = tqdm(dataloader, desc=cfg.config)

    if hasattr(cfg, 'run_dir'):
        checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')
    else:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        cfg.run_dir = os.path.join(DEFAULTS.experiment_root, dt_string)
        checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')
    print(f'run_epoch started, checkpoints will be saved in {checkpoints_dir}')

    for sample in prbar:
        global_step += 1
        if cfg.experiment.max_iter is not None and global_step > cfg.experiment.max_iter:
            break

        sample = move2device(sample, cfg.device)

        # add `iter` field to the sample (used in some criterions with dynamically changing weight)
        B = sample.num_graphs
        sample = add_field_to_pyg_batch(sample, 'iter', [global_step] * B, 'cloth', reference_key=None)

        # number of autoregressive steps to simulate for the training sample
        roll_steps = 1 + (global_step // training_module.mcfg.increase_roll_every)
        roll_steps = min(roll_steps, training_module.mcfg.roll_max)

        # ld_to_write is a dictionary of loss values averaged across a training sample
        # you can feed it to tensorboard or wandb writer
        optimizer_to_pass = optimizer if global_step >= training_module.mcfg.warmup_steps else None
        scheduler_to_pass = scheduler if global_step >= training_module.mcfg.warmup_steps else None
        ld_to_write = training_module(sample, roll_steps=roll_steps, optimizer=optimizer_to_pass,
                                      scheduler=scheduler_to_pass)

        # save checkpoint every `save_checkpoint_every` steps
        if global_step % cfg.experiment.save_checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
            save_checkpoint(training_module, aux_modules, cfg, checkpoint_path)

    return global_step
