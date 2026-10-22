"""
Run ContourCraft inference on multiple AMASS pose sequences in one go.

The model is loaded once; sequences are processed in order.
Output files use the same naming scheme as inference.py, derived per sequence.

Usage examples
--------------
# Multiple sequences, all other args shared:
    pixi run python inference.py \\
        --garment celina_002_combined \\
        --sequence seq1.npz seq2.npz seq3.npz

# Skip video rendering:
    pixi run python inference.py \\
        --garment celina_002_combined \\
        --sequence seq1.npz seq2.npz \\
        --no-video

# Custom output dir, male body:
    pixi run python inference.py \\
        --garment celina_002_combined \\
        --sequence seq1.npz seq2.npz \\
        --out-dir ccraft_data/results/batch_run \\
        --body-model smpl --gender male
"""
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from utils.arguments import load_params
from utils.common import move2device
from utils.io import pickle_dump
from utils.defaults import DEFAULTS
from utils.show import write_video
from utils import trace
from utils.validation import (
    apply_material_params,
    create_postcvpr_one_sequence_dataloader,
    load_runner_from_checkpoint,
)


_DATA_ROOT       = Path(DEFAULTS.data_root)
_DEFAULT_CONFIG  = 'contourcraft'
_DEFAULT_CKPT    = str(_DATA_ROOT / 'trained_models' / 'contourcraft.pth')
_DEFAULT_OUT_DIR = str(Path(DEFAULTS.project_dir) / 'outputs')
_DEFAULT_DEVICE  = 'cuda:0'

_DEFAULT_DENSITY       = 2.20022
_DEFAULT_LAME_MU       = 23600.0
_DEFAULT_LAME_LAMBDA   = 44400.0
_DEFAULT_BENDING_COEFF = 3.962e-05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='ContourCraft inference — batch of AMASS sequences, one model load.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--garment',  required=False, default='dress',
                   help='Garment name. Use comma-separated "name::layer" for multi-layer outfits.')
    p.add_argument('--sequence', required=False, nargs='+',
                   default=['./ccraft_data/fromanypose/pose_sequence.npz'],
                   help='One or more paths to AMASS pose sequences (.npz).')

    model = p.add_argument_group('Model')
    model.add_argument('--config',     default=_DEFAULT_CONFIG)
    model.add_argument('--checkpoint', default=_DEFAULT_CKPT)

    out = p.add_argument_group('Output')
    out.add_argument('--out-dir',  default=_DEFAULT_OUT_DIR)
    out.add_argument('--no-video', action='store_true')
    out.add_argument('--skip-rollout', action='store_true',
                     help='Skip the simulation (and model load) and re-render from the existing .pkl at the '
                          'derived path. Pass the same simulation-identifying flags (--mu, --garment, '
                          '--sequence, --checkpoint); --render-force is NOT part of the pkl name. Add '
                          '--render-force to colormap the force (the pkl must contain it — produced by any '
                          'bolt-on operator rollout).')

    sim = p.add_argument_group('Simulation')
    sim.add_argument('--steps',      type=int, default=500)
    sim.add_argument('--body-model', default='smpl', choices=['smpl', 'smplx'])
    sim.add_argument('--gender',     default='female', choices=['female', 'male'])

    mat = p.add_argument_group('Material parameters')
    mat.add_argument('--density',       type=float, default=_DEFAULT_DENSITY)
    mat.add_argument('--lame-mu',       type=float, default=_DEFAULT_LAME_MU)
    mat.add_argument('--lame-lambda',   type=float, default=_DEFAULT_LAME_LAMBDA)
    mat.add_argument('--bending-coeff', type=float, default=_DEFAULT_BENDING_COEFF)

    rt = p.add_argument_group('Runtime')
    rt.add_argument('--device', default=_DEFAULT_DEVICE)

    wb = p.add_argument_group('Wandb')
    wb.add_argument('--wandb',         action='store_true')
    wb.add_argument('--wandb-project', default='ccraft-inference')

    ren = p.add_argument_group('Render')
    ren.add_argument('--multi-view',         action='store_true')
    ren.add_argument('--render-force',       action='store_true',
                     help='Colormap the per-vertex normal force (bolt-on operator lambda_n) on the garment instead '
                          'of the preset color, to visually compare normal-force methods. Requires '
                          'contact_arm.mode=bolt_on. Purely visual — rollout/positions are unchanged.')

    arm = p.add_argument_group('Friction / bolt-on operator')
    arm.add_argument('--mu', type=float, default=None,
                     help='Coulomb friction coefficient for bolt-on operator inference-time projection. '
                          'Requires --config contourcraft_armBO (or any config with '
                          'contact_arm.mode=bolt_on). Overrides the config fallback mu. '
                          'Ignored when contact_arm.mode=none (default config).')

    return p.parse_args()


def derive_run_name(args: argparse.Namespace, sequence: str) -> str:
    """Rollout / .pkl name — identifies the SIMULATION only (checkpoint, sequence, garment, body, mu).
    Render-only distinctions (e.g. --render-force) are deliberately NOT included, so the pkl path is
    stable and re-derivable for --skip-rollout. The force video gets its '_force' suffix in main()."""
    ckpt_stem    = Path(args.checkpoint).stem
    seq_stem     = Path(sequence).stem
    garment_slug = args.garment.replace(' ', '').replace('::', '-').replace(',', '+')
    name = f'{garment_slug}_{ckpt_stem}_{seq_stem}_{args.body_model}_{args.gender}'
    if args.mu is not None:
        name += f'_mu{args.mu}'
    return name


def log_wandb(project: str, run_name: str, args: argparse.Namespace,
              sequence: str, trajectories_dict: dict) -> None:
    import wandb
    wandb.init(
        project=project,
        name=run_name,
        config=dict(
            config_name=args.config,
            checkpoint=args.checkpoint,
            garment=args.garment,
            sequence=sequence,
            body_model=args.body_model,
            gender=args.gender,
            steps=args.steps,
            density=args.density,
            lame_mu=args.lame_mu,
            lame_lambda=args.lame_lambda,
            bending_coeff=args.bending_coeff,
            friction_mu=args.mu,
        ),
    )
    metrics = trajectories_dict.get('metrics', {})
    if not metrics:
        print('[wandb] No metrics found — nothing to log.')
    else:
        keys = list(metrics.keys())
        n_frames = len(metrics[keys[0]])
        for frame_idx in range(n_frames):
            wandb.log({k: metrics[k][frame_idx] for k in keys}, step=frame_idx)
    wandb.finish()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)/ args.config / args.garment / Path(args.checkpoint).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model once (skipped entirely when re-rendering an existing rollout) ──────────────
    runner = None
    garment_dicts_dir = None
    if args.skip_rollout:
        print('[skip-rollout] not loading model; re-rendering from existing .pkl(s)')
        if args.no_video:
            print('[skip-rollout] --no-video with --skip-rollout: nothing to do.')
    else:
        trace.at('utils/arguments.py', 'load_params', f'loading config {args.config}')
        material_dict = dict(
            density=args.density,
            lame_mu=args.lame_mu,
            lame_lambda=args.lame_lambda,
            bending_coeff=args.bending_coeff,
        )
        modules, experiment_config = load_params(args.config)
        experiment_config = apply_material_params(experiment_config, material_dict)
        # --mu is applied to the CONFIG, before the runner is constructed (this used to be a
        # post-construction attribute write). A mu-feature config validates its mu
        # source at construction — the feature and the operator must read one resolved mu — so a
        # post-hoc write arrives too late and the run dies with "no mu source" despite --mu being
        # given. evaluate.py has always patched the config; this matches it.
        if args.mu is not None:
            runner_name = list(experiment_config.runner.keys())[0]
            OmegaConf.set_struct(experiment_config, False)
            experiment_config.runner[runner_name].material.friction_mu_override = args.mu
        _, runner = load_runner_from_checkpoint(args.checkpoint, modules, experiment_config)

        if args.mu is not None:
            arm_mode = getattr(runner.mcfg, 'contact_arm', None)
            arm_mode = arm_mode.mode if arm_mode is not None else 'none'
            print(f'bolt-on operator: friction_mu_override={args.mu}  (contact_arm.mode={arm_mode})')

        if runner.has_force_source():
            # Always capture the per-vertex contact force (read-only) so the pkl is force-renderable
            # later via --skip-rollout --render-force, regardless of whether this run colormaps it.
            # Covers both friction arms: bolt-on operator (bolt-on projection force) and trained-friction arm (loss-only
            # grav+inertia lambda_n via the attached friction_dissipative criterion) -- see
            # runners/ccraft.py::has_force_source / _capture_step_force.
            runner._capture_force = True
        elif args.render_force:
            arm = getattr(runner.mcfg, 'contact_arm', None)
            arm_mode = arm.mode if arm is not None else 'none'
            print(f'[render-force] WARNING: contact_arm.mode={arm_mode}, arm_br not enabled -- there '
                  f'is no contact force to render; the garment would be uniform. Use a bolt-on operator config'
                  f'(e.g. --config contourcraft_armBO_v11) or an trained-friction arm checkpoint.')

        garment_dicts_dir = Path(DEFAULTS.aux_data) / 'garment_dicts' / args.body_model

    # ── create renderer once (if needed) ─────────────────────────────────────
    renderer = None
    if not args.no_video:
        from aitviewer.headless import HeadlessRenderer
        renderer = HeadlessRenderer(window_type='headless', backend='egl')

    # ── loop over sequences ───────────────────────────────────────────────────
    n = len(args.sequence)
    for i, seq_path in enumerate(args.sequence, 1):
        print(f'\n[{i}/{n}] sequence: {seq_path}')

        run_name  = derive_run_name(args, seq_path)   # rollout / pkl name (render-agnostic)
        out_pkl   = out_dir / f'{run_name}.pkl'

        trace.at('inference.py', 'main', f'writing to {out_pkl}')

        if args.skip_rollout:
            if not out_pkl.exists():
                print(f'  [skip-rollout] SKIP: {out_pkl} not found — run once without --skip-rollout '
                      f'(with the same flags) first.')
                continue
            print(f'  [skip-rollout] re-rendering from existing {out_pkl}')
        else:
            trace.at('utils/validation.py', 'create_postcvpr_one_sequence_dataloader',
                     f'loading garment {args.garment} on {seq_path}')
            dataloader = create_postcvpr_one_sequence_dataloader(
                sequence_path=seq_path,
                garment_name=args.garment,
                config=args.config,
                sequence_loader=f'cmu_npz_{args.body_model}',
                obstacle_dict_file=None,
                gender=args.gender,
                garment_dicts_dir=str(garment_dicts_dir),
            )
            sequence = next(iter(dataloader))
            sequence = move2device(sequence, args.device)

            trajectories_dict = runner.valid_rollout(sequence, bare=True, n_steps=args.steps)
            pickle_dump(dict(trajectories_dict), out_pkl)
            trace.at('inference.py', 'main', f'wrote {out_pkl}')

            if args.wandb:
                print(f'  logging to wandb project="{args.wandb_project}" run="{run_name}" ...')
                log_wandb(args.wandb_project, run_name, args, seq_path, trajectories_dict)

        if renderer is not None:
            if args.render_force:
                force_video = out_dir / f'{run_name}_force'   # '_force' lives on the VIDEO, not the pkl
                print('  rendering FORCE-colormap video ...')
                write_video(out_pkl, force_video, renderer, multi_view=args.multi_view, render_force=True)
                print(f'  video    → {force_video}.*')
                # also render the preset-color video if it hasn't been rendered yet (for side-by-side)
                if not any(out_dir.glob(f'{run_name}_[0-9]*.mp4')):
                    print('  rendering preset-color video (not found) ...')
                    write_video(out_pkl, out_dir / run_name, renderer, multi_view=args.multi_view,
                                render_force=False)
                    print(f'  video    → {out_dir / run_name}.*')
            else:
                print('  rendering video ...')
                write_video(out_pkl, out_dir / run_name, renderer, multi_view=args.multi_view,
                            render_force=False)
                print(f'  video    → {out_dir / run_name}.*')

    print(f'\nDone. {n} sequence(s) processed.')


if __name__ == '__main__':
    main()

"""
python inference.py \
    --checkpoint ccraft_data/trained_models/contourcraft.pth \
    --config contourcraft \
    --sequence AMASS/smpl/CMU/05/05_08_poses.npz \
    --garment tshirt \ 
    --mu 0.0
"""