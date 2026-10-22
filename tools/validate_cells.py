#!/usr/bin/env python
"""Roll ONE evaluation cell (garment x motion) at one friction coefficient.

Built for code validation: the smallest unit `evaluate.py` can roll is all 40 cells of a split, and
a single cell is enough to exercise the same path -- same config loader, same EVAL_MATERIAL, same
`--mu` -> `material.friction_mu_override`, same dataset patch, same `runner.valid_rollout`.

Writes the trajectory in evaluate.py's own layout, rooted at `--out` instead of
`ccraft_data/eval/<run_name>`:

    <out>/rollouts/vs_<split>/<garment>/<motion>.pkl
    <out>/RUN.json

which is what `tools/r3/render_boundary_video.py --runs 'ARM=<out>'` reads.

The (garment, motion) set is read from the datasplit CSVs, so it cannot drift from the evaluation
set; `--list` prints it. `--mu` is bounded to [0.0, 0.6].

Usage:
  pixi run python tools/validate_cells.py --list
  CUDA_VISIBLE_DEVICES=0 pixi run python tools/validate_cells.py \
      --garment tank --motion 05_06 --mu 0.3 \
      --config contourcraft_armBO_v21aTC_itmw_b0cap_ev2_walkfast \
      --checkpoint ccraft_data/trained_models/contourcraft_frictionless.pth \
      --out output_code_validation/tank_05_06_mu0.3
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import evaluate as EV                                       # noqa: E402  harness helpers, unchanged
from utils.arguments import create_dataloader_modules       # noqa: E402
from utils.common import move2device                        # noqa: E402
from utils.io import pickle_dump                            # noqa: E402
from utils import trace                                     # noqa: E402

SPLITS = {'vs_snug': 'snug', 'vs_ssch': 'ssch'}
DEFAULT_VAL_SEQS = 'ccraft_data/aux_data/evaluation_sequences'
MU_MIN, MU_MAX = 0.0, 0.6


def read_eval_set(val_seqs_dir: Path):
    """{(garment, motion): split} for every cell of the evaluation set, read from the datasplits."""
    cells = {}
    for split, sota in SPLITS.items():
        csv = EV.find_datasplit(val_seqs_dir, sota)
        rows = pd.read_csv(csv)
        for g, q in zip(rows.garment, rows.id):
            cells[(str(g), str(q))] = split
    return cells


def print_eval_set(cells):
    garments, motions = [], []
    for g, q in cells:
        if g not in garments:
            garments.append(g)
        if q not in motions:
            motions.append(q)
    print(f'--garment   {" ".join(sorted(garments))}')
    print(f'--motion    {" ".join(sorted(motions))}')
    print(f'--mu        any value in [{MU_MIN}, {MU_MAX}]')
    print(f'\n{len(cells)} cells = {len(garments)} garments x {len(motions)} motions')


def build_runner(checkpoint, config, mu, device):
    """The runner evaluate.py would build for (checkpoint, config, --mu)."""
    trace.at('utils/arguments.py', 'load_params', f'loading config {config}')
    modules, cfg = EV.load_params(config)
    cfg = EV.apply_material_params(cfg, EV.EVAL_MATERIAL)
    runner_name = list(cfg.runner.keys())[0]
    OmegaConf.set_struct(cfg, False)
    cfg.runner[runner_name].material.friction_mu_override = float(mu)
    _, runner = EV.load_runner_from_checkpoint(checkpoint, modules, cfg)
    runner = runner.to(device)
    runner.eval()
    return runner, modules, cfg


def load_cell(modules, cfg, val_seqs_dir, betas_source, split, garment, motion, device):
    """The one batch for (garment, motion). The dataloader is sequential; other rows are skipped."""
    trace.at('datasets/ccraft.py', 'create_loader',
             f'loading {garment}/{motion} from {val_seqs_dir.name} (betas {betas_source})')
    split_csv = EV.find_datasplit(val_seqs_dir, SPLITS[split])
    rows = pd.read_csv(split_csv)
    exp_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(exp_cfg, False)
    exp_cfg = EV._patch_dataset_config(exp_cfg, val_seqs_dir, split_csv, betas_source=betas_source)
    dataloader = create_dataloader_modules(modules, exp_cfg)['inference'].create_dataloader(is_eval=True)
    for i, batch in enumerate(dataloader):
        if (str(rows.garment[i]), str(rows.id[i])) == (garment, motion):
            return move2device(batch, device)
    raise SystemExit(f'{garment}/{motion} not produced by the dataloader for {split_csv}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='print the evaluation set and exit')
    ap.add_argument('--garment')
    ap.add_argument('--motion')
    ap.add_argument('--mu', type=float, help=f'friction coefficient, {MU_MIN} to {MU_MAX}')
    ap.add_argument('--config', default='contourcraft_armBO_v21aTC_itmw_b0cap_ev2_walkfast')
    ap.add_argument('--checkpoint',
                    default='ccraft_data/trained_models/contourcraft_frictionless.pth')
    ap.add_argument('--out', default=None,
                    help='run directory (default output_code_validation/<garment>_<motion>_mu<mu>)')
    ap.add_argument('--val_seqs_dir', default=DEFAULT_VAL_SEQS)
    ap.add_argument('--betas_source', default='native', choices=['table', 'native'])
    ap.add_argument('--n-steps', type=int, default=-1, help='-1 = the whole sequence')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true', help='overwrite an existing rollout')
    args = ap.parse_args()

    val_seqs_dir = Path(args.val_seqs_dir)
    if not val_seqs_dir.is_absolute():
        val_seqs_dir = _REPO / val_seqs_dir
    cells = read_eval_set(val_seqs_dir)

    if args.list:
        print_eval_set(cells)
        return

    for name in ('garment', 'motion', 'mu'):
        if getattr(args, name) is None:
            raise SystemExit(f'--{name} is required (or pass --list)')
    if not MU_MIN <= args.mu <= MU_MAX:
        raise SystemExit(f'--mu {args.mu} outside the evaluated range [{MU_MIN}, {MU_MAX}]')
    key = (args.garment, args.motion)
    if key not in cells:
        print(f'{args.garment}/{args.motion} is not in the evaluation set.\n')
        print_eval_set(cells)
        raise SystemExit(2)
    split = cells[key]

    mu_tag = f'{args.mu:g}'
    out = Path(args.out) if args.out else Path('output_code_validation') / f'{args.garment}_{args.motion}_mu{mu_tag}'
    if not out.is_absolute():
        out = _REPO / out
    pkl = out / 'rollouts' / split / args.garment / f'{args.motion}.pkl'
    if pkl.exists() and not args.force:
        raise SystemExit(f'{pkl} exists; pass --force to overwrite')
    pkl.parent.mkdir(parents=True, exist_ok=True)

    trace.at('tools/validate_cells.py', 'main',
             f'rolling {split}/{args.garment}/{args.motion} at mu {args.mu}')
    runner, modules, cfg = build_runner(args.checkpoint, args.config, args.mu, args.device)
    batch = load_cell(modules, cfg, val_seqs_dir, args.betas_source,
                      split, args.garment, args.motion, args.device)

    t0 = time.perf_counter()
    traj = runner.valid_rollout(batch, bare=True, n_steps=args.n_steps)
    wall = time.perf_counter() - t0
    fail_reason = traj['metrics'].get('fail_reason', 'SUCCESS')
    traj['metrics'].pop('timeprof', None)
    pickle_dump(dict(traj), pkl)

    (out / 'RUN.json').write_text(json.dumps(dict(
        cell=f'{split}/{args.garment}/{args.motion}', garment=args.garment, motion=args.motion,
        mu=args.mu, config=args.config, checkpoint=args.checkpoint,
        val_seqs_dir=str(val_seqs_dir), betas_source=args.betas_source,
        frames=int(len(traj.get('pred', []))), fail_reason=fail_reason,
        wall_s=round(wall, 1), when=datetime.now().isoformat(timespec='seconds'),
    ), indent=1))

    trace.at('tools/validate_cells.py', 'main',
             f'{fail_reason} -- wrote {pkl.relative_to(_REPO)} '
             f'({len(traj.get("pred", []))} frames in {wall:.1f}s)')
    if fail_reason != 'SUCCESS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
