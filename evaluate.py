"""
ContourCraft Evaluation Harness — adapts HOOD/evaluate.py for ContourCraft.

Runs in three steps:
  1. rollout  – simulate ContourCraft on HOOD validation sequences, save trajectories
  2. metrics  – compute per-frame energy + collision metrics on saved trajectories
  3. tables   – aggregate and print results

Quick start
-----------
  # Vanilla ContourCraft (mu=0.5):
  pixi run python evaluate.py --run_name vanilla \\
      --checkpoint ccraft_data/trained_models/contourcraft.pth \\
      --model_config contourcraft --mu 0.5

  # Pre-named pretrained:
  pixi run python evaluate.py --run_name vanilla --pretrained contourcraft --mu 0.5

  # Our operator on the frictionless ContourCraft backbone (mu=0.3):
  pixi run python evaluate.py --run_name ours_mu0.3 \\
      --checkpoint ccraft_data/trained_models/contourcraft_frictionless.pth \\
      --model_config contourcraft_armBO_v21aTC_itmw_b0cap_ev2_walkfast --mu 0.3

  # Re-print tables from already-computed metrics:
  pixi run python evaluate.py --run_name vanilla --tables_only

  # Quick smoke test (~5% of sequences, at least 1 per garment):
  pixi run python evaluate.py --run_name smoke \\
      --checkpoint ccraft_data/trained_models/contourcraft.pth \\
      --model_config contourcraft --mu 0.5 --subsample 0.05

Notes
-----
- Validation sequences live in HOOD's hood_data (HOOD pkl format), which auto-detection
  looks for under DEFAULTS.aux_data/validation_sequences. To evaluate on the 40-cell grid
  shipped with this repository instead, pass
      --val_seqs_dir ccraft_data/aux_data/evaluation_sequences --betas_source native
  (that grid carries a body shape per sequence, so 'native' is required, not 'table').
- Each --run_name gets its own rollout + metrics subdirectory under
  ccraft_data/eval/<run_name>/ so different configs never overwrite each other.
- The metrics step uses a lightweight postcvpr runner (aux/metrics.yaml) — the
  ccraft model is not needed there; only energy/collision criterion evaluation.
"""
import argparse
import json
import math
import os
import pickle
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.arguments import load_params, create_runner, create_dataloader_modules
from utils.common import move2device
from utils.defaults import DEFAULTS
from utils.eval_contact_metrics import (aggregate_and_print_contact_metrics,
                                        compute_contact_metrics_for_seq,
                                        save_contact_metrics)
from utils.io import pickle_dump
from utils.timeprof import PROF
from utils.metrics import (
    calc_metrics_by_seq,
    make_canonicalpos_dict,
    print_metrics_vs_baselines,
    print_metrics_vs_sota,
)
from utils.validation import apply_material_params, load_runner_from_checkpoint


# ── constants ─────────────────────────────────────────────────────────────────

# Maps --pretrained name → (checkpoint stem, default model config name)
PRETRAINED = {
    'contourcraft': ('contourcraft', 'contourcraft'),
}

# Material parameters matching the HOOD paper evaluation
EVAL_MATERIAL = dict(
    density=0.20022,
    lame_mu=23600.0,
    lame_lambda=44400.0,
    bending_coeff=3.9625778333333325e-05,
)

_CHECKPOINT_DIR = Path(DEFAULTS.data_root) / 'trained_models'
_EVAL_DIR = Path(DEFAULTS.data_root) / 'eval'


# ── path helpers ──────────────────────────────────────────────────────────────

def find_val_seqs_dir() -> Path:
    """Auto-detect the HOOD validation_sequences directory."""
    candidates = [
        Path(DEFAULTS.aux_data) / 'validation_sequences',
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(
        'Cannot find validation_sequences/. Tried:\n'
        + '\n'.join(f'  {c}' for c in candidates)
        + '\nPass --val_seqs_dir explicitly.'
    )


def find_datasplit(val_seqs_dir: Path, sota: str) -> Path:
    """Locate comparison_seqs_to_{sota}.csv, tolerating datasplits vs data_splits naming."""
    filename = f'comparison_seqs_to_{sota}.csv'
    for folder in ('datasplits', 'data_splits'):
        p = val_seqs_dir / folder / filename
        if p.exists():
            return p
    sys.exit(f'Cannot find {filename} under {val_seqs_dir}')


# ── config helpers ────────────────────────────────────────────────────────────

def _patch_dataset_config(experiment_config, val_seqs_dir: Path, split_csv: Path,
                          betas_source: str = 'table'):
    """
    Rewrite the ccraft dataset config to load HOOD validation sequences via hood_pkl loader.
    Absolute paths bypass the DEFAULTS.data_root / DEFAULTS.aux_data prefix in ccraft.py.

    :param betas_source: 'table' (default, published behavior) substitutes betas_table[betas_id]
        per datasplit row; 'native' keeps each sequence pkl's own stored betas. The evaluation
        grid shipped with this repository carries a body shape per sequence and must be read with
        'native'; numbers from the two sources are not comparable with each other.
    """
    OmegaConf.set_struct(experiment_config, False)
    ds = experiment_config.dataloaders.inference.dataset
    dataset_name = list(ds.keys())[0]
    d = ds[dataset_name]

    betas_pkl = val_seqs_dir / 'betas.pkl'

    # Absolute → Path('/abs') / '/abs2' = Path('/abs2'), bypasses prefix prepend
    d.data_root = str(val_seqs_dir / 'pose_sequences')
    d.sequence_loader = 'hood_pkl'
    d.split_path = str(split_csv)       # absolute → os.path.join ignores DEFAULTS.aux_data prefix
    d.garment_dicts_dir = 'garment_dicts/smpl'   # relative to DEFAULTS.aux_data (stays on HDD2)
    d.obstacle_dict_file = 'smpl_aux.pkl'         # relative to DEFAULTS.aux_data
    d.wholeseq = True
    d.noise_scale = 0
    d.random_betas = False
    d.n_coarse_levels = 4
    d.separate_arms = True
    d.pinned_verts = True
    if betas_source == 'native':
        # No betas_table → hood_pkl loader keeps the pkl's stored betas.
        # rest_geometries lookups elsewhere still use the datasplit betas_id (energy zero-point).
        d.betas_file = None
    elif betas_pkl.exists():
        d.betas_file = str(betas_pkl)   # absolute → Path prepend has no effect

    experiment_config.dataloader.batch_size = 1
    experiment_config.dataloader.num_workers = 0
    return experiment_config


def _check_bolt_on(runner) -> bool:
    """Return True if the ccraft runner has bolt-on friction active."""
    ca = getattr(runner.mcfg, 'contact_arm', None)
    if ca is None:
        return False
    return getattr(ca, 'mode', 'none') == 'bolt_on'


def _check_mu_feature(runner) -> bool:
    """True if the model carries the friction-mu body-edge feature (checkpoints, and trained-friction arm).

    Worth printing: for a arm --mu drives BOTH the operator's cone and the network's mu
    conditioning, from one resolved value. A run scored with the feature OFF would
    load a 14-in checkpoint into a 13-in model and fail loudly — but a run scored with the feature
    ON and no --mu falls back to bolt_on.mu, which is a silent scoring error, so the header states
    the mu source explicitly."""
    return bool(getattr(getattr(getattr(runner, 'model', None), 'mcfg', None),
                        'enable_friction_mu_feature', False))


def _check_arm_br(runner):
    """Return (enabled, lambda_source) for trained-friction arm trained friction. BR has no inference-time
    operator — the model is a plain forward pass with a mu feature — so this is purely informational
    (distinguishes a BR checkpoint from a byte-identical vanilla one in the printed header)."""
    abr = getattr(runner.mcfg, 'arm_br', None)
    if abr is None:
        return False, None
    return getattr(abr, 'enabled', False), getattr(abr, 'lambda_source', None)


# ── rollout step ──────────────────────────────────────────────────────────────

def _subsample_indices(datasplit: pd.DataFrame, frac: float) -> list:
    """Stratified, DETERMINISTIC subsample — returns global row-positions to keep.

    Mirrors HOOD evaluate.py::_subsample_split_csv EXACTLY so the two harnesses evaluate the SAME
    sequences (a fair head-to-head). Per garment (first-appearance order, matching HOOD's
    groupby(sort=False)) keep k = ceil(frac*n) rows, chosen EVENLY-SPACED over the group in CSV order
    (>=1 each). No RNG — reproducible, and identical to HOOD's pick. Previously this used
    round()+np.random.choice (unseeded), which picked a different, non-reproducible subset than HOOD."""
    indices = []
    for g in datasplit.garment.unique():
        g_idx = datasplit.index[datasplit.garment == g].tolist()   # global positions, CSV order
        n = len(g_idx)
        k = max(1, math.ceil(frac * n))
        # 2026/07/11 update subsample selection: start from index 1
        # Skip the first sequence (too long)
        sel = np.unique(np.linspace(1, n - 1, num=k).round().astype(int))
        indices.extend(g_idx[s] for s in sel)
    return sorted(indices)


def _run_rollout_for_sota(runner, modules, experiment_config, val_seqs_dir: Path,
                          split_csv: Path, datasplit: pd.DataFrame,
                          rollouts_dir: Path, device: str, subsample: float,
                          betas_source: str = 'table'):
    """Run rollout for one comparison split and save trajectory pkls."""
    # Deep-copy config so patches don't bleed across splits
    exp_cfg = OmegaConf.create(OmegaConf.to_yaml(experiment_config))
    OmegaConf.set_struct(exp_cfg, False)
    exp_cfg = _patch_dataset_config(exp_cfg, val_seqs_dir, split_csv, betas_source=betas_source)

    dl_ms = create_dataloader_modules(modules, exp_cfg)
    dataloader = dl_ms['inference'].create_dataloader(is_eval=True)

    indices_set = (
        set(_subsample_indices(datasplit, subsample))
        if subsample < 1.0
        else set(range(len(datasplit)))
    )

    runner.eval()
    pbar = tqdm(total=len(indices_set), desc='  rollout')

    for i, batch in enumerate(dataloader):
        if i not in indices_set:
            continue

        seq_id = datasplit.id[i]
        garment = datasplit.garment[i]

        out_dir = rollouts_dir / garment
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'{seq_id}.pkl'

        # Resume: skip a cell only if everything it should have produced is there. Without the
        # second clause a profiling re-run over an existing tree would report a clean sweep and
        # write zero timing rows.
        if out_file.exists() and not (PROF.enabled and not _timeprof_path(out_file).exists()):
            pbar.update(1)
            continue

        batch = move2device(batch, device)
        traj = runner.valid_rollout(batch, bare=True)

        fail_reason = traj['metrics'].get('fail_reason', 'SUCCESS')
        if fail_reason != 'SUCCESS':
            tqdm.write(f'    [{fail_reason}] {garment}/{seq_id} -- rollout aborted, partial trajectory saved')

        # Time profile (--time_profile): written as a JSON sidecar and POPPED from the metrics dict,
        # so a profiled rollout pkl stays structurally identical to every un-profiled arm's. The
        # sidecar is the artifact the timing report reads; it survives pruning rollouts.
        tp = traj['metrics'].pop('timeprof', None)
        if tp is not None:
            tp['meta'].update(garment=garment, seq=seq_id, T=len(traj.get('pred', [])),
                              fail_reason=fail_reason)
            with open(_timeprof_path(out_file), 'w') as fh:
                json.dump(tp, fh)

        pickle_dump(dict(traj), out_file)
        pbar.update(1)

    pbar.close()


# ── metrics step ──────────────────────────────────────────────────────────────

def _get_metrics_runner_and_sample(val_seqs_dir: Path, device: str):
    """
    Build a postcvpr metrics runner (aux/metrics.yaml) and a sample template.
    The sample template's geometry is overwritten per-sequence in calc_metrics_by_seq,
    so any single valid validation sequence can seed it.
    """
    metrics_modules, metrics_config = load_params('aux/metrics')
    metrics_config = apply_material_params(metrics_config, EVAL_MATERIAL)

    # Patch to a real datasplit so the dataloader yields at least one sample
    split_csv = find_datasplit(val_seqs_dir, 'snug')
    metrics_config = _patch_dataset_config(metrics_config, val_seqs_dir, split_csv)
    # Only need one sample — stop after the first row
    ds = metrics_config.dataloaders.inference.dataset
    dataset_name = list(ds.keys())[0]
    # Leave split_path as-is; the full snug split is fine (we only call next(iter()))

    _, metrics_runner, _ = create_runner(metrics_modules, metrics_config, create_aux_modules=False)

    dl_ms = create_dataloader_modules(metrics_modules, metrics_config)
    dl = dl_ms['inference'].create_dataloader(is_eval=False)
    sample = next(iter(dl))
    sample = move2device(sample, device)

    return metrics_runner, sample


def _timeprof_path(out_file: Path) -> Path:
    return out_file.with_name(out_file.stem + '.timeprof.json')


def _coverage_path(out_file: Path) -> Path:
    return out_file.with_name(out_file.stem + '_coverage.json')


def _seq_fail_reason(seq_dict: dict) -> str:
    return seq_dict.get('metrics', {}).get('fail_reason', 'SUCCESS')


def _write_coverage(out_file: Path, n_scored: int, skipped: list):
    """Persist which sequences were excluded (and why) next to out_file, so tables can print a
    coverage line even under --tables_only, without re-scanning the (possibly large) rollout pkls.
    See."""
    payload = dict(
        n_scored=n_scored,
        n_total=n_scored + len(skipped),
        skipped=[dict(garment=g, seq=s, fail_reason=r) for g, s, r in skipped],
    )
    with open(_coverage_path(out_file), 'w') as f:
        json.dump(payload, f, indent=2)
    if skipped:
        print(f'  {len(skipped)} sequence(s) excluded from scoring (rollout aborted mid-sequence):')
        for g, s, r in skipped:
            print(f'    [{r}] {g}/{s}')


def _compute_metrics_for_sota(metrics_runner, sample_template, rollouts_dir: Path,
                               val_seqs_dir: Path, datasplit: pd.DataFrame,
                               out_file: Path, sota: str, device: str):
    """Compute energy + collision metrics for all saved trajectories in one split.

    A rollout whose fail_reason != SUCCESS (safecheck aborted it mid-sequence — see
    runners/ccraft.py::_rollout) is excluded here rather than scored: its per-sequence mean would
    only cover the frames before it blew up, which silently rewards whichever arm diverged
    earliest. See. Excluded sequences never enter
    `all_metrics` (so downstream aggregation/tools need no fail_reason-aware filtering of their
    own) and are instead recorded in a coverage manifest next to out_file.
    """
    rest_geo_path = val_seqs_dir / 'rest_geometries' / f'{sota}.pkl'
    if not rest_geo_path.exists():
        print(f'  WARNING: {rest_geo_path} not found — skipping metrics for {sota}')
        return

    with open(rest_geo_path, 'rb') as f:
        canonicalpos_by_beta = pickle.load(f)

    canonicalpos_dict = make_canonicalpos_dict(canonicalpos_by_beta, datasplit)

    all_metrics: dict = defaultdict(lambda: defaultdict(dict))
    skipped = []   # (garment, seq_id, fail_reason)

    garment_dirs = sorted(d for d in rollouts_dir.iterdir() if d.is_dir())
    if not garment_dirs:
        print(f'  No rollouts found in {rollouts_dir}')
        return

    for garment_dir in tqdm(garment_dirs, desc=f'  metrics vs {sota}'):
        garment = garment_dir.name
        for seq_pkl in sorted(garment_dir.glob('*.pkl')):
            seq_id = seq_pkl.stem
            if garment not in canonicalpos_dict or seq_id not in canonicalpos_dict[garment]:
                warnings.warn(f'No rest_pos for {garment}/{seq_id} in {sota}.pkl — skipping')
                continue

            with open(seq_pkl, 'rb') as f:
                seq_dict = pickle.load(f)
            fail_reason = _seq_fail_reason(seq_dict)
            if fail_reason != 'SUCCESS':
                skipped.append((garment, seq_id, fail_reason))
                continue

            rest_pos = canonicalpos_dict[garment][seq_id][0]
            metrics = calc_metrics_by_seq(str(seq_pkl), sample_template, metrics_runner,
                                          rest_pos, device=device, seq_dict=seq_dict)
            all_metrics[garment][seq_id] = metrics

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(dict(all_metrics), f)
    print(f'  saved → {out_file}')

    n_scored = sum(len(seqs) for seqs in all_metrics.values())
    _write_coverage(out_file, n_scored, skipped)


# ── contact metrics (position-based; matches HOOD) ──────────────────────
#
# These reproduce HOOD's proposal- contact metrics (penetration/contact rate, orientation-split
# sliding S / stick ratio, predicted-force fidelity) so a ContourCraft summary carries the same columns.
# They are computed purely from the saved rollout positions (the ccraft rollout already bakes bolt-on
# friction into `pred`), using HOOD's contact-edge definition rebuilt from positions — see
# utils/eval_contact_metrics.py for the why. Force metrics are NaN (force-free model), matching HOOD's
# bolt-on operator rows. This is pure evaluation — no rollout/model change.

def _load_body_vertex_type(val_seqs_dir: Path):
    """Body-vertex type array for the OBSTACLE_OMIT contact-edge mask (same smpl_aux.pkl the dataset
    uses; byte-identical to HOOD's). Read from DEFAULTS.aux_data, matching _patch_dataset_config's
    obstacle_dict_file='smpl_aux.pkl'."""
    aux_pkl = Path(DEFAULTS.aux_data) / 'smpl_aux.pkl'
    if not aux_pkl.exists():
        print(f'  WARNING: {aux_pkl} not found — contact metrics will keep all body verts (no'
              f'OBSTACLE_OMIT mask).')
        return None
    with open(aux_pkl, 'rb') as f:
        obstacle_dict = pickle.load(f)
    return obstacle_dict.get('vertex_type')


def _compute_contact_metrics_for_sota(rollouts_dir: Path, out_file: Path,
                                      body_vertex_type, device: str):
    """Compute the contact metrics for every saved trajectory in one split and save them.

    Same exclusion as _compute_metrics_for_sota, and for the same reason: these pkls are the same
    rollouts, so a fail_reason != SUCCESS sequence is skipped here too rather than being averaged
    in as a partial trajectory. The coverage manifest is written once by _compute_metrics_for_sota
    (identical skip set, no need to duplicate it here).
    """
    all_metrics: dict = defaultdict(dict)
    garment_dirs = sorted(d for d in rollouts_dir.iterdir() if d.is_dir())
    if not garment_dirs:
        print(f'  No rollouts found in {rollouts_dir}')
        return

    for garment_dir in tqdm(garment_dirs, desc=f'  contact {rollouts_dir.name}'):
        garment = garment_dir.name
        for seq_pkl in sorted(garment_dir.glob('*.pkl')):
            seq_id = seq_pkl.stem
            with open(seq_pkl, 'rb') as f:
                seq_dict = pickle.load(f)
            if _seq_fail_reason(seq_dict) != 'SUCCESS':
                continue
            all_metrics[garment][seq_id] = compute_contact_metrics_for_seq(
                seq_dict, body_vertex_type, device=device)

    save_contact_metrics(out_file, all_metrics)
    print(f'  saved → {out_file}')


def _print_coverage(metrics_dir: Path):
    """Print how many sequences actually scored vs. how many rollouts were attempted, and list any
    that were excluded for aborting mid-rollout. Reads the manifest _compute_metrics_for_sota wrote,
    so this also works under --tables_only. Silently does nothing for metrics computed by an
    evaluate.py predating this fix (no manifest) — coverage reporting is new, not a hard
    requirement of the pkl format. See."""
    blocks = []
    for sota in ('snug', 'ssch'):
        cov_path = metrics_dir / f'vs_{sota}_coverage.json'
        if not cov_path.exists():
            continue
        with open(cov_path) as f:
            cov = json.load(f)
        n_total, n_scored, skipped = cov['n_total'], cov['n_scored'], cov['skipped']
        line = f'  vs {sota.upper()}: {n_scored}/{n_total} scorable'
        if skipped:
            line += f' — {len(skipped)} excluded (aborted mid-rollout):'
        blocks.append((line, skipped))

    if not blocks:
        return
    print('\nCoverage (rollouts scored vs. attempted):')
    for line, skipped in blocks:
        print(line)
        for rec in skipped:
            print(f"    [{rec['fail_reason']}] {rec['garment']}/{rec['seq']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='ContourCraft evaluation harness.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--run_name', required=True,
                   help='Unique name for this eval run (determines output directory).')
    p.add_argument('--checkpoint', default=None,
                   help='Path to checkpoint .pth file.')
    p.add_argument('--model_config', default=None,
                   help='Config name (without .yaml) to load the model.')
    p.add_argument('--pretrained', default=None, choices=list(PRETRAINED),
                   help='Use a named pretrained model — sets checkpoint and model_config.')
    p.add_argument('--mu', type=float, default=None,
                   help='Coulomb friction mu override. Set 0.5 for vanilla, '
                        '0.0 for bolt-on (bolt-on operator) experiments.')
    p.add_argument('--subsample', type=float, default=1.0,
                   help='Fraction of sequences to evaluate (stratified by garment; '
                        '>=1 per garment). Use 0.05 for a quick smoke test.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--val_seqs_dir', default=None,
                   help='Path to HOOD validation_sequences directory. '
                        'Auto-detected if not given.')
    p.add_argument('--betas_source', default='table', choices=['table', 'native'],
                   help="'table' (default): betas_table[betas_id] per datasplit row — the "
                        "published behavior. 'native': keep each sequence pkl's own betas -- "
                        "required for the evaluation grid shipped here. Numbers from the two "
                        "sources are NOT comparable with each other.")
    p.add_argument('--time_profile', action='store_true',
                   help='Record a per-stage time profile of each rollout (utils/timeprof.py) and '
                        'write a <seq>.timeprof.json sidecar next to each trajectory pkl. Off by '
                        'default and byte-identical to the un-instrumented runner when off. '
                        'Assemble with an internal tool See.')
    p.add_argument('--skip_rollout', action='store_true',
                   help='Skip rollout (use existing saved trajectories).')
    p.add_argument('--skip_metrics', action='store_true',
                   help='Skip metrics computation.')
    p.add_argument('--tables_only', action='store_true',
                   help='Skip ; only reprint tables from saved metrics.')
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve checkpoint + model_config ──────────────────────────────────────
    if args.pretrained is not None:
        ckpt_stem, model_config = PRETRAINED[args.pretrained]
        checkpoint = str(_CHECKPOINT_DIR / f'{ckpt_stem}.pth')
        if args.model_config is not None:
            model_config = args.model_config   # --model_config overrides pretrained default
    else:
        if not args.tables_only and (args.checkpoint is None or args.model_config is None):
            sys.exit('ERROR: Specify either --pretrained or both --checkpoint and --model_config.')
        checkpoint = args.checkpoint
        model_config = args.model_config

    device = args.device
    if args.time_profile:
        # Preallocate the CUDA event pool BEFORE any rollout, so its one-off cost is never charged
        # to a measured stage.
        PROF.enable(device=device)
        PROF.set_run_meta(
            run_name=args.run_name, checkpoint=checkpoint, model_config=model_config,
            mu=args.mu, betas_source=args.betas_source, subsample=args.subsample,
            val_seqs_dir=str(args.val_seqs_dir or ''), device=device,
            gpu_name=(torch.cuda.get_device_name(device) if torch.cuda.is_available() else ''),
            driver=os.environ.get('NVIDIA_DRIVER_VERSION', ''),
            host=os.uname().nodename, n_cpu=os.cpu_count(),
            # Co-tenancy is a property of how the sweep was launched, not something a process can
            # observe. Declared by the launcher, mirroring p42_timeprof's `concurrency` column —
            # a contended sweep must never be mistakable for a clean one.
            concurrency=int(os.environ.get('TIMEPROF_CONCURRENCY', '1')),
            started_at=datetime.now().strftime('%F %T'),
            torch=torch.__version__,
        )

    # ── output paths ──────────────────────────────────────────────────────────
    eval_dir = _EVAL_DIR / args.run_name
    rollouts_dir = eval_dir / 'rollouts'
    metrics_dir = eval_dir / 'metrics'
    contact_dir = eval_dir / 'contact_metrics'   # contact metrics (HOOD parity)
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f'Eval dir: {eval_dir}')

    # ── validation sequences dir ───────────────────────────────────────────────
    # .resolve(): a RELATIVE --val_seqs_dir would leak relative split/data_root paths into the
    # dataset config, where ccraft.py prepends DEFAULTS.aux_data -> doubled path.
    val_seqs_dir = Path(args.val_seqs_dir).resolve() if args.val_seqs_dir else find_val_seqs_dir()
    print(f'Val seqs : {val_seqs_dir}')
    print(f'Betas    : {args.betas_source}'
          + ('  (sequence-native betas — NOT comparable to table-betas runs)'
             if args.betas_source == 'native' else ''))

    # ── load rollout model (skip if tables-only or both steps are skipped) ─────
    need_rollout_model = not args.tables_only and not args.skip_rollout
    if need_rollout_model:
        print(f'\nLoading config    : {model_config}')
        print(f'Loading checkpoint: {checkpoint}')

        modules, experiment_config = load_params(model_config)
        experiment_config = apply_material_params(experiment_config, EVAL_MATERIAL)

        if args.mu is not None:
            runner_name = list(experiment_config.runner.keys())[0]
            OmegaConf.set_struct(experiment_config, False)
            experiment_config.runner[runner_name].material.friction_mu_override = args.mu

        _, runner = load_runner_from_checkpoint(checkpoint, modules, experiment_config)
        runner = runner.to(device)

        bolt_on = _check_bolt_on(runner)
        arm_br_on, arm_br_lambda_source = _check_arm_br(runner)
        mu_str = f'mu_override={args.mu}' if args.mu is not None else 'mu from config'
        if arm_br_on:
            print(f'trained-friction arm: ON trained{arm_br_lambda_source}), {mu_str} — plain forward pass, '
                  f'no inference-time operator')
        else:
            print(f'bolt-on operator:{"ON  bolt_on, " + mu_str if bolt_on else "off (vanilla)"}')
        if _check_mu_feature(runner) and not arm_br_on:
            # one resolved mu drives the cone AND the network's conditioning.
            span = getattr(runner, '_mu_feature_span', lambda: ('?', '?'))()
            print(f'mu feat : ON  (body edges 13->14), span {span}, driven by the SAME resolved mu '
                  f'as the operator — {mu_str}')
            if args.mu is None:
                print('  WARNING: no --mu given; both the operator and the mu feature fall back to '
                      'contact_arm.bolt_on.mu. Pass --mu to sweep.')

        if runner.has_force_source():
            # Always capture the per-vertex contact force (read-only, cheap) so saved rollout pkls
            # are force-renderable later via render_test_set.py --eval-dir --render-force, regardless
            # of whether this evaluate.py run itself renders anything. Covers both friction arms:
            # bolt-on operator (bolt-on projection force) and trained-friction arm (loss-only grav+inertia lambda_n, captured
            # via the attached friction_dissipative criterion -- see runners/ccraft.py::has_force_source).
            runner._capture_force = True

    # ──: rollout ───────────────────────────────────────────────────────────
    if need_rollout_model:
        print('\n─── Rollout ───────────────────────────────────────────────────────────')
        for sota in ('snug', 'ssch'):
            split_csv = find_datasplit(val_seqs_dir, sota)
            datasplit = pd.read_csv(split_csv)
            sota_rollouts_dir = rollouts_dir / f'vs_{sota}'
            sota_rollouts_dir.mkdir(parents=True, exist_ok=True)
            print(f'  vs {sota.upper()} ({len(datasplit)} sequences):')
            _run_rollout_for_sota(
                runner, modules, experiment_config,
                val_seqs_dir, split_csv, datasplit,
                sota_rollouts_dir, device, args.subsample,
                betas_source=args.betas_source,
            )
            print(f'  → {sota_rollouts_dir}')

    # ──: metrics ───────────────────────────────────────────────────────────
    if not args.tables_only and not args.skip_metrics:
        print('\n─── Metrics ───────────────────────────────────────────────────────────')
        metrics_runner, sample_template = _get_metrics_runner_and_sample(val_seqs_dir, device)

        metrics_dir.mkdir(parents=True, exist_ok=True)
        for sota in ('snug', 'ssch'):
            sota_rollouts_dir = rollouts_dir / f'vs_{sota}'
            if not sota_rollouts_dir.exists():
                print(f'  skipping {sota}: {sota_rollouts_dir} not found')
                continue
            split_csv = find_datasplit(val_seqs_dir, sota)
            datasplit = pd.read_csv(split_csv)
            out_file = metrics_dir / f'vs_{sota}.pkl'
            _compute_metrics_for_sota(
                metrics_runner, sample_template,
                sota_rollouts_dir, val_seqs_dir, datasplit,
                out_file, sota, device,
            )

        # contact metrics (position-based; HOOD parity). Computed from the same saved rollouts.
        contact_dir.mkdir(parents=True, exist_ok=True)
        body_vertex_type = _load_body_vertex_type(val_seqs_dir)
        for sota in ('snug', 'ssch'):
            sota_rollouts_dir = rollouts_dir / f'vs_{sota}'
            if not sota_rollouts_dir.exists():
                continue
            _compute_contact_metrics_for_sota(
                sota_rollouts_dir, contact_dir / f'vs_{sota}.pkl', body_vertex_type, device)

    # ──: tables ────────────────────────────────────────────────────────────
    print('\n─── Tables ────────────────────────────────────────────────────────────')

    vs_snug = metrics_dir / 'vs_snug.pkl'
    vs_ssch = metrics_dir / 'vs_ssch.pkl'
    if not vs_snug.exists() or not vs_ssch.exists():
        missing = [str(p) for p in [vs_snug, vs_ssch] if not p.exists()]
        print(f'  Missing metrics files: {missing}')
        print('  Run the rollout first (remove --tables_only / --skip_metrics).')
    else:
        _print_coverage(metrics_dir)

        print('\nTable 1 & 2 (energy metrics — vs baseline methods):')
        print_metrics_vs_baselines(str(metrics_dir))

        print('\nTable supplementary (collision metrics — vs SNUG/SSCH):')
        print_metrics_vs_sota(str(metrics_dir))

    # contact metrics table (all garments + only dress). Printed if the contact pkls exist.
    if (contact_dir / 'vs_snug.pkl').exists() or (contact_dir / 'vs_ssch.pkl').exists():
        print('\nContact metrics (proposal v7 — HOOD parity):')
        aggregate_and_print_contact_metrics(str(contact_dir))

    print(f'\nDone. All outputs in {eval_dir}')


if __name__ == '__main__':
    main()
