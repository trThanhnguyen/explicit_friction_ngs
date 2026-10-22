# Explicit Frictional Contact Modeling in Neural Garment Simulation

Reference implementation of the Modular Contact Operator: a friction layer that attaches to a
**frozen** neural garment predictor at inference. It estimates the normal load, applies a
closed-form Coulomb correction along a path across body triangles, and propagates the induced
motion through the cloth as a linearized membrane wave — so the friction coefficient µ becomes a
value you set at run time, with no retraining and no iterative solve.

The predictor itself is unchanged upstream code: the ContourCraft README is kept verbatim at
[README_ContourCraft.md](README_ContourCraft.md), and installation and data follow
[INSTALL.md](INSTALL.md).

---

## Environment

Installation of the libraries and the ContourCraft data is described in [INSTALL.md](INSTALL.md).
That document offers three conda routes; the environment **actually used for every result here** is
pixi, which is what `pixi.toml` / `pixi.lock` in this repository pin:

```bash
pixi install
pixi run bash install_extra.sh      # CCCollisions, PyG and pytorch3d — the three pip installs pixi cannot manage
```

`install_extra.sh` needs `CCCollisions` and `cuda-samples` checked out; INSTALL.md §"Install
CCCollision" has the two `git clone` lines. After that either enter the environment once with
`pixi shell`, or prefix each command with `pixi run` as written below.

Select the GPU with `CUDA_VISIBLE_DEVICES` and always pass `--device cuda:0`.

## What you need to download

Nothing in this repository is a model or a body model. Three things must be placed next to the code
before anything will run:

1. **ContourCraft auxiliary data and SMPL(-X) body models** — the download links, the directory
   layout and the licence steps are in [INSTALL.md](INSTALL.md#download-data). This gives you
   `ccraft_data/aux_data/{body_models,garment_dicts,garment_meshes,smpl_aux.pkl}` and the released
   `ccraft_data/trained_models/{contourcraft,hood_final,hood_cvpr}.pth`.

2. **AMASS CMU pose sequences** — [INSTALL.md](INSTALL.md#amass-cmu-dataset). Needed for training,
   and for running `inference.py` on a motion of your choice.

3. **This paper's checkpoint**:

   Attached to the [`v1.0-wacv2027` release](../../releases/tag/v1.0-wacv2027):

   | unpack to | what it is |
   |---|---|
   | `ccraft_data/trained_models/contourcraft_frictionless.pth` | the ContourCraft backbone retrained with no friction term — the predictor our operator attaches to |

   No pose sequence ships here. The AMASS licence does not permit redistributing its sequences,
   so obtain motion from [AMASS](https://amass.is.tue.mpg.de/) directly (item 2 above) and point
   `--sequence` at whichever take you want.

   The 40-cell evaluation grid (`ccraft_data/aux_data/evaluation_sequences/`) *is* committed here —
   it is 5 MB and the code cannot enumerate the evaluation set without it.

The released HOOD checkpoint used below, `ccraft_data/trained_models/hood_final.pth`, comes from
item 1; it is not ours and is used as distributed.

---

## Quick start

Every command below writes to `output_code_validation/<run>/`, one folder per run. Common paths:

```bash
CC_CFG=contourcraft_armBO_v21aTC_itmw_b0cap_ev2_walkfast
CC_CKPT=ccraft_data/trained_models/contourcraft_frictionless.pth
HOOD_CFG=hood_v1_armBO_v21aTC_itmw_b0cap_ev2_walkfast
HOOD_CKPT=ccraft_data/trained_models/hood_final.pth
```

`tools/validate_cells.py` rolls one cell of the evaluation grid (a garment, a motion and a μ);
`tools/r3/render_boundary_video.py` renders the resulting rollouts side by side, one column per μ.
The two examples below use the cells shown in Figure 6 of the paper.

Tank top, ContourCraft + our operator, μ ∈ {0.0, 0.3}:

```bash
for MU in 0.0 0.3; do
  CUDA_VISIBLE_DEVICES=0 pixi run python tools/validate_cells.py \
    --garment tank --motion 05_06 --mu $MU \
    --config $CC_CFG --checkpoint $CC_CKPT --device cuda:0 \
    --out output_code_validation/tank_mu$MU
done

CUDA_VISIBLE_DEVICES=0 pixi run python tools/r3/render_boundary_video.py \
  --cell vs_snug/tank/05_06 --arms ours_walk --mus 0.0,0.3 \
  --runs 'ours_walk=output_code_validation/tank_mu{mu}' \
  --view 80 --view-frame 234 --elevation 8 --fill 0.88 --floor --floor-track --size 900x1200 --fps 30 \
  --out output_code_validation/tank
```

Long sleeve, HOOD + our operator, μ ∈ {0.1, 0.6}:

```bash
for MU in 0.1 0.6; do
  CUDA_VISIBLE_DEVICES=0 pixi run python tools/validate_cells.py \
    --garment longsleeve --motion 143_27 --mu $MU \
    --config $HOOD_CFG --checkpoint $HOOD_CKPT --device cuda:0 \
    --out output_code_validation/longsleeve_mu$MU
done

CUDA_VISIBLE_DEVICES=0 pixi run python tools/r3/render_boundary_video.py \
  --cell vs_snug/longsleeve/143_27 --arms hood_ours_walk --mus 0.1,0.6 \
  --runs 'hood_ours_walk=output_code_validation/longsleeve_mu{mu}' \
  --view 0 --view-frame 208 --elevation 8 --fill 0.88 --floor --floor-track --size 900x1200 --fps 30 \
  --out output_code_validation/longsleeve
```

Add `--keep-frames` to keep the PNG frames next to the mp4.

### Other cells

`--cell` is `vs_ssch/dress/<motion>` for the dress and `vs_snug/<garment>/<motion>` for the rest.
Selectable values — `pixi run python tools/validate_cells.py --list` prints them:

| | |
|---|---|
| `--garment` | `tshirt` `tank` `longsleeve` `dress` |
| `--motion` | `01_01` `05_06` `05_08` `07_02` `79_04` `104_54` `127_02` `133_02` `136_12` `143_27` |
| `--mu` | any value in `[0.0, 0.6]` |

Anything outside these is rejected.

### Your own garment or motion

For a garment or motion outside the evaluation grid, use `inference.py` with any SMPL sequence
from AMASS. It writes the rollout and an mp4 under `--out-dir`:

```bash
for MU in 0.05 0.5; do
  CUDA_VISIBLE_DEVICES=0 pixi run python inference.py \
    --garment longskirt --sequence AMASS/smpl/CMU/<subject>/<take>.npz \
    --config $CC_CFG --checkpoint $CC_CKPT --mu $MU \
    --out-dir output_code_validation/longskirt --device cuda:0
done
```

`pixi run python inference.py --help` lists the body-model, gender, material and length options.

### Operator configs

`$CC_CFG` is the full operator. The ablation configs drop its stages cumulatively and take the
same commands:

| stages | config |
|---|---|
| Coulomb correction | `contourcraft_armBO_v21aTC_core_ev2_chord` |
| + traversal | `contourcraft_armBO_v21aTC_ev2_walkfast` |
| + propagation | `contourcraft_armBO_v21aTC_itmw_b0_ev2_walkfast` |
| + strain limiting (full) | `contourcraft_armBO_v21aTC_itmw_b0cap_ev2_walkfast` |

The `hood_v1_*` configs are the same operator on the HOOD backbone.

---

## Evaluation

`evaluate.py` is the full harness: rollout → metrics → tables. It rolls a whole split, so it is
much heavier than `validate_cells.py` (which exists precisely because `evaluate.py` cannot select a
single cell).

```bash
# our operator on the ContourCraft backbone, over the evaluation grid at mu = 0.3
CUDA_VISIBLE_DEVICES=0 pixi run python evaluate.py --run_name ours_mu0.3 \
  --checkpoint ccraft_data/trained_models/contourcraft_frictionless.pth \
  --model_config contourcraft_armBO_v21aTC_itmw_b0cap_ev2_walkfast \
  --val_seqs_dir ccraft_data/aux_data/evaluation_sequences --betas_source native \
  --mu 0.3 --device cuda:0

# re-print tables from metrics already computed
pixi run python evaluate.py --run_name ours_mu0.3 --tables_only
```

**Both of those flags are required.** Left out, `evaluate.py` falls back to auto-detecting HOOD's
own `validation_sequences` directory — which this distribution does not ship — and to
`--betas_source table`. The grid committed here carries a body shape per sequence, so it must be
read with `native`. `tools/validate_cells.py` already defaults to both.

Each `--run_name` gets its own subdirectory under `ccraft_data/eval/<run_name>/`. Note that
`evaluate.py` **resumes** silently into an existing run directory — clear it before re-measuring.

---

## Training

Our operator requires no training: it attaches to a frozen predictor. What can be trained here is
the **ContourCraft backbone**, using the upstream two-stage recipe, unmodified — see
[TRAINING.md](TRAINING.md):

```bash
pixi run python train.py config=ccraft_train/stage1
pixi run python train.py config=ccraft_train/stage2
```

Point any of the operator configs above at the resulting checkpoint to run it with friction. The
checkpoint distributed with this repository is a ContourCraft backbone retrained with the friction
term removed, so that its frictional response cannot be inherited from training.

---

## Repository layout

```
utils/contact.py          the operator: Coulomb correction, traversal, normal completion
utils/itmw_wave.py        the membrane-wave propagation stage
tools/itmw/               its fused Triton kernels
runners/ccraft.py         the rollout loop the operator is attached to
configs/                  the shipped arm, the four ablation rungs, and upstream configs
tools/validate_cells.py   roll one evaluation cell
tools/r3/                 the side-by-side renderer and its helpers
evaluate.py               the full evaluation harness
```

`models/`, `runners/`, `criterions/` and `datasets/` are resolved **by name from the config at run
time**, so a module with no static importer is not unused.

## Citation

The paper appears at WACV 2027. Until the proceedings are published, cite it as to-appear:

```bibtex
@inproceedings{truong2027explicit,
  title     = {Explicit Frictional Contact Modeling in Neural Garment Simulation},
  author    = {Truong, Thanh-Nguyen and Tran, Vu-Hoang and Huang, Ching-Chun},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on
               Applications of Computer Vision (WACV)},
  year      = {2027},
  note      = {To appear}
}
```

Paper and supplementary material: **`<PAPER LINK - TBD>`**

Drop the `note` field once the proceedings are out and the page range is available. Please also
cite ContourCraft and HOOD, as below.

## License

MIT, as inherited from [ContourCraft](https://github.com/Dolorousrtur/ContourCraft) — see
[LICENSE](LICENSE). This repository is a derivative of ContourCraft, which itself follows the
structure of [HOOD](https://github.com/Dolorousrtur/HOOD); please cite both (the BibTeX entries are
in [README_ContourCraft.md](README_ContourCraft.md)).
