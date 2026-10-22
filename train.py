import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="chumpy")

import numpy as np
from material.utils import init_matstack
import torch

from utils.arguments import load_from_checkpoint, load_params, create_modules, resolve_run_dir
from utils.writer import WandbWriter
from utils.defaults import DEFAULTS


def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    modules, config = load_params()

    # `restart.step_start` is the single knob for resuming; propagate it to the top-level step_start
    # up front so create_modules' optimizer/scheduler reconstruction and the writer all see it.
    if config.restart.step_start is not None:
        config.step_start = config.restart.step_start

    # single owner of run_dir: timestamped folder for a fresh run, kept as-is when passed on resume.
    resolve_run_dir(config)

    # (b): config-driven allocator settings, so collision-heavy runs (trained-friction arm) no longer
    # depend on remembering the env var at launch. Must run before first CUDA use (create_modules).
    # A manually-set PYTORCH_CUDA_ALLOC_CONF wins over the config value.
    alloc_conf = config.experiment.pytorch_cuda_alloc_conf
    if alloc_conf:
        if os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
            print(f"PYTORCH_CUDA_ALLOC_CONF already set to '{os.environ['PYTORCH_CUDA_ALLOC_CONF']}' "
                  f"— keeping it over config value '{alloc_conf}'")
        elif torch.cuda.is_initialized():
            print(f"WARNING: CUDA already initialized — config pytorch_cuda_alloc_conf='{alloc_conf}' "
                  f"cannot take effect this run")
        else:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = alloc_conf
            print(f"PYTORCH_CUDA_ALLOC_CONF='{alloc_conf}' (from config)")

    dataloader_modules, runner_module, runner, aux_modules = create_modules(modules, config)

    if 'material_stack' in aux_modules:
        aux_modules = init_matstack(config, modules, aux_modules, dataloader_modules)
        DEFAULTS.project_name = 'gaugar_finetune'

    runner, aux_modules = load_from_checkpoint(config, runner, aux_modules)

    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if config.experiment.use_writer:
        writer = WandbWriter(config)
    else:
        writer = None

    global_step = config.step_start

    torch.manual_seed(57)
    np.random.seed(57)
    for i in range(config.experiment.n_epochs):
        dataloaders_dict = dict()

        for dataloader_name, dataloader in dataloader_modules.items():
            dataloaders_dict[dataloader_name] = dataloader.create_dataloader()

        global_step = runner_module.run_epoch(runner, aux_modules, dataloaders_dict, config, writer,
                                       global_step=global_step)

        if config.experiment.max_iter is not None and global_step > config.experiment.max_iter:
            break


if __name__ == '__main__':
    main()

""" command for resume (UPDATED 26/07/11):   
WANDB_RESUME=must WANDB_RUN_ID=<wandb-id> CU0 python train.py \
    config=<cfg-name> \
    restart.checkpoint_path=<ckpt-path-to-be-resume> \
    restart.step_start=<step-start> \
    restart.load_optimizer=True \
    run_dir="$PWD/ccraft_data/experiments/<timestampt>_<cfg-name>"
"""