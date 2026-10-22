import os
from pathlib import Path

from munch import munchify

# Every path here is DERIVED from where this file sits, so this module is
# identical on every machine and cannot be clobbered by a `git checkout`.
#
# Do not hard-code a machine's paths in this file. That is what bit and
# before: each carried a permanently-uncommitted edit, so any checkout
# wiped it and any pull conflicted. Machine-specific paths go in
# utils/defaults_local.py instead -- gitignored, one per host, overriding any
# key below:
#
#     # utils/defaults_local.py
# OVERRIDES = {'CMU_root': ''}
#
# See

PROJECT_DIR = Path(__file__).resolve().parents[1]

_over = {}
_local_py = PROJECT_DIR / 'utils' / 'defaults_local.py'
if _local_py.exists():
    import importlib.util

    _spec = importlib.util.spec_from_file_location('ccraft_defaults_local', _local_py)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _over = dict(getattr(_mod, 'OVERRIDES', {}))

project_dir = _over.get('project_dir', str(PROJECT_DIR))
data_root = _over.get('data_root', os.path.join(project_dir, 'ccraft_data'))

DEFAULTS = dict()

DEFAULTS['project_name'] = _over.get('project_name', 'ccraft')


DEFAULTS['data_root'] = data_root
DEFAULTS['aux_data'] = _over.get('aux_data', os.path.join(data_root, 'aux_data'))
DEFAULTS['project_dir'] = project_dir
DEFAULTS['experiment_root'] = _over.get('experiment_root', os.path.join(data_root, 'experiments'))

DEFAULTS['CMU_root'] = _over.get('CMU_root', os.path.join(project_dir, 'AMASS', 'smpl', 'CMU'))

DEFAULTS = munchify(DEFAULTS)
