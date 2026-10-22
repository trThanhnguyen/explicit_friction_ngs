#!/usr/bin/env bash
# install_extra.sh — pip dependencies that pixi cannot manage:
#
#   1. CCCollisions      — local CUDA extension (needs cuda-samples)
#   2. PyG packages      — custom wheel index, build tags like +pt25cu124
#   3. pytorch3d         — built from git source (slow CUDA compilation)
#
# All other pip-installed packages are declared in pixi.toml [pypi-dependencies]
# and are installed automatically by `pixi install`.
#
# Usage (from repo root):
#   pixi run bash install_extra.sh
#
# Or with the env already activated:
#   bash install_extra.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. CCCollisions (local CUDA extension) ─────────────────────────────────────
# Requires cuda-samples (bundled as submodule at ./cuda-samples/).
# Run `git submodule update --init` first if the directory is empty.
echo "==> [1/3] Installing CCCollisions..."
export CUDA_SAMPLES_INC="${REPO_ROOT}/cuda-samples/Common"
pip install "${REPO_ROOT}/CCCollisions"

# ── 2. PyTorch Geometric packages (custom wheel index) ────────────────────────
# These carry +pt25cu124 build tags that only exist on data.pyg.org, not PyPI.
# Pinned to the exact build that matches pytorch==2.5.1 + cuda==12.4.
echo "==> [2/3] Installing PyG packages..."
pip install \
  pyg-library==0.2.0 \
  "torch_scatter==2.1.2+pt25cu124" \
  "torch_sparse==0.6.18+pt25cu124" \
  "torch_cluster==1.6.3+pt25cu124" \
  "torch_spline_conv==1.2.2+pt25cu124" \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# ── 3. pytorch3d (git source — expect ~10–20 min for CUDA compilation) ─────────
# Pinned to the exact commit for pytorch3d==0.7.8 (branch: stable).
# If CUDA headers are missing, see INSTALL.md Troubleshooting section.
echo "==> [3/3] Installing pytorch3d (this will take a while)..."
pip install \
  "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

echo ""
echo "All done. Verify with:"
echo "  python -c 'import pytorch3d, torch_scatter, torch_sparse, torch_cluster, cccollisions; print(\"OK\")'"
