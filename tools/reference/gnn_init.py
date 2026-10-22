"""Re-initialise a GNN rollout from the Ando reference's drape state.

WHY THIS EXISTS
---------------
The reference solver cannot start where the GNN starts. The GNN is handed the
LBS-skinned frame 0, which penetrates the body (22-40 tri-tri intersections on
every VTO sequence -- see an internal tool), and a barrier method
cannot be initialised in a penetrating state. So the reference reaches frame 0 by
simulation instead, and lands on a physical drape that sits ~35 mm rms away from
the LBS state.

That difference is not cosmetic, and leaving it in place would mean the reference
and the GNN are rolling out from materially different worlds. Owner's decision
(2026-07-26) is to re-initialise the GNN from the SAME drape state rather than
write the difference off.

⚠️ PARITY WARNING. Whatever is compared against -- HOOD, EUNet, SENC,
ContourCraft, or this repo's own bolt-on operator/ -- must be given the same treatment.
If only some methods get the drape initial condition, the comparison silently
rewards whichever method happens to like the LBS one. That is a scoring artifact,
not a result.

USAGE
-----
Produce the state with the reference:

    <reference solver> --data <prep.npz> --mu 0.0 --init-out drape_init.npz ...

then point a rollout config at it:

    runner:
      ccraft:
        init_state_file: the reference outputs

Leaving `init_state_file` unset changes nothing -- the off path is byte-identical
to the prior behaviour.
"""

import numpy as np
import torch


def load_drape_init(path):
    """Read a --init-out npz. Returns (prev_pos, pos) as float32 (V,3) arrays."""
    d = np.load(path)
    for k in ("garment_prev", "garment_init"):
        if k not in d.files:
            raise KeyError(f"{path} is not an Ando --init-out file (missing {k!r}; "
                           f"has {sorted(d.files)})")
    return (np.asarray(d["garment_prev"], dtype=np.float32),
            np.asarray(d["garment_init"], dtype=np.float32))


def apply_drape_init(sample_step, path, *, verbose=True):
    """Overwrite cloth prev_pos/pos in-place with the reference's drape state.

    Applied AFTER the runner's own first-step setup, so it wins over whatever the
    dataset placed there. Vertex order is the garment dict's own order on both
    sides (the reference de-permutes ppf's internal ordering via
    ppf_env.object_vertex_map before exporting), so no remapping is needed --
    but the count is checked, because a silent mismatch here would be
    catastrophic and invisible.
    """
    prev, cur = load_drape_init(path)
    pos = sample_step['cloth'].pos
    n = pos.shape[0]
    if prev.shape[0] != n:
        raise ValueError(
            f"drape init has {prev.shape[0]} vertices but the rollout cloth has "
            f"{n}. These must be the same garment in the same order; check that "
            f"the init npz was produced from the same garment dict.")

    dev, dt = pos.device, pos.dtype
    lbs_pos = pos.detach().clone()
    sample_step['cloth'].prev_pos = torch.as_tensor(prev, device=dev, dtype=dt)
    sample_step['cloth'].pos = torch.as_tensor(cur, device=dev, dtype=dt)
    if verbose:
        shift = (sample_step['cloth'].pos - lbs_pos)
        print(f"[drape-init] re-initialised cloth from {path}: moved "
              f"{shift.norm(dim=-1).max().item()*1e3:.1f} mm max, "
              f"{shift.norm(dim=-1).mean().item()*1e3:.1f} mm mean off the LBS state")
    return sample_step
