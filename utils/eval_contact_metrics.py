"""
 contact-evaluation metrics for ContourCraft — position-based port of HOOD.

These reproduce, byte-for-byte in *definition*, the proposal-v7 metrics that HOOD logs in its
`evaluate.py --contact_metrics` pass (utils/eval_contact_metrics.py there), so a ContourCraft mu-sweep
SUMMARY carries the same columns as a HOOD one and the two are directly comparable.

Why a re-derivation instead of a verbatim port
----------------------------------------------
HOOD reads its metrics off the trained model's `('cloth','world_edge','obstacle')` bipartite
contact edges during a live grad-enabled forward. ContourCraft's trained model (`ccraft`) builds a
fundamentally different contact graph (cloth-cloth, node→face, contour/EdgeTypeCC-labeled), so those
edges are not available to read. Instead we rebuild HOOD's *definition* of the contact edge set from
positions — cloth vertex → nearest body vertex within `collision_radius` (k=`k_world_edges`), with the
same `OBSTACLE_OMIT` body-vertex mask (models/baselines.py `_add_positional_edges`) — using the shared,
identical `utils.connectivity.compute_connectivity_pt`. The per-edge geometry (`delta_n`, `delta_f`)
is then computed exactly as HOOD's `utils.contact.compute_contact_geometry`. Because the ccraft rollout
already bakes bolt-on friction into the saved `pred` positions, the slip/stick metrics reflect the
friction and vary with mu, just as in HOOD.

The force metrics (R_coulomb, R_cone, force_balance_residual) require a predicted contact force,
which the force-free bolt-on operator / vanilla ContourCraft model does not produce — they are logged as NaN
(rendered '---' in the SUMMARY), matching HOOD's bolt-on operator rows.

Frame convention (matches utils/metrics.py::set_step used by the energy/collision metrics):
for frame index i (i>=2, warmup 0/1 skipped), "current" = [i-1] and "next" = [i], for both cloth (`pred`)
and body (`obstacle`).
"""
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from utils.cloth_and_material import VertexNormals
from utils.common import NodeType
from utils.connectivity import compute_connectivity_pt
from utils.io import pickle_load
from utils.contact import node_face_signed_distance  # shared node/face geometry (== collision_penalty_perc)

_VERTEX_NORMALS = VertexNormals()  # area-weighted, normalized; same convention as HOOD vertex_normals()

# Defaults mirror HOOD's postcvpr eval (evaluate.py::_contact_params force-free branch +
# aux/comparisons model config + --stick_tau default).
COLLISION_RADIUS = 3e-2      # models/postcvpr collision_radius (proximity-edge build)
K_WORLD_EDGES = 1            # models/postcvpr k_world_edges
CONTACT_OFFSET_EPS = 2e-3    # ncp_contact contact_offset_eps fallback (active-set threshold)
STICK_TAU = 1e-3             # slip threshold (m)

# Up-axis for the orientation split: A+ = body normal points UP (top-facing, gravity-loaded,
# λ_n≈m·g·cosθ), A- = normal points DOWN (underside, gravity-unloaded, held by garment elasticity).
# The VTO/SMPL data is Y-UP — every config sets gravitational_energy.z_axis=1 and the body's height is
# axis 1. HOOD hardcodes [0,0,1] (Z) here, but Z is the left-right/arm-span axis in this data, so the
# Z split buckets contacts left-vs-right (meaningless) instead of top-vs-underside. Fixed 2026-07-03 to
# +Y; the same fix is applied in HOOD utils/eval_contact_metrics.py to keep the two comparable.
# See [[reference_hood_friction_loss]].
_UP_AXIS = (0., 1., 0.)

# Keys in the order they are printed. Identical to HOOD utils/eval_contact_metrics.CONTACT_METRIC_KEYS.
CONTACT_METRIC_KEYS = (
    'pen_rate_strict', 'pen_depth_mean', 'pen_depth_max', 'contact_rate',       #
    'sliding_S_pos_wcos', 'sliding_S_neg_wcos',                                #
    'stick_ratio_pos_wcos', 'stick_ratio_neg_wcos',                            #
    'R_coulomb', 'R_cone', 'force_balance_residual',                           # (force-free -> NaN)
)
_FORCE_KEYS = ('R_coulomb', 'R_cone', 'force_balance_residual')
# pos/neg wcos metrics are undefined when their A+/A- set is empty; NaN so nan-mean skips them.
_CONDITIONAL_KEYS = ('sliding_S_pos_wcos', 'sliding_S_neg_wcos',
                     'stick_ratio_pos_wcos', 'stick_ratio_neg_wcos')

# Print labels — identical strings to HOOD so the same parser reads both.
_PRINT_NAMES = {
    'pen_rate_strict': 'penetration rate strict (vertex/face, <0)',
    'pen_depth_mean': 'penetration depth mean (m)',
    'pen_depth_max': 'penetration depth max (m)',
    'contact_rate': 'contact rate (delta_n<eps_c)',
    'sliding_S_pos_wcos': 'sliding S pos cos-weighted (m)',
    'sliding_S_neg_wcos': 'sliding S neg cos-weighted (m)',
    'stick_ratio_pos_wcos': 'stick ratio pos cos-weighted',
    'stick_ratio_neg_wcos': 'stick ratio neg cos-weighted',
    'R_coulomb': 'Coulomb residual |phi_fb|',
    'R_cone': 'cone violation (N)',
    'force_balance_residual': 'force-balance residual',
}


def _contact_geometry(cloth_curr, cloth_next, body_curr, body_next, body_faces, body_vertex_type,
                      collision_radius, k):
    """Rebuild HOOD's contact-edge set from positions and its per-edge geometry.

    Edges: cloth vertex -> nearest body vertex within `collision_radius` (k), on the CURRENT frame
    (cloth_curr, body_curr), with OBSTACLE_OMIT body verts dropped — the exact construction of
    models/baselines.py::_add_positional_edges. Geometry mirrors utils.contact.compute_contact_geometry:
    n_hat is the outward body-vertex normal at the NEXT body pose (detached), the signed gap and slip are
    measured at the predicted step.

    :return: (n_hat (E,3), delta_n (E,1), delta_f (E,3)) or None if there are no contact edges.
    """
    idx_from, idx_to = compute_connectivity_pt(cloth_curr, body_curr, collision_radius, k=k)
    if idx_from.shape[0] > 0 and body_vertex_type is not None:
        keep = body_vertex_type[idx_to][..., 0] != NodeType.OBSTACLE_OMIT
        idx_from = idx_from[keep]
        idx_to = idx_to[keep]
    if idx_from.shape[0] == 0:
        return None

    vn = _VERTEX_NORMALS(body_next.unsqueeze(0), body_faces.unsqueeze(0))[0]   # (Vb,3) at next pose
    n_hat = vn[idx_to].detach()                                               # (E,3) constant outward dir

    garment_next = cloth_next[idx_from]
    garment_curr = cloth_curr[idx_from]
    b_next = body_next[idx_to]
    b_curr = body_curr[idx_to]

    delta_n = ((garment_next - b_next) * n_hat).sum(dim=-1, keepdim=True)      # (E,1) signed gap (>0 out)
    rel = (garment_next - garment_curr) - (b_next - b_curr)                    # (E,3)
    delta_f = rel - (rel * n_hat).sum(dim=-1, keepdim=True) * n_hat            # (E,3) tangential slip
    return n_hat, delta_n, delta_f


@torch.no_grad()
def compute_contact_metrics_pos(cloth_curr, cloth_next, body_curr, body_next, body_faces,
                                body_vertex_type, *, contact_offset_eps=CONTACT_OFFSET_EPS,
                                stick_tau=STICK_TAU, collision_radius=COLLISION_RADIUS,
                                k=K_WORLD_EDGES):
    """Per-frame / metrics from positions. Force metrics are NaN (force-free model).

    Numeric expressions match HOOD utils/eval_contact_metrics.compute_contact_metrics.
    """
    out = {key: 0.0 for key in CONTACT_METRIC_KEYS}
    for key in _FORCE_KEYS:
        out[key] = float('nan')
    for key in _CONDITIONAL_KEYS:
        out[key] = float('nan')

    # penetration (NODE/FACE, ALL cloth verts) — render-faithful; same geometry as collision_penalty_perc.
    sd = node_face_signed_distance(cloth_next, body_next, body_faces)          # (Vc,) signed; <0 inside
    inside = sd < 0.0
    out['pen_rate_strict'] = inside.float().mean().item()
    if inside.any():
        depth = -sd[inside]
        out['pen_depth_mean'] = depth.mean().item()
        out['pen_depth_max'] = depth.max().item()

    geo = _contact_geometry(cloth_curr, cloth_next, body_curr, body_next, body_faces,
                            body_vertex_type, collision_radius, k)
    if geo is None:
        # No proximity edges this frame -> no contact / slip to measure.
        return out
    n_hat, delta_n, delta_f = geo
    dfn = torch.norm(delta_f, dim=-1)                                          # (E,) ||delta_f||
    cos_theta = (n_hat * n_hat.new_tensor(_UP_AXIS)).sum(dim=-1)              # (E,) n_hat·up_hat (Y-up)
    active = delta_n < contact_offset_eps                                      # (E,1)
    active_b = active.squeeze(-1)                                             # (E,)

    out['contact_rate'] = active.float().mean().item()

    # orientation-split slip/stick (weights = |cos_theta|, purely geometric; A+/A- disjoint).
    if active_b.any():
        pos_b = active_b & (cos_theta > 0)
        neg_b = active_b & (cos_theta < 0)
        if pos_b.any():
            w = cos_theta[pos_b]
            dfa = dfn[pos_b]
            wsum = w.sum()
            out['sliding_S_pos_wcos'] = ((w * dfa).sum() / wsum).item()
            out['stick_ratio_pos_wcos'] = ((w * (dfa < stick_tau).float()).sum() / wsum).item()
        if neg_b.any():
            w = -cos_theta[neg_b]
            dfa = dfn[neg_b]
            wsum = w.sum()
            out['sliding_S_neg_wcos'] = ((w * dfa).sum() / wsum).item()
            out['stick_ratio_neg_wcos'] = ((w * (dfa < stick_tau).float()).sum() / wsum).item()

    return out


def aggregate_metric_lists(metric_lists, n_steps):
    """Per-sequence reduction: nan-mean over recorded frames. NaN-only metrics stay NaN.
    Identical to HOOD utils/eval_contact_metrics.aggregate_metric_lists."""
    out = {}
    for key, vals in metric_lists.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            out[key] = float('nan')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                out[key] = float(np.nanmean(arr))
    out['n_steps'] = n_steps
    return out


def compute_contact_metrics_for_seq(seq_dict, body_vertex_type, device='cuda:0', *,
                                    contact_offset_eps=CONTACT_OFFSET_EPS, stick_tau=STICK_TAU,
                                    collision_radius=COLLISION_RADIUS, k=K_WORLD_EDGES):
    """ per-sequence metrics from a saved rollout dict (keys: pred, obstacle, obstacle_faces).

    Frames 0,1 are velocity warmup and skipped (metrics start at i=2), matching calc_metrics_by_seq.
    """
    pred = torch.as_tensor(np.asarray(seq_dict['pred']), dtype=torch.float32, device=device)      # (T,Vc,3)
    obst = torch.as_tensor(np.asarray(seq_dict['obstacle']), dtype=torch.float32, device=device)  # (T,Vb,3)
    body_faces = torch.as_tensor(np.asarray(seq_dict['obstacle_faces']), dtype=torch.long, device=device)
    body_vertex_type = torch.as_tensor(np.asarray(body_vertex_type), dtype=torch.long, device=device)
    if body_vertex_type.dim() == 1:
        body_vertex_type = body_vertex_type[:, None]

    N = pred.shape[0]
    metric_lists = defaultdict(list)
    for i in range(2, N):
        cm = compute_contact_metrics_pos(
            pred[i - 1], pred[i], obst[i - 1], obst[i], body_faces, body_vertex_type,
            contact_offset_eps=contact_offset_eps, stick_tau=stick_tau,
            collision_radius=collision_radius, k=k)
        for key, val in cm.items():
            metric_lists[key].append(val)
    return aggregate_metric_lists(metric_lists, N - 2)


# ── aggregation + printing ────────────────────────────────────────────────────────────────────

def save_contact_metrics(out_file, all_metrics_dict):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(dict(all_metrics_dict), f)


def _accumulate(metrics_file, sums_all, sums_dress, skip_tshirt_for_ssch=True):
    """Sum each metric weighted by n_steps over a vs_{snug,ssch}.pkl, mirroring the energy-metric
    grouping in utils/metrics.py (tshirt excluded from the ssch group). Identical to HOOD."""
    metrics_dict = pickle_load(metrics_file)
    is_ssch = 'vs_ssch' in str(metrics_file)
    for garment, garment_dict in metrics_dict.items():
        if is_ssch and skip_tshirt_for_ssch and garment == 'tshirt':
            continue
        for seq, seq_dict in garment_dict.items():
            n_steps = seq_dict['n_steps']
            for key, val in seq_dict.items():
                w = val if key == 'n_steps' else val * n_steps
                sums_all[key] += w
                if garment == 'dress':
                    sums_dress[key] += w


def _print_block(title, sums):
    """Colon-format block (matches ContourCraft utils/metrics.py print style) so tools parse it the
    same way as the energy tables."""
    n_steps = sums.get('n_steps', 0)
    print(f'\n{title}:')
    if not n_steps:
        print('(no sequences)')
        return
    for key in CONTACT_METRIC_KEYS:
        if key not in sums:
            continue
        v = sums[key] / n_steps
        name = _PRINT_NAMES.get(key, key)
        print(f'{name}:\t\t\t {v:.3e}')


def aggregate_and_print_contact_metrics(contact_metrics_dir):
    """Print the contact-metrics table (all garments + only dress) from
    contact_metrics_dir/vs_{snug,ssch}.pkl. All-garments-then-dress ordering matches HOOD so
    mu_sweep_csv.sh can select by occurrence (1=all, 2=dress)."""
    contact_metrics_dir = Path(contact_metrics_dir)
    sums_all = defaultdict(float)
    sums_dress = defaultdict(float)
    for sota in ('snug', 'ssch'):
        f = contact_metrics_dir / f'vs_{sota}.pkl'
        if f.exists():
            _accumulate(f, sums_all, sums_dress)
        else:
            print(f'  (missing {f}, skipped)')
    _print_block('Contact metrics (proposal v7) — all garments', sums_all)
    _print_block('Contact metrics (proposal v7) — only dress', sums_dress)
