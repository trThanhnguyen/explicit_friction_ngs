#!/usr/bin/env python
"""Matched-frame still renderer for Fig 5.2-B (and any arm x mu drape comparison).

The repo's only render path is utils/show.py::write_video -> aitviewer HeadlessRenderer -> mp4,
which is wrong for a paper figure twice over: it needs ffmpeg (the .localbin shim is per-host and
not always present) and it renders one sequence at a time with a camera locked to THAT sequence.
A comparison figure needs the opposite: one frame, one camera, many rollouts.

Two things this file does that write_video deliberately cannot:

  * NO per-sequence floor shift. utils/show.py::place_meshes drops each sequence so its own
    lowest vertex sits at y=0. Across arms that is a per-row vertical offset, i.e. exactly the
    quantity the figure is trying to show. All arms here share one body trajectory (measured:
    max |body_ppf - body_gnn| = 6.2 mm, and the two neural arms are bit-identical), so raw world
    coordinates are already comparable and are used unchanged.
  * ONE camera for the whole panel, derived from the union bounding box over every (arm, mu)
    cell rendered, so nothing moves between tiles except the cloth.

Run (pixi env):
  pixi run python tools/r3/render_fig52b.py --panel a --out _drafts_v1/_figures/fig52b_raw
  pixi run python tools/r3/render_fig52b.py --panel a --azimuth 200 --elevation 12 --zoom 1.15
"""
import os

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse            # noqa: E402
import pickle              # noqa: E402
import sys                 # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np         # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MUS = ['0.0', '0.1', '0.3', '0.6']

# arm token -> {mu: eval directory under ccraft_data/eval/}.
#
# Deliberately EMPTY in this distribution: the rollout trees the figures were rendered from are
# hundreds of gigabytes and are not shipped. Supply rollout locations explicitly instead --
# `render_boundary_video.py --runs 'ARM=DIR'` (DIR may contain '{mu}') resolves an arm without
# consulting this table at all, which is the route README.md documents.
ARM_DIRS = {}

ARM_LABEL = {'ppf': 'PPF (solver reference)', 'ours': 'Frictionless ContourCraft + our op.',
             'ours_lowers': 'Frictionless ContourCraft + our op. (walk)',
             'cc_fric': 'ContourCraft + fric. energy', 'hood_fric': 'HOOD + fric. energy',
             'hood_ours': 'Released HOOD + our op.',
             'ours_ship': 'Frictionless ContourCraft + our op. (eps_v 1e-2)',
             'hood_ours_ship': 'Released HOOD + our op. (eps_v 1e-2)',
             'hood_ours_fl': 'Frictionless HOOD + our op. (eps_v 1e-2)',
             'ours_ev2': 'Frictionless ContourCraft + our op. (SHIPS)',
             'relcc_ours_ev2': 'Released ContourCraft + our op. (SHIPS)',
             'hood_ours_ev2': 'Released HOOD + our op. (SHIPS)',
             'hood_ours_fl_ev2': 'Frictionless HOOD + our op. (SHIPS)',
             'ours_walk': 'Frictionless ContourCraft + our op. (walk)',
             'relcc_ours_walk': 'Released ContourCraft + our op. (walk)',
             'hood_ours_walk': 'Released HOOD + our op. (walk)',
             'hood_ours_fl_walk': 'Frictionless HOOD + our op. (walk)',
             'flhood_walk_twin': 'Frictionless HOOD + our op. (walk, mu0.6 re-roll)',
             'relhood_walk_twin': 'Released HOOD + our op. (walk, mu0.6 re-roll)',
             'flhood_twinA': 'Frictionless HOOD + our op. (re-roll A)',
             'flhood_twinB': 'Frictionless HOOD + our op. (re-roll B)',
             'noop_ev2': 'Frictionless ContourCraft + our op., no propagation',
             'b0_ev2': 'Frictionless ContourCraft + our op., wave without cap',
             # the cumulative ladder, labelled as ADDITIONS, not removals
             'core_walkladder': 'Coulomb correction only',
             'noop_walk': '+ geometry-aware traversal',
             'b0_walk': '+ motion propagation'}
# NB the ladder's fourth column is `ours_walk`, whose label above is the one every other walk
# figure uses; the ablation composer draws its own column headers, so no ladder-specific alias
# is defined for it here.

# Panels chosen by the drape scan (all 40 cells x 6 mu pairs), then the frame chosen by the
# localisation scan. The cell ranking is on MEAN displacement; the frame is on the upper tail plus
# hem lift, because a 25 mm mean on a 1.5 m figure is invisible in print while a 26 mm hem lift is
# a silhouette change. Motion diversity is deliberate (136_12 / 133_02), .
PANELS = {
    # ppf p95 60 mm / hem +19 mm, ours p95 54 mm / hem +26 mm, baselines p95 38 mm
    'a': dict(cell='vs_ssch/dress/136_12',      mu_lo='0.0', mu_hi='0.6', frame=87),
    # largest localised change in the grid: ppf p95 77 mm, ours 60 mm, baselines 30 mm (2.5x)
    'b': dict(cell='vs_snug/longsleeve/133_02', mu_lo='0.0', mu_hi='0.6', frame=189),
}


# ---------------------------------------------------------------------------------------------
# Colours. The first pass used aitviewer's demo red-orange on a near-white body; the owner rejected
# both -- the body was inseparable from white paper in print, and the garment colour read as loud.
# These are muted, mid-value clothing colours (one per garment so a multi-panel figure stays
# legible), all kept off pure white/black so they survive greyscale printing with the body between
# them. Override with --cloth-color / --body-color.
PALETTE = {
    'dress':      (0.36, 0.45, 0.63),   # indigo / denim blue
    'longsleeve': (0.40, 0.52, 0.44),   # sage green
    'tshirt':     (0.72, 0.55, 0.38),   # camel / tan
    'tank':       (0.60, 0.42, 0.48),   # dusty rose
    'longskirt':  (0.36, 0.45, 0.63),
    'unpinned_hooded_tight_dress': (0.40, 0.52, 0.44),
}
CLOTH_FALLBACK = (0.45, 0.50, 0.60)
BODY_RGB = (0.52, 0.52, 0.55)           # owner's pick, 'body_dark' on the colour sheet.
                                        # Supersedes 'body_mid-grey' (0.66) chosen, which in
                                        # turn replaced a 0.74 warm tone that printed near-white.

# SMPL joint ids used for the facing/uprightness geometry
J_PELVIS, J_LHIP, J_RHIP, J_NECK, J_HEAD = 0, 1, 2, 12, 15


def smpl_joints(body_verts):
    """(24,3) joints from a posed SMPL mesh via the model's own J_regressor."""
    import pickle as _p
    with open(REPO / 'ccraft_data/aux_data/body_models/smpl/SMPL_FEMALE.pkl', 'rb') as f:
        Jr = np.asarray(_p.load(f, encoding='latin1')['J_regressor'].todense())
    return Jr @ body_verts


def body_frame(body_verts):
    """Body-relative axes + an uprightness score, so the camera can never land on a bad angle.

    A world-fixed azimuth is what put the first draft's panel (b) behind a crawling body. Deriving
    the view from the body's own facing direction makes "front" mean front in every pose.

    Returns (forward, right, up_torso, upright) where upright = cos of the torso axis against world
    up: 1.0 standing, ~0 horizontal. Frames below ~0.6 are lying/crawling and are not presentable.
    """
    J = smpl_joints(body_verts)
    right = J[J_RHIP] - J[J_LHIP]
    right[1] = 0.0
    n = np.linalg.norm(right)
    right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    world_up = np.array([0.0, 1.0, 0.0])
    forward = np.cross(world_up, right)
    forward /= max(np.linalg.norm(forward), 1e-9)
    torso = J[J_NECK] - J[J_PELVIS]
    upright = float(torso[1] / max(np.linalg.norm(torso), 1e-9))
    return forward, right, torso, upright


def load_frame(arm, mu, cell, frame):
    p = REPO / 'ccraft_data/eval' / ARM_DIRS[arm][mu] / 'rollouts' / f'{cell}.pkl'
    with open(p, 'rb') as f:
        d = pickle.load(f)
    T = d['pred'].shape[0]
    t = min(frame, T - 1)
    cf = d['cloth_faces']
    of = d['obstacle_faces']
    if cf.ndim == 3:
        cf, of = cf[0], of[0]
    return dict(cloth=d['pred'][t][None].astype(np.float32), cloth_faces=cf,
                body=d['obstacle'][t][None].astype(np.float32), body_faces=of, T=T, t=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default='a', choices=sorted(PANELS))
    ap.add_argument('--cell', default=None, help='override the panel cell')
    ap.add_argument('--mus', default=None, help='override, e.g. "0.1,0.6"')
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--arms', default='ppf,ours,cc_fric,hood_fric')
    ap.add_argument('--out', default='_figures/fig52b_raw')
    ap.add_argument('--size', default='900x1200', help='WxH per tile')
    ap.add_argument('--azimuth', default='180', help='degrees around +y; comma list sweeps views')
    ap.add_argument('--elevation', type=float, default=8.0, help='degrees above horizontal')
    ap.add_argument('--zoom', type=float, default=1.0, help='>1 moves the camera closer')
    ap.add_argument('--fov', type=float, default=45.0)
    ap.add_argument('--cloth-color', default=None,
                    help='RGB 0-1; default = the garment\'s entry in PALETTE')
    ap.add_argument('--body-color', default=None, help='RGB 0-1; default = BODY_RGB')
    ap.add_argument('--zoom-region', default=None,
                    help="aim a ZOOMED crop at one SMPL body region (e.g. 'upper arm', 'thigh') "
                         "instead of framing the whole body. a shoulder or "
                         "sleeve event is a few cm on a 1.5 m figure and does not survive full-body "
                         "framing. The crop box is shared by every tile, as the camera is.")
    ap.add_argument('--zoom-side', default=None, choices=['L', 'R', 'C'],
                    help="restrict --zoom-region to one limb. Without it, 'forearm' spans BOTH arms "
                         "and its bbox is ~1 m wide, so the 'zoom' is a 3/4-body crop. "
                         "'C' is the centre line -- torso, pelvis, neck, head live there, and an "
                         "L/R-only filter returns nothing for them.")
    ap.add_argument('--zoom-boundary', default=None,
                    help="frame the crop on a segment of the garment's FREE BOUNDARY, as "
                         "'<loop-rank>@<region><side>' (e.g. '0@thighR' = the longest boundary "
                         "loop, its vertices currently nearest the right thigh). Owner reading "
                         ": what reads on the page is the cloth's EDGE against the body "
                         "behind it, so that is what the crop should be built from. Loop ranks and "
                         "the regions available at this frame are printed when the flag is used. "
                         "Pair with an internal tool, which scores exactly this.")
    ap.add_argument('--zoom-body-cue', default='on', choices=['on', 'off'],
                    help="with --zoom-boundary: include the body part the segment sits against in "
                         "the crop box (default). 'off' frames the edge segment alone -- tighter, "
                         "and what sample 22 used at --zoom-margin 2.4, which the owner preferred "
                         "for tank 05_06. SMPL 'torso' spans spine1-3, so an @torsoC crop with the "
                         "cue on is nearly full-body; that is the case to turn it off for.")
    ap.add_argument('--zoom-margin', type=float, default=1.9,
                    help='framed radius as a multiple of the region bbox radius')
    ap.add_argument('--zoom-ref-arm', default='ours',
                    help="with --zoom-boundary: which arm's mu_hi tile LABELS the boundary segment "
                         "(see the note at the selection site). Default 'ours' is the historical "
                         "behaviour. Set it when the operator arm goes by another token -- otherwise "
                         "the lookup misses, silently falls back to the first arm at mu_lo, and "
                         "re-frames the whole panel: on tank 05_06 that is 22 edge verts -> 18 and "
                         "fov 12.8 -> 11.4 deg, with no warning printed.")
    ap.add_argument('--zoom-on', default='contact', choices=['contact', 'region'],
                    help="what the crop is centred on. 'contact' "
                         "centres on the CLOTH actually touching the region -- the cuff, the sleeve "
                         "cap -- so the event fills the tile. 'region' uses the body part's own bbox, "
                         "which left dead space and clipped a head sliver in sample 08.")
    ap.add_argument('--contact-radius', type=float, default=0.03,
                    help='m; cloth within this of the region counts as the contact patch')
    ap.add_argument('--zoom-mode', default='focal', choices=['focal', 'dolly'],
                    help="focal: keep the camera at its FULL-BODY "
                         "distance and narrow the field of view -- a long lens, so the region fills "
                         "the frame at near-orthographic proportions. dolly: move the camera in, "
                         "which is what produced the wide-angle, face-heavy sample 06.")
    ap.add_argument('--view', default=None,
                    help="BODY-RELATIVE view, overrides --azimuth: front | front34 | front34l | "
                         "side | back34. Comma list sweeps. Derived from the SMPL facing direction, "
                         "so it means the same thing in every pose.")
    ap.add_argument('--texture', default=None,
                    help="fabric name from tools/r3/garment_uv.TEX_DIMS_M, drawn through the "
                         "garment's own SEWING PATTERN as the UV map. a flat "
                         "single colour cannot express a garment sliding ALONG the body, which is "
                         "exactly what mu changes -- panels 10/11 were numerically real and "
                         "visually blank. The tint stays the garment's PALETTE entry.")
    ap.add_argument('--texture-dir', default='_figures/fig_qualitative/_fabrics',
                    help='directory of <name>.jpg diffuse maps (CC0, Poly Haven)')
    ap.add_argument('--texture-scale', type=float, default=1.0,
                    help='>1 shrinks the print (more repeats per metre of pattern)')
    ap.add_argument('--texture-native', action='store_true',
                    help="keep the photograph's own colour instead of tinting to the palette")
    ap.add_argument('--light-azimuth', type=float, default=None,
                    help="Shadow-casting key light, degrees OFFSET FROM THE CAMERA (0 = straight "
                         "down the lens, negative = to the camera's left). Resolved against the "
                         "body's own frame, so it means the same thing in any pose; the printed "
                         "value is the resulting body-relative angle. aitviewer's default key is "
                         "world-fixed at +z, which on longsleeve 143_27 threw the raised hand's "
                         "shadow across the face. Unset = aitviewer default, byte-identical to "
                         "every render before.")
    ap.add_argument('--light-elevation', type=float, default=35.0,
                    help='elevation of that light, degrees; only used with --light-azimuth')
    ap.add_argument('--ambient', type=float, default=None,
                    help="scene ambient strength; aitviewer's default is 2.0, which CLIPS a warm "
                         "body albedo to flat white and destroys exactly the shading that makes a "
                         "limb through cloth readable. ~1.1 keeps the skin off the ceiling. Unset "
                         "= aitviewer default, byte-identical to every render before.")
    ap.add_argument('--light-strength', type=float, default=None,
                    help='strength of BOTH directional lights (aitviewer default 1.0). Lower it '
                         'with --ambient to darken overall; raise it alone for more modelling.')
    ap.add_argument('--arm-run', default=None,
                    help="repoint an (arm, mu) cell at a specific rollout dir, comma-separated as "
                         "'ours:0.3=<run-dir>'. For visualisation only: it is how a re-rolled "
                         "twin gets into a panel without inventing a new arm name. Anything "
                         "rendered through this must never be quoted as a measurement.")
    ap.add_argument('--zoom-lock', default=None,
                    help="'cx,cy,cz,radius,body_radius' from another render's ZOOM-LOCK STRING: "
                         "reuse that crop and camera distance verbatim. Use it whenever two panels "
                         "must be read against each other (same frame, two mu pairs) -- without it "
                         "each render re-derives the crop from its own tiles and the two are "
                         "silently re-framed.")
    ap.add_argument('--transparent', action='store_true')
    args = ap.parse_args()

    spec = dict(PANELS[args.panel])
    if args.cell:
        spec['cell'] = args.cell
    if args.mus:
        spec['mu_lo'], spec['mu_hi'] = args.mus.split(',')
    if args.frame is not None:
        spec['frame'] = args.frame
    cell, mus, frame = spec['cell'], [spec['mu_lo'], spec['mu_hi']], spec['frame']
    arms = args.arms.split(',')
    W, H = (int(x) for x in args.size.split('x'))

    print(f'panel {args.panel}: {cell}  mu {mus[0]} -> {mus[1]}  frame {frame}')
    if args.arm_run:
        for spec in args.arm_run.split(','):
            k, _, run = spec.partition('=')
            a, _, m = k.partition(':')
            if a not in ARM_DIRS or m not in ARM_DIRS[a]:
                sys.exit(f'--arm-run {spec!r}: no such (arm, mu) cell')
            ARM_DIRS[a][m] = run
            print(f'  ARM-RUN OVERRIDE {a} mu{m} -> {run}')
    data = {(a, m): load_frame(a, m, cell, frame) for a in arms for m in mus}
    for (a, m), d in data.items():
        if d['t'] != frame:
            print(f'  ! {a} mu{m}: only {d["T"]} frames, clamped to {d["t"]}')

    # ---- one camera for the whole panel: union bbox over every tile ----
    if args.zoom_boundary:
        from scipy.spatial import cKDTree
        from tools.r3.garment_boundary import boundary_loops, smpl_vertex_labels
        from tools.r3.mu_interaction_map import JOINT_REGION as JR, JOINT_SIDE as JS
        rank, _, at_want = args.zoom_boundary.partition('@')
        loops = boundary_loops(next(iter(data.values()))['cloth_faces'])
        if not rank.isdigit() or int(rank) >= len(loops):
            sys.exit(f'--zoom-boundary loop rank must be 0..{len(loops)-1} '
                     f'(sizes {[len(l) for l in loops]})')
        lp = loops[int(rank)]
        vreg, vside = smpl_vertex_labels()
        # Which tile decides WHICH boundary vertices are "at" the region has to match the selector,
        # or the two disagree about what the panel is: select_boundary_frames labels the segment on
        # ours@mu_hi, and on a hem that has travelled a limb between mu the ppf@mu_lo label can be a
        # different body part entirely (this raised a hard error on dress/05_08 f97, not a silent
        # re-frame). The crop BOX is still the union over every tile, so no tile is favoured.
        # ⚠️ This lookup used to hard-code 'ours', so renaming the operator arm missed it and took
        # the fallback WITHOUT SAYING SO -- a different reference labels a different set of edge
        # verts, which re-frames every tile in the panel. Now the fallback announces itself.
        ref_key = (args.zoom_ref_arm, mus[-1])
        if ref_key not in data:
            print(f'  ! no {args.zoom_ref_arm!r} tile at mu{mus[-1]}; labelling the segment on '
                  f'{arms[0]}@mu{mus[0]} instead. Pass --zoom-ref-arm to pick deliberately -- this '
                  f'fallback CHANGES THE FRAMING of every tile in the panel.')
        ref = data.get(ref_key, data[(arms[0], mus[0])])
        _, nn = cKDTree(ref['body'][0]).query(ref['cloth'][0][lp])
        at = np.array([f'{r}{s}' for r, s in zip(vreg[nn], vside[nn])])
        vals, cnt = np.unique(at, return_counts=True)
        print(f'  boundary loop {rank} ({len(lp)} verts) sits against: '
              + ', '.join(f'{v}:{c}' for v, c in sorted(zip(vals, cnt), key=lambda x: -x[1])))
        sel_lp = lp[at == at_want] if at_want else lp
        if len(sel_lp) < 3:
            sys.exit(f'only {len(sel_lp)} boundary verts at {at_want!r}; pick from the list above')
        # The crop is the segment UNION the body part it is drawn against.
        # framing the edge alone "looks too close and we lose body-cue context" -- and the context is
        # the whole point, since a boundary is only legible against a landmark. This restores the
        # framing of the approved panels 09/12, which cropped on a full SMPL region, while still
        # centring on the segment rather than on the region's own bbox.
        import pickle as _p
        with open(REPO / 'ccraft_data/aux_data/body_models/smpl/SMPL_FEMALE.pkl', 'rb') as f:
            bw = np.asarray(_p.load(f, encoding='latin1')['weights'])
        bvj = bw.argmax(axis=1)
        bsel = np.array([f'{JR[int(j)]}{JS[int(j)]}' for j in bvj]) == at_want \
            if (at_want and args.zoom_body_cue == 'on') else np.zeros(len(bvj), bool)
        target = np.concatenate([d['cloth'][0][sel_lp] for d in data.values()]
                                + ([ref['body'][0][bsel]] if bsel.any() else []))
        if bsel.any():
            print(f'  + body cue: {int(bsel.sum())} verts of {at_want}')
        center = 0.5 * (target.min(0) + target.max(0))
        radius = float(np.linalg.norm(target.max(0) - target.min(0))) * 0.5 * args.zoom_margin
        print(f'  ZOOM on boundary {args.zoom_boundary!r}: {len(sel_lp)} edge verts, '
              f'centre {np.round(center,3)}, radius {radius:.3f} m')
    elif args.zoom_region:
        # Aim at the REGION, using the body's own vertices for that SMPL part. Body-side rather than
        # cloth-side on purpose: the body is common to every tile (max |body_ppf - body_gnn| = 6.2 mm),
        # so the crop box cannot shift between rows and re-frame the comparison.
        from tools.r3.mu_interaction_map import JOINT_REGION, JOINT_SIDE
        import pickle as _p
        with open(REPO / 'ccraft_data/aux_data/body_models/smpl/SMPL_FEMALE.pkl', 'rb') as f:
            w = np.asarray(_p.load(f, encoding='latin1')['weights'])
        vj = w.argmax(axis=1)
        vreg = np.array([JOINT_REGION[int(j)] for j in vj])
        sel = vreg == args.zoom_region
        if args.zoom_side:
            sel &= np.array([JOINT_SIDE[int(j)] for j in vj]) == args.zoom_side
        if not sel.any():
            sys.exit(f'unknown --zoom-region {args.zoom_region!r}; '
                     f'choose from {sorted(set(JOINT_REGION.values()))}')
        rp = np.concatenate([d['body'][0][sel] for d in data.values()])
        target, what = rp, f'{int(sel.sum())} body verts'
        if args.zoom_on == 'contact':
            from scipy.spatial import cKDTree
            patch = []
            for d in data.values():
                tree = cKDTree(d['body'][0][sel])
                dist, _ = tree.query(d['cloth'][0])
                patch.append(d['cloth'][0][dist < args.contact_radius])
            patch = [q for q in patch if len(q)]
            if patch:
                target = np.concatenate(patch)
                what = f'{len(target)} cloth verts within {1e2*args.contact_radius:.0f} cm'
            else:
                print('  ! no cloth within the contact radius; falling back to the region bbox')
        center = 0.5 * (target.min(0) + target.max(0))
        radius = float(np.linalg.norm(target.max(0) - target.min(0))) * 0.5 * args.zoom_margin
        print(f'  ZOOM on {args.zoom_region!r} ({args.zoom_on}): {what}, '
              f'centre {np.round(center,3)}, radius {radius:.3f} m')
    else:
        pts = np.concatenate([np.concatenate([d['cloth'][0], d['body'][0]])
                              for d in data.values()])
        center = 0.5 * (pts.min(0) + pts.max(0))
        radius = float(np.linalg.norm(pts.max(0) - pts.min(0))) * 0.5
    el = np.deg2rad(args.elevation)
    # Distance is always set by the WHOLE body; only the lens changes for a zoom. Dollying in on a
    # 45-degree lens foreshortens whatever is nearest the camera (the head, in sample 06).
    pts_all = np.concatenate([np.concatenate([d['cloth'][0], d['body'][0]])
                              for d in data.values()])
    body_radius = float(np.linalg.norm(pts_all.max(0) - pts_all.min(0))) * 0.5
    # Two panels of the SAME frame at different mu pairs do not share a crop by default: centre,
    # radius and the full-body distance are all unions over the tiles rendered, and the mu_hi tiles
    # differ. --zoom-lock replays another render's crop verbatim so the two are comparable tile for
    # tile; the string to pass is printed by every render.
    if args.zoom_lock:
        cx, cy, cz, radius, body_radius = (float(v) for v in args.zoom_lock.split(','))
        center = np.array([cx, cy, cz])
        print(f'  ZOOM-LOCK applied: centre {np.round(center, 3)}  radius {radius:.3f} m  '
              f'body_radius {body_radius:.3f} m')
    print('  ZOOM-LOCK STRING '
          f'{center[0]:.6f},{center[1]:.6f},{center[2]:.6f},{radius:.6f},{body_radius:.6f}')
    fov = args.fov
    if (args.zoom_region or args.zoom_boundary) and args.zoom_mode == 'focal':
        dist = 2.6 * body_radius / max(args.zoom, 1e-3)
        fov = float(np.rad2deg(2.0 * np.arctan(radius / dist)))
        print(f'  FOCAL zoom: camera held at {dist:.2f} m (full-body distance), '
              f'fov {args.fov:.0f} deg -> {fov:.1f} deg')
    else:
        dist = 2.6 * radius / max(args.zoom, 1e-3)

    # Reference body for the camera frame: the solver's if present, else the first tile. All arms
    # share the body trajectory here (max |body_ppf - body_gnn| = 6.2 mm), so the choice is cosmetic.
    ref_body = data[('ppf', mus[0])]['body'][0] if ('ppf', mus[0]) in data \
        else next(iter(data.values()))['body'][0]
    fwd, right, _, upright = body_frame(ref_body)
    print(f'  target {np.round(center,3)}  radius {radius:.3f} m')
    print(f'  body facing {np.round(fwd,3)}  uprightness {upright:.2f}'
          + ('  <-- LYING/CRAWLING, not presentable' if upright < 0.6 else ''))

    VIEWS = {'front': 0.0, 'front34': 40.0, 'front34l': -40.0, 'side': 90.0, 'back34': 140.0}
    if args.view:
        eyes = {}
        view_deg = {}
        for v in args.view.split(','):
            # a bare number is a body-relative azimuth in degrees: 0 = straight on, + rotates toward
            # the body's right. The named views are just handy values of it. Needed because 'front34'
            # put a raised arm in front of the head on longsleeve 143_27 and the
            # fix is a few degrees, not the next named preset.
            try:
                deg = float(v)
            except ValueError:
                if v not in VIEWS:
                    sys.exit(f'unknown --view {v!r}; choose from {sorted(VIEWS)} or a number')
                deg = VIEWS[v]
            th = np.deg2rad(deg)
            d = fwd * np.cos(th) + right * np.sin(th)          # rotate about world up
            eyes[v] = center + dist * (d * np.cos(el) + np.array([0.0, np.sin(el), 0.0]))
            view_deg[v] = deg
        azimuths = list(eyes)
    else:
        azimuths = [float(a) for a in str(args.azimuth).split(',')]
        eyes = {a: center + dist * np.array([np.sin(np.deg2rad(a)) * np.cos(el), np.sin(el),
                                             np.cos(np.deg2rad(a)) * np.cos(el)]) for a in azimuths}
        view_deg = {a: a for a in azimuths}

    from aitviewer.configuration import CONFIG as C
    C.update_conf({'window_width': W, 'window_height': H})
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.meshes import Meshes
    from aitviewer.scene.camera import PinholeCamera

    garment_name = cell.split('/')[1]
    cloth_rgb = tuple(float(x) for x in args.cloth_color.split(',')) if args.cloth_color \
        else PALETTE.get(garment_name, CLOTH_FALLBACK)
    body_rgb = tuple(float(x) for x in args.body_color.split(',')) if args.body_color else BODY_RGB
    print(f'  garment {garment_name}: cloth {np.round(cloth_rgb,3)}  body {np.round(body_rgb,3)}')

    uvspec = None
    if args.texture:
        from tools.r3.garment_uv import uv_map, uv_tiled, tint, tile_dims
        ref = next(iter(data.values()))
        v2v3, mf, uv_m = uv_map(garment_name, ref['cloth_faces'])
        src = REPO / args.texture_dir / f'{args.texture}.jpg'
        if not src.exists():
            sys.exit(f'no fabric {args.texture!r} at {src}')
        cache = REPO / args.texture_dir / '_tinted'
        cache.mkdir(parents=True, exist_ok=True)
        png = tint(str(src), None if args.texture_native else cloth_rgb,
                   str(cache / f'{garment_name}__{args.texture}'
                       f'{"__native" if args.texture_native else ""}.png'))
        texdir = REPO / args.texture_dir
        uvspec = dict(v2v3=v2v3, faces=mf,
                      uv=uv_tiled(uv_m, args.texture, args.texture_scale, texdir), path=png)
        w_m, _ = tile_dims(args.texture, texdir)
        print(f'  TEXTURE {args.texture} on the {garment_name} pattern '
              f'({len(uv_m)} pattern verts, one tile = {1e2*w_m/args.texture_scale:.0f} cm of cloth)')

    tag = f'{args.panel}_{cell.replace("/", "_")}_f{frame}'
    if args.texture:
        # the scale belongs in the tag: without it a print-size sweep silently overwrites itself
        tag += '_tex-' + args.texture + (f'@{args.texture_scale:g}' if args.texture_scale != 1.0
                                         else '')
        if args.texture_native:
            tag += '-native'
    if args.zoom_boundary:
        tag += '_bnd-' + args.zoom_boundary.replace('@', '-').replace(' ', '')
    if args.zoom_region:
        tag += '_zoom-' + args.zoom_region.replace(' ', '') + (args.zoom_side or '')
    out = REPO / args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    # window_type/backend exactly as render_test_set.py:178 -- moderngl picks X11 otherwise and
    # dies with "XOpenDisplay: cannot open display" on a headless box. PYOPENGL_PLATFORM does not
    # cover this: aitviewer renders through moderngl, not PyOpenGL.
    r = HeadlessRenderer(window_type='headless', backend='egl')
    # Exposure. Set once, before the per-view loop: these are scene properties, not per-frame ones.
    if args.ambient is not None:
        r.scene.ambient_strength = float(args.ambient)
    if args.light_strength is not None:
        for li in r.scene.lights:
            li.strength = float(args.light_strength)
    if args.ambient is not None or args.light_strength is not None:
        print(f'  exposure: ambient={r.scene.ambient_strength:.2f} '
              f'light_strength={r.scene.lights[0].strength:.2f}')
    for az in azimuths:
      eye = eyes[az]
      if args.light_azimuth is not None:
        # Aim the key light in the BODY's frame, like the camera, so "the shadow falls to the left
        # of the head" stays true whatever pose the frame is in. aitviewer stores a light as
        # (elevation, azimuth) of its direction, so convert through its own helper.
        from aitviewer.utils.utils import spherical_coordinates_from_direction
        thl = np.deg2rad(view_deg[az] + args.light_azimuth)
        ell = np.deg2rad(args.light_elevation)
        p = (fwd * np.cos(thl) + right * np.sin(thl)) * np.cos(ell) \
            + np.array([0.0, np.sin(ell), 0.0])
        th_e, th_a = spherical_coordinates_from_direction(-p / np.linalg.norm(p), degrees=True)
        key = next(li for li in r.scene.lights if li.shadow_enabled)
        key.elevation, key.azimuth = float(th_e), float(th_a)
        print(f'  key light at body-relative {view_deg[az] + args.light_azimuth:+.0f} deg, '
              f'elevation {args.light_elevation:.0f} deg')
      suffix = '' if len(azimuths) == 1 else (
          f'__{az}' if isinstance(az, str) else f'__az{az:g}')
      for a in arms:
        for m in mus:
            d = data[(a, m)]
            body = Meshes(d['body'], d['body_faces'], name='body', color=(*body_rgb, 1.0))
            body.backface_culling = False
            if uvspec is None:
                cloth = Meshes(d['cloth'], d['cloth_faces'], name='cloth', color=(*cloth_rgb, 1.0))
            else:
                # seam-split copy: same surface, vertices duplicated at the pattern seams so each
                # one can carry a single UV. Geometry is gathered, never recomputed.
                cloth = Meshes(d['cloth'][:, uvspec['v2v3']], uvspec['faces'], name='cloth',
                               uv_coords=uvspec['uv'], path_to_texture=uvspec['path'])
            cloth.backface_culling = False
            r.scene.add(body, cloth)

            r.scene.floor.enabled = False
            r.scene.origin.enabled = False
            # An explicit PinholeCamera, the same class write_video uses -- setting position/target
            # on the default ViewerCamera does not survive the render pass, which framed the subject
            # off-centre and cropped the head. Constant (1,3) tracks = a camera that never moves.
            cam = PinholeCamera(eye[None], center[None], W, H, fov=fov, viewer=r)
            r.scene.add(cam)
            r.set_temp_camera(cam)

            f = out / f'{a}__mu{m}{suffix}.png'
            # _init_scene() is what binds the GL context to the newly added nodes; HeadlessRenderer
            # .save_frame() calls it for you but takes no transparency flag, so call both by hand.
            r._init_scene()
            r.export_frame(str(f), transparent_background=args.transparent)
            print(f'  wrote {os.path.relpath(f, REPO)}')
            r.scene.remove(body)
            r.scene.remove(cloth)
            r.scene.remove(cam)

    (out / 'README.txt').write_text(
        f'cell   {cell}\nmu     {mus[0]} -> {mus[1]}\nframe  {frame}\n'
        f'camera target {list(np.round(center,4))} '
        f'(az {args.azimuth} el {args.elevation} zoom {args.zoom} fov {args.fov})\n'
        f'arms   {", ".join(ARM_LABEL[a] for a in arms)}\n'
        'Raw world coordinates -- no per-sequence floor shift; one camera for every tile.\n')
    print(f'\n{len(arms) * len(mus) * len(azimuths)} tiles -> {os.path.relpath(out, REPO)}')


if __name__ == '__main__':
    main()
