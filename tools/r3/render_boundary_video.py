#!/usr/bin/env python
"""Full-sequence mp4 per (arm, mu) from a rollout tree, one consistent camera.

Why a video tool at all, when Fig 5.2-B is a still: the boundary selector reports per-frame numbers,
and a per-frame number can be right about the geometry and still mislead about the EVENT. Owner
challenge on dress 05_08: "there's no way the leg doesn't make contact while it lifts" -- the
selector said the hem's contact fraction was low there, and the way to settle that is to watch the
whole motion rather than argue about one frame.

Differences from render_fig52b.py, all for sequences rather than stills:
  * the camera TRACKS the body (target = body bbox centre each frame, direction and distance fixed
    from a reference frame), because a subject that walks leaves a fixed frame;
  * distance is set from the union over every frame AND every arm, so no tile ever clips;
  * frames go to PNG then through ffmpeg. The pixi env's ffmpeg has libx264; the
    .localbin shim this box lacks is only needed by skvideo's path, which this does not use.

Run:
  pixi run python tools/r3/render_boundary_video.py --cell vs_ssch/dress/05_08 \\
      --arms ppf,ours,cc_fric --mus 0.0,0.3 --out _drafts_v1/_figures/fig52b_video_dress0508
"""
import os

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse            # noqa: E402
import pickle              # noqa: E402
import shutil              # noqa: E402
import subprocess          # noqa: E402
import sys                 # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np         # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.r3.render_fig52b import (ARM_DIRS, ARM_LABEL, BODY_RGB,   # noqa: E402
                                    CLOTH_FALLBACK, PALETTE, body_frame)


def load_seq(arm, mu, cell, roots=None):
    """Resolve one (arm, mu) rollout. `roots` is --runs: {arm: dir}, dir holding rollouts/<cell>.pkl
    and optionally containing '{mu}'. An arm absent from it resolves through the eval tree, as
    before, so every published figure re-renders unchanged."""
    if roots and arm in roots:
        root = Path(roots[arm].format(mu=mu))
        src = (root if root.is_absolute() else REPO / root) / 'rollouts' / f'{cell}.pkl'
    else:
        src = REPO / 'ccraft_data/eval' / ARM_DIRS[arm][mu] / 'rollouts' / f'{cell}.pkl'
    from utils import trace
    trace.at('tools/r3/render_boundary_video.py', 'load_seq',
             f'{arm} mu {mu} <- {os.path.relpath(src, REPO)}')
    with open(src, 'rb') as f:
        d = pickle.load(f)
    cf, of = d['cloth_faces'], d['obstacle_faces']
    if cf.ndim == 3:
        cf, of = cf[0], of[0]
    return d['pred'].astype(np.float32), d['obstacle'].astype(np.float32), cf, of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arms', default='ppf,ours,cc_fric')
    ap.add_argument('--runs', default=None,
                    help="read an arm's rollouts from somewhere other than the eval tree: "
                         "'ARM=DIR[,ARM=DIR]', DIR holding rollouts/<cell>.pkl and optionally "
                         "containing '{mu}'. Arms not named here are unaffected.")
    ap.add_argument('--mus', default='0.0,0.3')
    ap.add_argument('--out', required=True)
    ap.add_argument('--view', default='front34', help='named view or body-relative degrees')
    ap.add_argument('--elevation', type=float, default=8.0)
    ap.add_argument('--zoom', type=float, default=1.0, help='>1 moves the camera closer')
    ap.add_argument('--fov', type=float, default=45.0)
    ap.add_argument('--size', default='900x1100')
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1, help='-1 = to the end')
    ap.add_argument('--texture', default=None)
    ap.add_argument('--texture-dir', default='_drafts_v1/_figures/fig_qualitative/_fabrics')
    ap.add_argument('--cloth-color', default=None,
                    help='RGB 0-1, e.g. "0.40,0.52,0.44"; default = PALETTE.get(garment, '
                         'CLOTH_FALLBACK), same convention as render_fig52b.py --cloth-color.')
    ap.add_argument('--body-color', default=None,
                    help='RGB 0-1; default = BODY_RGB (grey), same convention as '
                         'render_fig52b.py --body-color.')
    ap.add_argument('--light-azimuth', type=float, default=None)
    ap.add_argument('--light-elevation', type=float, default=35.0)
    ap.add_argument('--box-boundary', default=None,
                    help="'<loop-rank>@<region><side>' -- draw a RED BOX around this boundary "
                         "segment during the pause, so the reader is pointed at the same region the "
                         "figure crops to. Same handle render_fig52b --zoom-boundary takes.")
    ap.add_argument('--pause-frame', type=int, default=None,
                    help='hold this frame (the one used in the figure) for --pause-seconds')
    ap.add_argument('--pause-seconds', type=float, default=1.0)
    ap.add_argument('--smooth-sigma', type=float, default=5.0,
                    help='gaussian smoothing of the camera path, in frames; show.py exposes the '
                         'same knob on lock_to_node')
    ap.add_argument('--camera', default='lock', choices=['lock', 'static', 'track'],
                    help="'lock' (default): the convention utils/show.py uses for every video in this repo -- aitviewer's path.lock_to_node, i.e. the camera sits at a CONSTANT OFFSET from the subject's centre and follows it, with the path gaussian-smoothed. A static camera framed on the union box (the earlier default) let the subject walk out of frame at both ends of the tank clip. 'static': ONE camera framed on the union "
                         "bounding box over every frame and every arm. 'track' re-centres on the "
                         "body each frame, which follows a walking subject but makes the clip jitter "
                         "-- the owner's verdict on the dress videos was that it 'looks very "
                         "shaky and causes headache', so tracking is no longer the default.")
    ap.add_argument('--fill', type=float, default=None,
                    help="frame the subject so its widest projected extent over EVERY frame, arm "
                         "and mu fills this fraction of the image "
                         "(the README uses 0.88). Unset keeps the old "
                         "`2.6 * union-bbox radius` rule, which leaves a walking subject small in "
                         "a lot of dead space. --zoom still divides the result.")
    ap.add_argument('--floor', action='store_true',
                    help='plain floor plane with a cast shadow. Off by '
                         'default so every already-published clip re-renders unchanged.')
    ap.add_argument('--floor-color', default='0.706,0.706,0.740')
    ap.add_argument('--floor-size', type=float, default=2000.0)
    ap.add_argument('--floor-track', action='store_true',
                    help="let the floor follow the LOWEST point of each frame instead of holding "
                         "one height for the clip. Mocap root drift otherwise sinks or floats the "
                         "feet; --floor only.")
    ap.add_argument('--light-strength', type=float, default=None,
                    help='key-light strength; unset leaves aitviewer\'s default untouched.')
    ap.add_argument('--fill-light-azimuth', type=float, default=None,
                    help='second (shadowless) light, body-relative degrees like --light-azimuth. '
                         'Unset leaves it where aitviewer put it.')
    ap.add_argument('--fill-light-strength', type=float, default=None)
    ap.add_argument('--keep-frames', action='store_true')
    ap.add_argument('--transparent', action='store_true',
                    help='RGBA frames with no background. '
                         'mp4 has no alpha channel, so this is only useful with --keep-frames.')
    # --- supplementary-video additions. All default-off: with none of them set, every
    # published clip re-renders unchanged (the pause/box code path below is untouched).
    ap.add_argument('--inset', action='store_true',
                    help='during the --pause-frame hold, paste a magnified crop of the red-box '
                         'region into the image corner farthest from the box, red-bordered, with a '
                         'thin connector line. With --inset or '
                         '--inset-track the box is the UNION over every arm and mu of the boundary '
                         'vertices at that frame, so it is the same rectangle in every tile and '
                         'contains every arm\'s boundary; without them the box stays per-arm as before.')
    ap.add_argument('--inset-scale', type=float, default=2.5,
                    help='magnification of the inset crop; reduced automatically if the inset would '
                         'exceed 45%% of the frame in either dimension')
    ap.add_argument('--inset-track', action='store_true',
                    help='draw the box (and the inset, if --inset) on EVERY frame, re-projected per '
                         'frame, not only during the pause hold. Needs --pause-frame and '
                         '--box-boundary (the pause frame is where the boundary labels are read).')
    ap.add_argument('--box-json', action='store_true',
                    help='write <out>/<tag>_box.json per clip: pause frame, hold length, box and '
                         'inset rectangles in pixels (per frame when tracking), fps, frame count -- '
                         'what a downstream composer needs to place labels and holds.')
    # --- focal zoom, render_fig52b.py's rule carried to a sequence (all default-off) ------------
    # The stills lane frames a paper panel by narrowing the LENS while the camera stays at its
    # full-body distance: dollying in on a 45-degree
    # lens foreshortens whatever is nearest. A clip needs the same crop to follow the body part,
    # so the target is the region's own body vertices per frame rather than a fixed point.
    ap.add_argument('--zoom-region', default=None,
                    help="SMPL region name (tools/r3/mu_interaction_map.JOINT_REGION) to frame on, "
                         "e.g. 'thigh', 'hand'. The camera holds its full-body distance and the fov "
                         "narrows; the target follows that region's BODY vertices, which are shared "
                         "by every arm (max |body_ppf - body_gnn| = 6.2 mm), so the crop cannot be "
                         "operator-dependent.")
    ap.add_argument('--zoom-side', default=None, choices=['L', 'R', 'C'],
                    help="restrict --zoom-region to one limb; without it 'forearm' spans both arms")
    ap.add_argument('--zoom-margin', type=float, default=1.9,
                    help='the framed half-extent is this multiple of the region half-diagonal')
    ap.add_argument('--zoom-ref-frame', type=int, default=None,
                    help='frame whose region extent sets the (constant) framed half-extent, and '
                         'which --zoom-lock is anchored to. Default: --pause-frame, else --start.')
    ap.add_argument('--view-frame', type=int, default=None,
                    help="frame whose SMPL facing resolves --view into a world direction. "
                         "Default --start, which is what every existing lane assumes. It matters "
                         "whenever the body turns during the clip: a still made at frame f and a "
                         "clip started at frame 0 then disagree by however far the body rotated "
                         "between them, and 'front34' means two different sides of the subject. "
                         "Pass the still's own frame to reproduce a published panel.")
    ap.add_argument('--ambient', type=float, default=None,
                    help="scene ambient strength; aitviewer's default is 2.0, which CLIPS a warm "
                         "body colour to flat white (render_fig52b.py:356). 0.7 keeps skin shaded.")
    ap.add_argument('--zoom-lock', default=None,
                    help="'cx,cy,cz,radius,body_radius' from a render_fig52b.py ZOOM-LOCK STRING: "
                         "reproduce that still's framing exactly at --zoom-ref-frame and then track. "
                         "radius and body_radius are taken verbatim (so the lens and the camera "
                         "distance are the still's); the centre is applied as an OFFSET from the "
                         "region's own centre at that frame, which is what lets it follow a limb. "
                         "--zoom-region must still be passed -- it is what is tracked.")
    # --- supplementary-PAGE additions. All default-off: with none of them set every
    # published clip re-renders unchanged.
    ap.add_argument('--zoom-shift', default=None,
                    help="'du,dv' -- slide the zoom framing so that whatever now sits at the "
                         "normalised point (0.5+du, 0.5+dv) of the zoomed frame ends up at its "
                         "CENTRE. Screen space, fractions of frame width and height, v downwards. "
                         "The lens and the camera distance are untouched, so the magnification is "
                         "exactly what --zoom-margin or --zoom-lock asked for; this only says "
                         "WHERE to point it. Applied after --zoom-lock's offset and carried by the "
                         "same tracking, so the held frame is exact and the clip follows the limb.")
    ap.add_argument('--zoom-boundary', default=None,
                    help="'<loop-rank>@<region><side>' -- frame on that free-boundary SEGMENT, the "
                         "way render_fig52b.py --zoom-boundary frames a still: what is framed is "
                         "the segment (union over every arm and mu, so no tile is favoured) plus "
                         "the body part it is drawn against, and the crop TRACKS that body part, "
                         "which every arm shares. Mutually exclusive with --zoom-region.")
    ap.add_argument('--zoom-body-cue', default='on', choices=['on', 'off'],
                    help='with --zoom-boundary: include the body part the segment sits against in '
                         'what is framed (render_fig52b.py --zoom-body-cue, -- '
                         'a boundary is only legible against a landmark)')
    ap.add_argument('--hold-zoom', action='store_true',
                    help='apply the zoom framing ONLY during the --pause-frame hold, ramping into '
                         'it and back out again, instead of for the whole clip: the motion plays at '
                         'the full-body lens and the held frame is what is magnified. A tight lens '
                         'that holds a still fine loses a moving limb for most of a rollout, which '
                         'is what this exists to avoid. Needs --pause-frame and a zoom spec.')
    ap.add_argument('--hold-zoom-in', type=float, default=1.0,
                    help='seconds of the hold spent ramping the lens in')
    ap.add_argument('--hold-zoom-out', type=float, default=1.0,
                    help='seconds spent ramping back out; 0 ends the clip zoomed')
    ap.add_argument('--end-at-hold', action='store_true',
                    help='stop when the hold ends -- nothing after --pause-frame is rendered. For '
                         'a looping tile whose payload is the held frame.')
    ap.add_argument('--cue-only', action='store_true',
                    help='write the --cue-json sidecars and NOTHING else: no frames, no mp4. The '
                         'overlay geometry is cheap and the render is not, so this is how a cue is '
                         'iterated on without re-rendering a clip that is already correct.')
    ap.add_argument('--cue-json', action='store_true',
                    help="write <out>/<tag>_cue.json: the framed boundary segment projected into "
                         "the HELD frame in normalised image coordinates, plus the hold timing and "
                         "the clip length. That is what a web page needs to draw an overlay ON TOP "
                         "of the tile instead of burning one into the picture.")
    args = ap.parse_args()

    arms, mus = args.arms.split(','), args.mus.split(',')
    W, H = (int(x) for x in args.size.split('x'))
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    garment = args.cell.split('/')[1]

    roots = {}
    if args.runs:
        for kv in args.runs.split(','):
            k, _, v = kv.partition('=')
            roots[k.strip()] = v.strip()
    data = {(a, m): load_seq(a, m, args.cell, roots) for a in arms for m in mus}
    T = min(len(v[0]) for v in data.values())
    t1 = T if args.end < 0 else min(args.end, T)
    if args.end_at_hold:
        # nothing after the held frame is ever shown, so nothing after it is rendered -- and the
        # --fill solve below then sizes the camera on the window the clip actually plays.
        if args.pause_frame is None:
            sys.exit('--end-at-hold needs --pause-frame')
        t1 = min(t1, args.pause_frame + 1)
    frames = range(args.start, t1)
    print(f'{args.cell}: {len(data)} tracks, {T} frames, rendering {args.start}..{t1 - 1}')

    # one camera geometry for every arm and mu: direction from the reference frame's body, distance
    # from the largest subject extent over the whole clip
    vframe = args.view_frame if args.view_frame is not None else args.start
    if not (0 <= vframe < T):
        sys.exit(f'--view-frame {vframe} outside 0..{T - 1}')
    ref_body = data[(arms[0], mus[0])][1][vframe]
    fwd, right, _, _ = body_frame(ref_body)
    VIEWS = {'front': 0.0, 'front34': 40.0, 'front34l': -40.0, 'side': 90.0, 'back34': 140.0}
    try:
        deg = float(args.view)
    except ValueError:
        deg = VIEWS[args.view]
    th, el = np.deg2rad(deg), np.deg2rad(args.elevation)
    dirv = (fwd * np.cos(th) + right * np.sin(th)) * np.cos(el) + np.array([0.0, np.sin(el), 0.0])
    radius, lo, hi = 0.0, None, None
    for cl, bo, _, _ in data.values():
        for t in frames:
            p = np.concatenate([cl[t], bo[t]])
            radius = max(radius, float(np.linalg.norm(p.max(0) - p.min(0))) * 0.5)
            lo = p.min(0) if lo is None else np.minimum(lo, p.min(0))
            hi = p.max(0) if hi is None else np.maximum(hi, p.max(0))
    dist = 2.6 * radius / max(args.zoom, 1e-3)
    # the static camera's target: the centre of everything the clip ever contains, so nothing leaves
    # the frame and nothing moves but the cloth
    fixed_ctr = 0.5 * (lo + hi)
    # 'lock' path: exactly utils/show.py's convention (aitviewer path.lock_to_node) -- target =
    # subject centre each frame, camera at a CONSTANT offset from it, both gaussian-smoothed. The
    # path is built from the REFERENCE arm only, so every arm and mu shares one camera and the
    # comparison is not re-framed between tiles.
    import scipy.ndimage as ndi
    rcl, rbo, _, _ = data[(arms[0], mus[0])]
    # target the BODY, not body+cloth: a swinging limb drags the joint bbox centre sideways and the
    # subject drifts off-centre for the whole clip
    tgt = np.stack([0.5 * (rbo[t].min(0) + rbo[t].max(0)) for t in frames])
    if args.smooth_sigma > 0:
        tgt = ndi.gaussian_filter1d(tgt, args.smooth_sigma, axis=0, mode='nearest')
    # --fill: re-solve the distance so the subject actually FILLS the frame. The union-bbox rule
    # above sizes on the widest thing the clip ever contains measured from a fixed centre, which on
    # a walking subject is dominated by the TRAVEL, not by the body -- so the human comes out small.
    # For every frame, arm and mu, the distance that
    # puts the subject's extent about its own camera target at `fill` of the half-frame, maximised.
    if args.fill is not None:
        fwd_c = -dirv
        right_c = np.cross(fwd_c, np.array([0.0, 1.0, 0.0]))
        right_c /= np.linalg.norm(right_c)
        up_c = np.cross(right_c, fwd_c)
        tan = np.tan(np.deg2rad(args.fov) * 0.5)
        need = 0.0
        for (a_, m_), (cl_, bo_, _, _) in data.items():
            for t in frames:
                if args.camera == 'lock':
                    c_ = tgt[t - args.start]
                elif args.camera == 'static':
                    c_ = fixed_ctr
                else:
                    c_ = 0.5 * (bo_[t].min(0) + bo_[t].max(0))
                u = np.concatenate([cl_[t], bo_[t]]) - c_
                d_off = u @ fwd_c
                need = max(need,
                           float(np.max(np.abs(u @ right_c) / (args.fill * tan * (W / H)) - d_off)),
                           float(np.max(np.abs(u @ up_c) / (args.fill * tan) - d_off)))
        dist = need / max(args.zoom, 1e-3)
        print(f'  --fill {args.fill}: camera distance {dist:.2f} m')

    # --- the zoom framing ---------------------------------------------------------------------
    # Solved once here whether it is applied to the whole clip (--zoom-region/--zoom-boundary on
    # their own, the behaviour) or only to the held frame (--hold-zoom). Either way it is
    # render_fig52b.py's focal rule: the camera keeps a full-body distance and the LENS narrows.
    fov_eff = args.fov
    zoom = None                 # (tracked centre per frame, camera distance, fov) when set
    zseg = None                 # the boundary segment the zoom was framed on, for --cue-json
    if args.zoom_region and args.zoom_boundary:
        sys.exit('--zoom-region and --zoom-boundary both frame the crop; pass one')
    if args.zoom_region or args.zoom_boundary:
        from tools.r3.mu_interaction_map import JOINT_REGION, JOINT_SIDE
        with open(REPO / 'ccraft_data/aux_data/body_models/smpl/SMPL_FEMALE.pkl', 'rb') as f:
            wgt = np.asarray(pickle.load(f, encoding='latin1')['weights'])
        vj = wgt.argmax(axis=1)
        tref = args.zoom_ref_frame if args.zoom_ref_frame is not None else (
            args.pause_frame if args.pause_frame is not None else args.start)
        if not (args.start <= tref < t1):
            sys.exit(f'--zoom-ref-frame {tref} outside the rendered range {args.start}..{t1 - 1}')
        zr = None
        if args.zoom_boundary:
            # render_fig52b.py:404-455 carried to a sequence. What is FRAMED is the segment over
            # every arm and mu (so the crop is not operator-dependent) plus the body part it is
            # drawn against; what is TRACKED is that body part alone, which every arm shares.
            from scipy.spatial import cKDTree
            from tools.r3.garment_boundary import boundary_loops, smpl_vertex_labels
            rank, _, at_want = args.zoom_boundary.partition('@')
            loops = boundary_loops(data[(arms[0], mus[0])][2])
            if not rank.isdigit() or int(rank) >= len(loops):
                sys.exit(f'--zoom-boundary loop rank must be 0..{len(loops) - 1} '
                         f'(sizes {[len(q) for q in loops]})')
            lp = loops[int(rank)]
            vreg, vside = smpl_vertex_labels()
            rcl_, rbo_, _, _ = data[(arms[0], mus[-1])]
            _, nn = cKDTree(rbo_[tref]).query(rcl_[tref][lp])
            at = np.array([f'{x}{y}' for x, y in zip(vreg[nn], vside[nn])])
            zseg = lp[at == at_want] if at_want else lp
            if len(zseg) < 3:
                vals, cnt = np.unique(at, return_counts=True)
                sys.exit(f'only {len(zseg)} boundary verts at {at_want!r}; loop {rank} sits against '
                         + ', '.join(f'{v}:{c}' for v, c in sorted(zip(vals, cnt), key=lambda q: -q[1])))
            sel = np.array([f'{JOINT_REGION[int(j)]}{JOINT_SIDE[int(j)]}' for j in vj]) == at_want \
                if (at_want and args.zoom_body_cue == 'on') else np.zeros(len(vj), bool)
            framed = np.concatenate([cl_[tref][zseg] for cl_, _, _, _ in data.values()]
                                    + ([rbo_[tref][sel]] if sel.any() else []))
            zr = float(np.linalg.norm(framed.max(0) - framed.min(0))) * 0.5 * args.zoom_margin
            zctr_ref = 0.5 * (framed.min(0) + framed.max(0))
            if not sel.any():          # no body cue: track the segment's own body neighbours
                sel = np.zeros(len(vj), bool)
                sel[np.unique(nn)] = True
            print(f'  ZOOM on boundary {args.zoom_boundary!r}: {len(zseg)} edge verts '
                  f'+ {int(sel.sum())} body-cue verts, framed half-extent {zr:.3f} m')
        else:
            sel = np.array([JOINT_REGION[int(j)] for j in vj]) == args.zoom_region
            if args.zoom_side:
                sel &= np.array([JOINT_SIDE[int(j)] for j in vj]) == args.zoom_side
            if not sel.any():
                sys.exit(f'unknown --zoom-region {args.zoom_region!r}; '
                         f'choose from {sorted(set(JOINT_REGION.values()))}')
        # target: the tracked part's centre per frame, smoothed on the same path as the body target
        # so a gait does not shake the crop. Body-side, so it is identical for every arm and mu.
        zc = np.stack([0.5 * (rbo[t][sel].min(0) + rbo[t][sel].max(0)) for t in frames])
        if args.smooth_sigma > 0:
            zc = ndi.gaussian_filter1d(zc, args.smooth_sigma, axis=0, mode='nearest')
        if zr is None:
            rp = rbo[tref][sel]
            zr = float(np.linalg.norm(rp.max(0) - rp.min(0))) * 0.5 * args.zoom_margin
        else:
            zc = zc + (zctr_ref - zc[tref - args.start])[None, :]
        zdist = dist
        if args.zoom_lock:
            cx, cy, cz, zr, br = (float(v) for v in args.zoom_lock.split(','))
            # the still's own camera distance, so the perspective matches it and not this clip's
            # union bbox; the centre becomes an offset the tracking then carries.
            zdist = 2.6 * br / max(args.zoom, 1e-3)
            zc = zc + (np.array([cx, cy, cz]) - zc[tref - args.start])[None, :]
            print(f'  ZOOM-LOCK at f{tref}: radius {zr:.3f} m, body_radius {br:.3f} m, '
                  f'camera distance {zdist:.2f} m')
        zfov = float(np.rad2deg(2.0 * np.arctan(zr / zdist)))
        if args.zoom_shift:
            # Re-point without re-scaling. A screen offset of (du, dv) at the target plane is
            # 2*du*zdist*tan(zfov/2)*(W/H) along the camera's right and -2*dv*zdist*tan(zfov/2)
            # along its up -- the inverse of project()'s mapping, evaluated at z = zdist. Adding
            # it to every tracked centre slides the crop and leaves the lens alone.
            du, dv = (float(v) for v in args.zoom_shift.split(','))
            f_ = -dirv
            side_ = np.cross(f_, np.array([0.0, 1.0, 0.0])); side_ /= np.linalg.norm(side_)
            up_ = np.cross(side_, f_)
            th_ = np.tan(np.deg2rad(zfov) * 0.5) * zdist
            zc = zc + (2.0 * du * th_ * (W / H) * side_ - 2.0 * dv * th_ * up_)[None, :]
            print(f'  ZOOM-SHIFT ({du:+.3f}, {dv:+.3f}) of frame: the crop moves '
                  f'{abs(2.0 * du * th_ * (W / H)) * 100:.1f} cm across and '
                  f'{abs(2.0 * dv * th_) * 100:.1f} cm down, lens unchanged at {zfov:.1f} deg')
        zoom = (zc, zdist, zfov)
        what = args.zoom_boundary or f'{args.zoom_region!r}{args.zoom_side or ""}'
        if args.hold_zoom:
            if args.pause_frame is None:
                sys.exit('--hold-zoom needs --pause-frame')
            print(f'  HOLD zoom on {what}: the clip plays at fov {args.fov:.0f} deg and the held '
                  f'frame ramps to {zfov:.1f} deg (half-extent {zr:.3f} m at {zdist:.2f} m)')
        else:
            tgt, dist, fov_eff = zc, zdist, zfov
            print(f'  FOCAL zoom on {what}: framed half-extent {zr:.3f} m, '
                  f'fov {args.fov:.0f} -> {fov_eff:.1f} deg, target tracks the region')
    elif args.hold_zoom:
        sys.exit('--hold-zoom needs --zoom-region or --zoom-boundary to say what to zoom to')

    rel = dist * dirv                                   # the 'relative_position' of lock_to_node
    cam_pos = tgt + rel[None, :]

    # lowest point of the whole subject per frame, over every arm and mu, so one floor serves the
    # panel and no arm's hem pokes through it
    lows = np.min(np.stack([np.minimum(cl_[args.start:t1, :, 1].min(1),
                                       bo_[args.start:t1, :, 1].min(1))
                            for cl_, bo_, _, _ in data.values()]), axis=0)
    floor_rgb = tuple(float(x) for x in args.floor_color.split(',')) + (1.0,)
    print(f'  view {deg:+.0f} deg (body facing read at f{vframe}), radius {radius:.2f} m, '
          f'camera distance {dist:.2f} m')

    uvspec = None
    cloth_rgb = tuple(float(x) for x in args.cloth_color.split(',')) if args.cloth_color \
        else PALETTE.get(garment, CLOTH_FALLBACK)
    body_rgb = tuple(float(x) for x in args.body_color.split(',')) if args.body_color else BODY_RGB
    if args.texture:
        from tools.r3.garment_uv import uv_map, uv_tiled, tint
        v2v3, mf, uv_m = uv_map(garment, data[(arms[0], mus[0])][2])
        texdir = REPO / args.texture_dir
        cache = texdir / '_tinted'
        cache.mkdir(parents=True, exist_ok=True)
        png = tint(str(texdir / f'{args.texture}.jpg'), cloth_rgb,
                   str(cache / f'{garment}__{args.texture}.png'))
        uvspec = dict(v2v3=v2v3, faces=mf, uv=uv_tiled(uv_m, args.texture, 1.0, texdir), path=png)

    from aitviewer.configuration import CONFIG as C
    C.update_conf({'window_width': W, 'window_height': H})
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.meshes import Meshes
    from aitviewer.renderables.plane import ChessboardPlane
    from aitviewer.scene.camera import PinholeCamera

    r = HeadlessRenderer(window_type='headless', backend='egl')
    if args.ambient is not None:
        r.scene.ambient_strength = float(args.ambient)
    if args.light_azimuth is not None:
        from aitviewer.utils.utils import spherical_coordinates_from_direction
        thl, ell = np.deg2rad(deg + args.light_azimuth), np.deg2rad(args.light_elevation)
        p = (fwd * np.cos(thl) + right * np.sin(thl)) * np.cos(ell) \
            + np.array([0.0, np.sin(ell), 0.0])
        e_, a_ = spherical_coordinates_from_direction(-p / np.linalg.norm(p), degrees=True)
        key = next(li for li in r.scene.lights if li.shadow_enabled)
        key.elevation, key.azimuth = float(e_), float(a_)
    # second, shadowless light, so the shaded side of a dress is not
    # a flat silhouette. Both strengths stay at aitviewer's defaults unless asked for.
    if args.fill_light_azimuth is not None:
        from aitviewer.utils.utils import spherical_coordinates_from_direction as _sph
        thf, elf = np.deg2rad(deg + args.fill_light_azimuth), np.deg2rad(25.0)
        pf_ = (fwd * np.cos(thf) + right * np.sin(thf)) * np.cos(elf) \
            + np.array([0.0, np.sin(elf), 0.0])
        fe_, fa_ = _sph(-pf_ / np.linalg.norm(pf_), degrees=True)
        fl = next(li for li in r.scene.lights if not li.shadow_enabled)
        fl.elevation, fl.azimuth = float(fe_), float(fa_)
    if args.light_strength is not None:
        next(li for li in r.scene.lights if li.shadow_enabled).strength = args.light_strength
    if args.fill_light_strength is not None:
        next(li for li in r.scene.lights if not li.shadow_enabled).strength = \
            args.fill_light_strength

    # boundary segment to highlight during the pause, resolved on the reference arm at the pause
    # frame so the box is the same rectangle in every arm's clip
    box_seg = None
    if args.box_boundary and args.pause_frame is not None:
        from scipy.spatial import cKDTree
        from tools.r3.garment_boundary import boundary_loops, smpl_vertex_labels
        from tools.r3.mu_interaction_map import JOINT_REGION as JR, JOINT_SIDE as JS
        rank, _, at_want = args.box_boundary.partition('@')
        lp = boundary_loops(data[(arms[0], mus[0])][2])[int(rank)]
        vreg, vside = smpl_vertex_labels()
        pcl, pbo, _, _ = data[(arms[0], mus[-1])]
        _, nn = cKDTree(pbo[args.pause_frame]).query(pcl[args.pause_frame][lp])
        at = np.array([f'{x}{y}' for x, y in zip(vreg[nn], vside[nn])])
        box_seg = lp[at == at_want] if at_want else lp
        print(f'  red box on {len(box_seg)} boundary vertices at {args.box_boundary!r}')

    def project(pts, eye, ctr, fovv=None):
        """World -> pixel, for the same PinholeCamera geometry (vertical fov, y down on screen)."""
        f = ctr - eye
        f /= np.linalg.norm(f)
        sidev = np.cross(f, np.array([0.0, 1.0, 0.0]))
        sidev /= np.linalg.norm(sidev)
        upv = np.cross(sidev, f)
        d = pts - eye[None, :]
        x, y, z = d @ sidev, d @ upv, d @ f
        th = np.tan(np.deg2rad(fov_eff if fovv is None else fovv) * 0.5)
        u = (x / (z * th * (W / H)) + 1.0) * 0.5 * W
        v = (1.0 - y / (z * th)) * 0.5 * H
        return np.stack([u, v], axis=1)

    use_inset = bool(args.inset or args.inset_track)

    def cam_at(t, bo_):
        if args.camera == 'lock':
            return cam_pos[t - args.start], tgt[t - args.start]
        if args.camera == 'static':
            return fixed_ctr + dist * dirv, fixed_ctr
        c_ = 0.5 * (bo_[t].min(0) + bo_[t].max(0))
        return c_ + dist * dirv, c_

    def box_rect_union(t):
        """Pixel rectangle around the boxed boundary at frame t, union over every arm and mu, so
        one rectangle serves every tile of the scene and every arm's boundary lies inside it."""
        pad = 0.045 * max(W, H)
        lo_, hi_ = None, None
        for (a_, m_), (cl_, bo_, _, _) in data.items():
            eye_, ctr_ = cam_at(t, bo_)
            uv_ = project(cl_[t][box_seg].astype(np.float64), eye_, ctr_)
            lo_ = uv_.min(0) if lo_ is None else np.minimum(lo_, uv_.min(0))
            hi_ = uv_.max(0) if hi_ is None else np.maximum(hi_, uv_.max(0))
        x0, y0 = lo_ - pad
        x1, y1 = hi_ + pad
        return [float(max(x0, 2)), float(max(y0, 2)), float(min(x1, W - 3)), float(min(y1, H - 3))]

    RED = (214, 27, 27)

    def annotate(im, rect):
        """Draw the red box; with --inset also paste the magnified crop in the farthest corner and
        a connector line. Returns (im, inset_rect or None). The crop is taken BEFORE the outline is
        drawn so the magnified region shows cloth, not the box."""
        from PIL import Image, ImageDraw
        x0, y0, x1, y1 = rect
        lw = max(3, W // 220)
        inset_rect = None
        if args.inset:
            crop = im.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
            bw, bh = crop.size
            sc = args.inset_scale
            sc = min(sc, 0.45 * W / max(bw, 1), 0.45 * H / max(bh, 1))
            iw, ih = max(int(round(bw * sc)), 1), max(int(round(bh * sc)), 1)
            crop = crop.resize((iw, ih), Image.LANCZOS)
            mg = max(12, W // 75)
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            corners = {'TL': (mg, mg), 'TR': (W - mg - iw, mg),
                       'BL': (mg, H - mg - ih), 'BR': (W - mg - iw, H - mg - ih)}
            best = max(corners, key=lambda k: (corners[k][0] + iw / 2 - cx) ** 2
                       + (corners[k][1] + ih / 2 - cy) ** 2)
            ix, iy = corners[best]
            im.paste(crop, (int(ix), int(iy)))
            inset_rect = [float(ix), float(iy), float(ix + iw), float(iy + ih)]
        dr = ImageDraw.Draw(im)
        if inset_rect is not None:
            ix0, iy0, ix1, iy1 = inset_rect
            # connector: from the box corner nearest the inset to the inset corner nearest the box
            bc = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            ic = [(ix0, iy0), (ix1, iy0), (ix0, iy1), (ix1, iy1)]
            pb, pi = min(((b, i) for b in bc for i in ic),
                         key=lambda q: (q[0][0] - q[1][0]) ** 2 + (q[0][1] - q[1][1]) ** 2)
            dr.line([pb, pi], fill=RED, width=max(2, lw // 2))
            dr.rectangle([ix0, iy0, ix1 - 1, iy1 - 1], outline=RED, width=lw)
        dr.rectangle([x0, y0, x1, y1], outline=RED, width=lw)
        return im, inset_rect

    def render_state(cl_, bo_, cf_, of_, t, eye, ctr, fovv, path):
        """One exported frame: the geometry at rollout step `t` seen from (eye, ctr) through a
        `fovv` lens. Split out of the frame loop so the --hold-zoom ramp can re-render the SAME
        state through a moving lens without duplicating any of the scene set-up."""
        body = Meshes(bo_[t][None], of_, name='body', color=(*body_rgb, 1.0))
        if uvspec is None:
            cloth = Meshes(cl_[t][None], cf_, name='cloth', color=(*cloth_rgb, 1.0))
        else:
            cloth = Meshes(cl_[t][None][:, uvspec['v2v3']], uvspec['faces'], name='cloth',
                           uv_coords=uvspec['uv'], path_to_texture=uvspec['path'])
        body.backface_culling = cloth.backface_culling = False
        r.scene.add(body, cloth)
        r.scene.floor.enabled = False
        r.scene.origin.enabled = False
        # --floor: a plain ground plane. Without --floor-track it is
        # one height for the clip, so mocap root drift lifts or sinks the feet; with it the
        # plane follows the frame's own lowest point and the feet stay planted.
        plane = None
        if args.floor:
            fy = float(lows[t - args.start]) if args.floor_track else float(lows.min())
            # ⚠️ ChessboardPlane's `height=` is STORED AND NEVER APPLIED -- only the
            # sibling `Chessboard` class writes it into the vertices (aitviewer
            # renderables/plane.py:361). Passing it leaves the floor at y=0, which on these
            # sequences (ground at y ~ -1.13) sits at hip height and hides the whole lower
            # body. Move the NODE instead. Centring it on the camera target as well keeps a
            # modest --floor-size under a subject that walks several metres.
            plane = ChessboardPlane(args.floor_size, 1, floor_rgb, floor_rgb, 'xz',
                                    name='PlainFloor',
                                    position=(float(ctr[0]), fy - 0.004, float(ctr[2])))
            plane.material.diffuse = 0.55
            r.scene.add(plane)
            # the shadow frustum is an ORTHOGRAPHIC box around the light's POSITION, so it
            # has to be parked on the subject or the shadow silently vanishes
            key_l = next(li for li in r.scene.lights if li.shadow_enabled)
            key_l.position = tuple(ctr - np.asarray(key_l.direction, float) * 4.0)
            key_l.shadow_map_size = 6.0
            key_l.shadow_map_near, key_l.shadow_map_far = 0.1, 20.0
            key_l._show_debug_lines = False
            if key_l._debug_lines is not None:
                key_l._debug_lines.enabled = False
            key_l.mesh.enabled = False
        cam = PinholeCamera(np.asarray(eye)[None], np.asarray(ctr)[None], W, H,
                            fov=fovv, viewer=r)
        r.scene.add(cam)
        r.set_temp_camera(cam)
        r._init_scene()
        r.export_frame(str(path), transparent_background=args.transparent)
        for n in (body, cloth, cam) + ((plane,) if plane is not None else ()):
            r.scene.remove(n)

    # the --hold-zoom ramp, as a list of (eye, ctr, fov) one per held frame. The lens and the
    # camera distance are interpolated in LOG, so the apparent size of the subject grows at a
    # constant rate and the move reads as a zoom rather than as a jump; the ease is a smoothstep.
    def hold_cameras(bo_):
        wide_eye, wide_ctr = cam_at(args.pause_frame, bo_)
        zc_, zdist_, zfov_ = zoom
        z_ctr = zc_[args.pause_frame - args.start]
        z_eye = z_ctr + zdist_ * dirv
        d_w = float(np.linalg.norm(wide_eye - wide_ctr))
        h_w, h_z = d_w * np.tan(np.deg2rad(fov_eff) * 0.5), zdist_ * np.tan(np.deg2rad(zfov_) * 0.5)
        n_in = max(int(round(args.hold_zoom_in * args.fps)), 0)
        n_out = max(int(round(args.hold_zoom_out * args.fps)), 0)
        n_still = max(int(round(args.pause_seconds * args.fps)), 1)
        out_ = []
        for k in range(n_in + n_still + n_out):
            u = (min(k + 1, n_in) / n_in) if (k < n_in and n_in) else (
                1.0 if k < n_in + n_still else
                max(0.0, 1.0 - (k - n_in - n_still + 1) / max(n_out, 1)))
            e = u * u * (3.0 - 2.0 * u)
            ctr_ = (1.0 - e) * np.asarray(wide_ctr) + e * np.asarray(z_ctr)
            d_ = float(np.exp((1.0 - e) * np.log(d_w) + e * np.log(zdist_)))
            h_ = float(np.exp((1.0 - e) * np.log(h_w) + e * np.log(h_z)))
            out_.append((ctr_ + d_ * dirv, ctr_, float(np.rad2deg(2.0 * np.arctan(h_ / d_)))))
        return out_, n_in, n_still, n_out

    for a in arms:
        for m in mus:
            cl, bo, cf, of = data[(a, m)]
            tag = f'{garment}__{a}__mu{m}'
            fdir = out / f'_frames_{tag}'
            fdir.mkdir(parents=True, exist_ok=True)
            for t in (() if args.cue_only else frames):
                # static by default (see --camera); 'track' follows the body and jitters
                if args.camera == 'lock':
                    ctr, eye = tgt[t - args.start], cam_pos[t - args.start]
                elif args.camera == 'static':
                    ctr, eye = fixed_ctr, fixed_ctr + dist * dirv
                else:
                    ctr = 0.5 * (bo[t].min(0) + bo[t].max(0))
                    eye = ctr + dist * dirv
                render_state(cl, bo, cf, of, t, eye, ctr, fov_eff, fdir / f'{t:05d}.png')
            # --inset-track: annotate every exported frame with the frame's own union rectangle
            box_json = {'cell': args.cell, 'arm': a, 'mu': m, 'fps': args.fps,
                        'start': args.start, 'end_exclusive': t1, 'size': [W, H],
                        'pause_frame': args.pause_frame, 'hold_frames': 0, 'box': None,
                        'inset': None, 'inset_scale': args.inset_scale if args.inset else None,
                        'tracked': bool(args.inset_track and box_seg is not None), 'boxes': {}}
            if args.inset_track and box_seg is not None:
                from PIL import Image
                for t in frames:
                    rect_t = box_rect_union(t)
                    im_t = Image.open(fdir / f'{t:05d}.png').convert(
                        'RGBA' if args.transparent else 'RGB')
                    im_t, ins_t = annotate(im_t, rect_t)
                    im_t.save(fdir / f'{t:05d}.png')
                    box_json['boxes'][int(t)] = {'box': rect_t, 'inset': ins_t}
            # --hold-zoom: RENDER the hold instead of duplicating one image -- same rollout
            # state throughout, a moving lens. Everything after the pause frame shifts up by
            # n_hold - 1 exactly as in the plain hold below (the pause frame's own slot is reused
            # by the ramp's first frame), so ffmpeg still sees one contiguous sequence.
            if zoom is not None and args.hold_zoom and args.start <= args.pause_frame < t1:
                cams, n_in, n_still, n_out = hold_cameras(bo)
                n_hold = len(cams)
                box_json['hold_frames'] = n_hold
                for t in (() if args.cue_only else range(t1 - 1, args.pause_frame, -1)):
                    (fdir / f'{t:05d}.png').rename(fdir / f'{t + n_hold - 1:05d}.png')
                prev = None
                for k, (eye_k, ctr_k, fov_k) in ([] if args.cue_only else enumerate(cams)):
                    dst_k = fdir / f'{args.pause_frame + k:05d}.png'
                    # the still part of the hold is one camera on one rollout state, so it is one
                    # picture; rendering it 200 times is the same picture 200 times
                    if prev is not None and np.allclose(eye_k, prev[0]) \
                            and np.allclose(ctr_k, prev[1]) and abs(fov_k - prev[2]) < 1e-9:
                        shutil.copyfile(prev[3], dst_k)
                    else:
                        render_state(cl, bo, cf, of, args.pause_frame, eye_k, ctr_k, fov_k, dst_k)
                    prev = (eye_k, ctr_k, fov_k, dst_k)
                if args.cue_json:
                    # The overlay the page draws lives in NORMALISED image coordinates on the
                    # fully-zoomed camera, so it is the same geometry whatever size the tile is
                    # displayed at -- and it is the projection this render actually used, not a
                    # hand-placed guess. Vertices whose normal faces away are dropped: the far
                    # side of an armhole would otherwise draw a line across the body.
                    import json as _json
                    seg = zseg if zseg is not None else box_seg
                    if seg is None:
                        sys.exit('--cue-json needs --zoom-boundary (or --box-boundary) to say '
                                 'which edge the overlay traces')
                    eye_z, ctr_z, fov_z = cams[min(n_in, len(cams) - 1)]
                    pf = args.pause_frame
                    vn = np.zeros_like(cl[pf])
                    tri = cl[pf][cf]
                    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    for c in range(3):
                        np.add.at(vn, cf[:, c], fn)
                    facing = ((np.asarray(eye_z)[None, :] - cl[pf][seg]) * vn[seg]).sum(1) > 0
                    # ...and hidden BY THE BODY. An armhole is a closed loop around a limb, so its
                    # far side faces the camera perfectly well and is still behind the arm; drawing
                    # it puts a line straight across the shoulder. Test in the image: a boundary
                    # vertex is occluded when a body vertex projecting within a neighbour's spacing
                    # of it sits a clear 2 cm nearer the eye.
                    from scipy.spatial import cKDTree as _KD
                    fh = np.asarray(ctr_z) - np.asarray(eye_z)
                    fh = fh / np.linalg.norm(fh)
                    buv = project(bo[pf].astype(np.float64), np.asarray(eye_z),
                                  np.asarray(ctr_z), fov_z)
                    suv_px = project(cl[pf][seg].astype(np.float64), np.asarray(eye_z),
                                     np.asarray(ctr_z), fov_z)
                    bz = (bo[pf] - np.asarray(eye_z)[None, :]) @ fh
                    sz = (cl[pf][seg] - np.asarray(eye_z)[None, :]) @ fh
                    tb = _KD(buv)
                    rad = 1.5 * float(np.median(tb.query(buv, k=2)[0][:, 1]))
                    for i_ in range(len(seg)):
                        nb = tb.query_ball_point(suv_px[i_], rad)
                        if nb and float(bz[nb].min()) < float(sz[i_]) - 0.02:
                            facing[i_] = False
                    uv = project(cl[pf][seg].astype(np.float64), np.asarray(eye_z),
                                 np.asarray(ctr_z), fov_z) / np.array([W, H])
                    # Written per VERTEX, in the loop's own order, with a visibility flag rather
                    # than as pre-cut polylines: the page compares one tile's edge against another
                    # tile's, and that comparison is only meaningful vertex to vertex. Cutting the
                    # runs here would throw the correspondence away.
                    (out / f'{tag}_cue.json').write_text(_json.dumps({
                        'cell': args.cell, 'arm': a, 'mu': m, 'fps': args.fps, 'size': [W, H],
                        'boundary': args.zoom_boundary or args.box_boundary,
                        'n_frames': (t1 - args.start) + n_hold - 1,
                        'hold_start': args.pause_frame - args.start,
                        'zoom_in_frames': n_in, 'still_frames': n_still, 'zoom_out_frames': n_out,
                        'verts': [int(v) for v in seg],
                        'uv': [[round(float(x), 5), round(float(y), 5)] for x, y in uv],
                        'vis': [int(bool(v)) for v in facing],
                    }, indent=1))
            # hold the figure's own frame, marked, so the clip says "this is the still you saw"
            elif args.pause_frame is not None and args.start <= args.pause_frame < t1:
                from PIL import Image, ImageDraw
                im = Image.open(fdir / f'{args.pause_frame:05d}.png').convert(
                    'RGBA' if args.transparent else 'RGB')
                if box_seg is not None and use_inset:
                    rect = box_rect_union(args.pause_frame)
                    if not args.inset_track:
                        im, ins = annotate(im, rect)
                    else:
                        ins = box_json['boxes'][int(args.pause_frame)]['inset']
                    box_json['box'], box_json['inset'] = rect, ins
                elif box_seg is not None:
                    if args.camera == 'lock':
                        eye, ctr = cam_pos[args.pause_frame - args.start], \
                            tgt[args.pause_frame - args.start]
                    elif args.camera == 'static':
                        ctr, eye = fixed_ctr, fixed_ctr + dist * dirv
                    else:
                        ctr = 0.5 * (bo[args.pause_frame].min(0) + bo[args.pause_frame].max(0))
                        eye = ctr + dist * dirv
                    uv = project(cl[args.pause_frame][box_seg].astype(np.float64), eye, ctr)
                    pad = 0.045 * max(W, H)
                    x0, y0 = uv.min(0) - pad
                    x1, y1 = uv.max(0) + pad
                    dr = ImageDraw.Draw(im)
                    dr.rectangle([max(x0, 2), max(y0, 2), min(x1, W - 3), min(y1, H - 3)],
                                 outline=(214, 27, 27), width=max(3, W // 220))
                n_hold = max(int(round(args.pause_seconds * args.fps)), 1)
                box_json['hold_frames'] = n_hold
                # Renumber everything after the pause so ffmpeg sees ONE CONTIGUOUS sequence.
                # The shift is n_hold - 1, not n_hold: the pause frame's own slot is reused by the
                # first held copy. Shifting by n_hold leaves a one-index hole, and ffmpeg's image
                # demuxer stops dead at the first missing number -- which silently truncated the
                # long-sleeve clips from 377 frames to 233 before this was caught by a frame count.
                for t in range(t1 - 1, args.pause_frame, -1):
                    (fdir / f'{t:05d}.png').rename(fdir / f'{t + n_hold - 1:05d}.png')
                for k in range(n_hold):
                    im.save(fdir / f'{args.pause_frame + k:05d}.png')

            if args.cue_only:
                print(f'  cue-only: wrote {tag}_cue.json')
                continue
            mp4 = out / f'{tag}.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                            '-framerate', str(args.fps), '-start_number', str(args.start),
                            '-i', str(fdir / '%05d.png'), '-c:v', 'libx264', '-crf', '18',
                            '-pix_fmt', 'yuv420p', str(mp4)], check=True)
            if not args.keep_frames:
                shutil.rmtree(fdir)
            print(f'  wrote {os.path.relpath(mp4, REPO)}  ({ARM_LABEL.get(a, a)}, mu {m})')
            if args.box_json:
                import json
                box_json['n_frames'] = (t1 - args.start) + max(box_json['hold_frames'] - 1, 0)
                (out / f'{tag}_box.json').write_text(json.dumps(box_json, indent=1))

    (out / 'README.txt').write_text(
        f'cell   {args.cell}\narms   {", ".join(arms)}\nmu     {", ".join(mus)}\n'
        f'frames {args.start}..{t1 - 1} at {args.fps} fps\n'
        f'view   {deg:+.0f} deg body-relative, elevation {args.elevation}\n'
        'Camera tracks the body centre; direction and distance fixed across every arm and mu.\n')


if __name__ == '__main__':
    main()
