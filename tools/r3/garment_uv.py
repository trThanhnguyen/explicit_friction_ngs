#!/usr/bin/env python
"""Texture a garment rollout using its own SEWING PATTERN as the UV map.

Why this exists: the owner's verdict on the flat-colour Fig 5.2-B candidates was that panels 10/11
"can't show any visible difference ... because of the uniform texture in the cloth (single color)".
That is a rendering fault, not a physics one -- the mu response there is 30-31 mm of p95 tangential
displacement, which a single-colour surface cannot express, because a flat garment sliding ALONG the
body changes only its silhouette.

The fix needs a spatial reference that is glued to the material. We already have one:
`ccraft_data/aux_data/garments_dict.pkl` carries `mesh_pos` (V2,2) + `mesh_faces` -- the garment's 2D
pattern layout in METRES, i.e. the same material space the stretching energy is written on. It is a
genuine UV map, already seam-split (V2 > V3 exactly at the seams), so a tiled fabric image advects
with the cloth and its cell size is set in physical units rather than screen units.

Two facts that make this usable, both verified rather than assumed (see `verify` below):
  * `mesh_faces[i,k]` and `faces[i,k]` are corner-aligned, so v2 -> v3 is a well-defined scatter.
  * the tshirt carries no `mesh_pos`, but it IS `tshirt_unzipped` with 31 seam vertices welded: same
    8710 faces, and the face-order relabelling is conflict-free onto all 4424 vertices. So its UVs
    transfer exactly. Do NOT try to recover this by nearest-rest-vertex matching -- the rest poses
    differ by up to 3.2 cm and that route silently produces a wrong map.

Tinting: the textured shader REPLACES the mesh colour (`lit_with_edges.glsl:231`,
`base_color.xyz = texture(...)`), so a palette cannot be applied at draw time. `tint()` bakes it in
instead -- luminance detail from the photograph, hue from the garment's palette entry. One fabric
tinted four ways keeps the four garment identities the owner fixed on while adding real weave.

Textures are CC0 (Poly Haven / ambientCG); `TEX_DIMS_M` records each scan's real-world extent so the
tile scale is physical, not guessed.
"""
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GARMENTS_DICT = REPO / 'ccraft_data/aux_data/garments_dict.pkl'

# garment -> the dict entry that actually carries `mesh_pos`
UV_SOURCE = {'tshirt': 'tshirt_unzipped'}

# Real-world extent of one texture tile, metres, from the Poly Haven asset API `dimensions` field.
# This is what makes "the mottling reads ~8 mm on the body" a measurable statement.
TEX_DIMS_M = {
    'jersey_melange':             (0.2906, 0.2762),
    'jogging_melange':            (0.2735, 0.2733),
    'cotton_jersey':              (0.2636, 0.2638),
    'knitted_fleece':             (0.2664, 0.2704),
    'rough_linen':                (0.2707, 0.2713),
    'poly_wool_herringbone':      (0.2701, 0.2757),
    'floral_jacquard':            (0.2529, 0.3836),
    'quatrefoil_jacquard_fabric': (0.2801, 0.2838),
    'denim_fabric_03':            (0.2714, 0.2710),
}


def _load_dict():
    with open(GARMENTS_DICT, 'rb') as f:
        return pickle.load(f)


def uv_map(garment, cloth_faces=None):
    """Seam-split render topology + UVs for `garment`.

    Returns (v2_to_v3, mesh_faces, uv_m) where
        v2_to_v3  (V2,)   gather index: render_verts = pred[frame][v2_to_v3]
        mesh_faces (F,3)  topology to draw with (indexes the seam-split vertices)
        uv_m      (V2,2)  pattern-space coordinates in METRES

    `cloth_faces` is the rollout pkl's own face array. Pass it: for the tshirt it is the only way to
    resolve the weld, and for the rest it is a free consistency check.
    """
    D = _load_dict()
    src = UV_SOURCE.get(garment, garment)
    gd = D[src]
    if 'mesh_pos' not in gd:
        raise KeyError(f'{src!r} carries no mesh_pos; no pattern UV available for {garment!r}')
    f3 = np.asarray(gd['faces']).astype(np.int64)
    mf = np.asarray(gd['mesh_faces']).astype(np.int64)
    uv = np.asarray(gd['mesh_pos']).astype(np.float64)

    if src != garment:
        # tshirt <- tshirt_unzipped: relabel through the shared face ORDER, not through geometry.
        if cloth_faces is None:
            raise ValueError(f'{garment!r} needs cloth_faces to resolve the weld from {src!r}')
        F = np.asarray(cloth_faces).astype(np.int64)
        if F.shape != f3.shape:
            raise ValueError(f'face-count mismatch {F.shape} vs {f3.shape}; the weld route is invalid')
        relabel = np.full(f3.max() + 1, -1, np.int64)
        for a, b in zip(f3.reshape(-1), F.reshape(-1)):
            if relabel[a] < 0:
                relabel[a] = b
            elif relabel[a] != b:
                raise ValueError(f'{garment!r}: face order does not correspond to {src!r}')
        f3 = relabel[f3]
    elif cloth_faces is not None and not np.array_equal(np.asarray(cloth_faces), f3):
        raise ValueError(f'{garment!r}: rollout faces differ from the garment dict')

    v2_to_v3 = np.full(len(uv), -1, np.int64)
    for a, b in zip(mf.reshape(-1), f3.reshape(-1)):
        if v2_to_v3[a] < 0:
            v2_to_v3[a] = b
        elif v2_to_v3[a] != b:
            raise ValueError(f'{garment!r}: mesh_faces / faces corners are not aligned')
    if (v2_to_v3 < 0).any():
        raise ValueError(f'{garment!r}: {(v2_to_v3 < 0).sum()} pattern vertices unreferenced')
    return v2_to_v3, mf, uv


def uv_cylinder(garment, cloth_faces=None, tile_w=None, up=1):
    """Fallback UVs for a garment with NO sewing pattern, by unwrapping it as a body of revolution.

    `longskirt` carries no `mesh_pos` and no same-topology donor exists -- the tshirt's
    face-order weld route has nothing to weld from. But a skirt IS very nearly a surface of
    revolution, so an angle-by-height unwrap of the REST pose is a decent metric map, and metric is
    all a tiled fabric needs.

    u = theta * r_hoop, v = height, both in METRES, so the result drops straight into uv_tiled and
    the "one tile = N cm of cloth" statement stays true.

    MEASURED distortion on longskirt's rest pose (uv edge length / 3d edge length, seam edges
    excluded): median 0.981, p05 0.851, p95 1.320, mean |log| 0.081. Two supposedly-better maps were
    tried and are WORSE, because the flare's cross-term beats what they fix: using each vertex's own
    radius for the hoop (mean |log| 0.137) and using profile slant length for v (0.112). Do not
    "improve" this without re-measuring -- `verify_cylinder` prints the number.

    r_hoop defaults to the median radius, which is also its optimum (a bounded 1-D fit lands on
    0.2509 m against a median of 0.2494). Given `tile_w` it is then nudged by <=1/2 tile so the
    circumference is a WHOLE number of tiles and the wrap seam matches instead of showing a
    mismatched stripe down the skirt.

    :return: (v2_to_v3, faces, uv_m), the same contract as uv_map.
    """
    D = _load_dict()
    gd = D[garment]
    if 'rest_pos' not in gd:
        raise KeyError(f'{garment!r} carries neither mesh_pos nor rest_pos; no UV route')
    R = np.asarray(gd['rest_pos']).astype(np.float64)
    f3 = np.asarray(gd['faces']).astype(np.int64)
    if cloth_faces is not None and not np.array_equal(np.asarray(cloth_faces), f3):
        raise ValueError(f'{garment!r}: rollout faces differ from the garment dict')

    ax = [i for i in range(3) if i != up]
    xy = R[:, ax] - R[:, ax].mean(axis=0)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    r_hoop = float(np.median(np.linalg.norm(xy, axis=1)))
    if tile_w:
        n = max(1, round(2 * np.pi * r_hoop / tile_w))
        r_hoop = n * tile_w / (2 * np.pi)

    # SEAM SPLIT. A face straddling theta = +-pi would otherwise run the texture backwards across
    # the entire tile. Duplicate its negative-theta corners at theta + 2pi; ~110 faces of 15231.
    tf = theta[f3]
    wrap = (tf.max(axis=1) - tf.min(axis=1)) > np.pi
    theta2, v2_to_v3 = list(theta), list(range(len(R)))
    faces = f3.copy()
    dup = {}
    for fi in np.nonzero(wrap)[0]:
        for ci in range(3):
            vi = f3[fi, ci]
            if theta[vi] < 0:
                if vi not in dup:
                    dup[vi] = len(theta2)
                    theta2.append(theta[vi] + 2 * np.pi)
                    v2_to_v3.append(vi)
                faces[fi, ci] = dup[vi]

    v2_to_v3 = np.asarray(v2_to_v3, np.int64)
    uv = np.stack([np.asarray(theta2) * r_hoop, R[v2_to_v3, up]], axis=1)
    return v2_to_v3, faces, uv


def verify_cylinder(garment, up=1):
    """Edge-length distortion of uv_cylinder against the rest pose. Seam edges excluded."""
    D = _load_dict()
    R = np.asarray(D[garment]['rest_pos']).astype(np.float64)
    v2_to_v3, faces, uv = uv_cylinder(garment, up=up)
    e = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                          faces[:, [2, 0]]]), axis=1), axis=0)
    L3 = np.linalg.norm(R[v2_to_v3[e[:, 0]]] - R[v2_to_v3[e[:, 1]]], axis=1)
    L2 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1)
    ratio = L2 / np.maximum(L3, 1e-9)
    return dict(median=float(np.median(ratio)), p05=float(np.percentile(ratio, 5)),
                p95=float(np.percentile(ratio, 95)),
                mean_abs_log=float(np.abs(np.log(np.clip(ratio, 1e-6, None))).mean()),
                n_verts=len(uv), n_dup=len(uv) - len(R))


def tile_dims(texture, texture_dir=None):
    """Real-world extent of one tile, metres. Falls back to the procedural registry."""
    if texture in TEX_DIMS_M:
        return TEX_DIMS_M[texture]
    if texture_dir is not None:
        import json
        reg = Path(texture_dir) / 'procedural.json'
        if reg.exists():
            d = json.loads(reg.read_text())
            if texture in d:
                return tuple(d[texture])
    raise KeyError(f'no tile size known for {texture!r}; generate it with '
                   f'a texture generator, or add it to TEX_DIMS_M')


def uv_tiled(uv_m, texture, scale=1.0, texture_dir=None):
    """Pattern metres -> texture units, so one tile covers the scan's real-world extent.

    scale > 1 shrinks the print (more repeats per metre). moderngl textures default to REPEAT wrap,
    so UVs outside [0,1] tile correctly and nothing needs pre-tiling.
    """
    w, h = tile_dims(texture, texture_dir)
    out = uv_m.copy()
    out[:, 0] = (uv_m[:, 0] - uv_m[:, 0].min()) / (w / scale)
    out[:, 1] = (uv_m[:, 1] - uv_m[:, 1].min()) / (h / scale)
    return out.astype(np.float32)


def uv_border(uv_m, tile_w, flip_v=False):
    """Pattern metres -> texture units for an ENGINEERED (border) print.

    Horizontal: tiles, exactly as uv_tiled, so the print repeats around the body.
    Vertical:   maps the garment's whole v range onto [0,1] ONCE, so the texture's own top-to-bottom
                gradient becomes waist-to-hem. This is what makes "sparse at the waist, peaked at
                the hem" a property of the garment rather than of the swatch.

    `flip_v` because whether uv.y = 0 is the image's top or bottom row depends on the loader; pick
    the one that puts the dense end at the hem and keep it.
    """
    out = uv_m.copy().astype(np.float64)
    out[:, 0] = (uv_m[:, 0] - uv_m[:, 0].min()) / tile_w
    v0, v1 = uv_m[:, 1].min(), uv_m[:, 1].max()
    t = (uv_m[:, 1] - v0) / max(v1 - v0, 1e-9)        # 0 at the hem, 1 at the waist
    out[:, 1] = t if flip_v else 1.0 - t
    return out.astype(np.float32)


def tint(src_png, rgb, out_png, contrast=0.85, gamma=0.8):
    """Bake `rgb` into the fabric's luminance detail. Returns out_png.

    rgb=None writes the photograph through unchanged (the fabric's native colour).
    """
    from PIL import Image
    img = np.asarray(Image.open(src_png).convert('RGB')).astype(np.float32) / 255.0
    if rgb is None:
        Image.fromarray((img * 255).astype(np.uint8)).save(out_png)
        return out_png
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    d = (lum / max(lum.mean(), 1e-6)) ** gamma           # mean-normalised, tone-compressed
    s = 1.0 + contrast * (d - 1.0)                       # contrast knob, keeps the mean on palette
    out = np.clip(np.asarray(rgb, np.float32)[None, None, :] * s[..., None], 0, 1)
    Image.fromarray((out * 255).astype(np.uint8)).save(out_png)
    return out_png


def tint_mask(src_png, base_rgb, out_png, ink_rgb=(1.0, 1.0, 1.0), lo=25.0, hi=99.5):
    """Composite an ENGINEERED print as INK ON CLOTH, rather than as a shade of the cloth.

    `tint` multiplies the garment colour by the swatch's luminance, which is right for a woven
    pattern (the colour is the yarn) but wrong for a lace border: at GROUND 0.62 / MARK 0.98 it can
    only reach base x ~1.3, so white lace comes out as pale cloth. Here the swatch is read as an
    alpha mask instead, so the ornament lands on `ink_rgb` whatever the base is.

    `lo`/`hi` are percentiles, not min/max: the 1 mm weave overlay puts outliers at both ends and a
    raw min/max would let a single noisy pixel set the whole mapping.
    """
    from PIL import Image
    img = np.asarray(Image.open(src_png).convert('RGB')).astype(np.float32) / 255.0
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    g, m = np.percentile(lum, lo), np.percentile(lum, hi)
    a = np.clip((lum - g) / max(m - g, 1e-6), 0.0, 1.0)[..., None]
    base = np.asarray(base_rgb, np.float32)[None, None, :]
    ink = np.asarray(ink_rgb, np.float32)[None, None, :]
    # keep the weave breathing on the cloth ground, or the fabric reads as flat vector art
    shade = np.clip(lum / max(g, 1e-6), 0.94, 1.06)[..., None]
    out = np.clip((base * (1 - a) + ink * a) * (1 - a * 0.7 + shade * a * 0.7), 0, 1)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((out * 255).astype(np.uint8)).save(out_png)
    return out_png


def verify(garment, cloth_faces, rest_pos=None):
    """Print the invariants this module relies on, for the garment at hand."""
    v2v3, mf, uv = uv_map(garment, cloth_faces)
    msg = [f'{garment:11s} V2={len(uv):6d} F={len(mf):6d} span='
           f'{uv[:, 0].ptp():.2f}x{uv[:, 1].ptp():.2f} m']
    if rest_pos is not None:
        # a pattern is developable at rest: 2D and 3D edge lengths should agree to within a few %
        e = np.concatenate([mf[:, [0, 1]], mf[:, [1, 2]], mf[:, [2, 0]]])
        l2 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1)
        p3 = np.asarray(rest_pos)[v2v3]
        l3 = np.linalg.norm(p3[e[:, 0]] - p3[e[:, 1]], axis=1)
        ok = l2 > 1e-9
        msg.append(f'2D/3D edge-length ratio median {np.median(l3[ok] / l2[ok]):.3f}')
    print('  '.join(msg))
