"""Standalone Blender script: extract rest-pose mesh from an FBX.

Invoked as:
    blender --background --python _blender_fbx_export.py -- <in.fbx> <out.npz>

Writes an NPZ with: verts (Nm, 3), faces (Fm, 3), uv (Nm, 2)|absent.

Coordinate convention: we pass `axis_forward='-Z'` and `axis_up='Y'` to the FBX
importer so the resulting mesh is in a right-handed Y-up frame matching SMPL-X.

Mixamo characters are rigged. We rely on the armature being in rest position
(no animation evaluated) so the mesh vertices come out in T/A-pose. We
explicitly set every armature to REST so any default animation pose can't
contaminate the bind pose we extract.
"""

import sys
import os

import bpy
import numpy as np

# Make the sibling mapping module importable. Blender's `--python` doesn't
# add the script's directory to sys.path automatically.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixamo_bone_map import MIXAMO_TO_SMPLX, strip_prefix, NUM_SMPLX_JOINTS  # noqa: E402


def _parse_args():
    # bpy strips its own args; positional after '--' are ours.
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected '-- <in.fbx> <out.npz>' on the command line.")
    after = argv[argv.index("--") + 1:]
    if len(after) != 2:
        raise SystemExit(f"Expected 2 positional args after '--', got {after!r}")
    return after[0], after[1]


def _clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_fbx(in_path):
    bpy.ops.import_scene.fbx(
        filepath=in_path,
        axis_forward="-Z",
        axis_up="Y",
        use_anim=False,
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
    )


def _force_rest_pose():
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    # Make sure depsgraph reflects the rest pose for the mesh evaluation below.
    bpy.context.view_layer.update()


def _collect_all_meshes():
    """Return every non-empty mesh in the scene. Mixamo characters typically
    ship as several meshes (body skin, hair, hoodie, pants, eyelashes, ...);
    we want ALL of them combined for the chamfer fit + rendering, otherwise
    only ~30% of the character's geometry is fit and the rendered result is
    missing clothes/hair."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    out = []
    for o in meshes:
        eo = o.evaluated_get(depsgraph)
        me = eo.to_mesh()
        if len(me.vertices) > 0:
            out.append(o)
        eo.to_mesh_clear()
    if not out:
        raise SystemExit("No non-empty meshes in FBX.")
    return out


def _pick_body_submesh_idx(meshes) -> int:
    """Pick the index of the BODY submesh (the humanoid skin/surface) for
    the chamfer fit. Other submeshes (hair, hoodie, pants, shoes) ride along
    via the same scale/R/t transform but are NOT used for the chamfer
    optimization -- including them blows up beta because they don't match
    SMPL-X's mean human silhouette.

    Heuristic: prefer names containing 'body' / 'surface' / 'skin'. Falls
    back to the mesh with the largest 3D bbox volume (which is usually the
    body for Mixamo's standard exports).
    """
    keywords = ["body", "surface", "skin"]
    for kw in keywords:
        for i, m in enumerate(meshes):
            if kw in m.name.lower():
                return i
    # Fallback: largest bbox volume.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    best_vol = -1.0
    best_idx = 0
    for i, m in enumerate(meshes):
        eo = m.evaluated_get(depsgraph)
        me = eo.to_mesh()
        coords = np.array([v.co for v in me.vertices], dtype=np.float32)
        if coords.shape[0]:
            extents = coords.max(0) - coords.min(0)
            vol = float(extents[0] * extents[1] * extents[2])
            if vol > best_vol:
                best_vol = vol; best_idx = i
        eo.to_mesh_clear()
    return best_idx


def _find_armature(mesh_obj):
    """Return the armature driving ``mesh_obj``'s skinning (or None)."""
    for mod in mesh_obj.modifiers:
        if mod.type == "ARMATURE" and mod.object is not None:
            return mod.object
    if mesh_obj.parent is not None and mesh_obj.parent.type == "ARMATURE":
        return mesh_obj.parent
    return None


def _extract_texture_bytes(mesh_obj):
    """Find the diffuse texture image bound to ``mesh_obj``'s first material
    and return its PNG-encoded bytes, or None if no texture is found.

    Mixamo FBXes usually pack one diffuse image per material via a
    ShaderNodeTexImage. For multi-material meshes we just take the first one
    with a valid image.
    """
    import tempfile

    materials = mesh_obj.data.materials
    if materials is None or len(materials) == 0:
        return None, None

    for mat in materials:
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE":
                continue
            img = node.image
            if img is None:
                continue
            # Two cases: image is packed (data in memory) or referenced via a
            # filepath. Prefer the packed bytes when available; fall back to
            # rendering pixels to a tempfile.
            if getattr(img, "packed_file", None) is not None and img.packed_file.size > 0:
                # Save to a tempfile then read bytes back, so the PNG header
                # is always correct regardless of original format (Mixamo
                # sometimes uses .tga/.psd).
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                old_filepath = img.filepath
                old_format = img.file_format
                img.file_format = "PNG"
                img.save_render(tmp_path)
                with open(tmp_path, "rb") as f:
                    data = f.read()
                # Restore original state in Blender (not strictly needed since
                # the scene is thrown away, but good practice).
                img.filepath = old_filepath
                img.file_format = old_format
                os.unlink(tmp_path)
                return np.frombuffer(data, dtype=np.uint8), mat.name
            # Pixels in memory but not packed -- save via Blender's render API.
            if img.has_data:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                img.file_format = "PNG"
                img.save_render(tmp_path)
                with open(tmp_path, "rb") as f:
                    data = f.read()
                os.unlink(tmp_path)
                return np.frombuffer(data, dtype=np.uint8), mat.name
    return None, None


def _extract_bone_rest_positions(mesh_obj):
    """Return (bone_names, bone_head_positions, smplx_joint_index_per_bone)
    for each bone in the armature driving ``mesh_obj``, with head positions
    in world space (Blender Z-up, pre-conversion to SMPL-X Y-up).

    For bones whose name doesn't map directly to a SMPL-X joint, the parent
    chain is walked (same logic as the vertex-group mapping).
    """
    arm = _find_armature(mesh_obj)
    if arm is None:
        return None, None, None
    world = np.array(arm.matrix_world)
    names: list[str] = []
    positions: list[np.ndarray] = []
    smplx_idx: list[int] = []
    for bone in arm.data.bones:
        head_local = np.array([bone.head_local.x, bone.head_local.y, bone.head_local.z, 1.0],
                              dtype=np.float32)
        head_world = world @ head_local
        names.append(bone.name)
        positions.append(head_world[:3].astype(np.float32))
        # Resolve SMPL-X joint via name + parent chain.
        j = MIXAMO_TO_SMPLX.get(strip_prefix(bone.name))
        if j is None:
            j = _bone_to_smplx_joint(bone, arm)
        smplx_idx.append(j)
    return (np.array(names),
            np.stack(positions, axis=0),
            np.array(smplx_idx, dtype=np.int32))


def _bone_to_smplx_joint(bone, armature):
    """Walk up the bone parent chain until we find one mapped in
    MIXAMO_TO_SMPLX. Returns the SMPL-X joint index, or 0 (pelvis) as a
    last-resort fallback so no weight is silently dropped."""
    cur = bone
    while cur is not None:
        j = MIXAMO_TO_SMPLX.get(strip_prefix(cur.name))
        if j is not None:
            return j
        cur = cur.parent
    return 0  # fallback: pelvis


def _extract_native_skinning(mesh_obj, n_verts):
    """Extract per-vertex SMPL-X LBS weights using the FBX's own skinning.

    Returns an (n_verts, NUM_SMPLX_JOINTS) float32 array with rows summing
    to ~1, or None if the mesh has no armature / vertex groups.
    """
    arm = _find_armature(mesh_obj)
    if arm is None or len(mesh_obj.vertex_groups) == 0:
        return None, {}

    # Map each vertex_group index to a SMPL-X joint via the bone's name
    # (walking parents for unmapped tips). Note: the vertex_group name and
    # the bone name are the same string for FBX skin clusters.
    vg_to_smplx: dict[int, int] = {}
    mapping_report: dict[str, str] = {}
    for vg in mesh_obj.vertex_groups:
        name = vg.name
        # Try direct mapping first.
        j = MIXAMO_TO_SMPLX.get(strip_prefix(name))
        if j is None:
            # Walk bone parents in the armature.
            bone = arm.data.bones.get(name)
            if bone is None:
                # Couldn't find the bone -- fall back to pelvis.
                j = 0
                mapping_report[name] = "(no bone) -> pelvis"
            else:
                j = _bone_to_smplx_joint(bone, arm)
                resolved = arm.data.bones[name].name
                if MIXAMO_TO_SMPLX.get(strip_prefix(resolved)) is None:
                    mapping_report[name] = f"-> via parent chain -> joint {j}"
                else:
                    mapping_report[name] = f"-> joint {j}"
        else:
            mapping_report[name] = f"-> joint {j}"
        vg_to_smplx[vg.index] = j

    weights = np.zeros((n_verts, NUM_SMPLX_JOINTS), dtype=np.float32)
    for vi, v in enumerate(mesh_obj.data.vertices):
        for ge in v.groups:
            j = vg_to_smplx.get(ge.group)
            if j is None:
                continue
            weights[vi, j] += ge.weight

    # Renormalize so each vertex row sums to 1 (Mixamo rigs typically already
    # sum to 1 per vertex, but parented-chain remappings can shift this).
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    weights = weights / row_sums

    return weights, mapping_report


def _saturation_boost_srgb(srgb_u8: np.ndarray,
                            target_sat: float = 0.55,
                            boost: float = 1.0,
                            min_orig_sat: float = 0.03) -> np.ndarray:
    """Boost saturation of uint8 sRGB to a TARGET saturation level.

    Mixamo's non-PBR textures bake the highlights into the diffuse, so
    even visibly-colored fabrics (skin, hair) have low per-channel range
    (S = ~0.2). A simple multiplicative boost (S *= 1.6) leaves them at
    ~0.3 which is still washed-out under typical lighting.

    Instead, we map current S -> max(boost*S, target_sat). Pure grays
    (S < min_orig_sat) are skipped so true-gray clothing stays gray.
    """
    rgba = srgb_u8.astype(np.float32) / 255.0
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    s = np.where(mx > 1e-8, delta / np.maximum(mx, 1e-8), 0.0)

    # New saturation: target if original has any hue, else unchanged.
    s_new = np.where(s > min_orig_sat, np.maximum(s * boost, target_sat), s)
    s_new = np.clip(s_new, 0.0, 1.0)

    # HSV-back: keep mx, push mn down to mx*(1-s_new), interpolate channels.
    mn_new = mx * (1.0 - s_new)
    scale = np.where(delta > 1e-8, (mx - mn_new) / np.maximum(delta, 1e-8), 1.0)
    r_new = mx - (mx - r) * scale
    g_new = mx - (mx - g) * scale
    b_new = mx - (mx - b) * scale
    out = np.stack([r_new, g_new, b_new], axis=-1)
    out = np.clip(out, 0.0, 1.0)
    full = np.concatenate([out, rgba[..., 3:4]], axis=-1)
    return (full * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def _linear_to_srgb_u8(linear_rgba_f32: np.ndarray) -> np.ndarray:
    """Convert linear-space float RGBA [0,1] to sRGB-space uint8.

    Blender stores Base Color and image.pixels in linear color space; the
    demo viewer (Viser via trimesh.visual.vertex_colors) displays as sRGB,
    so without this conversion the colors look unnaturally dark/saturated.
    Uses the standard piecewise sRGB transfer function.
    """
    x = np.clip(linear_rgba_f32, 0.0, 1.0)
    # Color channels (RGB) only; alpha is linear pass-through.
    rgb = x[..., :3]
    cutoff = rgb <= 0.0031308
    low = 12.92 * rgb
    hi = 1.055 * np.power(np.maximum(rgb, 1e-12), 1.0 / 2.4) - 0.055
    rgb_srgb = np.where(cutoff, low, hi)
    out = x.copy()
    out[..., :3] = rgb_srgb
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def _save_image_to_disk(img, out_path):
    """Save a Blender image's pixel array as raw .npy for offline inspection."""
    import numpy as _np
    pixels = _np.asarray(img.pixels[:], dtype=_np.float32)
    h, w, c = img.size[1], img.size[0], img.channels
    arr = pixels.reshape(h, w, c)
    _np.save(out_path, arr)
    print(f"[saved_image] {img.name} -> {out_path} ({h}x{w}x{c})")


def _dump_all_material_images(mesh_obj, prefix="[mat_images]"):
    """List every TEX_IMAGE node bound to this mesh's materials with the
    image name, dimensions, channels, colorspace, and a small sample of
    the median color. Use this to diagnose why a 'diffuse' image looks
    wrong (e.g. the FBX importer linked the normal map to Base Color)."""
    import numpy as _np
    mesh_name = getattr(mesh_obj, "name", "<mesh>")
    materials = mesh_obj.data.materials
    if materials is None or len(materials) == 0:
        return
    seen = set()
    for mat in materials:
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            img = node.image
            if img.name in seen:
                continue
            seen.add(img.name)
            # Is it linked to BSDF Base Color?
            linked_to = []
            for link in mat.node_tree.links:
                if link.from_node is node:
                    linked_to.append(f"{link.to_node.type}.{link.to_socket.name}")
            try:
                pixels = _np.asarray(img.pixels[:], dtype=_np.float32)
                w, h, c = img.size[0], img.size[1], img.channels
                if pixels.size == w * h * c:
                    rgb = pixels.reshape(h, w, c)[:, :, :3]
                    bright_mask = rgb.mean(axis=-1) > 0.05
                    med = (_np.median(rgb[bright_mask], axis=0)
                           if bright_mask.any() else _np.zeros(3))
                else:
                    med = _np.zeros(3)
            except Exception:
                med = _np.zeros(3)
            cs = img.colorspace_settings.name
            print(f"{prefix} {mesh_name} -> img='{img.name}' "
                  f"({img.size[0]}x{img.size[1]} {img.channels}ch {cs}) "
                  f"med_linear_rgb={med.round(3)} linked={linked_to}")


def _find_diffuse_image(mesh_obj):
    """Find the DIFFUSE texture bound to the mesh's materials.

    Mixamo "non-PBR" materials typically have 4 image nodes (Diffuse +
    Normal + Specular + Gloss). The naive "first TEX_IMAGE" pick often
    grabs the Normal map (grayscale/Non-Color) producing washed-out
    results. Filter by:
        1. Image is linked to a Principled BSDF Base Color input, OR
        2. Image's color space is sRGB (Non-Color = data textures only).
    """
    materials = mesh_obj.data.materials
    if materials is None or len(materials) == 0:
        return None

    def _is_usable(img):
        return img is not None and (
            img.has_data
            or (getattr(img, "packed_file", None) is not None
                and img.packed_file.size > 0)
        )

    mesh_name = getattr(mesh_obj, "name", "<mesh>")
    # Pass 1: image directly linked to Principled BSDF Base Color.
    for mat in materials:
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            base_in = node.inputs.get("Base Color")
            if base_in is None or not base_in.is_linked:
                continue
            src = base_in.links[0].from_node
            if src.type == "TEX_IMAGE" and _is_usable(src.image):
                print(f"[diffuse] {mesh_name} -> '{src.image.name}' (BSDF Base Color)")
                return src.image

    # Pass 2: scoring -- pick the image whose NAME most likely indicates a
    # diffuse / albedo / color map and least likely a normal / specular /
    # gloss / metallic map. Mixamo non-PBR materials often have all 4 maps
    # tagged sRGB by FBX import, so colorspace alone doesn't disambiguate.
    diffuse_keywords = ("diffuse", "albedo", "basecolor", "base_color", "color")
    skip_keywords = ("normal", "norm", "specular", "spec", "gloss", "roughness",
                     "metal", "metallic", "ao", "ambient", "occlusion",
                     "height", "displace", "bump")
    candidates = []
    for mat in materials:
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not _is_usable(node.image):
                continue
            img = node.image
            name_l = img.name.lower()
            score = 0
            if any(k in name_l for k in diffuse_keywords):
                score += 10
            if any(k in name_l for k in skip_keywords):
                score -= 10
            if img.colorspace_settings.name == "sRGB":
                score += 1
            candidates.append((score, img))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        best = candidates[0][1]
        print(f"[diffuse] {mesh_name} -> '{best.name}' (score-based; "
              f"all candidates: {[(s, i.name) for s, i in candidates]})")
        return best

    # Pass 3: last resort, any image at all.
    for mat in materials:
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and _is_usable(node.image):
                print(f"[diffuse] {mesh_name} -> '{node.image.name}' (fallback any image)")
                return node.image
    return None


def _representative_submesh_color(mesh_obj, uv_per_vert: np.ndarray | None) -> np.ndarray | None:
    """Compute ONE representative sRGB-uint8 color for the whole submesh.

    Sampling the texture per-vertex gives noisy results (UV seams, low-res
    texture detail, sampling padding pixels). For the demo it looks much
    cleaner to assign a single uniform color per submesh -- analogous to
    Y-Bot's flat-shaded BSDF base color, but for textured characters we
    derive that color from the texture itself (median of the bright pixels).

    Falls back to material BSDF base color, then default gray.
    """
    img = _find_diffuse_image(mesh_obj)
    if img is not None and uv_per_vert is not None:
        width, height = img.size
        channels = img.channels
        if width > 0 and height > 0:
            pixels = np.asarray(img.pixels[:], dtype=np.float32).reshape(height, width, channels)
            # Use the same V-flip convention as the historical representative
            # color baker (which gave correct sweater/pants/shoes colors on
            # the Mixamo characters). This is the OPPOSITE of per-vertex
            # baking (which removes the flip to make Body's skin sample
            # correctly) -- the Mixamo FBX UV conventions differ between
            # the Body submesh and the cloth/accessory submeshes.
            u = uv_per_vert[:, 0]
            v = 1.0 - uv_per_vert[:, 1]
            x = np.clip(np.round(u * (width - 1)).astype(np.int32), 0, width - 1)
            y = np.clip(np.round(v * (height - 1)).astype(np.int32), 0, height - 1)
            cols = pixels[y, x]  # (Nv, channels) linear
            rgb = cols[:, :3]
            # Filter out shadowed / UV-padding samples that pull the chosen
            # color toward gray. Threshold ~ linear 0.05 ≈ sRGB 65.
            brightness = rgb.mean(axis=1)
            bright = rgb[brightness > 0.05]
            if bright.shape[0] > 0:
                main_lin = np.percentile(bright, 75, axis=0)
                rgba_lin = np.concatenate([main_lin, [1.0]]).astype(np.float32)
                srgb = _linear_to_srgb_u8(rgba_lin[None, :])[0]
                # Bump saturation for visibility. Mixamo "non-PBR" textures
                # are intentionally low-saturation for PBR re-shading; we
                # bring it back so per-vertex-flat-shaded rendering shows
                # the character more vividly.
                return _saturation_boost_srgb(srgb[None, :])[0]
    # Fall back to material base color.
    solid = _solid_color_for_material(mesh_obj)
    if solid is not None:
        return _saturation_boost_srgb(solid[None, :])[0]
    return np.array([200, 200, 200, 255], dtype=np.uint8)


def _bake_per_vertex_rgba(mesh_obj, uv_per_vert: np.ndarray | None) -> np.ndarray | None:
    """Sample the mesh's diffuse texture at each vertex's UV. Returns
    (n_verts, 4) uint8 RGBA, or None if no texture / UVs are available.

    Verts get the FIRST UV encountered for them (boundary verts may have
    multiple loops with different UVs; this matches the existing per-vertex
    UV convention)."""
    img = _find_diffuse_image(mesh_obj)
    if img is None or uv_per_vert is None:
        return None
    n_verts = uv_per_vert.shape[0]

    # Read pixels into numpy. img.pixels is a flat list of floats in [0, 1].
    width, height = img.size
    channels = img.channels  # usually 4 (RGBA) or 3 (RGB)
    if width == 0 or height == 0:
        return None
    pixels = np.asarray(img.pixels[:], dtype=np.float32).reshape(height, width, channels)

    # Blender stores pixels bottom-up: pixels[0] = bottom row of image,
    # pixels[h-1] = top row. OpenGL UVs also have origin at the bottom-left
    # (V increases upward). So `y_storage = v * (h - 1)` directly, NO flip.
    # The previous `v = 1.0 - uv_v` was an extra flip that sent face UVs
    # (V ~ 0.85, near top of image) to the bottom of pixel storage (jeans
    # area on Mixamo textures).
    u = uv_per_vert[:, 0]
    v = uv_per_vert[:, 1]
    x = np.clip(np.round(u * (width - 1)).astype(np.int32), 0, width - 1)
    y = np.clip(np.round(v * (height - 1)).astype(np.int32), 0, height - 1)
    cols = pixels[y, x]                          # (N, channels)
    # Note: for an sRGB-tagged image (color textures), `img.pixels[]`
    # returns values ALREADY in sRGB space. For Non-Color images they
    # return linear values. We only call this for diffuse / Base Color
    # textures, which are sRGB, so the values are already display-ready
    # and no linear-to-sRGB conversion should be applied.
    is_srgb = (img.colorspace_settings.name == "sRGB")
    if cols.shape[-1] == 3:
        alpha = np.ones((n_verts, 1), dtype=np.float32)
        cols = np.concatenate([cols, alpha], axis=1)

    # Mixamo textures often have black UV padding around the laid-out island,
    # and some verts land on or just outside the texture boundary -- they end
    # up sampling near-black pixels, producing visible dark streaks across
    # clothing meshes. Replace very-dark sampled verts with the per-submesh
    # MEDIAN color of the bright-textured pixels so the streaks disappear.
    rgb_linear = cols[:, :3]
    brightness = rgb_linear.mean(axis=1)
    bright_mask = brightness > 0.02     # threshold in linear space ~= sRGB ~30
    if bright_mask.any():
        median_rgb = np.median(rgb_linear[bright_mask], axis=0)
        cols[~bright_mask, :3] = median_rgb

    # For sRGB-tagged textures, img.pixels[] is already in sRGB display
    # space (NOT linear) -- just clip + convert to uint8 directly. For
    # Non-Color images (rare here -- we already filter to diffuse), the
    # values are linear and need gamma encoding.
    if is_srgb:
        return (np.clip(cols, 0.0, 1.0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    return _linear_to_srgb_u8(cols)


def _solid_color_for_material(mesh_obj) -> np.ndarray | None:
    """Fallback: return the principled BSDF base color of the first material."""
    materials = mesh_obj.data.materials
    if materials is None or len(materials) == 0:
        return None
    for mat in materials:
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                c = node.inputs["Base Color"].default_value
                # default_value is linear; convert to sRGB for display parity.
                rgba_lin = np.array([c[0], c[1], c[2], c[3]], dtype=np.float32)
                return _linear_to_srgb_u8(rgba_lin[None, :])[0]
    return None


def _extract_mesh(mesh_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()

    # Apply object world transform so the resulting verts share a frame.
    world = np.array(eval_obj.matrix_world)
    n_verts = len(me.vertices)
    verts = np.zeros((n_verts, 3), dtype=np.float32)
    for i, v in enumerate(me.vertices):
        verts[i] = v.co
    verts_h = np.concatenate([verts, np.ones((n_verts, 1), dtype=np.float32)], axis=1)
    verts = (verts_h @ world.T)[:, :3].astype(np.float32)

    # Triangulate via loop_triangles to be robust to quads / n-gons.
    me.calc_loop_triangles()
    n_tri = len(me.loop_triangles)
    faces = np.zeros((n_tri, 3), dtype=np.int32)
    for i, lt in enumerate(me.loop_triangles):
        faces[i] = lt.vertices

    # UVs: pick the first layer if present. We collapse per-loop UVs to a
    # per-vertex UV by taking the FIRST loop UV encountered per vertex —
    # boundary seams may end up with arbitrary picks; downstream rendering
    # accepts that for a baked diffuse.
    uv = None
    if me.uv_layers:
        # List all UV layers for diagnostics.
        print(f"[uv_layers] {mesh_obj.name}: layers="
              f"{[(L.name, L.active, L.active_render) for L in me.uv_layers]}")
        # Pick the layer marked 'active for render' if any; this is the one
        # materials sample by default. Older Blender versions don't expose
        # active_render on the collection -- iterate layers manually.
        uv_layer = me.uv_layers.active.data
        for _L in me.uv_layers:
            if getattr(_L, "active_render", False):
                uv_layer = _L.data
                break
        uv = np.zeros((n_verts, 2), dtype=np.float32)
        seen = np.zeros(n_verts, dtype=bool)
        for poly in me.polygons:
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index
                if seen[vi]:
                    continue
                uv[vi] = uv_layer[li].uv
                seen[vi] = True

    eval_obj.to_mesh_clear()

    # Extract native skinning from the ORIGINAL (non-evaluated) mesh -- the
    # vertex groups + per-vert weights live on the source mesh, not the
    # modifier-evaluated copy. Verts are 1:1 in count when no destructive
    # modifiers (decimate, subdivide) are active, which is the Mixamo case.
    native_w, mapping_report = _extract_native_skinning(mesh_obj, n_verts)
    bone_names, bone_heads, bone_smplx_j = _extract_bone_rest_positions(mesh_obj)
    texture_bytes, texture_material = _extract_texture_bytes(mesh_obj)
    return (verts, faces, uv, native_w, mapping_report,
            bone_names, bone_heads, bone_smplx_j,
            texture_bytes, texture_material)


def main():
    in_path, out_path = _parse_args()
    if not os.path.isfile(in_path):
        raise SystemExit(f"Input FBX not found: {in_path}")

    _clear_scene()
    _import_fbx(in_path)
    _force_rest_pose()
    meshes = _collect_all_meshes()
    body_idx = _pick_body_submesh_idx(meshes)
    print(f"[fbx_export] found {len(meshes)} non-empty mesh(es) -- combining all "
          f"(body submesh for chamfer = '{meshes[body_idx].name}'):")

    all_verts, all_faces, all_uv, all_colors, all_w = [], [], [], [], []
    submesh_names, submesh_ranges = [], []   # (name, [vstart, vend))
    bone_names = bone_heads = bone_smplx_j = None
    aggregated_mapping: dict[str, str] = {}
    vert_offset = 0
    any_uv = False
    any_color = False

    for mi, mesh_obj in enumerate(meshes):
        (verts, faces, uv, native_w, mapping_report,
         bn, bh, bsj, _tex_bytes, _tex_mat) = _extract_mesh(mesh_obj)
        n_v = verts.shape[0]
        # Per-vertex texture sampling for ALL submeshes -- gives the actual
        # FBX texture colors at each vertex's UV (matches what the FBX
        # renders in Blender's Eevee/Cycles). The historical
        # _representative_submesh_color (75th percentile of bright pixels
        # with V-flip) gave artificially-bright tones that don't match
        # the FBX -- e.g. a "peach sweater" on Ch31 which is actually
        # dark navy in the source file.
        per_vert_rgba = _bake_per_vertex_rgba(mesh_obj, uv)
        if per_vert_rgba is None:
            rep_rgba = _representative_submesh_color(mesh_obj, uv)
            per_vert_rgba = np.tile(rep_rgba, (n_v, 1))
            print(f"[color] {mesh_obj.name}: representative fallback={tuple(rep_rgba[:3])}")
        else:
            # No saturation boost -- the user wants colors to match the
            # FBX as rendered (which uses the raw texture). The previous
            # boost made dark navy clothing look like bright blue.
            med = np.median(per_vert_rgba[:, :3], axis=0).astype(int)
            print(f"[color] {mesh_obj.name}: per-vertex bake median={tuple(med)}")
        any_color = True

        all_verts.append(verts)
        all_faces.append((faces + vert_offset).astype(np.int32))
        all_w.append(native_w if native_w is not None
                     else np.zeros((n_v, NUM_SMPLX_JOINTS), dtype=np.float32))
        all_colors.append(per_vert_rgba)
        if uv is not None:
            all_uv.append(uv); any_uv = True
        else:
            all_uv.append(np.zeros((n_v, 2), dtype=np.float32))

        submesh_names.append(mesh_obj.name)
        submesh_ranges.append((vert_offset, vert_offset + n_v))
        print(f"    [{mi}] {mesh_obj.name:30s} V={n_v:6d} F={faces.shape[0]:6d}  "
              f"texture={'yes' if any_color else 'no'}  weights={'yes' if native_w is not None else 'no'}")
        vert_offset += n_v
        # Bones (same armature across all meshes) -- extract once.
        if bone_names is None and bh is not None:
            bone_names, bone_heads, bone_smplx_j = bn, bh, bsj
        aggregated_mapping.update(mapping_report)

    verts_all  = np.concatenate(all_verts,  axis=0).astype(np.float32)
    faces_all  = np.concatenate(all_faces,  axis=0).astype(np.int32)
    weights_all = np.concatenate(all_w,     axis=0).astype(np.float32)
    colors_all = np.concatenate(all_colors, axis=0).astype(np.uint8)
    uv_all     = np.concatenate(all_uv,     axis=0).astype(np.float32) if any_uv else None

    save_kwargs = {
        "verts": verts_all, "faces": faces_all,
        "mesh_name": "combined",
        "submesh_names": np.array(submesh_names),
        "submesh_ranges": np.array(submesh_ranges, dtype=np.int32),
        "body_submesh_idx": np.int32(body_idx),
        # Convenience: explicit (vstart, vend) of the body submesh range.
        "body_vert_range": np.array(submesh_ranges[body_idx], dtype=np.int32),
    }
    if uv_all is not None:
        save_kwargs["uv"] = uv_all
    if any_color or True:  # always save colors (fallback gray for untextured submeshes)
        save_kwargs["vertex_colors"] = colors_all
    save_kwargs["native_lbs_weights"] = weights_all
    if bone_heads is not None:
        save_kwargs["bone_names"] = bone_names
        save_kwargs["bone_heads"] = bone_heads
        save_kwargs["bone_smplx_joint_idx"] = bone_smplx_j
    np.savez(out_path, **save_kwargs)
    print(f"[fbx_export] wrote {out_path}: total V={verts_all.shape[0]} F={faces_all.shape[0]}  "
          f"submeshes={len(meshes)}  uv={'yes' if uv_all is not None else 'no'}  "
          f"vertex_colors=yes  bones={'yes' if bone_heads is not None else 'no'}")
    if aggregated_mapping:
        print(f"[fbx_export] bone -> SMPL-X joint mapping ({len(aggregated_mapping)} unique vertex groups):")
        for name, info in sorted(aggregated_mapping.items()):
            print(f"    {name:35s} {info}")


if __name__ == "__main__":
    main()
