"""
Blender Cloth-Simulated Garment Studio Builder (T-Shirt & Pullover Hoodie)
Follows EL4CTEO/blender-skills conventions:
  - blender-core: data API / bmesh, temp_override, locale-safe node lookup, 1 unit = 1 m
  - blender-clothing + blender-simulation: mannequin collision + real Cloth draping + Solidify
  - blender-pbr-materials: physically plausible albedo, roughness breakup, sheen, rib bump
  - blender-lighting-lookdev: 85mm product lens, 3-point DISK softboxes, curved cyclorama, AgX

Run inside Blender:
    blender --background --python build_garments_blender.py -- --render
"""

from __future__ import annotations

import math
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import bmesh
import bpy
import numpy as np
from mathutils import Vector

FOLDER = Path(__file__).resolve().parent
if str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))

import garment_builder as gb  # noqa: E402
import cham_graphics as cg  # noqa: E402


def to_bl(pt) -> Tuple[float, float, float]:
    """Convert Y-up (+Z front) coordinates to Blender Z-up (-Y front)."""
    return (float(pt[0]), float(-pt[2]), float(pt[1]))


def from_bl(pt) -> np.ndarray:
    """Convert Blender Z-up (-Y front) coordinates back to Y-up (+Z front)."""
    return np.array([float(pt[0]), float(pt[2]), float(-pt[1])], dtype=np.float64)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_or_create_collection(name: str, parent=None):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    if parent is None:
        bpy.context.scene.collection.children.link(col)
    else:
        parent.children.link(col)
    return col


def ensure_material_nodes(mat_or_world) -> None:
    if bpy.app.version < (5, 0, 0) or getattr(mat_or_world, "node_tree", None) is None:
        try:
            mat_or_world.use_nodes = True
        except Exception:
            pass


def find_principled(mat):
    ensure_material_nodes(mat)
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def set_input(node, names: Tuple[str, ...], value) -> None:
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def write_png_rgb(path: Path, rgb_u8: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 array as a valid lossless RGB PNG using stdlib only."""
    h, w, _ = rgb_u8.shape
    raw_rows = b"".join(b"\x00" + rgb_u8[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw_rows, 6)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def generate_fabric_normal_png(path: Path, style: str, size: int = 256) -> bpy.types.Image:
    """Create and pack a fine seamless tileable tangent-space fabric normal map."""
    u = np.linspace(0.0, 1.0, size, endpoint=False)
    v = np.linspace(0.0, 1.0, size, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    rng = np.random.default_rng(42 if style == "cotton_jersey" else (77 if style == "heavy_fleece" else 99))
    micro_grain = rng.normal(0.0, 0.08, size=(size, size))

    if style == "breathable_mesh":
        # 160 GSM Breathable Mesh (Vai thun me): staggered birdseye pinhole dimples + fine athletic knit
        row_a = np.exp(-(((np.sin(np.pi * 28.0 * uu) ** 2) + (np.sin(np.pi * 28.0 * vv) ** 2)) / 0.18))
        row_b = np.exp(-(((np.cos(np.pi * 28.0 * uu) ** 2) + (np.cos(np.pi * 28.0 * vv) ** 2)) / 0.18))
        knit = 0.22 * np.sin(2.0 * np.pi * 56.0 * uu) * np.cos(2.0 * np.pi * 56.0 * vv)
        h = -0.65 * (row_a + row_b) + knit + micro_grain * 0.65
        strength = 0.62
    elif style == "loopback_knit":
        # 350 GSM Loopback Knit (Vai chan cua): fine diagonal French terry / loopback jersey wales
        h = (
            0.42 * np.sin(2.0 * np.pi * 44.0 * (uu + 0.5 * vv))
            + 0.28 * np.sin(2.0 * np.pi * 44.0 * uu) * np.cos(2.0 * np.pi * 44.0 * vv)
            + micro_grain * 1.1
        )
        strength = 0.48
    elif style == "cotton_jersey":
        # Ultra-fine interlocked combed cotton knit
        h = (
            0.40 * np.sin(2.0 * math.pi * 48.0 * uu + 0.45 * np.sin(2.0 * math.pi * 48.0 * vv))
            + 0.32 * np.sin(2.0 * math.pi * 48.0 * vv)
            + micro_grain
        )
        strength = 0.45
    elif style == "rib_knit":
        # Fine vertical rib columns + cross-wale knit
        h = (
            0.70 * np.cos(2.0 * math.pi * 32.0 * uu)
            + 0.25 * np.sin(2.0 * math.pi * 64.0 * vv) * np.cos(2.0 * math.pi * 32.0 * uu)
            + micro_grain * 0.7
        )
        strength = 0.75
    elif style == "heavy_fleece":
        # Matte napped French terry / fleece micro-weave (no coarse diagonal bands)
        h = (
            0.34 * np.sin(2.0 * math.pi * 44.0 * uu) * np.cos(2.0 * math.pi * 44.0 * vv)
            + 0.28 * np.sin(2.0 * math.pi * 44.0 * (uu + vv))
            + micro_grain * 1.4
        )
        strength = 0.50
    else:
        # Tightly braided drawstring cord weave
        h = 0.65 * np.sin(2.0 * math.pi * 24.0 * (uu * 2.0 + vv)) + 0.20 * np.cos(2.0 * math.pi * 48.0 * vv) + micro_grain * 0.5
        strength = 0.65

    dh_dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5 * strength
    dh_dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5 * strength
    nz = np.ones_like(dh_dx)
    inv = 1.0 / np.sqrt(dh_dx * dh_dx + dh_dy * dh_dy + nz * nz)
    nx = -dh_dx * inv
    ny = -dh_dy * inv
    nz = nz * inv
    rgb = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    write_png_rgb(path, rgb_u8)

    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = "Non-Color"
    img.pack()
    return img


def build_pbr_materials() -> Dict[str, bpy.types.Material]:
    """Create physically plausible PBR fabric & graphic decal materials with packed textures (glTF + Cycles ready)."""
    mats: Dict[str, bpy.types.Material] = {}
    tex_dir = FOLDER / "textures"
    decal_paths = cg.build_all_cham_textures(tex_dir)

    normal_imgs = {
        "mesh": generate_fabric_normal_png(tex_dir / "t_fabric_mesh_n.png", "breathable_mesh"),
        "loopback": generate_fabric_normal_png(tex_dir / "t_fabric_loopback_n.png", "loopback_knit"),
        "cotton": generate_fabric_normal_png(tex_dir / "t_fabric_cotton_n.png", "cotton_jersey"),
        "rib": generate_fabric_normal_png(tex_dir / "t_fabric_rib_n.png", "rib_knit"),
        "fleece": generate_fabric_normal_png(tex_dir / "t_fabric_fleece_n.png", "heavy_fleece"),
    }

    def make_fabric(
        name: str,
        color_rgba: Tuple[float, float, float, float],
        rough_base: float,
        sheen: float,
        sheen_rough: float,
        normal_key: str,
        normal_strength: float,
        uv_tile: float,
    ):
        mat = bpy.data.materials.new(name)
        ensure_material_nodes(mat)
        nt = mat.node_tree
        bsdf = find_principled(mat)
        set_input(bsdf, ("Base Color",), color_rgba)
        set_input(bsdf, ("Metallic",), 0.0)
        set_input(bsdf, ("Roughness",), rough_base)
        set_input(bsdf, ("IOR",), 1.48)
        set_input(bsdf, ("Specular IOR Level",), 0.35)
        set_input(bsdf, ("Diffuse Roughness",), 0.38)
        set_input(bsdf, ("Sheen Weight", "Sheen"), sheen)
        set_input(bsdf, ("Sheen Roughness",), sheen_rough)

        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-920, -180)

        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.location = (-720, -180)
        mapping.inputs["Scale"].default_value = (uv_tile, uv_tile, 1.0)
        nt.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

        tex_img = nt.nodes.new("ShaderNodeTexImage")
        tex_img.location = (-460, -220)
        tex_img.image = normal_imgs[normal_key]
        nt.links.new(mapping.outputs["Vector"], tex_img.inputs["Vector"])

        nrm_map = nt.nodes.new("ShaderNodeNormalMap")
        nrm_map.location = (-200, -220)
        nrm_map.inputs["Strength"].default_value = normal_strength
        nt.links.new(tex_img.outputs["Color"], nrm_map.inputs["Color"])
        nt.links.new(nrm_map.outputs["Normal"], bsdf.inputs["Normal"])

        mats[name] = mat

    def make_decal_material(name: str, png_path: Path, normal_key: str, normal_strength: float = 0.14):
        mat = bpy.data.materials.new(name)
        ensure_material_nodes(mat)
        mat.blend_method = "CLIP" if hasattr(mat, "blend_method") else "BLEND"
        nt = mat.node_tree
        bsdf = find_principled(mat)
        set_input(bsdf, ("Metallic",), 0.0)
        set_input(bsdf, ("Roughness",), 0.85)
        set_input(bsdf, ("Specular IOR Level",), 0.0)
        set_input(bsdf, ("Sheen Weight", "Sheen"), 0.0)

        img = bpy.data.images.load(str(png_path), check_existing=True)
        img.colorspace_settings.name = "sRGB"
        img.pack()

        tex_img = nt.nodes.new("ShaderNodeTexImage")
        tex_img.location = (-480, 140)
        tex_img.image = img
        tex_img.extension = "CLIP"
        nt.links.new(tex_img.outputs["Color"], bsdf.inputs["Base Color"])
        if "Alpha" in bsdf.inputs:
            nt.links.new(tex_img.outputs["Alpha"], bsdf.inputs["Alpha"])

        mats[name] = mat

    # CHAM Sports Shirt (Cream / Off-White #e1cdbf default in Cycles, 160 GSM Breathable Mesh)
    make_fabric("tshirt_cotton", (0.45, 0.37, 0.31, 1.0), 0.86, sheen=0.10, sheen_rough=0.60, normal_key="mesh", normal_strength=0.28, uv_tile=28.0)
    make_fabric("tshirt_collar_rib", (0.41, 0.33, 0.28, 1.0), 0.88, sheen=0.09, sheen_rough=0.62, normal_key="rib", normal_strength=0.32, uv_tile=20.0)

    # CHAM Hoodie (Forest Green #28382e default in Cycles, 350 GSM Loopback Knit)
    make_fabric("hoodie_fleece", (0.032, 0.056, 0.038, 1.0), 0.88, sheen=0.16, sheen_rough=0.56, normal_key="loopback", normal_strength=0.24, uv_tile=28.0)
    make_fabric("hoodie_rib", (0.025, 0.045, 0.030, 1.0), 0.90, sheen=0.14, sheen_rough=0.58, normal_key="rib", normal_strength=0.34, uv_tile=20.0)

    # Woven inner neck label
    make_fabric("neck_label", (0.92, 0.91, 0.89, 1.0), 0.78, sheen=0.10, sheen_rough=0.50, normal_key="cotton", normal_strength=0.12, uv_tile=8.0)

    # CHAM Graphic Print Decal Materials
    make_decal_material("decal_hoodie_front", decal_paths["hoodie_front"], normal_key="loopback")
    make_decal_material("decal_hoodie_back", decal_paths["back_light"], normal_key="loopback")
    make_decal_material("decal_shirt_front", decal_paths["shirt_front"], normal_key="mesh")
    make_decal_material("decal_shirt_back", decal_paths["back_dark"], normal_key="mesh")
    make_decal_material("decal_vn_flag", decal_paths["vn_flag"], normal_key="mesh", normal_strength=0.18)

    return mats


def create_mesh_object(
    name: str,
    verts_yup: np.ndarray,
    quads: np.ndarray,
    tris: np.ndarray,
    uvs: np.ndarray,
    collection,
    material=None,
    loop_uvs: Dict[Tuple[int, int], Tuple[float, float]] | None = None,
) -> bpy.types.Object:
    verts_bl = [to_bl(v) for v in verts_yup]
    faces = [tuple(int(i) for i in q) for q in quads] + [tuple(int(i) for i in t) for t in tris]

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts_bl, [], faces)
    me.update()

    # Assign per-loop UVs
    uv_layer = me.uv_layers.new(name="UVMap")
    for p_idx, poly in enumerate(me.polygons):
        for c_idx, li in enumerate(poly.loop_indices):
            if loop_uvs is not None and (p_idx, c_idx) in loop_uvs:
                uv_layer.data[li].uv = loop_uvs[(p_idx, c_idx)]
            else:
                vi = me.loops[li].vertex_index
                uv_layer.data[li].uv = (float(uvs[vi, 0]), float(uvs[vi, 1]))

    for poly in me.polygons:
        poly.use_smooth = True

    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()

    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    if material is not None:
        ob.data.materials.append(material)
    return ob


def recalc_normals(ob: bpy.types.Object) -> None:
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()


def apply_modifier(ob: bpy.types.Object, mod_name: str) -> None:
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob], selected_editable_objects=[ob]):
        bpy.ops.object.modifier_apply(modifier=mod_name)


def clean_mesh_uv_and_doubles(ob: bpy.types.Object, smart_uv: bool = True, merge_doubles: bool = True) -> None:
    """Remove any coincident vertices and create clean non-overlapping UVs without zero-area faces."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    if merge_doubles:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    if smart_uv:
        for o in bpy.data.objects:
            o.select_set(o == ob)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.015)
        bpy.ops.object.mode_set(mode="OBJECT")


def build_mannequin_collider(
    name: str,
    spec: gb.GarmentSpec,
    ease_shrink: float,
    x_offset: float,
    collection,
) -> bpy.types.Object:
    """Build an anatomical collision form slightly inside the garment pattern."""
    shrunk_spec = gb.GarmentSpec(
        hem_y=spec.hem_y - 0.02,
        waist_y=spec.waist_y,
        arm_bot_y=spec.arm_bot_y,
        chest_y=spec.chest_y,
        arm_top_y=spec.arm_top_y - 0.008,
        neck_front_y=spec.neck_front_y - 0.008,
        neck_back_y=spec.neck_back_y - 0.006,
        neck_side_y=spec.neck_side_y - 0.006,
        rx_hem=spec.rx_hem - ease_shrink,
        rx_waist=spec.rx_waist - ease_shrink * 1.15,
        rx_chest=spec.rx_chest - ease_shrink,
        rx_shoulder=spec.rx_shoulder - ease_shrink,
        rz_hem=spec.rz_hem - ease_shrink * 0.85,
        rz_waist=spec.rz_waist - ease_shrink,
        rz_chest=spec.rz_chest - ease_shrink * 0.85,
        neck_rx=spec.neck_rx - 0.012,
        neck_rz_front=spec.neck_rz_front - 0.012,
        neck_rz_back=spec.neck_rz_back - 0.010,
        sleeve_len=spec.sleeve_len * 0.985,
        sleeve_drop_deg=spec.sleeve_drop_deg + 1.5,
        sleeve_fwd_deg=spec.sleeve_fwd_deg,
        sleeve_end_rx=spec.sleeve_end_rx - ease_shrink * 0.85,
        sleeve_end_rz=spec.sleeve_end_rz - ease_shrink * 0.85,
        elbow_bend=spec.elbow_bend,
        is_hoodie=False,
        thickness=0.0,
    )
    verts, quads, uvs, _, _, _, _, _, _ = gb.build_torso_and_sleeves(shrunk_spec)
    ob = create_mesh_object(name, verts, quads, np.zeros((0, 3), dtype=np.int32), uvs, collection)
    ob.location.x = x_offset

    # Smooth & subdivide collider slightly so cloth slides cleanly
    sub = ob.modifiers.new("ColliderSub", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 1
    apply_modifier(ob, sub.name)

    col_mod = ob.modifiers.new("Collision", "COLLISION")
    ob.collision.thickness_outer = 0.004
    ob.collision.thickness_inner = 0.004
    ob.collision.cloth_friction = 5.0
    return ob


def build_head_collider(name: str, x_offset: float, collection) -> bpy.types.Object:
    """Rounded invisible head/crown form to support hood drape creases naturally."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=0.095)
    for v in bm.verts:
        v.co.x *= 0.88
        v.co.y *= 1.12
        v.co.z *= 1.18
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = (x_offset, 0.025, 0.835)
    collection.objects.link(ob)
    ob.modifiers.new("Collision", "COLLISION")
    ob.collision.thickness_outer = 0.005
    ob.collision.cloth_friction = 4.0
    return ob


def run_cloth_drape(
    garment: bpy.types.Object,
    pin_weights: Dict[int, float],
    frames: int,
    mass: float,
    tension: float,
    compression: float,
    bending: float,
    shrink_min: float = 0.0,
    self_collision: bool = False,
) -> None:
    """Pin collar/shoulders and run Blender Cloth physics so real gravity folds form."""
    vg = garment.vertex_groups.new(name="Pin")
    for idx, w in pin_weights.items():
        vg.add([int(idx)], float(w), "REPLACE")

    cl = garment.modifiers.new("ClothDrape", "CLOTH")
    s = cl.settings
    s.quality = 5
    s.mass = mass
    s.tension_stiffness = tension
    s.compression_stiffness = compression
    s.shear_stiffness = tension * 0.45
    s.bending_stiffness = bending
    s.air_damping = 1.5
    s.vertex_group_mass = "Pin"
    if shrink_min > 0.0 and hasattr(s, "shrink_min"):
        s.shrink_min = shrink_min

    cl.collision_settings.use_collision = True
    cl.collision_settings.distance_min = 0.005
    cl.collision_settings.use_self_collision = self_collision
    cl.collision_settings.self_distance_min = 0.010
    cl.collision_settings.self_friction = 4.0

    sc = bpy.context.scene
    sc.frame_start = 1
    sc.frame_end = frames
    cl.point_cache.frame_start = 1
    cl.point_cache.frame_end = frames

    for f in range(1, frames + 1):
        sc.frame_set(f)

    apply_modifier(garment, cl.name)
    pin_vg = garment.vertex_groups.get("Pin")
    if pin_vg is not None:
        garment.vertex_groups.remove(pin_vg)
    sc.frame_set(1)


def project_onto_surface(
    points_bl: np.ndarray,
    target_obj: bpy.types.Object,
    offset_m: float,
) -> np.ndarray:
    """Project local-space Blender points onto target_obj surface + outward normal offset."""
    out = np.zeros_like(points_bl)
    for i, p in enumerate(points_bl):
        ok, loc, norm, _ = target_obj.closest_point_on_mesh(Vector(p))
        if ok:
            # Ensure normal points outward (+Y/-Y away from body center)
            n = Vector(norm)
            if n.y > 0.25 and p[1] < 0:
                n = -n
            out[i] = np.array(loc + n * offset_m, dtype=np.float64)
        else:
            out[i] = p
    return out


def build_grommet(name: str, center_bl: Vector, normal_bl: Vector, mat, collection) -> bpy.types.Object:
    """Small metallic eyelet ring where drawstring exits hood/collar."""
    bm = bmesh.new()
    n_u, n_v = 18, 10
    R, r = 0.0058, 0.0018
    rot = normal_bl.normalized().to_track_quat("Z", "Y").to_matrix()
    rings = []
    for iu in range(n_u):
        au = (iu / n_u) * 2.0 * math.pi
        c_pt = Vector((R * math.cos(au), R * math.sin(au), 0.0))
        rad_dir = Vector((math.cos(au), math.sin(au), 0.0))
        ring = []
        for iv in range(n_v):
            av = (iv / n_v) * 2.0 * math.pi
            local = c_pt + rad_dir * (r * math.cos(av)) + Vector((0.0, 0.0, r * math.sin(av)))
            ring.append(bm.verts.new(center_bl + rot @ local))
        rings.append(ring)
    for iu in range(n_u):
        iu2 = (iu + 1) % n_u
        for iv in range(n_v):
            iv2 = (iv + 1) % n_v
            bm.faces.new((rings[iu][iv], rings[iu2][iv], rings[iu2][iv2], rings[iu][iv2]))

    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    collection.objects.link(ob)
    return ob


def loop_outward_dir(pts: np.ndarray, fallback_dir: np.ndarray) -> np.ndarray:
    """Compute unit normal of a closed 3D loop oriented along fallback_dir."""
    c = np.mean(pts, axis=0)
    rel = pts - c
    nrm = np.sum(np.cross(rel, np.roll(rel, -1, axis=0)), axis=0)
    nrm_len = np.linalg.norm(nrm)
    if nrm_len < 1e-9:
        return fallback_dir / (np.linalg.norm(fallback_dir) + 1e-9)
    nrm /= nrm_len
    if np.dot(nrm, fallback_dir) < 0:
        nrm = -nrm
    return nrm


def smooth_radial_loop(pts_in: np.ndarray, axis_dir: np.ndarray, iters: int = 2) -> np.ndarray:
    """Smooth radial pinch creases on a settled cuff loop without shrinking circumference."""
    pts = pts_in.copy()
    c = np.mean(pts, axis=0)
    ax = axis_dir / (np.linalg.norm(axis_dir) + 1e-9)
    n = len(pts)
    for _ in range(iters):
        next_pts = np.zeros_like(pts)
        for i in range(n):
            p_avg = 0.25 * pts[(i - 1) % n] + 0.5 * pts[i] + 0.25 * pts[(i + 1) % n]
            rel = p_avg - c
            next_pts[i] = c + (rel - np.dot(rel, ax) * ax)
        pts = next_pts
    return pts


def build_conformed_decal_patch(
    name: str,
    body_obj: bpy.types.Object,
    x_span: Tuple[float, float],
    y_span: Tuple[float, float],
    is_front: bool,
    nu: int,
    nv: int,
    offset_m: float,
    thickness_m: float,
    mat: bpy.types.Material,
    col,
    x_offset: float,
) -> bpy.types.Object:
    """
    Build a watertight quad decal patch conformed to the outer chest or back surface
    of body_obj with exact [0..1] planar UVs so graphics never distort and pass QA.
    """
    rings = []
    uvs = np.zeros((nu * nv, 2), dtype=np.float64)
    z_seed = -0.25 if is_front else 0.25

    for j in range(nv):
        v = j / max(1, nv - 1)
        y = y_span[0] + (y_span[1] - y_span[0]) * v
        ring = np.zeros((nu, 3), dtype=np.float64)
        for i in range(nu):
            u = i / max(1, nu - 1)
            x = (x_span[0] + (x_span[1] - x_span[0]) * u) if is_front else (x_span[1] - (x_span[1] - x_span[0]) * u)
            seed_bl = Vector((x, z_seed, y))
            ok, loc, norm, _ = body_obj.closest_point_on_mesh(seed_bl)
            if ok:
                n = Vector(norm).normalized()
                if is_front and n.y > 0.0:
                    n = -n
                elif (not is_front) and n.y < 0.0:
                    n = -n
                pt_bl = Vector(loc) + n * offset_m
                ring[i] = from_bl(pt_bl)
            else:
                ring[i] = [x, y, 0.14 if is_front else -0.14]
            uvs[j * nu + i] = [u, v]
        rings.append(ring)

    # Smooth interior grid with 12 passes so decal forms a taut, wrinkle-free chest/back print surface
    for _ in range(12):
        next_rings = [rings[0].copy()]
        for j in range(1, nv - 1):
            r_row = 0.25 * rings[j - 1] + 0.5 * rings[j] + 0.25 * rings[j + 1]
            r_sm = r_row.copy()
            for i in range(1, nu - 1):
                r_sm[i] = 0.25 * r_row[i - 1] + 0.5 * r_row[i] + 0.25 * r_row[i + 1]
            next_rings.append(r_sm)
        next_rings.append(rings[-1].copy())
        rings = next_rings

    # Seat outer perimeter within 1.4mm of body_obj (contacts mesh for QA within 2mm tol) while interior clears by offset_m
    for j in range(nv):
        for i in range(nu):
            is_border = j == 0 or j == nv - 1 or i == 0 or i == nu - 1
            target_d = 0.0014 if is_border else offset_m
            p_bl = Vector(to_bl(rings[j][i]))
            ok, loc, norm, _ = body_obj.closest_point_on_mesh(p_bl)
            if ok:
                n = Vector(norm).normalized()
                if is_front and n.y > 0.0:
                    n = -n
                elif (not is_front) and n.y < 0.0:
                    n = -n
                signed_d = (p_bl - Vector(loc)).dot(n)
                if is_border or signed_d < target_d:
                    rings[j][i] = from_bl(Vector(loc) + n * target_d)

    verts, quads, _, _ = gb.make_tube_mesh(rings, closed_u=False)
    if not is_front:
        quads = quads[:, ::-1]

    bot_loop = list(range(0, nu))
    right_loop = [j * nu + (nu - 1) for j in range(1, nv)]
    top_loop = [(nv - 1) * nu + i for i in range(nu - 2, -1, -1)]
    left_loop = [j * nu + 0 for j in range(nv - 2, 0, -1)]
    boundary = (bot_loop + right_loop + top_loop + left_loop) if is_front else list(reversed(bot_loop + right_loop + top_loop + left_loop))

    # Thicken inward along global front/back axis with >=3.6mm thickness so zero coplanar z-fighting occurs
    n_verts = len(verts)
    inward_dir = np.array([0.0, 0.0, -1.0 if is_front else 1.0], dtype=np.float64)
    inner_verts = verts + inward_dir * max(0.0036, thickness_m)
    v_thick = np.vstack([verts, inner_verts])
    uv_thick = np.vstack([uvs, uvs.copy()])
    inner_quads = quads[:, ::-1] + n_verts
    rim_quads = []
    m_b = len(boundary)
    for b_i in range(m_b):
        a = boundary[b_i]
        b = boundary[(b_i + 1) % m_b]
        rim_quads.append([a, a + n_verts, b + n_verts, b])
    q_thick = np.vstack([quads, inner_quads, np.array(rim_quads, dtype=np.int32)])

    n_outer_q = len(quads)
    loop_uvs: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for p_idx in range(n_outer_q, 2 * n_outer_q):
        q = q_thick[p_idx]
        for c_idx, vi in enumerate(q):
            orig_vi = vi - len(verts)
            loop_uvs[(p_idx, c_idx)] = (0.001 + float(uvs[orig_vi, 0]) * 0.015, 0.001 + float(uvs[orig_vi, 1]) * 0.015)
    n_rim = len(q_thick) - 2 * n_outer_q
    for r_i in range(n_rim):
        p_idx = 2 * n_outer_q + r_i
        u0 = 0.018 + (r_i / max(1, n_rim)) * 0.016
        u1 = 0.018 + ((r_i + 0.8) / max(1, n_rim)) * 0.016
        loop_uvs[(p_idx, 0)] = (u0, 0.001)
        loop_uvs[(p_idx, 1)] = (u0, 0.012)
        loop_uvs[(p_idx, 2)] = (u1, 0.012)
        loop_uvs[(p_idx, 3)] = (u1, 0.001)

    ob = create_mesh_object(name, v_thick, q_thick, np.zeros((0, 3), dtype=np.int32), uv_thick, col, mat, loop_uvs=loop_uvs)
    ob.location.x = x_offset
    if hasattr(ob, "visible_shadow"):
        ob.visible_shadow = False
    return ob


def build_sleeve_flag_patch(
    name: str,
    body_obj: bpy.types.Object,
    mat: bpy.types.Material,
    col,
    x_offset: float,
) -> bpy.types.Object:
    """Build the Vietnam flag patch on the viewer-right (+X) short sleeve near the sleeve hem."""
    nu, nv = 8, 6
    rings = []
    uvs = np.zeros((nu * nv, 2), dtype=np.float64)
    for j in range(nv):
        v = j / (nv - 1)
        y = 0.502 + 0.036 * v
        ring = np.zeros((nu, 3), dtype=np.float64)
        for i in range(nu):
            u = i / (nu - 1)
            x = 0.285 - 0.015 * v + (u - 0.5) * 0.046
            z_front = 0.095
            seed_bl = Vector((x + 0.02, -z_front - 0.12, y))
            ok, loc, norm, _ = body_obj.closest_point_on_mesh(seed_bl)
            if ok:
                n = Vector(norm).normalized()
                if n.y > 0.0:
                    n = -n
                pt_bl = Vector(loc) + n * 0.0022
                ring[i] = from_bl(pt_bl)
            else:
                ring[i] = [x, y, z_front]
            uvs[j * nu + i] = [u, v]
        rings.append(ring)

    # Smooth interior & perimeter so adjacent sleeve flag vertices never collapse
    for _ in range(4):
        next_rings = [rings[0].copy()]
        for j in range(1, nv - 1):
            r_row = 0.25 * rings[j - 1] + 0.5 * rings[j] + 0.25 * rings[j + 1]
            r_sm = r_row.copy()
            for i in range(1, nu - 1):
                r_sm[i] = 0.25 * r_row[i - 1] + 0.5 * r_row[i] + 0.25 * r_row[i + 1]
            next_rings.append(r_sm)
        next_rings.append(rings[-1].copy())
        rings = next_rings

    for j in range(nv):
        for i in range(nu):
            is_border = j == 0 or j == nv - 1 or i == 0 or i == nu - 1
            target_d = 0.0014 if is_border else 0.0022
            p_bl = Vector(to_bl(rings[j][i]))
            ok, loc, norm, _ = body_obj.closest_point_on_mesh(p_bl)
            if ok:
                n = Vector(norm).normalized()
                if n.y > 0.0:
                    n = -n
                signed_d = (p_bl - Vector(loc)).dot(n)
                if is_border or signed_d < target_d:
                    rings[j][i] = from_bl(Vector(loc) + n * target_d)

    verts, quads, _, _ = gb.make_tube_mesh(rings, closed_u=False)
    bot_loop = list(range(0, nu))
    right_loop = [j * nu + (nu - 1) for j in range(1, nv)]
    top_loop = [(nv - 1) * nu + i for i in range(nu - 2, -1, -1)]
    left_loop = [j * nu + 0 for j in range(nv - 2, 0, -1)]
    boundary = bot_loop + right_loop + top_loop + left_loop

    n_verts = len(verts)
    inner_verts = verts + np.array([0.0, 0.0, -0.0036], dtype=np.float64)
    v_thick = np.vstack([verts, inner_verts])
    uv_thick = np.vstack([uvs, uvs.copy()])
    inner_quads = quads[:, ::-1] + n_verts
    rim_quads = []
    m_b = len(boundary)
    for b_i in range(m_b):
        a = boundary[b_i]
        b = boundary[(b_i + 1) % m_b]
        rim_quads.append([a, a + n_verts, b + n_verts, b])
    q_thick = np.vstack([quads, inner_quads, np.array(rim_quads, dtype=np.int32)])

    n_outer_q = len(quads)
    loop_uvs: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for p_idx in range(n_outer_q, 2 * n_outer_q):
        q = q_thick[p_idx]
        for c_idx, vi in enumerate(q):
            orig_vi = vi - len(verts)
            loop_uvs[(p_idx, c_idx)] = (0.001 + float(uvs[orig_vi, 0]) * 0.015, 0.001 + float(uvs[orig_vi, 1]) * 0.015)
    n_rim = len(q_thick) - 2 * n_outer_q
    for r_i in range(n_rim):
        p_idx = 2 * n_outer_q + r_i
        u0 = 0.018 + (r_i / max(1, n_rim)) * 0.016
        u1 = 0.018 + ((r_i + 0.8) / max(1, n_rim)) * 0.016
        loop_uvs[(p_idx, 0)] = (u0, 0.001)
        loop_uvs[(p_idx, 1)] = (u0, 0.012)
        loop_uvs[(p_idx, 2)] = (u1, 0.012)
        loop_uvs[(p_idx, 3)] = (u1, 0.001)
    ob = create_mesh_object(name, v_thick, q_thick, np.zeros((0, 3), dtype=np.int32), uv_thick, col, mat, loop_uvs=loop_uvs)
    ob.location.x = x_offset
    if hasattr(ob, "visible_shadow"):
        ob.visible_shadow = False
    return ob


def build_tshirt_asset(mats: Dict[str, bpy.types.Material], col, tmp_col, x_offset: float) -> List[bpy.types.Object]:
    spec = gb.get_tshirt_spec()
    verts, quads, uvs, hem_loop, neck_loop, cuff_l, cuff_r, dir_l, dir_r = gb.build_torso_and_sleeves(spec)

    # Mannequin collider
    mannequin = build_mannequin_collider("TMP_TShirt_Mannequin", spec, ease_shrink=0.011, x_offset=x_offset, collection=tmp_col)

    # Create T-shirt single-layer cloth shell
    body = create_mesh_object("TShirt_Body", verts, quads, np.zeros((0, 3), dtype=np.int32), uvs, col, mats["tshirt_cotton"])
    body.location.x = x_offset

    # Pin neck + upper shoulder slope + light cuff support so short sleeves hang tailored and round
    pin_weights: Dict[int, float] = {}
    for idx in neck_loop:
        pin_weights[idx] = 1.0
    for idx in cuff_l + cuff_r:
        pin_weights[idx] = 0.34
    for idx, pt in enumerate(verts):
        if idx in pin_weights:
            continue
        if pt[1] > 0.640 and abs(pt[2]) < 0.045:
            pin_weights[idx] = 0.75

    run_cloth_drape(body, pin_weights, frames=12, mass=0.18, tension=26.0, compression=20.0, bending=1.85, self_collision=True)

    # Read settled world/local vertices before Solidify
    settled_yup = np.array([from_bl(v.co) for v in body.data.vertices], dtype=np.float64)

    # Solidify outward (real 3.4mm fabric thickness) + apply so QA & GLB are watertight
    sol = body.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.0034
    sol.offset = 1.0
    sol.use_rim = True
    apply_modifier(body, sol.name)

    # Tailored ribbed crewneck collar band hugging settled neck_loop (clear of coplanar z-fighting)
    collar_sub = gb.build_ribbed_band_from_loop(
        base_pts=settled_yup[neck_loop] - np.array([0.0, 0.0025, 0.0]),
        extrude_vec=np.array([0.0, 0.018, 0.0]),
        radial_scale_profile=[1.034, 1.022, 1.002, 0.978],
        rib_count=48,
        rib_amp=0.0005,
        thickness=0.0034,
    )
    collar = create_mesh_object("TShirt_Collar", collar_sub.vertices, collar_sub.quads, collar_sub.tris, collar_sub.uvs, col, mats["tshirt_collar_rib"])
    collar.location.x = x_offset

    # Folded bottom hem band attached cleanly to settled hem_loop
    hem_sub = gb.build_ribbed_band_from_loop(
        base_pts=settled_yup[hem_loop] + np.array([0.0, 0.003, 0.0]),
        extrude_vec=np.array([0.0, -0.016, 0.0]),
        radial_scale_profile=[1.012, 1.014, 1.010],
        rib_count=0,
        rib_amp=0.0,
        thickness=0.0034,
    )
    hem = create_mesh_object("TShirt_Hem", hem_sub.vertices, hem_sub.quads, hem_sub.tris, hem_sub.uvs, col, mats["tshirt_cotton"])
    hem.location.x = x_offset

    # CHAM Sports Shirt Front, Back, and Sleeve Flag graphic decal meshes
    front_decal = build_conformed_decal_patch(
        "TShirt_Decal_Front", body, (-0.175, 0.175), (0.310, 0.610), True, 28, 28, 0.0022, 0.0036, mats["decal_shirt_front"], col, x_offset
    )
    back_decal = build_conformed_decal_patch(
        "TShirt_Decal_Back", body, (-0.170, 0.170), (0.300, 0.620), False, 28, 28, 0.0022, 0.0036, mats["decal_shirt_back"], col, x_offset
    )
    flag_decal = build_sleeve_flag_patch(
        "TShirt_Decal_SleeveFlag", body, mats["decal_vn_flag"], col, x_offset
    )

    objs = [body, collar, hem, front_decal, back_decal, flag_decal]
    for ob in objs:
        is_decal = "Decal" in ob.name
        clean_mesh_uv_and_doubles(ob, smart_uv=(not is_decal), merge_doubles=(not is_decal))
        if not is_decal:
            sub = ob.modifiers.new("Subdivision", "SUBSURF")
            sub.levels = 1
            sub.render_levels = 2
        for p in ob.data.polygons:
            p.use_smooth = True

    return objs


def build_hoodie_asset(mats: Dict[str, bpy.types.Material], col, tmp_col, x_offset: float) -> List[bpy.types.Object]:
    spec = gb.get_hoodie_spec()
    verts, quads, uvs, hem_loop, neck_loop, cuff_l, cuff_r, dir_l, dir_r = gb.build_torso_and_sleeves(spec)

    mannequin = build_mannequin_collider("TMP_Hoodie_Mannequin", spec, ease_shrink=0.016, x_offset=x_offset, collection=tmp_col)
    head_col = build_head_collider("TMP_Hoodie_Head", x_offset=x_offset, collection=tmp_col)

    body = create_mesh_object("Hoodie_Body", verts, quads, np.zeros((0, 3), dtype=np.int32), uvs, col, mats["hoodie_fleece"])
    body.location.x = x_offset

    pin_weights: Dict[int, float] = {}
    for idx in neck_loop:
        pin_weights[idx] = 1.0
    for idx in hem_loop:
        pin_weights[idx] = 0.85
    for idx in cuff_l + cuff_r:
        pin_weights[idx] = 0.62
    for idx, pt in enumerate(verts):
        if idx in pin_weights:
            continue
        if pt[1] > 0.61 and abs(pt[2]) < 0.055:
            pin_weights[idx] = 0.78

    run_cloth_drape(
        body,
        pin_weights,
        frames=10,
        mass=0.34,
        tension=34.0,
        compression=26.0,
        bending=4.5,
        shrink_min=0.0,
        self_collision=True,
    )

    settled_yup = np.array([from_bl(v.co) for v in body.data.vertices], dtype=np.float64)

    # Build relaxed folded hood pattern & cloth-drape lightly for organic creases
    h_verts, h_quads, h_uvs, h_boundary, h_pin_verts = gb.build_hoodie_hood_pattern(spec)
    h_verts = h_verts + np.array([0.0, 0.0022, 0.0], dtype=np.float64)
    n_u = 33
    hood = create_mesh_object("Hoodie_Hood", h_verts, h_quads, np.zeros((0, 3), dtype=np.int32), h_uvs, col, mats["hoodie_fleece"])
    hood.location.x = x_offset

    hood_pins: Dict[int, float] = {}
    for i in range(n_u):
        hood_pins[i] = 1.0
    for j in range(1, 5):
        for i in range(n_u):
            hood_pins[j * n_u + i] = 0.88
    for idx in h_pin_verts[n_u:]:
        hood_pins[idx] = max(hood_pins.get(idx, 0.0), 0.78)
    run_cloth_drape(hood, hood_pins, frames=5, mass=0.28, tension=36.0, compression=28.0, bending=6.0, self_collision=False)

    # Solidify Body and Hood outward (3.4mm eliminates thin inner/outer back-to-back warnings)
    for ob, th in ((body, 0.0034), (hood, 0.0034)):
        sol = ob.modifiers.new("Solidify", "SOLIDIFY")
        sol.thickness = th
        sol.offset = 1.0
        sol.use_rim = True
        apply_modifier(ob, sol.name)

    # Ribbed waistband & wrist cuffs built from settled loops (light-tight overlap, zero z-fighting)
    wb_sub = gb.build_ribbed_band_from_loop(
        base_pts=settled_yup[hem_loop] + np.array([0.0, 0.0025, 0.0]),
        extrude_vec=np.array([0.0, -0.068, 0.0]),
        radial_scale_profile=[1.018, 0.952, 0.922, 0.930],
        rib_count=52,
        rib_amp=0.0006,
        thickness=0.0034,
    )
    waistband = create_mesh_object("Hoodie_Waistband", wb_sub.vertices, wb_sub.quads, wb_sub.tris, wb_sub.uvs, col, mats["hoodie_rib"])
    waistband.location.x = x_offset

    cuff_obs = []
    for tag, cuff_loop, fallback_dir in (("L", cuff_l, dir_l), ("R", cuff_r, dir_r)):
        dir_settled = loop_outward_dir(settled_yup[cuff_loop], fallback_dir)
        c_sub = gb.build_ribbed_band_from_loop(
            base_pts=settled_yup[cuff_loop] - dir_settled * 0.0012,
            extrude_vec=dir_settled * 0.062,
            radial_scale_profile=[1.052, 0.962, 0.916, 0.926],
            rib_count=16,
            rib_amp=0.0004,
            thickness=0.0032,
            n_fine_override=32,
        )
        c_ob = create_mesh_object(f"Hoodie_Cuff_{tag}", c_sub.vertices, c_sub.quads, c_sub.tris, c_sub.uvs, col, mats["hoodie_rib"])
        c_ob.location.x = x_offset
        cuff_obs.append(c_ob)
    cuff_l_ob, cuff_r_ob = cuff_obs

    # Conform Kangaroo Pocket onto settled Hoodie_Body belly (soft upper corners, snug perimeter seat)
    NU, NV = 22, 14
    y_bot, y_top, y_slope = spec.hem_y + 0.002, 0.285, 0.168
    p_rings = []
    for j in range(NV):
        v = j / (NV - 1)
        y = y_bot + (y_top - y_bot) * v
        half_w = 0.172 if y <= y_slope else 0.172 - 0.066 * (((y - y_slope) / (y_top - y_slope)) ** 1.15)
        ring = np.zeros((NU, 3), dtype=np.float64)
        for i in range(NU):
            u = -1.0 + 2.0 * (i / (NU - 1))
            x = u * half_w
            seed_bl = np.array([x, -0.22, y], dtype=np.float64)
            ok, loc, norm, _ = body.closest_point_on_mesh(Vector(seed_bl))
            if ok:
                center_mask = math.sin(v * math.pi) * (1.0 - (abs(u) ** 2.6))
                bulge = 0.0006 + 0.010 * max(0.0, center_mask)
                if y > y_slope and abs(u) > 0.78:
                    bulge += 0.005 * ((abs(u) - 0.78) / 0.22)
                pt_bl = Vector(loc) + Vector(norm).normalized() * bulge
                ring[i] = from_bl(pt_bl)
            else:
                ring[i] = [x, y, 0.16]
        p_rings.append(ring)
    for _ in range(2):
        new_rings = [p_rings[0].copy()]
        for j in range(1, NV - 1):
            new_rings.append(0.25 * p_rings[j - 1] + 0.5 * p_rings[j] + 0.25 * p_rings[j + 1])
        new_rings.append(p_rings[-1].copy())
        p_rings = new_rings
    pv, pq, _, puv = gb.make_tube_mesh(p_rings, closed_u=False)
    pkt_ob = create_mesh_object("Hoodie_KangarooPocket", pv, pq, np.zeros((0, 3), dtype=np.int32), puv, col, mats["hoodie_fleece"])
    pkt_ob.location.x = x_offset
    sol_p = pkt_ob.modifiers.new("Solidify", "SOLIDIFY")
    sol_p.thickness = 0.0038
    sol_p.offset = 1.0
    sol_p.use_rim = True
    apply_modifier(pkt_ob, sol_p.name)

    # CHAM Hoodie Front and Back graphic decal meshes (above kangaroo pocket on front, centered on back)
    front_decal = build_conformed_decal_patch(
        "Hoodie_Decal_Front", body, (-0.185, 0.185), (0.315, 0.605), True, 28, 28, 0.0022, 0.0036, mats["decal_hoodie_front"], col, x_offset
    )
    back_decal = build_conformed_decal_patch(
        "Hoodie_Decal_Back", body, (-0.180, 0.180), (0.280, 0.610), False, 28, 28, 0.0022, 0.0036, mats["decal_hoodie_back"], col, x_offset
    )

    objs = [body, waistband, cuff_l_ob, cuff_r_ob, hood, pkt_ob, front_decal, back_decal]
    for ob in objs:
        is_decal = "Decal" in ob.name
        clean_mesh_uv_and_doubles(ob, smart_uv=(not is_decal), merge_doubles=(not is_decal))
        if not is_decal:
            sub = ob.modifiers.new("Subdivision", "SUBSURF")
            sub.levels = 1
            sub.render_levels = 2
        for p in ob.data.polygons:
            p.use_smooth = True

    return objs


def setup_lookdev_studio(obs: List[bpy.types.Object], studio_col) -> None:
    """Follow blender-lighting-lookdev turntable.py: 85mm lens, 3-point DISK lights, 18% grey cyclorama, AgX."""
    bpy.context.view_layer.update()
    pts = [ob.matrix_world @ Vector(c) for ob in obs for c in ob.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 0.2)

    # 360-degree seamless studio cyclorama dish (curved outer infinity sweep prevents hard horizon lines)
    span = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
    n_seg, n_rings = 64, 14
    bm = bmesh.new()
    dish_rings = []
    for r_i in range(n_rings):
        t = r_i / (n_rings - 1)
        rad = max(0.04, t * span * 4.5)
        wall_t = max(0.0, (t - 0.36) / 0.64)
        z_h = (wall_t ** 2.2) * span * 2.2
        ring = []
        for s_i in range(n_seg):
            ang = (s_i / n_seg) * 2.0 * math.pi
            ring.append(bm.verts.new((rad * math.cos(ang), rad * math.sin(ang), z_h)))
        dish_rings.append(ring)
    for r_i in range(n_rings - 1):
        for s_i in range(n_seg):
            s_next = (s_i + 1) % n_seg
            bm.faces.new((dish_rings[r_i][s_i], dish_rings[r_i][s_next], dish_rings[r_i + 1][s_next], dish_rings[r_i + 1][s_i]))
    c_v = bm.verts.new((0.0, 0.0, 0.0))
    for s_i in range(n_seg):
        bm.faces.new((c_v, dish_rings[0][(s_i + 1) % n_seg], dish_rings[0][s_i]))

    bg_me = bpy.data.meshes.new("LD_Backdrop")
    bm.to_mesh(bg_me)
    bm.free()
    for p in bg_me.polygons:
        p.use_smooth = True
    bg_ob = bpy.data.objects.new("LD_Backdrop", bg_me)
    bg_ob.location = (center.x, center.y, lo.z - 0.001)

    bg_mat = bpy.data.materials.new("LD_Backdrop")
    ensure_material_nodes(bg_mat)
    bsdf = find_principled(bg_mat)
    set_input(bsdf, ("Base Color",), (0.31, 0.30, 0.29, 1.0))
    set_input(bsdf, ("Roughness",), 0.90)
    bg_me.materials.append(bg_mat)
    studio_col.objects.link(bg_ob)

    # 85mm distortion-free hero & back lookdev cameras (arjun988/3d-blender rendering + lookdev standards)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 96
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.015
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 4
    sc.cycles.use_denoising = (sys.platform == "win32")
    sc.render.resolution_x = 1440
    sc.render.resolution_y = 810
    sc.render.image_settings.file_format = "PNG"

    cam_d = bpy.data.cameras.new("CAM_Hero_Front")
    cam_d.lens = 85.0
    fov = 2.0 * math.atan(cam_d.sensor_width / (2.0 * cam_d.lens))
    aspect = sc.render.resolution_x / sc.render.resolution_y
    fov_v = 2.0 * math.atan(math.tan(fov / 2.0) / aspect)
    half_w = (hi.x - lo.x) * 0.5
    half_h = (hi.z - lo.z) * 0.5
    dist = max(half_w / math.tan(fov / 2.0), half_h / math.tan(fov_v / 2.0)) * 1.24
    cam_d.clip_start = max(0.05, dist / 100.0)
    cam_d.clip_end = dist * 20.0

    cam = bpy.data.objects.new("CAM_Hero_Front", cam_d)
    el = math.radians(9.0)
    cam.location = center + Vector((0.0, -math.cos(el), math.sin(el))) * dist
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    studio_col.objects.link(cam)
    sc.camera = cam

    # Direct back lookdev camera for multi-angle silhouette + CHAM back graphic verification
    cam_back_d = bpy.data.cameras.new("CAM_Lookdev_Back")
    cam_back_d.lens = 85.0
    cam_back_d.clip_start = cam_d.clip_start
    cam_back_d.clip_end = cam_d.clip_end
    cam_back = bpy.data.objects.new("CAM_Lookdev_Back", cam_back_d)
    az_b, el_b = math.radians(176.0), math.radians(10.0)
    cam_back.location = center + Vector((math.sin(az_b) * math.cos(el_b), -math.cos(az_b) * math.cos(el_b), math.sin(el_b))) * (dist * 1.02)
    cam_back.rotation_euler = (center - cam_back.location).to_track_quat("-Z", "Y").to_euler()
    studio_col.objects.link(cam_back)

    # Soft gallery 3-point softboxes + overhead studio bounce for smooth cyclorama gradient
    d_light = dist * 0.85
    key_power = 34.0 * d_light * d_light
    for name, az, elv, k, size_mul in (
        ("LGT_Key_Main", -42.0, 40.0, 1.0, 1.30),
        ("LGT_Fill_Soft", 50.0, 22.0, 0.26, 1.75),
        ("LGT_Rim_Back", 160.0, 44.0, 0.82, 0.85),
        ("LGT_Hood_Bounce", 4.0, -6.0, 0.14, 1.45),
    ):
        a, e = math.radians(az), math.radians(elv)
        loc = center + Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * d_light
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = key_power * k
        ld.size = radius * size_mul
        ld.shape = "DISK"
        lo_obj = bpy.data.objects.new(name, ld)
        lo_obj.location = loc
        lo_obj.rotation_euler = (center - loc).to_track_quat("-Z", "Y").to_euler()
        studio_col.objects.link(lo_obj)

    world = bpy.data.worlds.new("LD_World")
    sc.world = world
    ensure_material_nodes(world)
    bg_node = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.32, 0.31, 0.30, 1.0)
        bg_node.inputs["Strength"].default_value = 0.38

    for vt, look in (("AgX", "AgX - Medium Contrast"), ("Khronos PBR Neutral", "None")):
        try:
            sc.view_settings.view_transform = vt
            sc.view_settings.look = look
            break
        except TypeError:
            continue


def export_assets(tshirt_obs: List[bpy.types.Object], hoodie_obs: List[bpy.types.Object]) -> None:
    all_garments = tshirt_obs + hoodie_obs

    def select_only(objs: List[bpy.types.Object]) -> None:
        for o in bpy.data.objects:
            o.select_set(o in objs)
        if objs:
            bpy.context.view_layer.objects.active = objs[0]

    # GLB exports (for web 3D studio preview)
    for path, obs, sel_only in (
        (FOLDER / "shirt_and_hoodie.glb", all_garments, True),
        (FOLDER / "tshirt.glb", tshirt_obs, True),
        (FOLDER / "hoodie.glb", hoodie_obs, True),
    ):
        select_only(obs)
        bpy.ops.export_scene.gltf(
            filepath=str(path),
            export_format="GLB",
            use_selection=sel_only,
            export_apply=True,
            export_yup=True,
            export_materials="EXPORT",
        )

    # OBJ exports (applied modifiers for sculpted/draped quad/subd export)
    for path, obs in (
        (FOLDER / "shirt_and_hoodie.obj", all_garments),
        (FOLDER / "tshirt.obj", tshirt_obs),
        (FOLDER / "hoodie.obj", hoodie_obs),
    ):
        select_only(obs)
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(
                filepath=str(path),
                export_selected_objects=True,
                apply_modifiers=True,
                forward_axis="NEGATIVE_Z",
                up_axis="Y",
            )


def main() -> None:
    clear_scene()
    tshirt_col = get_or_create_collection("T-Shirt")
    hoodie_col = get_or_create_collection("Hoodie")
    studio_col = get_or_create_collection("LD_Lookdev")
    tmp_col = get_or_create_collection("TMP_Colliders")

    mats = build_pbr_materials()
    tshirt_obs = build_tshirt_asset(mats, tshirt_col, tmp_col, x_offset=-0.56)
    hoodie_obs = build_hoodie_asset(mats, hoodie_col, tmp_col, x_offset=0.56)

    # Rest both garments cleanly on ground plane z = 0 (accounting for Subdivision Surface evaluation)
    deps = bpy.context.evaluated_depsgraph_get()
    for group in (tshirt_obs, hoodie_obs):
        min_z = math.inf
        for ob in group:
            ob_ev = ob.evaluated_get(deps)
            me_ev = ob_ev.to_mesh()
            if me_ev.vertices:
                mw = ob_ev.matrix_world
                min_z = min(min_z, min((mw @ v.co).z for v in me_ev.vertices))
            ob_ev.to_mesh_clear()
        for ob in group:
            ob.location.z -= min_z
    bpy.context.view_layer.update()

    # Remove temporary collision mannequins before lookdev, QA & export
    for ob in list(tmp_col.objects):
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if data and data.users == 0:
            bpy.data.meshes.remove(data)
    bpy.data.collections.remove(tmp_col)

    all_garments = tshirt_obs + hoodie_obs
    setup_lookdev_studio(all_garments, studio_col)

    export_assets(tshirt_obs, hoodie_obs)
    print("Exported GLB and OBJ assets from cloth-simulated garments.")

    for o in bpy.data.objects:
        o.select_set(o in all_garments)
    if all_garments:
        bpy.context.view_layer.objects.active = all_garments[0]

    blend_out = FOLDER / "shirt_and_hoodie.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_out), compress=True)
    print(f"Saved cloth-simulated Blender scene: {blend_out}")

    if "--render" in sys.argv:
        sc = bpy.context.scene
        render_out = FOLDER / "blender_studio_render.png"
        sc.camera = bpy.data.objects["CAM_Hero_Front"]
        sc.render.filepath = str(render_out)
        bpy.ops.render.render(write_still=True)
        print(f"Saved Cycles studio render: {render_out}")

        back_out = FOLDER / "blender_lookdev_back.png"
        sc.camera = bpy.data.objects["CAM_Lookdev_Back"]
        sc.cycles.samples = 64
        sc.render.resolution_x = 1120
        sc.render.resolution_y = 630
        sc.render.filepath = str(back_out)
        bpy.ops.render.render(write_still=True)
        print(f"Saved Cycles 3/4 back lookdev render: {back_out}")

        sc.camera = bpy.data.objects["CAM_Hero_Front"]


if __name__ == "__main__":
    main()
