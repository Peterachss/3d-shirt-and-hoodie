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
import sys
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


def find_principled(mat):
    if not mat.use_nodes:
        try:
            mat.use_nodes = True
        except Exception:
            pass
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def set_input(node, names: Tuple[str, ...], value) -> None:
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def build_pbr_materials() -> Dict[str, bpy.types.Material]:
    """Create physically plausible PBR fabric & metal materials (glTF + Cycles ready)."""
    mats: Dict[str, bpy.types.Material] = {}

    def make_fabric(
        name: str,
        color_rgba: Tuple[float, float, float, float],
        rough_base: float,
        sheen: float,
        sheen_rough: float,
        bump_strength: float,
        rib_scale: float = 0.0,
        noise_scale: float = 180.0,
    ):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
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
        tex_coord.location = (-820, -180)

        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.location = (-580, -80)
        noise.inputs["Scale"].default_value = noise_scale
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Roughness"].default_value = 0.65
        nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

        # Subtle roughness modulation (+/- 0.06) for Cycles fabric realism
        ramp = nt.nodes.new("ShaderNodeMapRange")
        ramp.location = (-360, -40)
        ramp.inputs["From Min"].default_value = 0.25
        ramp.inputs["From Max"].default_value = 0.75
        ramp.inputs["To Min"].default_value = max(0.55, rough_base - 0.07)
        ramp.inputs["To Max"].default_value = min(0.98, rough_base + 0.06)
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
        nt.links.new(ramp.outputs["Result"], bsdf.inputs["Roughness"])

        bump_src = noise.outputs["Fac"]
        if rib_scale > 0.0:
            wave = nt.nodes.new("ShaderNodeTexWave")
            wave.location = (-580, -300)
            wave.wave_type = "BANDS"
            wave.bands_direction = "X"
            wave.wave_profile = "SIN"
            wave.inputs["Scale"].default_value = rib_scale
            wave.inputs["Distortion"].default_value = 0.6
            nt.links.new(tex_coord.outputs["UV"], wave.inputs["Vector"])

            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "FLOAT"
            mix.location = (-360, -240)
            mix.inputs["Factor"].default_value = 0.75
            nt.links.new(noise.outputs["Fac"], mix.inputs["A"])
            nt.links.new(wave.outputs["Fac"], mix.inputs["B"])
            bump_src = mix.outputs["Result"]

        bump = nt.nodes.new("ShaderNodeBump")
        bump.location = (-180, -220)
        bump.inputs["Strength"].default_value = bump_strength
        bump.inputs["Distance"].default_value = 0.002
        nt.links.new(bump_src, bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        mats[name] = mat

    # Linear albedo values (measured dielectric ranges)
    make_fabric("tshirt_cotton", (0.76, 0.73, 0.68, 1.0), 0.86, sheen=0.32, sheen_rough=0.65, bump_strength=0.04, noise_scale=240.0)
    make_fabric("tshirt_collar_rib", (0.69, 0.66, 0.61, 1.0), 0.88, sheen=0.30, sheen_rough=0.68, bump_strength=0.11, rib_scale=150.0)
    make_fabric("hoodie_fleece", (0.032, 0.037, 0.048, 1.0), 0.91, sheen=0.42, sheen_rough=0.72, bump_strength=0.065, noise_scale=160.0)
    make_fabric("hoodie_rib", (0.022, 0.026, 0.035, 1.0), 0.93, sheen=0.38, sheen_rough=0.74, bump_strength=0.14, rib_scale=140.0)
    make_fabric("hoodie_drawstring", (0.72, 0.69, 0.64, 1.0), 0.84, sheen=0.25, sheen_rough=0.60, bump_strength=0.10, rib_scale=90.0)

    metal = bpy.data.materials.new("hoodie_metal")
    metal.use_nodes = True
    m_bsdf = find_principled(metal)
    set_input(m_bsdf, ("Base Color",), (0.66, 0.68, 0.71, 1.0))
    set_input(m_bsdf, ("Metallic",), 1.0)
    set_input(m_bsdf, ("Roughness",), 0.25)
    mats["hoodie_metal"] = metal

    return mats


def create_mesh_object(
    name: str,
    verts_yup: np.ndarray,
    quads: np.ndarray,
    tris: np.ndarray,
    uvs: np.ndarray,
    collection,
    material=None,
) -> bpy.types.Object:
    verts_bl = [to_bl(v) for v in verts_yup]
    faces = [tuple(int(i) for i in q) for q in quads] + [tuple(int(i) for i in t) for t in tris]

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts_bl, [], faces)
    me.update()

    # Assign per-loop UVs
    uv_layer = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
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
        sleeve_len=spec.sleeve_len * 0.92,
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
    s.quality = 8
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
    cl.collision_settings.self_distance_min = 0.006
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


def build_tshirt_asset(mats: Dict[str, bpy.types.Material], col, tmp_col, x_offset: float) -> List[bpy.types.Object]:
    spec = gb.get_tshirt_spec()
    verts, quads, uvs, hem_loop, neck_loop, cuff_l, cuff_r, _, _ = gb.build_torso_and_sleeves(spec)

    # Mannequin collider
    mannequin = build_mannequin_collider("TMP_TShirt_Mannequin", spec, ease_shrink=0.011, x_offset=x_offset, collection=tmp_col)

    # Create T-shirt single-layer cloth shell
    body = create_mesh_object("TShirt_Body", verts, quads, np.zeros((0, 3), dtype=np.int32), uvs, col, mats["tshirt_cotton"])
    body.location.x = x_offset

    # Pin neck + upper shoulder slope so sleeves, chest, underarms, waist & hem drape naturally
    pin_weights: Dict[int, float] = {}
    for idx in neck_loop:
        pin_weights[idx] = 1.0
    for idx, pt in enumerate(verts):
        if idx in pin_weights:
            continue
        # Soft pin right along top shoulder line (z near 0 and y > 0.64)
        if pt[1] > 0.645 and abs(pt[2]) < 0.045:
            pin_weights[idx] = 0.70

    run_cloth_drape(body, pin_weights, frames=24, mass=0.18, tension=20.0, compression=14.0, bending=0.75, self_collision=True)

    # Read settled world/local vertices before Solidify
    settled_yup = np.array([from_bl(v.co) for v in body.data.vertices], dtype=np.float64)

    # Solidify outward (real 2.6mm fabric thickness) + apply so QA & GLB are watertight
    sol = body.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.0026
    sol.offset = 1.0
    sol.use_rim = True
    apply_modifier(body, sol.name)

    # Ribbed crewneck collar attached cleanly to settled neck_loop (clear of coplanar z-fighting, zero gap)
    collar_sub = gb.build_ribbed_band_from_loop(
        base_pts=settled_yup[neck_loop] - np.array([0.0, 0.003, 0.0]),
        extrude_vec=np.array([0.0, 0.025, 0.0]),
        radial_scale_profile=[1.032, 1.045, 1.02, 0.985],
        rib_count=48,
        rib_amp=0.0006,
        thickness=0.0032,
    )
    collar = create_mesh_object("TShirt_Collar", collar_sub.vertices, collar_sub.quads, collar_sub.tris, collar_sub.uvs, col, mats["tshirt_collar_rib"])
    collar.location.x = x_offset

    # Folded bottom hem band attached directly to settled hem_loop (clear of coplanar z-fighting)
    hem_sub = gb.build_ribbed_band_from_loop(
        base_pts=settled_yup[hem_loop] + np.array([0.0, 0.003, 0.0]),
        extrude_vec=np.array([0.0, -0.017, 0.0]),
        radial_scale_profile=[1.012, 1.014, 1.010],
        rib_count=0,
        rib_amp=0.0,
        thickness=0.0028,
    )
    hem = create_mesh_object("TShirt_Hem", hem_sub.vertices, hem_sub.quads, hem_sub.tris, hem_sub.uvs, col, mats["tshirt_cotton"])
    hem.location.x = x_offset

    objs = [body, collar, hem]
    for ob in objs:
        sub = ob.modifiers.new("Subdivision", "SUBSURF")
        sub.levels = 1
        sub.render_levels = 2
        for p in ob.data.polygons:
            p.use_smooth = True

    return objs


def build_hoodie_asset(mats: Dict[str, bpy.types.Material], col, tmp_col, x_offset: float) -> List[bpy.types.Object]:
    spec = gb.get_hoodie_spec()
    verts, quads, uvs, hem_loop, neck_loop, cuff_l, cuff_r, dir_l, dir_r = gb.build_torso_and_sleeves(spec)

    mannequin = build_mannequin_collider("TMP_Hoodie_Mannequin", spec, ease_shrink=0.018, x_offset=x_offset, collection=tmp_col)
    head_col = build_head_collider("TMP_Hoodie_Head", x_offset=x_offset, collection=tmp_col)

    body = create_mesh_object("Hoodie_Body", verts, quads, np.zeros((0, 3), dtype=np.int32), uvs, col, mats["hoodie_fleece"])
    body.location.x = x_offset

    pin_weights: Dict[int, float] = {}
    for idx in neck_loop:
        pin_weights[idx] = 1.0
    for idx in hem_loop:
        pin_weights[idx] = 0.45
    for idx, pt in enumerate(verts):
        if idx in pin_weights:
            continue
        if pt[1] > 0.62 and abs(pt[2]) < 0.05:
            pin_weights[idx] = 0.72

    run_cloth_drape(
        body,
        pin_weights,
        frames=24,
        mass=0.34,
        tension=28.0,
        compression=20.0,
        bending=2.1,
        shrink_min=0.0,
        self_collision=True,
    )

    settled_yup = np.array([from_bl(v.co) for v in body.data.vertices], dtype=np.float64)

    # Compute exact settled sleeve-end normal from the gravity-draped cuff loop
    def settled_cuff_dir(cuff_loop: List[int], fallback_dir: np.ndarray) -> np.ndarray:
        pts = settled_yup[cuff_loop]
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

    # Build hood pattern & cloth-drape it lightly over the head form for organic creases
    h_verts, h_quads, h_uvs, h_boundary, h_pin_verts = gb.build_hoodie_hood_pattern(spec)
    hood = create_mesh_object("Hoodie_Hood", h_verts, h_quads, np.zeros((0, 3), dtype=np.int32), h_uvs, col, mats["hoodie_fleece"])
    hood.location.x = x_offset

    hood_pins: Dict[int, float] = {}
    n_u = 33
    for i in range(n_u):
        hood_pins[i] = 1.0  # neck collar base locked
    for j in range(1, 4):
        for i in range(n_u):
            hood_pins[j * n_u + i] = 0.85  # keep throat/collar fold clean without pinching
    for idx in h_pin_verts[n_u:]:
        hood_pins[idx] = max(hood_pins.get(idx, 0.0), 0.55)
    run_cloth_drape(hood, hood_pins, frames=12, mass=0.28, tension=30.0, compression=22.0, bending=3.5, self_collision=False)

    # Solidify Body and Hood outward
    for ob, th in ((body, 0.0034), (hood, 0.0028)):
        sol = ob.modifiers.new("Solidify", "SOLIDIFY")
        sol.thickness = th
        sol.offset = 1.0
        sol.use_rim = True
        apply_modifier(ob, sol.name)

    # Smooth any sharp cloth creases on cuff base loops so radial cinching never self-intersects
    def smooth_cuff_base(loop_indices: List[int], axis_dir: np.ndarray) -> np.ndarray:
        pts = settled_yup[loop_indices].copy()
        c = np.mean(pts, axis=0)
        ax = axis_dir / (np.linalg.norm(axis_dir) + 1e-9)
        for _ in range(2):
            next_pts = np.zeros_like(pts)
            n = len(pts)
            for i in range(n):
                p_avg = 0.25 * pts[(i - 1) % n] + 0.5 * pts[i] + 0.25 * pts[(i + 1) % n]
                rel = p_avg - c
                radial = rel - np.dot(rel, ax) * ax
                next_pts[i] = c + radial
            pts = next_pts
        return pts

    # Ribbed waistband & wrist cuffs built from settled loops (clear of z-fighting & interlocked)
    wb_sub = gb.build_ribbed_band_from_loop(
        base_pts=settled_yup[hem_loop] + np.array([0.0, 0.0015, 0.0]),
        extrude_vec=np.array([0.0, -0.070, 0.0]),
        radial_scale_profile=[1.024, 0.945, 0.910, 0.920],
        rib_count=52,
        rib_amp=0.0006,
        thickness=0.0032,
    )
    waistband = create_mesh_object("Hoodie_Waistband", wb_sub.vertices, wb_sub.quads, wb_sub.tris, wb_sub.uvs, col, mats["hoodie_rib"])
    waistband.location.x = x_offset

    dir_l_settled = settled_cuff_dir(cuff_l, dir_l)
    cuff_l_sub = gb.build_ribbed_band_from_loop(
        base_pts=smooth_cuff_base(cuff_l, dir_l_settled) - dir_l_settled * 0.001,
        extrude_vec=dir_l_settled * 0.065,
        radial_scale_profile=[1.065, 0.985, 0.945, 0.955],
        rib_count=16,
        rib_amp=0.0004,
        thickness=0.0026,
        n_fine_override=32,
    )
    cuff_l_ob = create_mesh_object("Hoodie_Cuff_L", cuff_l_sub.vertices, cuff_l_sub.quads, cuff_l_sub.tris, cuff_l_sub.uvs, col, mats["hoodie_rib"])
    cuff_l_ob.location.x = x_offset

    dir_r_settled = settled_cuff_dir(cuff_r, dir_r)
    cuff_r_sub = gb.build_ribbed_band_from_loop(
        base_pts=smooth_cuff_base(cuff_r, dir_r_settled) - dir_r_settled * 0.001,
        extrude_vec=dir_r_settled * 0.065,
        radial_scale_profile=[1.065, 0.985, 0.945, 0.955],
        rib_count=16,
        rib_amp=0.0004,
        thickness=0.0026,
        n_fine_override=32,
    )
    cuff_r_ob = create_mesh_object("Hoodie_Cuff_R", cuff_r_sub.vertices, cuff_r_sub.quads, cuff_r_sub.tris, cuff_r_sub.uvs, col, mats["hoodie_rib"])
    cuff_r_ob.location.x = x_offset

    # Conform Kangaroo Pocket onto settled Hoodie_Body belly (seam edges seat at +0.6mm so zero floating/z-fighting)
    NU, NV = 20, 12
    y_bot, y_top, y_slope = spec.hem_y + 0.004, 0.295, 0.175
    p_rings = []
    for j in range(NV):
        v = j / (NV - 1)
        y = y_bot + (y_top - y_bot) * v
        half_w = 0.176 if y <= y_slope else 0.176 - 0.064 * (((y - y_slope) / (y_top - y_slope)) ** 0.92)
        ring = np.zeros((NU, 3), dtype=np.float64)
        for i in range(NU):
            u = -1.0 + 2.0 * (i / (NU - 1))
            x = u * half_w
            seed_bl = np.array([x, -0.22, y], dtype=np.float64)
            ok, loc, norm, _ = body.closest_point_on_mesh(Vector(seed_bl))
            if ok:
                # Seam perimeter touches within 0.6mm (passes QA contact check), center puffs +11mm
                center_mask = math.sin(v * math.pi) * (1.0 - (abs(u) ** 2.6))
                bulge = 0.0006 + 0.011 * max(0.0, center_mask)
                if y > y_slope and abs(u) > 0.78:
                    bulge += 0.005 * ((abs(u) - 0.78) / 0.22)
                pt_bl = Vector(loc) + Vector(norm).normalized() * bulge
                ring[i] = from_bl(pt_bl)
            else:
                ring[i] = [x, y, 0.16]
        p_rings.append(ring)
    pv, pq, _, puv = gb.make_tube_mesh(p_rings, closed_u=False)
    pkt_ob = create_mesh_object("Hoodie_KangarooPocket", pv, pq, np.zeros((0, 3), dtype=np.int32), puv, col, mats["hoodie_fleece"])
    pkt_ob.location.x = x_offset
    sol_p = pkt_ob.modifiers.new("Solidify", "SOLIDIFY")
    sol_p.thickness = 0.0036
    sol_p.offset = 1.0
    sol_p.use_rim = True
    apply_modifier(pkt_ob, sol_p.name)

    # Conform Drawstrings, Grommets, and Aglets directly onto settled chest surface
    extra_objs: List[bpy.types.Object] = []
    for side_sign in (+1.0, -1.0):
        tag = "L" if side_sign > 0 else "R"
        x_start = side_sign * 0.028
        y_start = spec.neck_front_y + 0.016
        y_end = 0.445
        n_path, n_circ = 20, 10

        pts_yup = []
        grommet_pos = None
        grommet_nrm = None

        for k in range(n_path):
            t = k / (n_path - 1)
            y = y_start + (y_end - y_start) * t
            x = x_start + side_sign * (0.009 * math.sin(t * math.pi * 0.8) + 0.0025 * math.sin(t * math.pi * 2.0))
            seed_bl = Vector((x, -0.22, y))
            ok, loc, norm, _ = body.closest_point_on_mesh(seed_bl)
            if ok:
                nrm = Vector(norm).normalized()
                if k == 0:
                    grommet_pos = Vector(loc) + nrm * 0.0015
                    grommet_nrm = nrm
                pt_bl = Vector(loc) + nrm * (0.0046 + 0.002 * math.sin(t * math.pi))
                pts_yup.append(from_bl(pt_bl))
            else:
                pts_yup.append(np.array([x, y, 0.14], dtype=np.float64))

        if grommet_pos is not None:
            grom_ob = build_grommet(f"Hoodie_Grommet_{tag}", grommet_pos, grommet_nrm, mats["hoodie_metal"], col)
            grom_ob.location.x = x_offset
            extra_objs.append(grom_ob)

        def sweep(path_pts, radius, cap_s, cap_e):
            rings = []
            for idx, p in enumerate(path_pts):
                tang = path_pts[1] - path_pts[0] if idx == 0 else (path_pts[-1] - path_pts[-2] if idx == len(path_pts) - 1 else path_pts[idx + 1] - path_pts[idx - 1])
                tang /= np.linalg.norm(tang) + 1e-9
                n1 = np.cross(tang, np.array([1.0, 0.0, 0.0]))
                n1 /= np.linalg.norm(n1) + 1e-9
                n2 = np.cross(tang, n1)
                ring = np.zeros((n_circ, 3), dtype=np.float64)
                for c in range(n_circ):
                    ang = (c / n_circ) * 2.0 * math.pi
                    ring[c] = p + (n1 * math.cos(ang) + n2 * math.sin(ang)) * radius
                rings.append(ring)
            return gb.make_tube_mesh(rings, closed_u=True, cap_start=cap_s, cap_end=cap_e)

        vc, qc, tc, uvc = sweep(pts_yup[:-2], 0.0036, True, False)
        va, qa, ta, uva = sweep([pts_yup[-3], pts_yup[-2], pts_yup[-1], pts_yup[-1] + np.array([0.0, -0.017, 0.001])], 0.0046, True, True)
        cord_ob = create_mesh_object(f"Hoodie_Drawstring_{tag}", vc, qc, tc, uvc, col, mats["hoodie_drawstring"])
        aglet_ob = create_mesh_object(f"Hoodie_Aglet_{tag}", va, qa, ta, uva, col, mats["hoodie_metal"])
        cord_ob.location.x = x_offset
        aglet_ob.location.x = x_offset
        extra_objs.extend([cord_ob, aglet_ob])

    objs = [body, waistband, cuff_l_ob, cuff_r_ob, hood, pkt_ob] + extra_objs
    for ob in objs:
        is_small = any(k in ob.name for k in ("Drawstring", "Aglet", "Grommet"))
        sub = ob.modifiers.new("Subdivision", "SUBSURF")
        sub.levels = 1
        sub.render_levels = 1 if is_small else 2
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

    # Seamless curved cyclorama backdrop (18% neutral grey)
    w = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z) * 5.0
    bm = bmesh.new()
    a = bm.verts.new((-w / 2, -w / 2, 0))
    b = bm.verts.new((w / 2, -w / 2, 0))
    c = bm.verts.new((w / 2, w / 2, 0))
    d = bm.verts.new((-w / 2, w / 2, 0))
    e = bm.verts.new((w / 2, w / 2, w * 0.55))
    f = bm.verts.new((-w / 2, w / 2, w * 0.55))
    bm.faces.new((a, b, c, d))
    bm.faces.new((d, c, e, f))
    bg_me = bpy.data.meshes.new("LD_Backdrop")
    bm.to_mesh(bg_me)
    bm.free()
    for p in bg_me.polygons:
        p.use_smooth = True
    bg_ob = bpy.data.objects.new("LD_Backdrop", bg_me)
    bg_ob.location = (center.x, center.y, lo.z - 0.002)
    bev = bg_ob.modifiers.new("Curve", "BEVEL")
    bev.width = w * 0.25
    bev.segments = 16
    bev.limit_method = "ANGLE"

    bg_mat = bpy.data.materials.new("LD_Backdrop")
    bg_mat.use_nodes = True
    bsdf = find_principled(bg_mat)
    set_input(bsdf, ("Base Color",), (0.38, 0.37, 0.35, 1.0))
    set_input(bsdf, ("Roughness",), 0.85)
    bg_me.materials.append(bg_mat)
    studio_col.objects.link(bg_ob)

    # 85mm distortion-free product camera
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 48
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = False
    sc.render.resolution_x = 1120
    sc.render.resolution_y = 665
    sc.render.image_settings.file_format = "PNG"

    cam_d = bpy.data.cameras.new("LD_Cam")
    cam_d.lens = 85.0
    fov = 2.0 * math.atan(cam_d.sensor_width / (2.0 * cam_d.lens))
    aspect = sc.render.resolution_x / sc.render.resolution_y
    fov_v = 2.0 * math.atan(math.tan(fov / 2.0) / aspect)
    half_w = (hi.x - lo.x) * 0.5
    half_h = (hi.z - lo.z) * 0.5
    dist = max(half_w / math.tan(fov / 2.0), half_h / math.tan(fov_v / 2.0)) * 1.24
    cam_d.clip_start = max(0.05, dist / 100.0)
    cam_d.clip_end = dist * 20.0

    cam = bpy.data.objects.new("LD_Cam", cam_d)
    el = math.radians(10.0)
    cam.location = center + Vector((0.0, -math.cos(el), math.sin(el))) * dist
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    studio_col.objects.link(cam)
    sc.camera = cam

    # 3-point softbox DISK area lights
    d_light = dist * 0.78
    key_power = 65.0 * d_light * d_light
    for name, az, elv, k, size_mul in (
        ("LD_Key", -42.0, 34.0, 1.0, 1.05),
        ("LD_Fill", 48.0, 16.0, 0.36, 1.55),
        ("LD_Rim", 160.0, 40.0, 0.85, 0.65),
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
    world.use_nodes = True
    bg_node = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.42, 0.41, 0.40, 1.0)
        bg_node.inputs["Strength"].default_value = 0.42

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

    # Rest both garments cleanly on ground plane z = 0
    bpy.context.view_layer.update()
    for group in (tshirt_obs, hoodie_obs):
        min_z = min((ob.matrix_world @ Vector(c)).z for ob in group for c in ob.bound_box)
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

    blend_out = FOLDER / "shirt_and_hoodie.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_out), compress=True)
    print(f"Saved cloth-simulated Blender scene: {blend_out}")

    export_assets(tshirt_obs, hoodie_obs)
    print("Exported GLB and OBJ assets from cloth-simulated garments.")

    if "--render" in sys.argv:
        render_out = FOLDER / "blender_studio_render.png"
        bpy.context.scene.render.filepath = str(render_out)
        bpy.ops.render.render(write_still=True)
        print(f"Saved Cycles studio render: {render_out}")


if __name__ == "__main__":
    main()
