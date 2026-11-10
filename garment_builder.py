"""
Parametric 3D Garment Mesh Builder (Crewneck T-Shirt & Pullover Hoodie)
Generates clean, thickened, quad-dominant 3D meshes in OBJ+MTL (quads preserved
for Blender Subdivision Surface / sculpting) and binary glTF 2.0 (.glb) with PBR
materials and clean UVs.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


# ---------------------------------------------------------------------------
# Mesh Data Container & Geometry Helpers
# ---------------------------------------------------------------------------

@dataclass
class MaterialDef:
    name: str
    base_color: Tuple[float, float, float, float]
    roughness: float = 0.85
    metallic: float = 0.0


@dataclass
class SubMesh:
    name: str
    vertices: np.ndarray        # (N, 3) float64
    quads: np.ndarray           # (M, 4) int32
    tris: np.ndarray            # (K, 3) int32
    uvs: np.ndarray             # (N, 2) float64
    material_name: str

    @property
    def normals(self) -> np.ndarray:
        return compute_vertex_normals(self.vertices, self.quads, self.tris)


def compute_vertex_normals(
    verts: np.ndarray, quads: np.ndarray, tris: np.ndarray
) -> np.ndarray:
    normals = np.zeros_like(verts, dtype=np.float64)

    if len(quads) > 0:
        v0 = verts[quads[:, 0]]
        v1 = verts[quads[:, 1]]
        v2 = verts[quads[:, 2]]
        v3 = verts[quads[:, 3]]
        fn = np.cross(v2 - v0, v3 - v1)
        for col in range(4):
            np.add.at(normals, quads[:, col], fn)

    if len(tris) > 0:
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for col in range(3):
            np.add.at(normals, tris[:, col], fn)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    return normals / lengths


def thicken_shell(
    verts: np.ndarray,
    quads: np.ndarray,
    uvs: np.ndarray,
    boundary_loops: List[List[int]],
    thickness: float,
    inset_normals: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a closed solid shell from an open quad surface by duplicating
    vertices offset inward along vertex normals, reversing inner quad winding,
    and bridging each boundary edge loop with quad strips.
    """
    n_verts = len(verts)
    normals = compute_vertex_normals(verts, quads, np.zeros((0, 3), dtype=np.int32))
    offset_dir = -normals if inset_normals else normals
    inner_verts = verts + offset_dir * thickness

    all_verts = np.vstack([verts, inner_verts])
    inner_uvs = uvs.copy()
    inner_uvs[:, 0] = (inner_uvs[:, 0] + 0.003) % 1.0
    inner_uvs[:, 1] = np.clip(inner_uvs[:, 1] + 0.003, 0.0, 1.0)
    all_uvs = np.vstack([uvs, inner_uvs])
    inner_quads = quads[:, ::-1] + n_verts

    rim_quads_list = []
    for loop in boundary_loops:
        m = len(loop)
        for i in range(m):
            a = loop[i]
            b = loop[(i + 1) % m]
            a_in = a + n_verts
            b_in = b + n_verts
            rim_quads_list.append([a, a_in, b_in, b])

    if rim_quads_list:
        rim_quads = np.array(rim_quads_list, dtype=np.int32)
        all_quads = np.vstack([quads, inner_quads, rim_quads])
    else:
        all_quads = np.vstack([quads, inner_quads])

    return all_verts, all_quads, all_uvs


def make_tube_mesh(
    rings: List[np.ndarray],
    closed_u: bool = True,
    cap_start: bool = False,
    cap_end: bool = False,
    u_range: Tuple[float, float] = (0.0, 1.0),
    v_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a quad-grid tube from a list of (Nu, 3) rings."""
    n_rings = len(rings)
    nu = len(rings[0])
    verts = np.vstack(rings)
    quads = []
    tris = []

    uvs = np.zeros((len(verts), 2), dtype=np.float64)
    for j in range(n_rings):
        v_coord = v_range[0] + (v_range[1] - v_range[0]) * (j / max(1, n_rings - 1))
        for i in range(nu):
            u_coord = u_range[0] + (u_range[1] - u_range[0]) * (i / max(1, nu if closed_u else nu - 1))
            uvs[j * nu + i] = [u_coord, v_coord]

    u_steps = nu if closed_u else nu - 1
    for j in range(n_rings - 1):
        for i in range(u_steps):
            i_next = (i + 1) % nu
            a = j * nu + i
            b = j * nu + i_next
            c = (j + 1) * nu + i_next
            d = (j + 1) * nu + i
            quads.append([a, b, c, d])

    if cap_start and closed_u:
        center = np.mean(rings[0], axis=0, keepdims=True)
        c_idx = len(verts)
        verts = np.vstack([verts, center])
        uvs = np.vstack([uvs, [[0.5, v_range[0]]]])
        for i in range(nu):
            i_next = (i + 1) % nu
            tris.append([c_idx, i_next, i])

    if cap_end and closed_u:
        center = np.mean(rings[-1], axis=0, keepdims=True)
        c_idx = len(verts)
        verts = np.vstack([verts, center])
        uvs = np.vstack([uvs, [[0.5, v_range[1]]]])
        base = (n_rings - 1) * nu
        for i in range(nu):
            i_next = (i + 1) % nu
            tris.append([c_idx, base + i, base + i_next])

    return (
        verts,
        np.array(quads, dtype=np.int32) if quads else np.zeros((0, 4), dtype=np.int32),
        np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32),
        uvs,
    )


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def superellipse_pt(theta: float, rx: float, rz: float, power: float = 2.55) -> Tuple[float, float]:
    sx = math.sin(theta)
    cz = math.cos(theta)
    ax = abs(sx)
    az = abs(cz)
    exp = 2.0 / power
    x = math.copysign(rx * (ax ** exp), sx) if ax > 1e-9 else 0.0
    z = math.copysign(rz * (az ** exp), cz) if az > 1e-9 else 0.0
    return x, z


# ---------------------------------------------------------------------------
# Torso + Seamless Armholes + Sleeves Parametric Core
# ---------------------------------------------------------------------------

@dataclass
class GarmentSpec:
    hem_y: float
    waist_y: float
    arm_bot_y: float
    chest_y: float
    arm_top_y: float
    neck_front_y: float
    neck_back_y: float
    neck_side_y: float

    rx_hem: float
    rx_waist: float
    rx_chest: float
    rx_shoulder: float
    rz_hem: float
    rz_waist: float
    rz_chest: float

    neck_rx: float
    neck_rz_front: float
    neck_rz_back: float

    sleeve_len: float
    sleeve_drop_deg: float
    sleeve_fwd_deg: float
    sleeve_end_rx: float
    sleeve_end_rz: float
    elbow_bend: float = 0.0
    is_hoodie: bool = False
    thickness: float = 0.0028


def eval_torso_profile(spec: GarmentSpec, y: float, theta: float) -> Tuple[float, float]:
    """Continuous torso cross-section evaluated identically below and above underarm."""
    if y <= spec.waist_y:
        t = smoothstep(spec.hem_y, spec.waist_y, y)
        rx = spec.rx_hem + (spec.rx_waist - spec.rx_hem) * t
        rz = spec.rz_hem + (spec.rz_waist - spec.rz_hem) * t
        if spec.is_hoodie:
            puff = math.sin(min(1.0, max(0.0, (y - spec.hem_y) / 0.14)) * math.pi) * 0.017
            rx += puff
            rz += puff * 0.65
    elif y <= spec.chest_y:
        t = smoothstep(spec.waist_y, spec.chest_y, y)
        rx = spec.rx_waist + (spec.rx_chest - spec.rx_waist) * t
        rz = spec.rz_waist + (spec.rz_chest - spec.rz_waist) * t
    else:
        t = smoothstep(spec.chest_y, spec.arm_top_y, y)
        rx = spec.rx_chest + (spec.rx_shoulder - spec.rx_chest) * t
        rz = spec.rz_chest * (1.0 - 0.12 * t)

    x, z = superellipse_pt(theta, rx, rz, power=2.55)

    # Gentle vertical drape folds fading smoothly toward chest
    v_fade = 1.0 - smoothstep(spec.hem_y, spec.chest_y, y)
    drape_amp = (0.0042 if spec.is_hoodie else 0.0026) * v_fade
    drape = drape_amp * math.sin(6.0 * theta + 0.5) + (drape_amp * 0.45) * math.cos(10.0 * theta)
    r_len = math.hypot(x, z) + 1e-9
    x += (x / r_len) * drape
    z += (z / r_len) * drape

    # Anatomical front chest fullness and upper-back shoulder blade contour
    if z > 0:
        z += 0.010 * math.exp(-((abs(x) - 0.085) ** 2) / 0.006) * smoothstep(0.22, 0.47, y)
    else:
        z += 0.008 * math.exp(-(x ** 2) / 0.004) * smoothstep(0.20, 0.47, y)

    return x, z


def build_torso_and_sleeves(spec: GarmentSpec) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int], List[int], np.ndarray, np.ndarray
]:
    """
    Build a seamless pure-quad open shell for torso + both sleeves.
    N_THETA = 64 around torso:
      Front upper: k in [53..63, 0..11]  (22 intervals, s = -11..+11)
      +X armhole:  k in [11..21]         (10 intervals)
      Back upper:  k in [21..43]         (22 intervals, s = +11..-11)
      -X armhole:  k in [43..53]         (10 intervals)
      Shoulder seam +X: front s=+7..+11 (k=7..11) stitched to back k=25..21
      Shoulder seam -X: front s=-7..-11 (k=57..53) stitched to back k=39..43
      Neck loop: front s=-7..+7 (14 edges) + back k=25..39 (14 edges) = 28 edges
    """
    N_THETA = 64
    J_ARM = 18
    J_TOP = 32

    vert_list: List[np.ndarray] = []
    uv_list: List[List[float]] = []
    idx_map: Dict[Tuple[int, int], int] = {}

    # 1) Lower torso rings j = 0 .. J_ARM
    for j in range(J_ARM + 1):
        v = j / J_ARM
        y = spec.hem_y + (spec.arm_bot_y - spec.hem_y) * v
        for k in range(N_THETA):
            theta = (k / N_THETA) * 2.0 * math.pi
            x, z = eval_torso_profile(spec, y, theta)
            idx_map[(j, k)] = len(vert_list)
            vert_list.append(np.array([x, y, z], dtype=np.float64))
            uv_list.append([k / N_THETA, 0.75 * (y / spec.neck_back_y)])

    # 2) Upper torso front and back panels j = J_ARM + 1 .. J_TOP
    front_ks = list(range(53, 64)) + list(range(0, 12))
    back_ks = list(range(21, 44))
    NECK_SPAN = 7   # s in [-7..+7] is neck; |s| in [7..11] is shoulder seam (4 intervals)
    MAX_S = 11

    def upper_panel_pt(j: int, k: int, is_front: bool) -> Tuple[np.ndarray, Tuple[float, float]]:
        t = (j - J_ARM) / (J_TOP - J_ARM)
        s = (k if k <= MAX_S else k - N_THETA) if is_front else (32 - k)
        abs_s = abs(s)
        sign_s = 1.0 if s >= 0 else -1.0

        # Target top-edge position at j = J_TOP
        if abs_s <= NECK_SPAN:
            frac = abs_s / NECK_SPAN
            ang = frac * (math.pi * 0.5)
            tx = sign_s * spec.neck_rx * math.sin(ang)
            if is_front:
                tz = spec.neck_rz_front * math.cos(ang)
                ty = spec.neck_front_y + (spec.neck_side_y - spec.neck_front_y) * (frac ** 1.35)
            else:
                tz = -spec.neck_rz_back * math.cos(ang)
                ty = spec.neck_back_y + (spec.neck_side_y - spec.neck_back_y) * (frac ** 1.45)
        else:
            frac = (abs_s - NECK_SPAN) / (MAX_S - NECK_SPAN)
            tx = sign_s * (spec.neck_rx + (spec.rx_shoulder - spec.neck_rx) * frac)
            ty = spec.neck_side_y + (spec.arm_top_y - spec.neck_side_y) * frac
            tz = 0.0

        # Smooth height interpolation starting continuously from arm_bot_y
        y = spec.arm_bot_y + (ty - spec.arm_bot_y) * t
        theta = (k / N_THETA) * 2.0 * math.pi
        bx, bz = eval_torso_profile(spec, y, theta)

        # Blend smoothly from continuous torso contour (at t=0) to shoulder/neck contour (at t=1)
        # Use identical blending curve across NECK_SPAN boundary so chest/back stay C1-smooth
        blend_x = t ** 1.30
        blend_z = smoothstep(0.0, 1.0, t)

        x = bx + (tx - bx) * blend_x
        if abs_s >= NECK_SPAN:
            sh_frac = (abs_s - NECK_SPAN) / (MAX_S - NECK_SPAN)
            z_arch = math.sin(t * math.pi) * (0.015 if is_front else -0.015) * sh_frac
            z = bz * (1.0 - blend_z) + tz * blend_z + z_arch
        else:
            z = bz * (1.0 - blend_z) + tz * blend_z

        return np.array([x, y, z], dtype=np.float64), (k / N_THETA, 0.75 * (y / spec.neck_back_y))

    for j in range(J_ARM + 1, J_TOP + 1):
        for k in front_ks:
            pt, uv = upper_panel_pt(j, k, is_front=True)
            idx_map[(j, k)] = len(vert_list)
            vert_list.append(pt)
            uv_list.append([uv[0], uv[1]])

        for k in back_ks:
            if j == J_TOP and 21 <= k <= 25:
                # k=21..25 maps to front k=11..7
                front_match = 11 - (k - 21)
                idx_map[(j, k)] = idx_map[(j, front_match)]
                continue
            if j == J_TOP and 39 <= k <= 43:
                # k=39..43 maps to front k=57..53
                front_match = 57 - (k - 39)
                idx_map[(j, k)] = idx_map[(j, front_match)]
                continue

            pt, uv = upper_panel_pt(j, k, is_front=False)
            idx_map[(j, k)] = len(vert_list)
            vert_list.append(pt)
            uv_list.append([uv[0], uv[1]])

    quads: List[List[int]] = []
    for j in range(J_ARM):
        for k in range(N_THETA):
            k_next = (k + 1) % N_THETA
            quads.append([
                idx_map[(j, k)],
                idx_map[(j, k_next)],
                idx_map[(j + 1, k_next)],
                idx_map[(j + 1, k)],
            ])

    for j in range(J_ARM, J_TOP):
        for idx_k in range(len(front_ks) - 1):
            k0, k1 = front_ks[idx_k], front_ks[idx_k + 1]
            quads.append([idx_map[(j, k0)], idx_map[(j, k1)], idx_map[(j + 1, k1)], idx_map[(j + 1, k0)]])
        for idx_k in range(len(back_ks) - 1):
            k0, k1 = back_ks[idx_k], back_ks[idx_k + 1]
            quads.append([idx_map[(j, k0)], idx_map[(j, k1)], idx_map[(j + 1, k1)], idx_map[(j + 1, k0)]])

    arm_l_loop = (
        [idx_map[(J_ARM, k)] for k in range(11, 21)]
        + [idx_map[(j, 21)] for j in range(J_ARM, J_TOP)]
        + [idx_map[(j, 11)] for j in range(J_TOP, J_ARM, -1)]
    )
    arm_r_loop = (
        [idx_map[(J_ARM, k)] for k in range(43, 53)]
        + [idx_map[(j, 53)] for j in range(J_ARM, J_TOP)]
        + [idx_map[(j, 43)] for j in range(J_TOP, J_ARM, -1)]
    )

    def add_sleeve(arm_loop: List[int], side_sign: float) -> Tuple[List[int], np.ndarray]:
        arm_pts = np.array([vert_list[idx] for idx in arm_loop], dtype=np.float64)
        arm_center = np.mean(arm_pts, axis=0)
        n_seg = len(arm_loop)

        drop_rad = math.radians(spec.sleeve_drop_deg)
        fwd_rad = math.radians(spec.sleeve_fwd_deg)
        axis_dir = np.array([
            side_sign * math.cos(drop_rad) * math.cos(fwd_rad),
            -math.sin(drop_rad),
            math.sin(fwd_rad) * math.cos(drop_rad),
        ], dtype=np.float64)
        axis_dir /= np.linalg.norm(axis_dir)

        world_up = np.array([0.0, 1.0, 0.0])
        fwd_vec = np.cross(axis_dir, world_up)
        if side_sign < 0:
            fwd_vec = -fwd_vec
        fwd_vec /= np.linalg.norm(fwd_vec)
        up_vec = np.cross(fwd_vec, axis_dir) if side_sign > 0 else np.cross(axis_dir, fwd_vec)
        up_vec /= np.linalg.norm(up_vec)

        angles = [math.atan2(np.dot(p - arm_center, up_vec), np.dot(p - arm_center, fwd_vec)) for p in arm_pts]
        n_sleeve_rings = 16 if spec.is_hoodie else 10

        prev_ring = arm_loop
        last_center = arm_center
        prev_center = arm_center

        for s_step in range(1, n_sleeve_rings + 1):
            t = s_step / n_sleeve_rings
            bend = math.sin(t * math.pi) * spec.elbow_bend
            ring_center = arm_center + axis_dir * (spec.sleeve_len * t) + np.array([0.0, 0.0, bend])
            ring_center[1] -= 0.007 * math.sin(t * math.pi)

            cur_ring: List[int] = []
            for i in range(n_seg):
                p0 = arm_pts[i]
                ang = angles[i]
                cuff_pt = (
                    ring_center
                    + fwd_vec * (spec.sleeve_end_rz * math.cos(ang))
                    + up_vec * (spec.sleeve_end_rx * math.sin(ang))
                )
                p_start_shifted = p0 + (ring_center - arm_center)
                ease = smoothstep(0.0, 0.88, t)
                pt = p_start_shifted * (1.0 - ease) + cuff_pt * ease

                if spec.is_hoodie and t > 0.58:
                    stack = 0.0040 * math.sin((t - 0.58) * 24.0) * math.sin(ang * 2.0 + 0.7)
                    radial = pt - ring_center
                    pt = pt + radial / (np.linalg.norm(radial) + 1e-6) * stack

                idx = len(vert_list)
                vert_list.append(pt)
                uv_list.append([i / n_seg, 0.25 + 0.5 * t])
                cur_ring.append(idx)

            for i in range(n_seg):
                i_next = (i + 1) % n_seg
                quads.append([prev_ring[i], prev_ring[i_next], cur_ring[i_next], cur_ring[i]])

            prev_center = last_center
            last_center = ring_center
            prev_ring = cur_ring

        end_dir = last_center - prev_center
        end_dir /= np.linalg.norm(end_dir) + 1e-9
        return prev_ring, end_dir

    cuff_l_loop, dir_l = add_sleeve(arm_l_loop, +1.0)
    cuff_r_loop, dir_r = add_sleeve(arm_r_loop, -1.0)

    hem_loop = [idx_map[(0, k)] for k in range(N_THETA)]
    neck_loop = (
        [idx_map[(J_TOP, 7)]]
        + [idx_map[(J_TOP, k)] for k in range(6, -1, -1)]
        + [idx_map[(J_TOP, k)] for k in range(63, 56, -1)]
        + [idx_map[(J_TOP, k)] for k in range(38, 25, -1)]
    )
    cuff_r_loop_oriented = list(reversed(cuff_r_loop))

    return (
        np.array(vert_list, dtype=np.float64),
        np.array(quads, dtype=np.int32),
        np.array(uv_list, dtype=np.float64),
        hem_loop,
        neck_loop,
        cuff_l_loop,
        cuff_r_loop_oriented,
        dir_l,
        dir_r,
    )


# ---------------------------------------------------------------------------
# Garment Details (Ribbed Collar, Ribbed Cuffs/Waistband, Hood, Pocket, Cords)
# ---------------------------------------------------------------------------

def build_ribbed_band_from_loop(
    base_pts: np.ndarray,
    extrude_vec: np.ndarray,
    radial_scale_profile: Sequence[float],
    rib_count: int = 36,
    rib_amp: float = 0.0012,
    thickness: float = 0.003,
) -> SubMesh:
    """Build a ribbed closed loop band perpendicular to the extrude axis."""
    n_orig = len(base_pts)
    n_fine = max(n_orig * 3, rib_count * 2)
    fine_base = np.zeros((n_fine, 3), dtype=np.float64)
    for i in range(n_fine):
        f = (i / n_fine) * n_orig
        i0 = int(math.floor(f)) % n_orig
        i1 = (i0 + 1) % n_orig
        frac = f - math.floor(f)
        fine_base[i] = base_pts[i0] * (1.0 - frac) + base_pts[i1] * frac

    center = np.mean(fine_base, axis=0)
    ext_len = np.linalg.norm(extrude_vec) + 1e-9
    ext_axis = extrude_vec / ext_len

    rings = []
    n_steps = len(radial_scale_profile)
    for s_idx, scale in enumerate(radial_scale_profile):
        t = s_idx / max(1, n_steps - 1)
        ring = np.zeros_like(fine_base)
        for i in range(n_fine):
            rel = fine_base[i] - center
            # Separate component along extrude axis vs perpendicular radial plane!
            axial = np.dot(rel, ext_axis) * ext_axis
            radial = rel - axial
            r_len = np.linalg.norm(radial) + 1e-9
            rib = rib_amp * math.sin(2.0 * math.pi * rib_count * (i / n_fine)) * math.sin(t * math.pi * 0.85 + 0.15)
            pt = center + axial + radial * scale + extrude_vec * t + (radial / r_len) * rib
            ring[i] = pt
        rings.append(ring)

    verts, quads, _, uvs = make_tube_mesh(rings, closed_u=True)
    start_loop = list(range(0, n_fine))
    end_loop = list(reversed(range((n_steps - 1) * n_fine, n_steps * n_fine)))
    v_thick, q_thick, uv_thick = thicken_shell(
        verts, quads, uvs, [start_loop, end_loop], thickness=thickness, inset_normals=True
    )
    return SubMesh("ribbed_band", v_thick, q_thick, np.zeros((0, 3), dtype=np.int32), uv_thick, "rib_knit")


def get_tshirt_spec() -> GarmentSpec:
    return GarmentSpec(
        hem_y=0.015,
        waist_y=0.25,
        arm_bot_y=0.435,
        chest_y=0.47,
        arm_top_y=0.648,
        neck_front_y=0.626,
        neck_back_y=0.706,
        neck_side_y=0.718,
        rx_hem=0.244,
        rx_waist=0.228,
        rx_chest=0.252,
        rx_shoulder=0.242,
        rz_hem=0.118,
        rz_waist=0.106,
        rz_chest=0.128,
        neck_rx=0.092,
        neck_rz_front=0.088,
        neck_rz_back=0.074,
        sleeve_len=0.205,
        sleeve_drop_deg=40.0,
        sleeve_fwd_deg=5.0,
        sleeve_end_rx=0.084,
        sleeve_end_rz=0.072,
        elbow_bend=0.0,
        is_hoodie=False,
        thickness=0.0026,
    )


def get_hoodie_spec() -> GarmentSpec:
    return GarmentSpec(
        hem_y=0.068,
        waist_y=0.23,
        arm_bot_y=0.415,
        chest_y=0.46,
        arm_top_y=0.628,
        neck_front_y=0.648,
        neck_back_y=0.706,
        neck_side_y=0.718,
        rx_hem=0.278,
        rx_waist=0.285,
        rx_chest=0.298,
        rx_shoulder=0.292,
        rz_hem=0.144,
        rz_waist=0.148,
        rz_chest=0.156,
        neck_rx=0.105,
        neck_rz_front=0.095,
        neck_rz_back=0.082,
        sleeve_len=0.535,
        sleeve_drop_deg=55.0,
        sleeve_fwd_deg=9.0,
        sleeve_end_rx=0.068,
        sleeve_end_rz=0.064,
        elbow_bend=0.022,
        is_hoodie=True,
        thickness=0.0038,
    )


def build_tshirt() -> Tuple[List[SubMesh], Dict[str, MaterialDef]]:
    spec = get_tshirt_spec()

    verts, quads, uvs, hem_loop, neck_loop, cuff_l, cuff_r, _, _ = build_torso_and_sleeves(spec)
    v_thick, q_thick, uv_thick = thicken_shell(
        verts, quads, uvs, [hem_loop, neck_loop, cuff_l, cuff_r], thickness=spec.thickness, inset_normals=True
    )
    body_mesh = SubMesh("TShirt_Body", v_thick, q_thick, np.zeros((0, 3), dtype=np.int32), uv_thick, "tshirt_cotton")

    collar_mesh = build_ribbed_band_from_loop(
        base_pts=verts[neck_loop],
        extrude_vec=np.array([0.0, 0.022, 0.0]),
        radial_scale_profile=[1.02, 1.035, 1.01, 0.97],
        rib_count=48,
        rib_amp=0.0007,
        thickness=0.0032,
    )
    collar_mesh.name = "TShirt_Collar"
    collar_mesh.material_name = "tshirt_collar_rib"

    hem_mesh = build_ribbed_band_from_loop(
        base_pts=verts[hem_loop] + np.array([0.0, 0.003, 0.0]),
        extrude_vec=np.array([0.0, -0.018, 0.0]),
        radial_scale_profile=[1.002, 1.004, 1.001],
        rib_count=0,
        rib_amp=0.0,
        thickness=0.0028,
    )
    hem_mesh.name = "TShirt_Hem"
    hem_mesh.material_name = "tshirt_cotton"

    materials = {
        "tshirt_cotton": MaterialDef("tshirt_cotton", (0.91, 0.89, 0.85, 1.0), roughness=0.88),
        "tshirt_collar_rib": MaterialDef("tshirt_collar_rib", (0.87, 0.85, 0.81, 1.0), roughness=0.90),
    }
    return [body_mesh, collar_mesh, hem_mesh], materials


def build_hoodie_hood_pattern(spec: GarmentSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Build the open single-layer quad hood pattern plus neck-base and face-brim vertex lists
    so Blender Cloth Physics can drape it naturally before Solidify.
    """
    N_U = 33
    N_V = 18
    rings: List[np.ndarray] = []

    for j in range(N_V):
        v = j / (N_V - 1)
        ring = np.zeros((N_U, 3), dtype=np.float64)

        for i in range(N_U):
            u = i / (N_U - 1)
            s = (u - 0.5) * 2.0
            abs_s = abs(s)
            sign_s = 1.0 if s >= 0 else -1.0

            ang_neck = (1.0 - abs_s) * math.pi
            x_neck = sign_s * (0.024 * (abs_s ** 2) + (spec.neck_rx + 0.008) * math.sin(ang_neck))
            z_neck = 0.098 * math.cos(ang_neck)
            if z_neck < 0:
                z_neck *= (spec.neck_rz_back + 0.006) / 0.098
            y_neck = (spec.neck_back_y - 0.014) + 0.016 * math.sin(ang_neck) - 0.042 * (abs_s ** 3.2)

            ang_face = abs_s * math.pi
            x_face = sign_s * (0.112 * math.sin(ang_face * 0.92) + 0.024 * (abs_s ** 4))
            y_face = 0.820 + 0.135 * math.cos(ang_face)
            z_face = 0.108 + 0.008 * math.sin(ang_face)

            back_dome = math.sin(v * math.pi) * math.cos(abs_s * math.pi * 0.5)
            side_dome = math.sin(v * math.pi) * math.sin(abs_s * math.pi)

            x = x_neck * (1.0 - v) + x_face * v + sign_s * 0.036 * side_dome
            y = y_neck * (1.0 - v) + y_face * v + 0.055 * back_dome
            z = z_neck * (1.0 - v) + z_face * v - 0.145 * back_dome

            z -= 0.009 * math.sin(v * math.pi * 2.5) * math.exp(-(abs_s ** 2) / 0.15) * (1.0 - v)
            seam = 0.0025 * math.exp(-(abs_s ** 2) / 0.003) * math.sin(v * math.pi)
            y += seam * 0.7
            z -= seam * 0.7

            if v > 0.88:
                brim = (v - 0.88) / 0.12
                z += 0.004 * brim
                x += sign_s * 0.003 * math.sin(brim * math.pi) * math.sin(ang_face)

            ring[i] = [x, y, z]
        rings.append(ring)

    verts, quads, _, uvs = make_tube_mesh(rings, closed_u=False)
    quads = quads[:, ::-1]

    bot_loop = list(range(0, N_U))
    right_loop = [j * N_U + (N_U - 1) for j in range(1, N_V)]
    top_loop = [(N_V - 1) * N_U + i for i in range(N_U - 2, -1, -1)]
    left_loop = [j * N_U + 0 for j in range(N_V - 2, 0, -1)]
    boundary = list(reversed(bot_loop + right_loop + top_loop + left_loop))
    face_rim = [(N_V - 1) * N_U + i for i in range(N_U)]
    return verts, quads, uvs, boundary, bot_loop + face_rim


def build_hoodie_hood(spec: GarmentSpec) -> SubMesh:
    verts, quads, uvs, boundary, _ = build_hoodie_hood_pattern(spec)
    v_thick, q_thick, uv_thick = thicken_shell(
        verts, quads, uvs, [boundary], thickness=0.0044, inset_normals=True
    )
    return SubMesh("Hoodie_Hood", v_thick, q_thick, np.zeros((0, 3), dtype=np.int32), uv_thick, "hoodie_fleece")


def build_kangaroo_pocket(spec: GarmentSpec) -> SubMesh:
    NU = 20
    NV = 12
    y_bot = spec.hem_y + 0.003
    y_top = 0.295
    y_slope_start = 0.175

    rings: List[np.ndarray] = []
    for j in range(NV):
        v = j / (NV - 1)
        y = y_bot + (y_top - y_bot) * v
        half_w = 0.178 if y <= y_slope_start else 0.178 - 0.066 * (((y - y_slope_start) / (y_top - y_slope_start)) ** 0.92)

        if y <= spec.waist_y:
            tw = smoothstep(spec.hem_y, spec.waist_y, y)
            rx = spec.rx_hem + (spec.rx_waist - spec.rx_hem) * tw
            rz = spec.rz_hem + (spec.rz_waist - spec.rz_hem) * tw
            puff = math.sin(min(1.0, max(0.0, (y - spec.hem_y) / 0.14)) * math.pi) * 0.017
            rx += puff
            rz += puff * 0.65
        else:
            tw = smoothstep(spec.waist_y, spec.chest_y, y)
            rx = spec.rx_waist + (spec.rx_chest - spec.rx_waist) * tw
            rz = spec.rz_waist + (spec.rz_chest - spec.rz_waist) * tw

        ring = np.zeros((NU, 3), dtype=np.float64)
        for i in range(NU):
            u = -1.0 + 2.0 * (i / (NU - 1))
            x = u * half_w
            x_ratio = min(0.95, abs(x) / rx)
            z_torso = rz * ((1.0 - (x_ratio ** 2.55)) ** (1.0 / 2.55))

            edge_tuck = 1.0 - (abs(u) ** 3.5) * 0.45
            v_bulge = math.sin(v * math.pi) * 0.008 + 0.007
            if y > y_slope_start and abs(u) > 0.78:
                v_bulge += 0.006 * ((abs(u) - 0.78) / 0.22)
            z = z_torso + v_bulge * edge_tuck + 0.0025
            ring[i] = [x, y, z]
        rings.append(ring)

    verts, quads, _, uvs = make_tube_mesh(rings, closed_u=False)
    bot_loop = list(range(0, NU))
    right_loop = [j * NU + (NU - 1) for j in range(1, NV)]
    top_loop = [(NV - 1) * NU + i for i in range(NU - 2, -1, -1)]
    left_loop = [j * NU + 0 for j in range(NV - 2, 0, -1)]
    boundary = bot_loop + right_loop + top_loop + left_loop

    v_thick, q_thick, uv_thick = thicken_shell(
        verts, quads, uvs, [boundary], thickness=0.0036, inset_normals=True
    )
    return SubMesh("Hoodie_KangarooPocket", v_thick, q_thick, np.zeros((0, 3), dtype=np.int32), uv_thick, "hoodie_fleece")


def build_drawstring_and_aglet(side_sign: float, spec: GarmentSpec) -> List[SubMesh]:
    n_path = 20
    n_circ = 10
    r_cord = 0.0036
    r_aglet = 0.0046

    y_start = spec.neck_front_y + 0.014
    y_end = 0.445
    x_start = side_sign * 0.026

    pts = []
    for k in range(n_path):
        t = k / (n_path - 1)
        y = y_start + (y_end - y_start) * t
        x = x_start + side_sign * (0.009 * math.sin(t * math.pi * 0.8) + 0.0025 * math.sin(t * math.pi * 2.0))
        theta = math.atan2(x, spec.rz_chest)
        _, z_surf = eval_torso_profile(spec, min(y, spec.chest_y), theta * 0.55)
        if y > spec.arm_bot_y:
            # Follow upper chest slope smoothly toward neck_rz_front
            t_up = smoothstep(spec.arm_bot_y, spec.neck_front_y + 0.014, y)
            z_surf = z_surf * (1.0 - t_up) + spec.neck_rz_front * t_up
        z = z_surf + 0.0055 + 0.002 * math.sin(t * math.pi)
        pts.append(np.array([x, y, z], dtype=np.float64))

    def sweep_tube(path_pts: List[np.ndarray], radius: float, cap_s: bool, cap_e: bool):
        rings = []
        for idx, p in enumerate(path_pts):
            if idx == 0:
                tang = path_pts[1] - path_pts[0]
            elif idx == len(path_pts) - 1:
                tang = path_pts[-1] - path_pts[-2]
            else:
                tang = path_pts[idx + 1] - path_pts[idx - 1]
            tang /= np.linalg.norm(tang) + 1e-9
            n1 = np.cross(tang, np.array([1.0, 0.0, 0.0]))
            n1 /= np.linalg.norm(n1) + 1e-9
            n2 = np.cross(tang, n1)
            ring = np.zeros((n_circ, 3), dtype=np.float64)
            for c in range(n_circ):
                ang = (c / n_circ) * 2.0 * math.pi
                ring[c] = p + (n1 * math.cos(ang) + n2 * math.sin(ang)) * radius
            rings.append(ring)
        return make_tube_mesh(rings, closed_u=True, cap_start=cap_s, cap_end=cap_e)

    cord_pts = pts[:-2]
    aglet_pts = [pts[-3], pts[-2], pts[-1], pts[-1] + np.array([0.0, -0.016, 0.001])]

    vc, qc, tc, uvc = sweep_tube(cord_pts, r_cord, True, False)
    va, qa, ta, uva = sweep_tube(aglet_pts, r_aglet, True, True)

    tag = "L" if side_sign > 0 else "R"
    return [
        SubMesh(f"Hoodie_Drawstring_{tag}", vc, qc, tc, uvc, "hoodie_drawstring"),
        SubMesh(f"Hoodie_Aglet_{tag}", va, qa, ta, uva, "hoodie_metal"),
    ]


def build_hoodie() -> Tuple[List[SubMesh], Dict[str, MaterialDef]]:
    spec = get_hoodie_spec()

    verts, quads, uvs, hem_loop, neck_loop, cuff_l, cuff_r, dir_l, dir_r = build_torso_and_sleeves(spec)

    v_thick, q_thick, uv_thick = thicken_shell(
        verts, quads, uvs, [hem_loop, neck_loop, cuff_l, cuff_r], thickness=spec.thickness, inset_normals=True
    )
    body_mesh = SubMesh("Hoodie_Body", v_thick, q_thick, np.zeros((0, 3), dtype=np.int32), uv_thick, "hoodie_fleece")

    waistband = build_ribbed_band_from_loop(
        base_pts=verts[hem_loop] + np.array([0.0, 0.005, 0.0]),
        extrude_vec=np.array([0.0, -0.073, 0.0]),
        radial_scale_profile=[0.95, 0.91, 0.89, 0.90],
        rib_count=68,
        rib_amp=0.0014,
        thickness=0.0042,
    )
    waistband.name = "Hoodie_Waistband"
    waistband.material_name = "hoodie_rib"

    cuff_l_mesh = build_ribbed_band_from_loop(
        base_pts=verts[cuff_l] - dir_l * 0.005,
        extrude_vec=dir_l * 0.067,
        radial_scale_profile=[0.95, 0.86, 0.82, 0.84],
        rib_count=34,
        rib_amp=0.0012,
        thickness=0.0040,
    )
    cuff_l_mesh.name = "Hoodie_Cuff_L"
    cuff_l_mesh.material_name = "hoodie_rib"

    cuff_r_mesh = build_ribbed_band_from_loop(
        base_pts=verts[cuff_r] - dir_r * 0.005,
        extrude_vec=dir_r * 0.067,
        radial_scale_profile=[0.95, 0.86, 0.82, 0.84],
        rib_count=34,
        rib_amp=0.0012,
        thickness=0.0040,
    )
    cuff_r_mesh.name = "Hoodie_Cuff_R"
    cuff_r_mesh.material_name = "hoodie_rib"

    hood_mesh = build_hoodie_hood(spec)
    pocket_mesh = build_kangaroo_pocket(spec)
    cords_l = build_drawstring_and_aglet(+1.0, spec)
    cords_r = build_drawstring_and_aglet(-1.0, spec)

    materials = {
        "hoodie_fleece": MaterialDef("hoodie_fleece", (0.09, 0.10, 0.12, 1.0), roughness=0.92),
        "hoodie_rib": MaterialDef("hoodie_rib", (0.075, 0.085, 0.105, 1.0), roughness=0.94),
        "hoodie_drawstring": MaterialDef("hoodie_drawstring", (0.86, 0.84, 0.80, 1.0), roughness=0.85),
        "hoodie_metal": MaterialDef("hoodie_metal", (0.72, 0.73, 0.75, 1.0), roughness=0.32, metallic=0.88),
    }

    parts = [body_mesh, waistband, cuff_l_mesh, cuff_r_mesh, hood_mesh, pocket_mesh] + cords_l + cords_r
    return parts, materials


# ---------------------------------------------------------------------------
# Exporters: OBJ+MTL (quads preserved) and Binary glTF 2.0 (.glb)
# ---------------------------------------------------------------------------

def translate_submeshes(meshes: List[SubMesh], offset: Tuple[float, float, float]) -> List[SubMesh]:
    off = np.array(offset, dtype=np.float64)
    return [
        SubMesh(m.name, m.vertices + off, m.quads.copy(), m.tris.copy(), m.uvs.copy(), m.material_name)
        for m in meshes
    ]


def export_obj(meshes: List[SubMesh], materials: Dict[str, MaterialDef], obj_path: Path) -> None:
    mtl_path = obj_path.with_suffix(".mtl")
    with mtl_path.open("w", encoding="utf-8") as fm:
        fm.write("# Material library generated by garment_builder.py\n")
        for mat in materials.values():
            r, g, b, _ = mat.base_color
            fm.write(f"newmtl {mat.name}\nKd {r:.4f} {g:.4f} {b:.4f}\nKa 0.0500 0.0500 0.0500\n")
            spec = 0.08 * (1.0 - mat.roughness) + 0.5 * mat.metallic
            ns = max(5.0, (1.0 - mat.roughness) * 400.0)
            fm.write(f"Ks {spec:.4f} {spec:.4f} {spec:.4f}\nNs {ns:.2f}\nd 1.0\nillum 2\n\n")

    with obj_path.open("w", encoding="utf-8") as fo:
        fo.write(f"# Quad-topology 3D garment export\nmtllib {mtl_path.name}\n\n")
        v_offset = 1
        for m in meshes:
            fo.write(f"o {m.name}\nusemtl {m.material_name}\n")
            norms = m.normals
            for vx, vy, vz in m.vertices:
                fo.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for u, v in m.uvs:
                fo.write(f"vt {u:.6f} {v:.6f}\n")
            for nx, ny, nz in norms:
                fo.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for q in m.quads:
                a, b, c, d = q + v_offset
                fo.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c} {d}/{d}/{d}\n")
            for t in m.tris:
                a, b, c = t + v_offset
                fo.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
            fo.write("\n")
            v_offset += len(m.vertices)


def export_glb(meshes: List[SubMesh], materials: Dict[str, MaterialDef], glb_path: Path) -> None:
    mat_list = list(materials.values())
    mat_index = {m.name: i for i, m in enumerate(mat_list)}
    gltf_materials = [
        {
            "name": m.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(c) for c in m.base_color],
                "metallicFactor": float(m.metallic),
                "roughnessFactor": float(m.roughness),
            },
            "doubleSided": False,
        }
        for m in mat_list
    ]

    bin_chunks: List[bytes] = []
    buffer_views, accessors, gltf_meshes, gltf_nodes = [], [], [], []
    byte_cursor = 0

    def append_bin(arr: np.ndarray, target: int) -> int:
        nonlocal byte_cursor
        raw = arr.tobytes()
        pad = (4 - (len(raw) % 4)) % 4
        if pad:
            raw += b"\x00" * pad
        bv_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_cursor, "byteLength": len(arr.tobytes()), "target": target})
        bin_chunks.append(raw)
        byte_cursor += len(raw)
        return bv_idx

    for m_idx, m in enumerate(meshes):
        tri_list = []
        if len(m.quads) > 0:
            q = m.quads
            tri_list.append(q[:, [0, 1, 2]])
            tri_list.append(q[:, [0, 2, 3]])
        if len(m.tris) > 0:
            tri_list.append(m.tris)
        indices = np.vstack(tri_list).astype(np.uint32).ravel()

        pos = m.vertices.astype(np.float32)
        nrm = m.normals.astype(np.float32)
        uv = np.column_stack([m.uvs[:, 0], 1.0 - m.uvs[:, 1]]).astype(np.float32)

        bv_pos, bv_nrm, bv_uv, bv_idx = append_bin(pos, 34962), append_bin(nrm, 34962), append_bin(uv, 34962), append_bin(indices, 34963)
        acc_pos = len(accessors)
        accessors.append({"bufferView": bv_pos, "componentType": 5126, "count": len(pos), "type": "VEC3",
                          "min": [float(x) for x in pos.min(axis=0)], "max": [float(x) for x in pos.max(axis=0)]})
        acc_nrm = len(accessors)
        accessors.append({"bufferView": bv_nrm, "componentType": 5126, "count": len(nrm), "type": "VEC3"})
        acc_uv = len(accessors)
        accessors.append({"bufferView": bv_uv, "componentType": 5126, "count": len(uv), "type": "VEC2"})
        acc_idx = len(accessors)
        accessors.append({"bufferView": bv_idx, "componentType": 5125, "count": len(indices), "type": "SCALAR"})

        gltf_meshes.append({
            "name": m.name,
            "primitives": [{"attributes": {"POSITION": acc_pos, "NORMAL": acc_nrm, "TEXCOORD_0": acc_uv},
                            "indices": acc_idx, "material": mat_index[m.material_name], "mode": 4}],
        })
        gltf_nodes.append({"name": m.name, "mesh": m_idx})

    gltf_doc = {
        "asset": {"version": "2.0", "generator": "garment_builder.py"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(gltf_nodes)))}],
        "nodes": gltf_nodes,
        "meshes": gltf_meshes,
        "materials": gltf_materials,
        "bufferViews": buffer_views,
        "accessors": accessors,
        "buffers": [{"byteLength": byte_cursor}],
    }
    json_bytes = json.dumps(gltf_doc, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - (len(json_bytes) % 4)) % 4)
    bin_bytes = b"".join(bin_chunks)
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with glb_path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_len))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        f.write(bin_bytes)


# ---------------------------------------------------------------------------
# Verification Renderer
# ---------------------------------------------------------------------------

def render_turntable_preview(meshes: List[SubMesh], materials: Dict[str, MaterialDef], out_png: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    views = [
        ("Front View", 0.0, 8.0),
        ("Three-Quarter View", 32.0, 14.0),
        ("Back View", 180.0, 10.0),
        ("Side Profile", 90.0, 8.0),
    ]
    W, H = 640, 640
    canvas = Image.new("RGB", (W * 2, H * 2), (242, 241, 237))
    draw_all = ImageDraw.Draw(canvas)

    all_v = np.vstack([m.vertices for m in meshes])
    center = (all_v.min(axis=0) + all_v.max(axis=0)) * 0.5
    span = float(np.max(all_v.max(axis=0) - all_v.min(axis=0)))

    light1 = np.array([0.45, 0.72, 0.68])
    light1 /= np.linalg.norm(light1)
    light2 = np.array([-0.6, 0.25, -0.5])
    light2 /= np.linalg.norm(light2)

    for idx_v, (label, yaw_deg, pitch_deg) in enumerate(views):
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        Ry = np.array([[math.cos(yaw), 0.0, -math.sin(yaw)], [0.0, 1.0, 0.0], [math.sin(yaw), 0.0, math.cos(yaw)]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(pitch), math.sin(pitch)], [0.0, -math.sin(pitch), math.cos(pitch)]])
        R = Rx @ Ry

        panel = Image.new("RGB", (W, H), (242, 241, 237))
        pdraw = ImageDraw.Draw(panel)
        scale = (W * 0.80) / max(span, 1e-6)
        faces_to_draw = []

        for m in meshes:
            color = np.array(materials[m.material_name].base_color[:3]) * 255.0
            v_rot = (m.vertices - center) @ R.T
            if len(m.quads) > 0:
                q_pts = v_rot[m.quads]
                fn = np.cross(q_pts[:, 2] - q_pts[:, 0], q_pts[:, 3] - q_pts[:, 1])
                fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
                fn = fn / np.where(fn_len < 1e-12, 1.0, fn_len)
                vis = fn[:, 2] > -0.05
                q_pts, fn = q_pts[vis], fn[vis]
                z_depth = np.mean(q_pts[:, :, 2], axis=1)
                for qi in range(len(q_pts)):
                    n = fn[qi]
                    kd = max(0.0, float(np.dot(n, light1))) * 0.72 + max(0.0, float(np.dot(n, light2))) * 0.22 + 0.22
                    shade = np.clip(color * min(1.08, kd), 0, 255).astype(int)
                    pts_2d = [(float(p[0] * scale + W * 0.5), float(-p[1] * scale + H * 0.52)) for p in q_pts[qi]]
                    faces_to_draw.append((float(z_depth[qi]), pts_2d, tuple(shade.tolist())))

            if len(m.tris) > 0:
                t_pts = v_rot[m.tris]
                fn = np.cross(t_pts[:, 1] - t_pts[:, 0], t_pts[:, 2] - t_pts[:, 0])
                fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
                fn = fn / np.where(fn_len < 1e-12, 1.0, fn_len)
                vis = fn[:, 2] > -0.05
                t_pts, fn = t_pts[vis], fn[vis]
                z_depth = np.mean(t_pts[:, :, 2], axis=1)
                for ti in range(len(t_pts)):
                    n = fn[ti]
                    kd = max(0.0, float(np.dot(n, light1))) * 0.72 + max(0.0, float(np.dot(n, light2))) * 0.22 + 0.22
                    shade = np.clip(color * min(1.08, kd), 0, 255).astype(int)
                    pts_2d = [(float(p[0] * scale + W * 0.5), float(-p[1] * scale + H * 0.52)) for p in t_pts[ti]]
                    faces_to_draw.append((float(z_depth[ti]), pts_2d, tuple(shade.tolist())))

        faces_to_draw.sort(key=lambda item: item[0])
        for _, pts_2d, col in faces_to_draw:
            pdraw.polygon(pts_2d, fill=col)

        pdraw.text((22, 20), label, fill=(35, 37, 42))
        canvas.paste(panel, ((idx_v % 2) * W, (idx_v // 2) * H))

    draw_all.line([(W, 0), (W, H * 2)], fill=(215, 213, 208), width=2)
    draw_all.line([(0, H), (W * 2, H)], fill=(215, 213, 208), width=2)
    canvas.save(out_png)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    tshirt_meshes, tshirt_mats = build_tshirt()
    hoodie_meshes, hoodie_mats = build_hoodie()

    export_obj(tshirt_meshes, tshirt_mats, out_dir / "tshirt.obj")
    export_glb(tshirt_meshes, tshirt_mats, out_dir / "tshirt.glb")

    export_obj(hoodie_meshes, hoodie_mats, out_dir / "hoodie.obj")
    export_glb(hoodie_meshes, hoodie_mats, out_dir / "hoodie.glb")

    combined_meshes = (
        translate_submeshes(tshirt_meshes, (-0.56, 0.0, 0.0))
        + translate_submeshes(hoodie_meshes, (0.56, 0.0, 0.0))
    )
    combined_mats = {**tshirt_mats, **hoodie_mats}
    export_obj(combined_meshes, combined_mats, out_dir / "shirt_and_hoodie.obj")
    export_glb(combined_meshes, combined_mats, out_dir / "shirt_and_hoodie.glb")

    render_turntable_preview(combined_meshes, combined_mats, out_dir / "preview_render.png")

    total_quads = sum(len(m.quads) for m in combined_meshes)
    total_verts = sum(len(m.vertices) for m in combined_meshes)
    print(f"Generated 3D garments in {out_dir}")
    print(f"Combined vertices: {total_verts}, quads: {total_quads}")


if __name__ == "__main__":
    main()
