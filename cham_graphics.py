"""
High-Resolution CHAM Apparel Graphic Decal & Fabric Swatch Generator
Generates transparent RGBA PNG decals and macro fabric swatch textures matching
the ISHCMC CHAM Hoodie (350 GSM Loopback Knit) and CHAM Sports Shirt (160 GSM Breathable Mesh).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFilter = None
    ImageFont = None


def _load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/calibrib.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/calibri.ttf"]
    )
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _load_serif_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = (
        ["C:/Windows/Fonts/georgiab.ttf", "C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/palab.ttf"]
        if bold
        else ["C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/pala.ttf"]
    )
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size=size)
            except Exception:
                pass
    return _load_font(size, bold=bold)


def draw_ishcmc_badge(size: int = 512) -> Image.Image:
    """Render the circular white ISHCMC badge with navy architecture, serif title, and motto."""
    SS = 2
    W = size * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx, cy = W // 2, W // 2
    r_outer = int(W * 0.475)
    r_ring = int(W * 0.442)
    navy = (25, 58, 108, 255)
    offwhite = (249, 248, 245, 255)

    # Outer white circle & inner navy ring
    d.ellipse((cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer), fill=offwhite)
    d.ellipse((cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring), outline=navy, width=max(4, W // 95))

    # Classical school building illustration in top half
    by = int(W * 0.445)
    bw = int(W * 0.56)
    left_x = cx - bw // 2
    right_x = cx + bw // 2

    # Base ground line
    d.line((left_x - 10, by, right_x + 10, by), fill=navy, width=6)

    # Side wings
    wing_w = int(bw * 0.24)
    wing_h = int(W * 0.095)
    for wx in (left_x, right_x - wing_w):
        d.rectangle((wx, by - wing_h, wx + wing_w, by), outline=navy, width=5, fill=offwhite)
        # Pitched roof on wing
        d.polygon([(wx - 6, by - wing_h), (wx + wing_w // 2, by - wing_h - int(W * 0.025)), (wx + wing_w + 6, by - wing_h)], outline=navy, fill=offwhite)
        # 2x3 windows on wing
        for r_i in range(2):
            for c_i in range(3):
                win_x = wx + int(wing_w * (0.16 + c_i * 0.26))
                win_y = by - wing_h + int(wing_h * (0.20 + r_i * 0.42))
                d.rectangle((win_x, win_y, win_x + int(wing_w * 0.14), win_y + int(wing_h * 0.26)), fill=navy)

    # Two flanking towers
    tw_w = int(bw * 0.14)
    tw_h = int(W * 0.135)
    for tx in (left_x + wing_w - 4, right_x - wing_w - tw_w + 4):
        d.rectangle((tx, by - tw_h, tx + tw_w, by), outline=navy, width=5, fill=offwhite)
        # Pyramid roof + finial
        spire_top = (tx + tw_w // 2, by - tw_h - int(W * 0.055))
        d.polygon([(tx - 5, by - tw_h), spire_top, (tx + tw_w + 5, by - tw_h)], outline=navy, fill=offwhite)
        d.line((spire_top[0], spire_top[1], spire_top[0], spire_top[1] - int(W * 0.022)), fill=navy, width=5)
        # Arched door/window on tower
        arch_x0 = tx + int(tw_w * 0.26)
        arch_x1 = tx + int(tw_w * 0.74)
        d.rectangle((arch_x0, by - int(tw_h * 0.52), arch_x1, by), fill=navy)
        d.ellipse((arch_x0, by - int(tw_h * 0.65), arch_x1, by - int(tw_h * 0.38)), fill=navy)

    # Central pavilion + pediment + clock/bell spire
    c_w = int(bw * 0.30)
    c_h = int(W * 0.125)
    cx0, cx1 = cx - c_w // 2, cx + c_w // 2
    # Central spire behind pediment
    sp_w = int(c_w * 0.28)
    sp_top_y = by - c_h - int(W * 0.165)
    d.rectangle((cx - sp_w // 2, by - c_h - int(W * 0.095), cx + sp_w // 2, by - c_h), outline=navy, width=5, fill=offwhite)
    d.polygon([(cx - sp_w // 2 - 4, by - c_h - int(W * 0.095)), (cx, sp_top_y), (cx + sp_w // 2 + 4, by - c_h - int(W * 0.095))], outline=navy, fill=navy)
    # Weathervane cross at top of spire
    d.line((cx, sp_top_y - int(W * 0.032), cx, sp_top_y), fill=navy, width=5)
    d.line((cx - int(W * 0.016), sp_top_y - int(W * 0.020), cx + int(W * 0.016), sp_top_y - int(W * 0.020)), fill=navy, width=5)

    # Central facade & pediment
    d.rectangle((cx0, by - c_h, cx1, by), outline=navy, width=5, fill=offwhite)
    d.polygon([(cx0 - 8, by - c_h), (cx, by - c_h - int(W * 0.062)), (cx1 + 8, by - c_h)], outline=navy, width=5, fill=offwhite)
    d.polygon([(cx0 + 14, by - c_h - 4), (cx, by - c_h - int(W * 0.044)), (cx1 - 14, by - c_h - 4)], fill=navy)

    # Three central arched doorways
    for a_i in (-1, 0, 1):
        ax = cx + a_i * int(c_w * 0.30)
        aw = int(c_w * 0.18)
        ah = int(c_h * 0.68)
        d.rectangle((ax - aw // 2, by - ah, ax + aw // 2, by), fill=navy)
        d.ellipse((ax - aw // 2, by - ah - aw // 2, ax + aw // 2, by - ah + aw // 2), fill=navy)

    # Typography
    f_title1 = _load_serif_font(int(W * 0.068), bold=True)
    f_title2 = _load_serif_font(int(W * 0.066), bold=True)
    f_motto = _load_font(int(W * 0.040), bold=True)

    d.text((cx, int(W * 0.515)), "International School", font=f_title1, fill=navy, anchor="mm")
    d.text((cx, int(W * 0.592)), "HO CHI MINH CITY", font=f_title2, fill=navy, anchor="mm")
    d.line((cx - int(W * 0.33), int(W * 0.648), cx + int(W * 0.33), int(W * 0.648)), fill=(160, 175, 195, 255), width=3)
    d.text((cx, int(W * 0.692)), "Energized · Engaged · Empowered", font=f_motto, fill=navy, anchor="mm")

    return img.resize((size, size), Image.Resampling.LANCZOS)


def draw_sakura_blossom(size: int = 420) -> Image.Image:
    """Render the 5-petal pink cherry blossom (sakura) graphic for the CHAM Hoodie left chest."""
    SS = 2
    W = size * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx, cy = W * 0.5, W * 0.52
    crimson = (196, 43, 68, 255)
    pink_fill = (252, 178, 195, 255)
    inner_pink = (238, 122, 148, 255)

    # Draw 5 notched heart-shaped petals
    for p_idx in range(5):
        base_ang = math.radians(-90.0 + p_idx * 72.0)
        ca, sa = math.cos(base_ang), math.sin(base_ang)

        pts_outer = []
        pts_inner = []
        n_steps = 80
        for i in range(n_steps + 1):
            t = (i / n_steps) * 2.0 - 1.0  # -1 .. +1 across petal
            ang_local = t * math.radians(33.0)
            # Outer lobe with characteristic cherry-blossom V-notch at tip (t=0)
            notch = 0.13 * math.exp(-((t / 0.16) ** 2))
            r = W * (0.06 + (0.37 - notch) * (math.cos(t * math.pi * 0.5) ** 0.55))
            lx = r * math.cos(ang_local)
            ly = r * math.sin(ang_local)
            px = cx + lx * ca - ly * sa
            py = cy + lx * sa + ly * ca
            pts_outer.append((px, py))

            r_in = r * 0.84
            lx_i = r_in * math.cos(ang_local * 0.92)
            ly_i = r_in * math.sin(ang_local * 0.92)
            pts_inner.append((cx + lx_i * ca - ly_i * sa, cy + lx_i * sa + ly_i * ca))

        d.polygon([(cx, cy)] + pts_outer, fill=crimson)
        d.polygon([(cx, cy)] + pts_inner, fill=pink_fill)

        # Subtle inner petal vein lines
        for v_off in (-0.10, 0.0, 0.10):
            vang = base_ang + v_off
            d.line(
                (cx + W * 0.06 * math.cos(vang), cy + W * 0.06 * math.sin(vang),
                 cx + W * 0.20 * math.cos(vang), cy + W * 0.20 * math.sin(vang)),
                fill=inner_pink,
                width=5,
            )

    # Stamens radiating from center
    for s_idx in range(10):
        sang = math.radians(-90.0 + s_idx * 36.0 + 18.0)
        r_len = W * (0.12 if s_idx % 2 == 0 else 0.095)
        ex = cx + r_len * math.cos(sang)
        ey = cy + r_len * math.sin(sang)
        d.line((cx, cy, ex, ey), fill=crimson, width=6)
        dot_r = W * 0.015
        d.ellipse((ex - dot_r, ey - dot_r, ex + dot_r, ey + dot_r), fill=crimson)

    c_r = W * 0.032
    d.ellipse((cx - c_r, cy - c_r, cx + c_r, cy + c_r), fill=crimson)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def draw_smiling_sun(size: int = 420) -> Image.Image:
    """Render the smiling yellow sun graphic for the CHAM Sports Shirt left chest."""
    SS = 2
    W = size * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx, cy = W * 0.5, W * 0.5
    sun_gold = (252, 198, 56, 255)
    sun_warm = (247, 168, 48, 255)
    ray_yellow = (248, 208, 74, 255)
    face_brown = (62, 38, 28, 255)
    cheek_coral = (242, 118, 96, 215)

    # Alternating long and short rounded rays around the sun disc
    n_rays = 16
    for i in range(n_rays):
        ang = (i / n_rays) * 2.0 * math.pi
        r0 = W * 0.285
        r1 = W * (0.435 if i % 2 == 0 else 0.385)
        x0, y0 = cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)
        x1, y1 = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
        d.line((x0, y0, x1, y1), fill=ray_yellow, width=int(W * 0.028))
        cr = W * 0.014
        d.ellipse((x1 - cr, y1 - cr, x1 + cr, y1 + cr), fill=ray_yellow)

    # Outer & inner sun circle
    r_disc = int(W * 0.245)
    d.ellipse((cx - r_disc, cy - r_disc, cx + r_disc, cy + r_disc), fill=sun_warm)
    r_in = int(W * 0.228)
    d.ellipse((cx - r_in, cy - r_in - 6, cx + r_in - 6, cy + r_in), fill=sun_gold)

    # Rosy cheeks
    ch_r = int(W * 0.038)
    for sx in (-1, 1):
        chx = cx + sx * int(W * 0.115)
        chy = cy + int(W * 0.028)
        d.ellipse((chx - ch_r, chy - int(ch_r * 0.7), chx + ch_r, chy + int(ch_r * 0.7)), fill=cheek_coral)

    # Eyes
    eye_r = int(W * 0.020)
    for sx in (-1, 1):
        ex = cx + sx * int(W * 0.072)
        ey = cy - int(W * 0.025)
        d.ellipse((ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r), fill=face_brown)

    # Smile arc
    sm_r = int(W * 0.062)
    d.arc((cx - sm_r, cy - int(W * 0.02), cx + sm_r, cy + sm_r + int(W * 0.02)), start=20, end=160, fill=face_brown, width=int(W * 0.018))

    return img.resize((size, size), Image.Resampling.LANCZOS)


def draw_clothesline(width: int = 1200, height: int = 380) -> Image.Image:
    """Render the curved clothesline with 5 colorful hanging baby/kids rompers clipped by wooden pins."""
    SS = 2
    W, H = width * SS, height * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Curved white/cream clothesline string from top-left to top-right
    x_left, y_left = W * 0.05, H * 0.14
    x_right, y_right = W * 0.95, H * 0.12
    sag = H * 0.20

    def line_y(x_val: float) -> float:
        u = (x_val - x_left) / (x_right - x_left)
        return y_left * (1.0 - u) + y_right * u + sag * 4.0 * u * (1.0 - u)

    line_pts = []
    for i in range(121):
        xv = x_left + (x_right - x_left) * (i / 120.0)
        line_pts.append((xv, line_y(xv)))
    d.line(line_pts, fill=(246, 242, 234, 255), width=14)
    # Little end knots
    for kx, ky in (line_pts[0], line_pts[-1]):
        d.ellipse((kx - 12, ky - 12, kx + 12, ky + 12), fill=(198, 146, 108, 255))

    # 5 hanging outfits matching the reference:
    # 1. Coral-pink romper with cream top
    # 2. Sky-teal long-sleeve top & pants
    # 3. Cobalt-blue overalls with pink straps/cuffs
    # 4. Golden-yellow overalls with pink/cream top
    # 5. Cerulean-blue romper with pink collar trim
    outfits = [
        {"u": 0.15, "tilt": -13.0, "kind": "romper", "main": (232, 106, 122, 255), "top": (246, 224, 192, 255), "trim": (162, 74, 88, 255)},
        {"u": 0.32, "tilt": -6.0, "kind": "sweater", "main": (74, 184, 212, 255), "top": (244, 218, 176, 255), "trim": (52, 136, 168, 255)},
        {"u": 0.50, "tilt": 0.0, "kind": "overall", "main": (64, 118, 188, 255), "top": (232, 92, 124, 255), "trim": (232, 92, 124, 255)},
        {"u": 0.68, "tilt": 6.0, "kind": "overall", "main": (246, 176, 52, 255), "top": (236, 112, 144, 255), "trim": (176, 82, 112, 255)},
        {"u": 0.85, "tilt": 13.0, "kind": "romper", "main": (76, 158, 206, 255), "top": (198, 226, 244, 255), "trim": (224, 98, 126, 255)},
    ]

    for spec in outfits:
        u = spec["u"]
        cx = x_left + (x_right - x_left) * u
        cy = line_y(cx)

        # Render individual outfit on a local layer then rotate by `tilt`
        ow, oh = int(W * 0.22), int(H * 0.78)
        card = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        mx = ow // 2
        top_y = int(oh * 0.14)

        w_sh = int(ow * 0.30)
        w_hip = int(ow * 0.34)
        bot_y = int(oh * 0.84)
        crotch_y = int(oh * 0.56)

        if spec["kind"] == "sweater":
            # Long-sleeved cyan baby outfit
            pts_body = [
                (mx - w_sh, top_y),
                (mx + w_sh, top_y),
                (mx + int(ow * 0.44), int(oh * 0.72)),
                (mx + int(ow * 0.31), int(oh * 0.75)),
                (mx + int(ow * 0.24), int(oh * 0.42)),
                (mx + int(ow * 0.22), bot_y),
                (mx + int(ow * 0.04), bot_y),
                (mx, crotch_y),
                (mx - int(ow * 0.04), bot_y),
                (mx - int(ow * 0.22), bot_y),
                (mx - int(ow * 0.24), int(oh * 0.42)),
                (mx - int(ow * 0.31), int(oh * 0.75)),
                (mx - int(ow * 0.44), int(oh * 0.72)),
            ]
            cd.polygon(pts_body, fill=spec["main"])
            # Cream yoke collar
            cd.pieslice((mx - int(ow * 0.20), top_y - int(oh * 0.10), mx + int(ow * 0.20), top_y + int(oh * 0.16)), 0, 180, fill=spec["top"])
            cd.arc((mx - int(ow * 0.20), top_y - int(oh * 0.10), mx + int(ow * 0.20), top_y + int(oh * 0.16)), 0, 180, fill=spec["trim"], width=7)
        else:
            # Sleeveless/short-sleeve romper or overall with two legs
            pts_legs = [
                (mx - w_sh, top_y),
                (mx + w_sh, top_y),
                (mx + w_hip, bot_y),
                (mx + int(ow * 0.06), bot_y),
                (mx, crotch_y),
                (mx - int(ow * 0.06), bot_y),
                (mx - w_hip, bot_y),
            ]
            cd.polygon(pts_legs, fill=spec["main"])
            # Upper chest bib / yoke
            pts_bib = [
                (mx - w_sh, top_y),
                (mx + w_sh, top_y),
                (mx + int(w_sh * 1.05), int(oh * 0.38)),
                (mx - int(w_sh * 1.05), int(oh * 0.38)),
            ]
            cd.polygon(pts_bib, fill=spec["top"])
            # Neck scoop cutout
            cd.pieslice((mx - int(ow * 0.16), top_y - int(oh * 0.08), mx + int(ow * 0.16), top_y + int(oh * 0.12)), 0, 180, fill=(0, 0, 0, 0))
            cd.arc((mx - int(ow * 0.18), top_y - int(oh * 0.09), mx + int(ow * 0.18), top_y + int(oh * 0.15)), 0, 180, fill=spec["trim"], width=10)
            # Cuffs at bottom of legs
            cd.line((mx - w_hip, bot_y - 6, mx - int(ow * 0.06), bot_y - 6), fill=spec["trim"], width=12)
            cd.line((mx + int(ow * 0.06), bot_y - 6, mx + w_hip, bot_y - 6), fill=spec["trim"], width=12)

        # Two clothespins at shoulders
        for sx in (-1, 1):
            px = mx + sx * int(w_sh * 0.68)
            cd.rounded_rectangle((px - 8, top_y - 26, px + 8, top_y + 18), radius=5, fill=(214, 152, 112, 255), outline=(142, 88, 58, 255), width=3)

        rot_card = card.rotate(spec["tilt"], resample=Image.Resampling.BICUBIC, center=(mx, top_y))
        img.alpha_composite(rot_card, (int(cx - mx), int(cy - top_y)))

    return img.resize((width, height), Image.Resampling.LANCZOS)


def draw_handprint_with_heart(size: int, color_main: Tuple[int, int, int], color_dark: Tuple[int, int, int], tilt_deg: float) -> Image.Image:
    """Render a stylized painted handprint with a cutout heart in the palm."""
    SS = 2
    W = size * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx, cy = W * 0.52, W * 0.62
    rx, ry = W * 0.21, W * 0.18

    # Palm base (solid painted organic shape)
    d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(*color_dark, 255))
    d.ellipse((cx - rx * 0.94, cy - ry * 0.94, cx + rx * 0.88, cy + ry * 0.86), fill=(*color_main, 255))

    # 5 fingers (thumb on left + 4 fingers fanning above)
    fingers = [
        (-54.0, W * 0.21, W * 0.20, W * 0.044, -0.05, 0.04),  # Thumb
        (-24.0, W * 0.20, W * 0.30, W * 0.042, -0.02, -0.01), # Index
        (-4.0,  W * 0.21, W * 0.34, W * 0.043,  0.00, -0.02), # Middle
        (15.0,  W * 0.20, W * 0.31, W * 0.041,  0.02, -0.01), # Ring
        (34.0,  W * 0.19, W * 0.25, W * 0.037,  0.04,  0.01), # Pinky
    ]
    for ang_deg, r_start, r_end, f_rad, ox, oy in fingers:
        ang = math.radians(-90.0 + ang_deg)
        bx = cx + ox * W + r_start * math.cos(ang)
        by = cy + oy * W + r_start * math.sin(ang)
        tx = cx + ox * W + (r_start + r_end) * math.cos(ang)
        ty = cy + oy * W + (r_start + r_end) * math.sin(ang)
        # Draw segmented painted finger pads with a subtle joint gap near base (u ~ 0.18)
        for s_i in range(18):
            u = s_i / 17.0
            if 0.12 < u < 0.21:
                continue
            px = bx + (tx - bx) * u
            py = by + (ty - by) * u
            rad = f_rad * (1.0 - 0.10 * u) * (1.0 + 0.07 * math.sin(u * math.pi * 3.0))
            d.ellipse((px - rad, py - rad, px + rad, py + rad), fill=(*color_main, 255))

    # Soft cream-white heart in center of palm
    hx, hy = cx - W * 0.01, cy + W * 0.01
    hs = W * 0.082
    heart_pts = []
    for i in range(60):
        t = (i / 60.0) * 2.0 * math.pi
        x_h = 16.0 * (math.sin(t) ** 3)
        y_h = -(13.0 * math.cos(t) - 5.0 * math.cos(2.0 * t) - 2.0 * math.cos(3.0 * t) - math.cos(4.0 * t))
        heart_pts.append((hx + x_h * (hs / 16.0), hy + y_h * (hs / 16.0)))
    d.polygon(heart_pts, fill=(246, 241, 232, 242))

    rot = img.rotate(tilt_deg, resample=Image.Resampling.BICUBIC)
    return rot.resize((size, size), Image.Resampling.LANCZOS)


def draw_bee_with_trail(size: int = 380, dark_trail: bool = True) -> Image.Image:
    """Render the cute yellow-and-black bumblebee with blue wings and dotted flight loop."""
    SS = 2
    W = size * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    trail_col = (32, 42, 62, 230) if dark_trail else (242, 244, 248, 230)

    # Dotted flight loop below-right of the bee
    n_dots = 28
    for i in range(n_dots):
        t = i / (n_dots - 1)
        ang = t * math.pi * 1.65 - 0.4
        rx = W * (0.56 + 0.22 * math.sin(ang) + 0.06 * math.cos(ang * 2.0))
        ry = W * (0.64 + 0.16 * math.cos(ang))
        d.ellipse((rx - 6, ry - 6, rx + 6, ry + 6), fill=trail_col)

    # Bee body (tilted -18 deg)
    bee_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bee_layer)
    bx, by = int(W * 0.46), int(W * 0.36)
    bw, bh = int(W * 0.22), int(W * 0.145)

    # Two sky-blue wings above body
    wing_blue = (58, 168, 232, 255)
    wing_light = (120, 204, 248, 255)
    bd.ellipse((bx - int(bw * 0.85), by - int(bh * 1.55), bx - int(bw * 0.05), by - int(bh * 0.15)), fill=wing_blue)
    bd.ellipse((bx + int(bw * 0.02), by - int(bh * 1.62), bx + int(bw * 0.82), by - int(bh * 0.20)), fill=wing_light)

    # Stinger on left rear
    bd.polygon([(bx - bw - int(W * 0.045), by), (bx - bw + 8, by - 18), (bx - bw + 8, by + 18)], fill=(30, 28, 28, 255))

    # Yellow oval body + black stripes clipped inside body
    body_mask = Image.new("L", (W, W), 0)
    md = ImageDraw.Draw(body_mask)
    md.ellipse((bx - bw, by - bh, bx + bw, by + bh), fill=255)

    body_col = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    bcd = ImageDraw.Draw(body_col)
    bcd.ellipse((bx - bw, by - bh, bx + bw, by + bh), fill=(250, 198, 42, 255))
    # Two vertical dark stripes
    for sx in (-int(bw * 0.38), int(bw * 0.08)):
        bcd.rectangle((bx + sx - int(bw * 0.16), by - bh - 10, bx + sx + int(bw * 0.16), by + bh + 10), fill=(30, 26, 24, 255))
    # Eye on right front
    ex, ey = bx + int(bw * 0.58), by - int(bh * 0.18)
    bcd.ellipse((ex - 12, ey - 12, ex + 12, ey + 12), fill=(30, 26, 24, 255))

    bee_layer.paste(body_col, (0, 0), body_mask)
    bee_rot = bee_layer.rotate(18.0, resample=Image.Resampling.BICUBIC, center=(bx, by))
    img.alpha_composite(bee_rot)

    return img.resize((size, size), Image.Resampling.LANCZOS)


def draw_vietnam_flag(width: int = 320, height: int = 214) -> Image.Image:
    """Render the Vietnam national flag patch with a 10% transparent margin so edge UVs are invisible."""
    SS = 2
    W, H = width * SS, height * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    mx, my = int(W * 0.08), int(H * 0.08)
    # Embroidered patch border & crimson-red field
    d.rounded_rectangle((mx, my, W - mx, H - my), radius=16, fill=(210, 32, 39, 255), outline=(168, 22, 28, 255), width=8)

    cx, cy = W * 0.5, H * 0.5
    r_out = (H - 2 * my) * 0.30
    r_in = r_out * 0.382
    pts = []
    for i in range(10):
        ang = math.radians(-90.0 + i * 36.0)
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    d.polygon(pts, fill=(255, 208, 28, 255))

    return img.resize((width, height), Image.Resampling.LANCZOS)


def generate_front_decal(is_hoodie: bool, size: int = 1024) -> Image.Image:
    """
    Assemble the complete front chest graphic decal in normalized chest coordinates:
      - Viewer-left (right chest): ISHCMC circular badge
      - Viewer-right (left chest): Sakura flower (hoodie) or Smiling sun (sports shirt)
      - Mid-chest below badges: 5-outfit curved clothesline
    """
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    badge = draw_ishcmc_badge(size=int(size * 0.255))
    canvas.alpha_composite(badge, (int(size * 0.155), int(size * 0.175)))

    if is_hoodie:
        right_icon = draw_sakura_blossom(size=int(size * 0.255))
        canvas.alpha_composite(right_icon, (int(size * 0.595), int(size * 0.165)))
    else:
        right_icon = draw_smiling_sun(size=int(size * 0.275))
        canvas.alpha_composite(right_icon, (int(size * 0.580), int(size * 0.155)))

    line_w = int(size * 0.78)
    line_h = int(size * 0.26)
    clothesline = draw_clothesline(width=line_w, height=line_h)
    canvas.alpha_composite(clothesline, (int((size - line_w) * 0.5), int(size * 0.455)))

    return canvas


def generate_back_decal(dark_text: bool, size: int = 1024) -> Image.Image:
    """
    Assemble the complete CHAM back graphic decal:
      - Top-left: olive-green painted handprint with heart
      - Top-right: bumblebee with dotted flight trail
      - Center: bold uppercase 'CHAM'
      - Bottom-left: coral-pink painted handprint with heart
      - Bottom-right: sky-blue painted handprint with heart
      - Bottom subtitle: 'CHANGING LIVES THROUGH / COMPASSION'
    """
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)

    text_col = (18, 34, 58, 255) if dark_text else (246, 247, 250, 255)

    # Top-left green handprint
    hp_green = draw_handprint_with_heart(int(size * 0.29), (146, 174, 58), (98, 126, 34), tilt_deg=12.0)
    canvas.alpha_composite(hp_green, (int(size * 0.25), int(size * 0.08)))

    # Top-right bee
    bee = draw_bee_with_trail(int(size * 0.26), dark_trail=dark_text)
    canvas.alpha_composite(bee, (int(size * 0.54), int(size * 0.13)))

    # Center bold 'CHAM'
    f_cham = _load_font(int(size * 0.125), bold=True)
    d.text((int(size * 0.49), int(size * 0.425)), "CHAM", font=f_cham, fill=text_col, anchor="mm")

    # Bottom-left pink handprint
    hp_pink = draw_handprint_with_heart(int(size * 0.28), (242, 118, 146), (198, 72, 102), tilt_deg=16.0)
    canvas.alpha_composite(hp_pink, (int(size * 0.20), int(size * 0.48)))

    # Bottom-right cyan/blue handprint
    hp_blue = draw_handprint_with_heart(int(size * 0.28), (44, 156, 224), (24, 108, 176), tilt_deg=-10.0)
    canvas.alpha_composite(hp_blue, (int(size * 0.51), int(size * 0.48)))

    # Subtitle 'CHANGING LIVES THROUGH / COMPASSION'
    f_sub = _load_font(int(size * 0.036), bold=True)
    d.text((int(size * 0.50), int(size * 0.815)), "CHANGING LIVES THROUGH", font=f_sub, fill=text_col, anchor="mm")
    d.text((int(size * 0.50), int(size * 0.862)), "COMPASSION", font=f_sub, fill=text_col, anchor="mm")

    return canvas


def build_all_cham_textures(tex_dir: Path, force: bool = False) -> dict:
    tex_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hoodie_front": tex_dir / "cham_hoodie_front.png",
        "shirt_front": tex_dir / "cham_shirt_front.png",
        "back_dark": tex_dir / "cham_back_dark.png",
        "back_light": tex_dir / "cham_back_light.png",
        "vn_flag": tex_dir / "cham_vn_flag.png",
        "ishcmc_badge": tex_dir / "cham_ishcmc_badge.png",
    }
    if Image is not None and (force or not all(p.exists() for p in paths.values())):
        generate_front_decal(is_hoodie=True, size=1024).save(paths["hoodie_front"])
        generate_front_decal(is_hoodie=False, size=1024).save(paths["shirt_front"])
        generate_back_decal(dark_text=True, size=1024).save(paths["back_dark"])
        generate_back_decal(dark_text=False, size=1024).save(paths["back_light"])
        draw_vietnam_flag(width=320, height=214).save(paths["vn_flag"])
        draw_ishcmc_badge(size=512).save(paths["ishcmc_badge"])
    return paths


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "textures"
    res = build_all_cham_textures(out, force=True)
    for k, p in res.items():
        print(f"Generated {k}: {p}")
