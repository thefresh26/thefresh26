#!/usr/bin/env python3
"""Generate the animated GitHub profile banners.

Run from the repository root:
    python scripts/banner/generate.py
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/avatar.png"
ASSETS = ROOT / "assets"
LOGOS = Path(__file__).resolve().parent / "logos"
DATA = Path(__file__).resolve().parent / "data"

W, H = 1180, 610
INTRO_SECONDS = 3.2
INTRO_HOLD = 3.0    # seconds the portrait holds before the first morph
TRANSITION = 2.4    # seconds for each morph between shapes (slower = smoother)
HOLD = 2.6          # seconds each logo silhouette holds before morphing on
TRAVELLER_COUNT = 2200
SEED = 314159

# Order the portrait morphs through. Add/remove names here (and their
# silhouette in make_logos) to change the sequence — everything else
# (timing, labels, transport chain) is computed from this list.
LOGO_SEQUENCE = ["python", "supabase", "neon", "angular"]
LOOP_SECONDS = INTRO_HOLD + len(LOGO_SEQUENCE) * (TRANSITION + HOLD) + TRANSITION

ROWS = [
    ("Subject", "Santiago Escorcia"),
    ("Role", "Desarrollo · Soporte TI · Mantenimientos"),
    ("Origin", "Colombia"),
    ("Education", "Autodidacta · Vibe coding"),
    ("Status", "Building + Learning + Automating"),
    ("ToolChain", "VS Code · Git · GitHub"),
    ("Core.Lang", "Python · Kotlin"),
    ("Core.Frontend", "Angular"),
    ("Core.Backend", "FastAPI · Django"),
    ("Core.Database", "MySQL · Supabase · Neon"),
    ("Core.Infra", "Automatizacion con IA"),
    ("Grid.Mail", "thefresh334@gmail.com"),
    ("Grid.LinkedIn", "/in/santiago-escorcia-8b0204367"),
    ("Grid.GitHub", "thefresh26"),
    ("Grid.X", "@Thefresh334"),
]

THEMES = {
    "dark": {
        "bg": "#0A101F",
        "panel": "#0D1628",
        "panel2": "#101B30",
        "line": "#25344C",
        "muted": "#8291A8",
        "text": "#DDE7F5",
        "portrait": "#A78BFA",
        "chrome": "#22D3EE",
        "accent": "#10B981",
        "shadow": "#02050B",
    },
    "light": {
        "bg": "#F6F8FA",
        "panel": "#FFFFFF",
        "panel2": "#EDF3F7",
        "line": "#CBD7E1",
        "muted": "#64748B",
        "text": "#172033",
        "portrait": "#5B21B6",
        "chrome": "#0891B2",
        "accent": "#10B981",
        "shadow": "#AAB7C4",
    },
}


def make_logos() -> dict[str, Image.Image]:
    """Create clean 400px black-on-transparent silhouette sources."""
    LOGOS.mkdir(parents=True, exist_ok=True)
    size = 400
    logos: dict[str, Image.Image] = {}

    # Python: real logo silhouette (scripts/banner/logos/python.png), rebuilt
    # from the official mark's own alpha channel so the shape is authentic.
    # If that file is ever missing, fall back to a simple abstraction so the
    # script still runs.
    python_path = LOGOS / "python.png"
    if python_path.exists():
        python_logo = Image.open(python_path).convert("RGBA").resize((size, size))
    else:
        python_logo = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(python_logo)
        d.rounded_rectangle((92, 58, 302, 196), radius=64, fill="black")
        d.rounded_rectangle((98, 204, 308, 342), radius=64, fill="black")
        d.ellipse((118, 92, 156, 130), fill=(0, 0, 0, 0))
        d.ellipse((244, 270, 282, 308), fill=(0, 0, 0, 0))
    logos["python"] = python_logo

    # Angular-inspired rounded shield with an open "A" cut into it.
    angular = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(angular)
    shield = [
        (200, 34), (352, 96), (324, 298), (200, 372),
        (76, 298), (48, 96),
    ]
    d.polygon(shield, fill="black")
    d.polygon([(200, 122), (275, 300), (125, 300)], fill=(0, 0, 0, 0))
    d.rectangle((160, 236, 240, 258), fill="black")
    logos["angular"] = angular

    # Supabase-inspired lightning bolt.
    supabase = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(supabase)
    bolt = [
        (222, 18), (84, 232), (176, 232), (150, 382),
        (322, 152), (218, 152),
    ]
    d.polygon(bolt, fill="black")
    logos["supabase"] = supabase

    # Neon-inspired bold "N" glyph, built from the same thick-stroke technique
    # as the code mark.
    neon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(neon)
    stroke = 54
    d.line([(96, 322), (96, 78)], fill="black", width=stroke)
    d.line([(96, 78), (304, 322)], fill="black", width=stroke, joint="curve")
    d.line([(304, 322), (304, 78)], fill="black", width=stroke)
    logos["neon"] = neon

    for name, image in logos.items():
        image.save(LOGOS / f"{name}.png", optimize=True)
    return logos


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit Floyd-Steinberg diffusion; True means a lit pixel."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    height, width = work.shape
    for y in range(height):
        left_to_right = y % 2 == 0
        xs = range(width) if left_to_right else range(width - 1, -1, -1)
        direction = 1 if left_to_right else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < width:
                work[y, nx] += err * 7 / 16
            if y + 1 < height:
                if 0 <= x - direction < width:
                    work[y + 1, x - direction] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < width:
                    work[y + 1, nx] += err * 1 / 16
    return out


def subject_alpha(rgb: Image.Image) -> np.ndarray:
    """Build a synthetic alpha mask (0..255) separating the subject from a
    flat illustrated background, since the source avatar has no real alpha
    channel (unlike a cut-out photo). Background colour is sampled from the
    four corners, which the avatar leaves untouched."""
    arr = np.asarray(rgb).astype(float)
    corners = np.concatenate(
        [arr[0, :6], arr[-1, -6:], arr[:6, 0], arr[-6:, -1]]
    )
    bg_color = corners.mean(axis=0)
    dist = np.linalg.norm(arr - bg_color, axis=-1)
    dist = dist / (dist.max() + 1e-6)
    mask = np.clip((dist - 0.10) / 0.5, 0, 1)
    mask_img = Image.fromarray(np.uint8(mask * 255), "L").filter(
        ImageFilter.GaussianBlur(1.2)
    )
    return np.asarray(mask_img)


def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    """Return sampled x/y banner coordinates from a 300x340 dither grid."""
    raw = Image.open(SOURCE).convert("RGB")
    alpha_full = subject_alpha(raw)
    source = raw.convert("RGBA")
    source.putalpha(Image.fromarray(alpha_full, "L"))

    # Tighter head + shoulders crop so face detail fills the VISUAL.MAP frame.
    # Computed relative to the source's own size so it works for any avatar,
    # matching the original photo crop's aspect ratio (372:422).
    sw, sh = source.size
    target_ratio = 372 / 422
    crop_w = min(sw, int(sh * target_ratio))
    crop_h = min(sh, int(crop_w / target_ratio))
    left = (sw - crop_w) // 2
    crop = source.crop((left, 0, left + crop_w, crop_h)).resize(
        (300, 340), Image.Resampling.LANCZOS
    )
    rgb = crop.convert("RGB")
    alpha = np.asarray(crop.getchannel("A"), dtype=np.float32) / 255.0

    # Composite onto white and threshold the dark strokes as "ink". This is
    # theme-independent by design: a flat two-tone icon (or a photo) should
    # trace the same silhouette whether the dots end up drawn light-on-dark
    # or dark-on-light — only the render colour differs, never the shape.
    white = Image.new("RGBA", crop.size, "white")
    white.alpha_composite(crop)
    prepared = ImageOps.grayscale(white.convert("RGB"))
    prepared = ImageOps.autocontrast(prepared, cutoff=1)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.35)
    prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2, percent=175, threshold=1))
    bits = floyd_steinberg(np.asarray(prepared))
    active = ~bits & (alpha > 0.08)

    # Keep the full 300×340 lattice — skipping 2×2 cells was the soft/blurry look.
    ys, xs = np.where(active)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.column_stack((74 + xs, 154 + ys)).astype(np.float32)
    if len(points) > 18000:
        points = points[rng.choice(len(points), 18000, replace=False)]
    return points


def sample_logo_points(
    image: Image.Image, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Sample a silhouette into the portrait frame's visual coordinate space."""
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 127)
    chosen = rng.choice(len(xs), count, replace=len(xs) < count)
    # Logo occupies a centered 270x270 square inside VISUAL.MAP.
    return np.column_stack((89 + xs[chosen] * 0.675, 188 + ys[chosen] * 0.675)).astype(
        np.float32
    )


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Order target points by minimum-cost assignment from source points."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    """Aggregate adjacent horizontal one-pixel dots into compact SVG path runs."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def dotted_leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{x} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def text_width(text: str, font_size: float) -> float:
    """Stable monospace width used both for textLength and leader placement."""
    return len(text) * font_size * 0.605


def animate_values(points: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in points)


def label_group(c: dict, text: str, x: float, y: float, fade_in0: float,
                 fade_in1: float, fade_out0: float, fade_out1: float) -> str:
    """A small caption that fades in/out at specific points in the loop,
    naming which silhouette the dots currently form."""
    key_times = ";".join(
        num(v / LOOP_SECONDS)
        for v in (0, fade_in0, fade_in1, fade_out0, fade_out1, LOOP_SECONDS)
    )
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="end" fill="{c["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11" '
        f'font-weight="700" letter-spacing=".6" opacity="0">{html.escape(text)}'
        f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" '
        f'repeatCount="indefinite" keyTimes="{key_times}" values="0;0;1;1;0;0"/>'
        f'</text>'
    )


def render_svg(
    theme_name: str,
    portrait: np.ndarray,
    logo_points: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme_name]
    n = min(TRAVELLER_COUNT, len(portrait))
    source = portrait[rng.choice(len(portrait), n, replace=False)]

    # Build the morph chain: portrait -> logo[0] -> logo[1] -> ... -> back.
    chain = [source]
    prev = source
    for name in LOGO_SEQUENCE:
        prev = transport(prev, logo_points[name][:n])
        chain.append(prev)

    # Schedule: an initial portrait hold, then (transition + hold) per logo,
    # then one final transition back to the portrait to close the loop.
    times = [0.0, INTRO_HOLD]
    for _ in LOGO_SEQUENCE:
        times.append(times[-1] + TRANSITION)
        times.append(times[-1] + HOLD)
    times.append(times[-1] + TRANSITION)
    key_times = ";".join(num(v / LOOP_SECONDS) for v in times)

    # Returning each traveller to its exact starting portrait coordinate keeps
    # the repeat boundary seamless. All logo-to-logo morphs use optimal transport.
    frames = [source, source]
    for pts in chain[1:]:
        frames += [pts, pts]
    frames.append(source)
    opacity_values = ";".join(["0", "0"] + ["1"] * (2 * len(LOGO_SEQUENCE)) + ["0"])

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        "<title id=\"title\">Santiago's live system profile</title>",
        '<desc id="desc">Animated terminal profile with a dithered portrait that '
        "morphs through Python, Supabase, Neon, and Angular silhouettes.</desc>",
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".35"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'letter-spacing=".4">profile.sh --live</text>',
        # Left visual frame.
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">300×340 / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        # Loop layer stays visible at t=0 so camo/static first frames still show the face.
        # Intro duplicate below shimmers on top, then hands off at 3.2s.
        '<g opacity="1">',
    ]

    # Dense portrait drift: 94 independently noisy bands moving toward the
    # first logo's centroid.
    first_logo_centroid = chain[1].mean(axis=0)
    band_ids = rng.integers(0, 94, size=len(portrait))
    noise = rng.normal(0, 2, size=(94, 2))
    # These two animations share the same keyTimes as the traveller morph, so
    # their values lists must have exactly as many entries: visible only
    # during the initial portrait hold (indices 0-1), then gone for every
    # logo phase, back at the very end (index -1) to match the loop close.
    n_key = len(times)
    drift_opacity = ";".join([".94", ".94"] + ["0"] * (n_key - 3) + [".94"])
    for band in range(94):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        centroid = pts.mean(axis=0)
        delta = (first_logo_centroid - centroid) * 0.18 + noise[band]
        moved = f"{num(delta[0])} {num(delta[1])}"
        drift_translate = ";".join(
            ["0 0", "0 0", moved, moved] + ["0 0"] * (n_key - 4)
        )
        d = point_path(pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{t["portrait"]}" stroke-width="1" '
            'opacity=".94">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{drift_translate}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" '
            f'repeatCount="indefinite" keyTimes="{key_times}" '
            f'values="{drift_opacity}"/></path>'
        )

    # Optimal-transport travellers, represented as tiny path squares (never glyphs).
    for i in range(n):
        positions = animate_values(frames, i)
        parts.append(
            f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{t["portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{positions}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{opacity_values}"/></path>'
        )
    parts.append("</g>")

    # One-shot scattered intro: sixty random, interleaved point groups.
    intro_ids = rng.integers(0, 60, size=len(portrait))
    order = rng.permutation(60)
    starts = np.empty(60)
    starts[order] = np.linspace(0.05, 1.2, 60)
    for group in range(60):
        pts = portrait[intro_ids == group]
        if not len(pts):
            continue
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{num(starts[group])}s" dur=".8s" '
            'values="0;1" fill="freeze"/>'
            '<animate attributeName="opacity" begin="3.08s" dur=".12s" values="1;0" fill="freeze"/>'
            "</path>"
        )
    parts.extend(
        [
            "</g>",
            # Small frame telemetry.
            f'<text x="58" y="551" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">'
            f'PTS {len(portrait):05d} · FS/SERPENTINE</text>',
        ]
    )
    for i, name in enumerate(LOGO_SEQUENCE):
        transition_end = times[2 + 2 * i]
        hold_end = times[3 + 2 * i]
        parts.append(
            label_group(
                t, f"MODE: {name.upper()}", 438, 551,
                transition_end - 0.4, transition_end, hold_end, hold_end + 0.4,
            )
        )
    parts.extend(
        [
            # Right information panel.
            f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" '
            f'stroke="{t["line"]}"/>',
            f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
            f'<text x="490" y="111" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
            # LIVE badge and handle pill.
            '<g filter="url(#glow)"><circle cx="915" cy="106" r="4" fill="#FF4D5A">'
            '<animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/>'
            '</circle></g>',
            '<text x="927" y="111" fill="#FF4D5A" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" '
            'font-weight="700">LIVE</text>',
            f'<rect x="982" y="94" width="146" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" '
            f'stroke="{t["chrome"]}"/>',
            f'<text x="1055" y="111" text-anchor="middle" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14" '
            'font-weight="700">@thefresh26</text>',
        ]
    )

    value_right = 1127.0
    row_y = 153.0
    for label, value in ROWS:
        value_len = text_width(value, 14)
        label_len = text_width(label, 14)
        leader_start = 491 + label_len + 12
        leader_end = value_right - value_len - 12
        parts.extend(
            [
                f'<text x="491" y="{num(row_y)}" fill="{t["muted"]}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14">'
                f"{html.escape(label)}</text>",
                f'<path d="{dotted_leader(leader_start, leader_end, row_y - 4)}" '
                f'fill="none" stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
                f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" '
                f'fill="{t["text"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
                f'font-size="14" textLength="{num(value_len)}" lengthAdjust="spacingAndGlyphs">'
                f"{html.escape(value)}</text>",
            ]
        )
        row_y += 23

    parts.extend(
        [
            f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
            f'<text x="491" y="548" fill="{t["accent"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
            "● ALL SYSTEMS NOMINAL</text>",
            f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
            "UTC-5 · LATAM NODE</text>",
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source avatar: {SOURCE}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    logos = make_logos()

    # Cache theme-specific dither points as reproducible source data.
    portraits: dict[str, np.ndarray] = {}
    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + index)
        points = portrait_points(theme, rng)
        portraits[theme] = points
        np.save(DATA / f"portrait-{theme}.npy", points)

    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + 100 + index)
        sampled = {
            name: sample_logo_points(image, rng, TRAVELLER_COUNT)
            for name, image in logos.items()
        }
        for name, points in sampled.items():
            np.save(DATA / f"{name}-{theme}.npy", points)
        svg = render_svg(theme, portraits[theme], sampled, rng)
        output = ASSETS / f"banner-{theme}.svg"
        output.write_text(svg, encoding="utf-8")
        byte_size = output.stat().st_size
        print(
            f"{output.relative_to(ROOT)}: {byte_size:,} bytes "
            f"({byte_size / 1024:.1f} KiB), {len(portraits[theme]):,} portrait dots, "
            f"{TRAVELLER_COUNT} travellers"
        )

    for name in logos:
        output = LOGOS / f"{name}.png"
        print(f"{output.relative_to(ROOT)}: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
