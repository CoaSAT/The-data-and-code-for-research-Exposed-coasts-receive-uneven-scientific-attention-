from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree.ElementTree import iterparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PROFILE_LAT_COL = "AG"
PROFILE_LON_COL = "AH"


ATTENTION_BINS = [
    ("<25", -np.inf, 25, (182, 221, 232), "cold spots"),
    ("25-200", 25, 200, (104, 172, 210), ""),
    ("200-400", 200, 400, (230, 180, 74), ""),
    ("400-600", 400, 600, (213, 117, 54), ""),
    (">600", 600, np.inf, (128, 64, 144), "hot spots"),
]


def configure_attention_bins(cold_threshold: float, mid_edges: list[float], hot_threshold: float) -> None:
    """Set the five attention classes used in the map and share panel."""
    global ATTENTION_BINS
    if len(mid_edges) != 2:
        raise ValueError("--mid-edges must contain exactly two comma-separated values")
    m1, m2 = sorted(float(v) for v in mid_edges)
    if not (cold_threshold < m1 < m2 < hot_threshold):
        raise ValueError("Expected cold_threshold < mid_edges[0] < mid_edges[1] < hot_threshold")
    ATTENTION_BINS = [
        (f"<{int(cold_threshold):g}", -np.inf, cold_threshold, (182, 221, 232), "cold spots"),
        (f"{int(cold_threshold):g}-{int(m1):g}", cold_threshold, m1, (104, 172, 210), ""),
        (f"{int(m1):g}-{int(m2):g}", m1, m2, (230, 180, 74), ""),
        (f"{int(m2):g}-{int(hot_threshold):g}", m2, hot_threshold, (213, 117, 54), ""),
        (f">{int(hot_threshold):g}", hot_threshold, np.inf, (128, 64, 144), "hot spots"),
    ]


def cold_threshold() -> float:
    return ATTENTION_BINS[0][2]


def hot_threshold() -> float:
    return ATTENTION_BINS[-1][1]

# ---------------------------------------------------------------------------
# Manual adjustment knobs
# ---------------------------------------------------------------------------
# If you want to tune the figure manually, start here. Coordinates are pixels
# in the final canvas. The script exports at 300 dpi.

CANVAS_W = 2400
CANVAS_H = 1920
MARGIN = 70

# Main map panel. Increase MAP_BOTTOM to make panel a taller; decrease it to
# give panels b/c more vertical space.
MAP_LEFT = MARGIN
MAP_TOP = 125
MAP_RIGHT = 2040
MAP_BOTTOM = 955

LON_PROFILE_TOP = 0
LON_PROFILE_BOTTOM = 0
LAT_PROFILE_LEFT = 2080
LAT_PROFILE_RIGHT = 2325

# Legend for panel a. These are offsets from the map box.
LEGEND_X_OFFSET = 72
LEGEND_Y_FROM_MAP_BOTTOM = 325

# Lower panel b.
COAST_SHARE_TOP = 1015
COAST_SHARE_BOTTOM = 1290

# Panel c: Lorenz curve / Coastal Attention Concentration Index.
CACI_TOP = 1325
CACI_BOTTOM = 1840
CACI_PLOT_LEFT = MAP_LEFT + 86
CACI_PLOT_RIGHT = LAT_PROFILE_RIGHT

# Font sizes. All fonts use Arial.
FONT_PANEL = 58
FONT_TITLE = 45
FONT_BODY = 32
FONT_SMALL = 26
FONT_AXIS = 26
FONT_TINY = 22
FONT_LEGEND = 34
FONT_SHARE_VALUE = 40
FONT_SHARE_AXIS = 31
FONT_CACI_AXIS = 30

# Point sizes for attention classes on panel a.
POINT_RADIUS_LOW = 1.9
POINT_RADIUS_HIGH = 2.2

# Base-map styling.
LAND_FILL = (226, 226, 226)
COASTLINE = (62, 66, 70)
GRATICULE = (218, 222, 226)
BOUNDARY = (8, 10, 12)
INK = (6, 8, 10)
MUTED = (88, 92, 97)
MAP_LAT_MIN = -60.0
MAP_LAT_MAX = 75.0

ROBINSON_LAT = np.array(
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
    dtype=float,
)
ROBINSON_X = np.array(
    [
        1.0000,
        0.9986,
        0.9954,
        0.9900,
        0.9822,
        0.9730,
        0.9600,
        0.9427,
        0.9216,
        0.8962,
        0.8679,
        0.8350,
        0.7986,
        0.7597,
        0.7186,
        0.6732,
        0.6213,
        0.5722,
        0.5322,
    ],
    dtype=float,
)
ROBINSON_Y = np.array(
    [
        0.0000,
        0.0620,
        0.1240,
        0.1860,
        0.2480,
        0.3100,
        0.3720,
        0.4340,
        0.4958,
        0.5571,
        0.6176,
        0.6769,
        0.7346,
        0.7903,
        0.8435,
        0.8936,
        0.9394,
        0.9761,
        1.0000,
    ],
    dtype=float,
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [r"C:\Windows\Fonts\arialbd.ttf"]
        if bold
        else [r"C:\Windows\Fonts\arial.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box
    x = x0 + (lon + 180.0) / 360.0 * (x1 - x0)
    y = y1 - (lat - MAP_LAT_MIN) / (MAP_LAT_MAX - MAP_LAT_MIN) * (y1 - y0)
    return x, y


def mollweide_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to a Mollweide-style ellipse scaled to the map box."""
    lon_rad = np.radians(lon)
    lat_rad = np.radians(np.clip(lat, -89.999, 89.999))
    theta = lat_rad.copy()
    for _ in range(12):
        denom = 2 + 2 * np.cos(2 * theta)
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat_rad)) / denom
    x_m = (2 * np.sqrt(2) / np.pi) * lon_rad * np.cos(theta)
    y_m = np.sqrt(2) * np.sin(theta)

    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    sx = (x1 - x0) / (4 * np.sqrt(2))
    sy = (y1 - y0) / (2 * np.sqrt(2))
    return cx + x_m * sx, cy - y_m * sy


def robinson_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to a flat lon-lat map focused on evaluated coasts."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    return lonlat_to_xy(lon, lat, box)


def draw_graticule(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], f_axis) -> None:
    for lon in [-120, -60, 0, 60, 120]:
        lats = np.linspace(MAP_LAT_MIN, MAP_LAT_MAX, 260)
        lons = np.full_like(lats, lon, dtype=float)
        xs, ys = robinson_xy(lons, lats, box)
        draw.line(list(zip(xs, ys)), fill=GRATICULE, width=1)
    for lat in [-60, -30, 0, 30, 60]:
        lons = np.linspace(-180, 180, 360)
        lats = np.full_like(lons, lat, dtype=float)
        xs, ys = robinson_xy(lons, lats, box)
        draw.line(list(zip(xs, ys)), fill=GRATICULE, width=1)

    draw_projection_boundary(draw, box)

    # Coordinate labels are intentionally omitted to keep the global map clean.


def draw_projection_boundary(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rectangle(box, outline=BOUNDARY, width=2)


def draw_land_base(draw: ImageDraw.ImageDraw, geojson_path: Path, box: tuple[int, int, int, int]) -> None:
    if not geojson_path.exists():
        return
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    def draw_ring(coords: list[list[float]]) -> None:
        arr = np.asarray(coords, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 3:
            return
        # Split rings where Natural Earth polygons cross the dateline.
        jumps = np.where(np.abs(np.diff(arr[:, 0])) > 180)[0]
        starts = [0] + (jumps + 1).tolist()
        ends = (jumps + 1).tolist() + [len(arr)]
        for start, end in zip(starts, ends):
            seg = arr[start:end]
            if len(seg) < 3:
                continue
            if np.nanmax(seg[:, 1]) < MAP_LAT_MIN or np.nanmin(seg[:, 1]) > MAP_LAT_MAX:
                continue
            seg = seg.copy()
            seg[:, 1] = np.clip(seg[:, 1], MAP_LAT_MIN, MAP_LAT_MAX)
            xs, ys = robinson_xy(seg[:, 0], seg[:, 1], box)
            pts = list(zip(xs.tolist(), ys.tolist()))
            draw.polygon(pts, fill=LAND_FILL)
            draw.line(pts, fill=COASTLINE, width=1)

    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "Polygon":
            for ring in geom.get("coordinates", []):
                draw_ring(ring)
        elif geom.get("type") == "MultiPolygon":
            for polygon in geom.get("coordinates", []):
                for ring in polygon:
                    draw_ring(ring)


def draw_point_map(draw: ImageDraw.ImageDraw, df: pd.DataFrame, box: tuple[int, int, int, int]) -> None:
    lon = df["lon"].to_numpy(float)
    lat = df["lat"].to_numpy(float)
    att = df["All_attention_research_count"].to_numpy(float)
    x, y = robinson_xy(lon, lat, box)
    visible = np.isfinite(x) & np.isfinite(y) & (lat >= MAP_LAT_MIN) & (lat <= MAP_LAT_MAX)

    for label, low, high, color, _ in ATTENTION_BINS:
        if np.isneginf(low):
            mask = (att < high) & visible
        elif np.isposinf(high):
            mask = (att > low) & visible
        else:
            mask = (att >= low) & (att <= high) & visible

        # Draw low-attention points first and high-attention points last.
        radius = POINT_RADIUS_LOW if label in ["<25", "25-200"] else POINT_RADIUS_HIGH
        for px, py in zip(x[mask], y[mask]):
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)


def draw_marginal_profiles(
    draw: ImageDraw.ImageDraw,
    df: pd.DataFrame,
    map_box: tuple[int, int, int, int],
    lon_box: tuple[int, int, int, int],
    lat_box: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    profile_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source = profile_df if profile_df is not None else df
    lon = source["lon"].to_numpy(float)
    lat = source["lat"].to_numpy(float)
    if "All_attention_research_count" in source.columns:
        attention = source["All_attention_research_count"].to_numpy(float)
    else:
        attention = np.ones(len(source), dtype=float)
    visible = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(attention) & (lat >= MAP_LAT_MIN) & (lat <= MAP_LAT_MAX)
    lon = lon[visible]
    lat = lat[visible]
    attention = attention[visible]

    lon_edges = np.arange(-180, 181, 2)
    lat_edges = np.arange(MAP_LAT_MIN, MAP_LAT_MAX + 1, 1)
    profile = pd.read_csv(Path(__file__).parent / "data/profile_histogram.csv")
    lon_totals = profile.loc[profile.axis.eq("longitude"), "count"].to_numpy()
    lat_totals = profile.loc[profile.axis.eq("latitude"), "count"].to_numpy()

    lx0, ly0, lx1, ly1 = lat_box
    fill = (128, 64, 144)
    pale = (236, 230, 240)
    axis = (162, 168, 174)
    grid = (228, 231, 234)
    line_color = (111, 48, 130)

    def fmt_studies(value: float) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f} million"
        if value >= 1_000:
            return f"{value / 1_000:.0f}k"
        return f"{value:.0f}"

    def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
        if len(values) < window:
            return values
        kernel = np.ones(window, dtype=float) / window
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    lon_scaled = smooth(lon_totals, window=5)
    lon_max = float(lon_scaled.max()) if len(lon_scaled) else 1.0
    lon_max = max(lon_max, 1.0)

    draw.rectangle(lat_box, outline=axis, width=1)
    draw.text((lx0, ly0 - 30), "Latitudinal summed attention", fill=MUTED, font=fonts["small"])
    lat_scaled = smooth(lat_totals, window=5)
    lat_max = float(lat_scaled.max()) if len(lat_scaled) else 1.0
    lat_max = max(lat_max, 1.0)
    for lat_tick in [-60, -30, 0, 30, 60]:
        _, ty_arr = robinson_xy(np.array([0.0]), np.array([lat_tick]), map_box)
        ty = float(ty_arr[0])
        draw.line((lx0, ty, lx1, ty), fill=grid, width=1)
    for tick_value in [0, lat_max / 2, lat_max]:
        tx = lx0 + tick_value / lat_max * (lx1 - lx0 - 12)
        draw.line((tx, ly0, tx, ly1), fill=grid, width=1)
        label = fmt_studies(tick_value)
        label_box = draw.textbbox((0, 0), label, font=fonts["tiny"])
        draw.text((tx - (label_box[2] - label_box[0]) / 2, ly1 + 8), label, fill=MUTED, font=fonts["tiny"])
    lat_points = []
    for i, value in enumerate(lat_scaled):
        center = (lat_edges[i] + lat_edges[i + 1]) / 2
        _, yy_arr = robinson_xy(np.array([0.0]), np.array([center]), map_box)
        px = lx0 + value / lat_max * (lx1 - lx0 - 12)
        lat_points.append((px, float(yy_arr[0])))
    if lat_points:
        draw.polygon([(lx0, ly0), *lat_points, (lx0, ly1)], fill=tuple(int(c * 0.25 + 255 * 0.75) for c in fill))
        draw.line(lat_points, fill=line_color, width=3)
    draw.text((lx0, ly1 + 28), "Latitude", fill=MUTED, font=fonts["tiny"])

    return lon_totals, lat_totals


def attention_coastline_shares(df: pd.DataFrame) -> dict[str, float]:
    attention = pd.to_numeric(df["All_attention_research_count"], errors="coerce")
    valid = attention.notna()
    total = int(valid.sum())
    if total == 0:
        return {"Cold spots": 0.0, "Intermediate": 0.0, "Hot spots": 0.0}
    cold = int((attention[valid] < cold_threshold()).sum())
    hot = int((attention[valid] > hot_threshold()).sum())
    other = total - cold - hot
    return {
        "Cold spots": cold / total * 100,
        "Intermediate": other / total * 100,
        "Hot spots": hot / total * 100,
    }


def draw_coastline_share_panel(
    draw: ImageDraw.ImageDraw,
    df: pd.DataFrame,
    box: tuple[int, int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    x0, y0, x1, y1 = box
    shares = attention_coastline_shares(df)
    colors = {
        "Cold spots": ATTENTION_BINS[0][3],
        "Intermediate": (226, 230, 234),
        "Hot spots": ATTENTION_BINS[-1][3],
    }
    bar_y = y0 + 64
    bar_h = 86
    bar_w = x1 - x0

    draw.text((x0, y0), "b", fill=INK, font=fonts["panel"])
    draw.text((x0 + 62, y0 + 7), "Coastline share by attention class", fill=INK, font=fonts["title"])

    cursor = x0
    segment_bounds = []
    for label in ["Cold spots", "Intermediate", "Hot spots"]:
        value = shares[label]
        width = bar_w * value / 100
        x_end = cursor + width
        draw.rectangle((cursor, bar_y, x_end, bar_y + bar_h), fill=colors[label], outline=(255, 255, 255), width=2)
        segment_bounds.append((label, cursor, x_end, value))
        if width > 160:
            text_color = (255, 255, 255) if label == "Hot spots" else INK
            value_text = f"{value:.1f}%"
            bbox = draw.textbbox((0, 0), value_text, font=fonts["share_value"])
            draw.text(
                (cursor + width / 2 - (bbox[2] - bbox[0]) / 2, bar_y + 22),
                value_text,
                fill=text_color,
                font=fonts["share_value"],
            )
        cursor = x_end

    for tick in [0, 25, 50, 75, 100]:
        tx = x0 + bar_w * tick / 100
        draw.line((tx, bar_y + bar_h + 7, tx, bar_y + bar_h + 25), fill=INK, width=2)
        label = f"{tick}%"
        bbox = draw.textbbox((0, 0), label, font=fonts["share_axis"])
        draw.text((tx - (bbox[2] - bbox[0]) / 2, bar_y + bar_h + 35), label, fill=INK, font=fonts["share_axis"])

    legend_y = y1 - 54
    legend_items = [
        (f"Cold spots (<{int(cold_threshold()):g})", colors["Cold spots"]),
        (f"Intermediate ({int(cold_threshold()):g}-{int(hot_threshold()):g})", colors["Intermediate"]),
        (f"Hot spots (>{int(hot_threshold()):g})", colors["Hot spots"]),
    ]
    lx = x0
    for label, color in legend_items:
        draw.rectangle((lx, legend_y, lx + 24, legend_y + 24), fill=color, outline=(190, 196, 202))
        draw.text((lx + 36, legend_y - 4), label, fill=INK, font=fonts["small"])
        lx += 390


def draw_caci_panel(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    lorenz_path: Path,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    """Draw panel c directly in the original Figure 1 canvas.

    The Lorenz curve is precomputed from the same GCC attention table used for
    panel a/b, sorted from low to high attention per evaluated coastline point.
    """
    if not lorenz_path.exists():
        return
    lorenz = pd.read_csv(lorenz_path)
    x = pd.to_numeric(lorenz["cumulative_coastline_share"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(lorenz["cumulative_mapped_record_share"], errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.clip(x[valid], 0, 1)
    y = np.clip(y[valid], 0, 1)
    if len(x) < 2:
        return
    summary_path = lorenz_path.with_name("figure1_caci_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    g = summary.get("gini_gcc_points")
    g_ci = summary.get("gini_gcc_points_bootstrap95", [np.nan, np.nan])
    top10 = summary.get("top_decile_record_share_gcc_points")

    left, right = CACI_PLOT_LEFT, CACI_PLOT_RIGHT
    top, bottom = CACI_TOP + 50, CACI_BOTTOM - 38
    width = right - left
    height = bottom - top

    def px(value: float) -> float:
        return left + value * width

    def py(value: float) -> float:
        return bottom - value * height

    # Panel label.
    draw.text((MAP_LEFT, CACI_TOP - 58), "c", fill=INK, font=fonts["panel"])
    draw.text((MAP_LEFT + 66, CACI_TOP - 47), "Coastal Attention Concentration Index (Lorenz-Gini)", fill=INK, font=fonts["title"])

    # Grid and axes.
    for tick in np.linspace(0, 1, 6):
        x_tick = px(float(tick))
        y_tick = py(float(tick))
        draw.line((x_tick, top, x_tick, bottom), fill=GRATICULE, width=2)
        draw.line((left, y_tick, right, y_tick), fill=GRATICULE, width=2)

    # Area between equal-attention line and Lorenz curve.
    equality = [(px(float(v)), py(float(v))) for v in x]
    lorenz_pts = [(px(float(v)), py(float(w))) for v, w in zip(x, y)]
    if len(lorenz_pts) >= 3:
        draw.polygon(equality + list(reversed(lorenz_pts)), fill=(220, 244, 248))

    # Top-decile region.
    y90 = float(np.interp(0.90, x, y))
    draw.rectangle((px(0.90), top, right, bottom), fill=(246, 240, 247))

    # Redraw grid above fills.
    for tick in np.linspace(0, 1, 6):
        x_tick = px(float(tick))
        y_tick = py(float(tick))
        draw.line((x_tick, top, x_tick, bottom), fill=GRATICULE, width=2)
        draw.line((left, y_tick, right, y_tick), fill=GRATICULE, width=2)

    # Lines and markers.
    draw.line((left, bottom, right, top), fill=(170, 181, 188), width=2)
    if len(lorenz_pts) >= 2:
        draw.line(lorenz_pts, fill=(31, 111, 139), width=7, joint="curve")
    x90 = px(0.90)
    dash = 12
    yy = top
    while yy < bottom:
        draw.line((x90, yy, x90, min(yy + dash, bottom)), fill=ATTENTION_BINS[-1][3], width=3)
        yy += dash * 2
    for cx, cy, radius in [(x90, py(y90), 8), (right, top, 9)]:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=ATTENTION_BINS[-1][3], outline=(255, 255, 255), width=2)

    # Compact method note: keep it inside panel c so readers immediately see
    # that CACI is a Lorenz-curve Gini concentration measure.
    if g is not None and top10 is not None:
        method_lines = [
            "Lorenz-Gini method",
            f"CACI Gini = {g:.3f} (95% CI {g_ci[0]:.3f}-{g_ci[1]:.3f})",
            f"Top 10% coastline = {top10 * 100:.1f}% of records",
        ]
        text_x = left + 170
        text_y = top + 34
        line_gap = 34
        max_w = max(draw.textbbox((0, 0), line, font=fonts["small"])[2] for line in method_lines)
        draw.rounded_rectangle(
            (text_x - 16, text_y - 14, text_x + max_w + 18, text_y + line_gap * len(method_lines) + 6),
            radius=10,
            fill=(248, 251, 252),
            outline=(207, 217, 222),
            width=2,
        )
        for i, line in enumerate(method_lines):
            draw.text(
                (text_x, text_y + i * line_gap),
                line,
                fill=INK,
                font=fonts["small"] if i else fonts["body"],
            )

    # Axes and tick labels.
    draw.rectangle((left, top, right, bottom), outline=(170, 180, 188), width=2)
    for tick in np.linspace(0, 1, 6):
        x_tick = px(float(tick))
        y_tick = py(float(tick))
        draw.line((x_tick, bottom, x_tick, bottom + 10), fill=INK, width=2)
        draw.line((left - 10, y_tick, left, y_tick), fill=INK, width=2)
        x_label = f"{int(tick * 100)}%"
        y_label = f"{int(tick * 100)}%"
        xb = draw.textbbox((0, 0), x_label, font=fonts["caci_axis"])
        yb = draw.textbbox((0, 0), y_label, font=fonts["caci_axis"])
        draw.text((x_tick - (xb[2] - xb[0]) / 2, bottom + 20), x_label, fill=INK, font=fonts["caci_axis"])
        y_text_y = y_tick - (yb[3] - yb[1]) / 2
        draw.text((left - 18 - (yb[2] - yb[0]), y_text_y), y_label, fill=INK, font=fonts["caci_axis"])

    xlabel = "Cumulative share of evaluated coastline"
    xb = draw.textbbox((0, 0), xlabel, font=fonts["title"])
    draw.text((left + width / 2 - (xb[2] - xb[0]) / 2, bottom + 58), xlabel, fill=INK, font=fonts["body"])

    ylabel = "Mapped records (%)"
    ybbox = draw.textbbox((0, 0), ylabel, font=fonts["body"])
    y_label_img = Image.new("RGBA", (ybbox[2] - ybbox[0] + 12, ybbox[3] - ybbox[1] + 12), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label_img)
    y_label_draw.text((6, 4), ylabel, fill=INK, font=fonts["body"])
    y_label_img = y_label_img.rotate(90, expand=True)
    image.paste(y_label_img, (left - 140, int(top + height / 2 - y_label_img.height / 2)), y_label_img)


def parse_population_summary(path: Path) -> dict[str, float]:
    pop = pd.read_excel(path)
    first_col = pop.columns[0]
    total_pop = float(pop["人口总数"].dropna().iloc[0])
    cold_row = pop[pop[first_col].astype(str).str.strip().eq("<25")].iloc[0]
    hot_row = pop[pop[first_col].astype(str).str.strip().eq(">600")].iloc[0]
    return {
        "total_pop": total_pop,
        "cold_points": float(cold_row["Unnamed: 1"]),
        "cold_coast_share": float(cold_row["Unnamed: 2"]) * 100,
        "cold_pop": float(cold_row["覆盖的人口"]),
        "cold_pop_share": float(cold_row["Unnamed: 5"]) * 100,
        "hot_points": float(hot_row["Unnamed: 1"]),
        "hot_coast_share": float(hot_row["Unnamed: 2"]) * 100,
        "hot_pop": float(hot_row["覆盖的人口"]),
        "hot_pop_share": float(hot_row["Unnamed: 5"]) * 100,
    }


def draw_stacked_share_panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    stats: dict[str, float],
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    ink = INK
    muted = MUTED
    grey = (229, 233, 237)
    cold = ATTENTION_BINS[0][3]
    hot = ATTENTION_BINS[-1][3]
    other = grey
    bar_x = x + 318
    bar_w = 560
    bar_h = 38
    row_gap = 112
    top_y = y + 72
    bottom_y = top_y + row_gap
    rows = [
        ("Coastline length", stats["cold_coast_share"], 100 - stats["cold_coast_share"] - stats["hot_coast_share"], stats["hot_coast_share"]),
        ("Coastal population", stats["cold_pop_share"], 100 - stats["cold_pop_share"] - stats["hot_pop_share"], stats["hot_pop_share"]),
    ]
    colors = [cold, other, hot]
    labels = ["cold spots", "other attention classes", "hot spots"]

    def segments(values: tuple[float, float, float], yy: int) -> list[tuple[float, float, tuple[int, int, int]]]:
        cursor = bar_x
        out = []
        for value, color in zip(values, colors):
            w = bar_w * value / 100
            out.append((cursor, cursor + w, color))
            cursor += w
        return out

    top_segments = segments(rows[0][1:], top_y)
    bottom_segments = segments(rows[1][1:], bottom_y)

    # Sankey-like ribbons. The same attention class is linked from coastline
    # share to population share, making demographic concentration visible.
    for (x0_top, x1_top, color), (x0_bot, x1_bot, _) in zip(top_segments, bottom_segments):
        ribbon = [
            (x0_top, top_y + bar_h),
            (x1_top, top_y + bar_h),
            (x1_bot, bottom_y),
            (x0_bot, bottom_y),
        ]
        draw.polygon(ribbon, fill=tuple(int(c * 0.72 + 255 * 0.28) for c in color))

    for i, (label, cold_v, other_v, hot_v) in enumerate(rows):
        yy = top_y + i * row_gap
        draw.text((x, yy + 5), label, fill=ink, font=fonts["body"])
        for x0, x1, color in segments((cold_v, other_v, hot_v), yy):
            draw.rectangle((x0, yy, x1, yy + bar_h), fill=color, outline=(255, 255, 255), width=1)
        cold_end = bar_x + bar_w * cold_v / 100
        hot_start = bar_x + bar_w * (cold_v + other_v) / 100
        draw.text((cold_end + 10, yy + 9), f"{cold_v:.1f}%", fill=muted, font=fonts["small"])
        draw.text((hot_start - 60, yy + 9), f"{hot_v:.1f}%", fill=muted, font=fonts["small"])

    legend_x = x
    legend_y = y + 255
    for i, (label, color) in enumerate(zip(labels, colors)):
        xx = legend_x + [0, 190, 500][i]
        draw.rectangle((xx, legend_y, xx + 17, legend_y + 17), fill=color, outline=(210, 214, 218) if color == grey else None)
        draw.text((xx + 26, legend_y - 2), label, fill=muted, font=fonts["small"])


def draw_gdp_panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    cold_gdp: float,
    hot_gdp: float,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    ink = INK
    muted = MUTED
    grid = (233, 236, 239)
    axis = (205, 211, 216)
    cold = ATTENTION_BINS[0][3]
    hot = ATTENTION_BINS[-1][3]
    plot_x = x + 150
    plot_y = y + 95
    plot_w = 650
    max_v = max(50000, np.ceil(max(cold_gdp, hot_gdp) / 10000) * 10000)

    for tick in np.arange(0, max_v + 1, 10000):
        px = plot_x + int(plot_w * tick / max_v)
        draw.line((px, plot_y - 20, px, plot_y + 130), fill=grid, width=1)
        draw.text((px - 12, plot_y + 145), f"{int(tick / 1000)}", fill=muted, font=fonts["small"])
    draw.line((plot_x, plot_y + 130, plot_x + plot_w, plot_y + 130), fill=axis, width=1)
    draw.text((plot_x + 8, plot_y + 180), "GDP per capita (US$ thousands)", fill=muted, font=fonts["small"])

    for i, (label, value, color) in enumerate([("Cold spots", cold_gdp, cold), ("Hot spots", hot_gdp, hot)]):
        yy = plot_y + i * 84
        px = plot_x + int(plot_w * value / max_v)
        draw.text((x, yy - 8), label, fill=ink, font=fonts["body"])
        draw.line((plot_x, yy, px, yy), fill=(145, 151, 158), width=2)
        draw.ellipse((px - 8, yy - 8, px + 8, yy + 8), fill=color, outline=(255, 255, 255), width=2)
        draw.text((px + 16, yy - 10), f"{value / 1000:.1f}", fill=ink, font=fonts["body"])

    ratio = hot_gdp / cold_gdp if cold_gdp else np.nan
    # Ratio is reported in the caption/text rather than as an in-figure annotation.


def draw_figure(raw_path: Path, population_path: Path | None, output_path: Path, land_path: Path, lorenz_path: Path | None = None) -> None:
    cols = ["lon", "lat", "All_attention_research_count", "gdp_ppp_us"]
    if raw_path.suffix.lower() == ".csv":
        df = pd.read_csv(raw_path, usecols=cols)
    else:
        df = pd.read_excel(raw_path, usecols=cols)
    profile_df = pd.read_csv(Path(__file__).parent / "data/profile_histogram.csv")

    fonts = {
        "panel": font(FONT_PANEL, True),
        "title": font(FONT_TITLE, True),
        "body": font(FONT_BODY),
        "small": font(FONT_SMALL),
        "axis": font(FONT_AXIS),
        "tiny": font(FONT_TINY),
        "legend": font(FONT_LEGEND),
        "share_value": font(FONT_SHARE_VALUE, True),
        "share_axis": font(FONT_SHARE_AXIS),
        "caci_axis": font(FONT_CACI_AXIS),
    }
    ink = INK
    muted = MUTED

    w, h = CANVAS_W, CANVAS_H
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    margin = MARGIN
    map_box = (MAP_LEFT, MAP_TOP, MAP_RIGHT, MAP_BOTTOM)
    draw.text((margin, 50), "a", fill=ink, font=fonts["panel"])
    draw.text((margin + 62, 57), "Spatial distribution of coastal scientific attention", fill=ink, font=fonts["title"])
    lon_totals, lat_totals = draw_marginal_profiles(
        draw,
        df,
        map_box,
        (0, 0, 0, 0),
        (LAT_PROFILE_LEFT, MAP_TOP, LAT_PROFILE_RIGHT, MAP_BOTTOM),
        fonts,
        profile_df=profile_df,
    )

    draw_graticule(draw, map_box, fonts["axis"])
    draw_land_base(draw, land_path, map_box)
    draw_point_map(draw, df, map_box)
    draw_projection_boundary(draw, map_box)

    # Compact transparent legend, placed over the lower-left ocean space.
    leg_x, leg_y = map_box[0] + LEGEND_X_OFFSET, map_box[3] - LEGEND_Y_FROM_MAP_BOTTOM
    draw.text((leg_x, leg_y), "Studies", fill=ink, font=fonts["legend"])
    for i, (label, _, _, color, note) in enumerate(ATTENTION_BINS):
        yy = leg_y + 54 + i * 48
        draw.ellipse((leg_x, yy + 4, leg_x + 28, yy + 32), fill=color)
        row_label = label
        if note == "cold spots":
            row_label = f"{label} cold"
        elif note == "hot spots":
            row_label = f"{label} hot"
        draw.text((leg_x + 44, yy - 1), row_label, fill=ink, font=fonts["legend"])

    draw_coastline_share_panel(
        draw,
        df,
        (MAP_LEFT, COAST_SHARE_TOP, LAT_PROFILE_RIGHT, COAST_SHARE_BOTTOM),
        fonts,
    )
    if lorenz_path is not None:
        draw_caci_panel(draw, img, lorenz_path, fonts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, dpi=(300, 300))
    pdf_path = output_path.with_suffix(".pdf")
    page_width = CANVAS_W * 72 / 300
    page_height = CANVAS_H * 72 / 300
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height))
    pdf.drawImage(ImageReader(img), 0, 0, width=page_width, height=page_height)
    pdf.showPage()
    pdf.save()

    summary_path = output_path.with_suffix(".summary.txt")
    summary_path.write_text(
        "\n".join(
            [
                f"raw_path={raw_path}",
                f"n_points={len(df)}",
                f"longitude_bins={len(lon_totals)}",
                f"latitude_bins={len(lat_totals)}",
                f"longitude_total_studies_max={float(np.nanmax(lon_totals)):.4f}",
                f"latitude_total_studies_max={float(np.nanmax(lat_totals)):.4f}",
                f"profile_source={population_path if profile_df is not None else raw_path}",
                f"profile_n_points={len(profile_df) if profile_df is not None else len(df)}",
                "attention_bins=" + json.dumps(
                    [{"label": label, "low": None if np.isneginf(low) else low, "high": None if np.isposinf(high) else high} for label, low, high, _, _ in ATTENTION_BINS],
                    ensure_ascii=False,
                ),
                "coastline_share=" + json.dumps(attention_coastline_shares(df), ensure_ascii=False),
                "panel_b=coastline share by cold/intermediate/hot attention class",
                f"panel_c_lorenz={lorenz_path if lorenz_path is not None else 'not drawn'}",
            ]
        ),
        encoding="utf-8",
    )


def load_profile_points(path: Path) -> pd.DataFrame:
    """Load original geolocated records for marginal profiles.

    For the large 2-degree-buffer workbook, latitude/longitude are stored in
    columns AG/AH. CSV inputs should contain lon/lat or longitude/latitude.
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        rename = {}
        if "longitude" in df.columns and "lon" not in df.columns:
            rename["longitude"] = "lon"
        if "latitude" in df.columns and "lat" not in df.columns:
            rename["latitude"] = "lat"
        df = df.rename(columns=rename)
        return df[["lon", "lat"]].dropna()
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Unsupported profile source: {path}")

    points: list[tuple[float, float]] = []
    cell_ref_re = re.compile(r"([A-Z]+)([0-9]+)")
    with zipfile.ZipFile(path) as archive:
        with archive.open("xl/worksheets/sheet1.xml") as sheet_xml:
            row_lat: tuple[int, float] | None = None
            row_lon: tuple[int, float] | None = None
            for _, elem in iterparse(sheet_xml, events=("end",)):
                if elem.tag != f"{NS_MAIN}c":
                    continue
                ref = elem.attrib.get("r", "")
                match = cell_ref_re.match(ref)
                if not match:
                    elem.clear()
                    continue
                col, row_text = match.groups()
                if col not in (PROFILE_LAT_COL, PROFILE_LON_COL):
                    elem.clear()
                    continue
                row_number = int(row_text)
                value_node = elem.find(f"{NS_MAIN}v")
                if row_number == 1 or value_node is None or value_node.text is None:
                    elem.clear()
                    continue
                try:
                    value = float(value_node.text)
                except ValueError:
                    elem.clear()
                    continue
                if col == PROFILE_LAT_COL:
                    row_lat = (row_number, value)
                else:
                    row_lon = (row_number, value)
                if row_lat and row_lon and row_lat[0] == row_lon[0]:
                    lat = row_lat[1]
                    lon = row_lon[1]
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        points.append((lon, lat))
                    row_lat = None
                    row_lon = None
                elem.clear()
    return pd.DataFrame(points, columns=["lon", "lat"])




