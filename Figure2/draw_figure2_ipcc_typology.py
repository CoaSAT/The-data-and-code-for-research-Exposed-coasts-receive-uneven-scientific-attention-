from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "data" / "matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shapefile
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
STATS = OUT / "ipcc_ar6_typology_attention"
IPCC_SHP = (
    ROOT
    / "data"
    / "cvi_zenodo_17513575"
    / "unzipped"
    / "CVI_Data"
    / "IPCC_Regions_CVI"
    / "CVI_for_IPCC_regions.shp"
)
TYPOLOGY_SHP = ROOT / "data" / "durr_2011_coastal_typology" / "unzipped" / "typology_coastline.shp"
LAND = OUT / "ne_110m_land.geojson"
ASSIGNMENTS = STATS / "literature_ipcc_ar6_typology_assignments.csv"

INK = "#111111"
MUTED = "#5F666D"
GRID = "#E6EAEE"
LAND_FILL = "#D9DEE3"
LAND_EDGE = "#B9C0C7"
OCEAN = "#FFFFFF"

MAP_YLIM = (-75, 85)

TYPOLOGY_LABELS = {
    1: "Small deltas",
    2: "Tidal systems",
    3: "Lagoons",
    4: "Fjords and fjaerds",
    5: "Large Rivers",
    51: "Large Rivers",
    6: "Karst",
    7: "Arheic",
}

TYPOLOGY_ORDER = [
    "Tidal systems",
    "Small deltas",
    "Lagoons",
    "Fjords and fjaerds",
    "Arheic",
    "Karst",
    "Large Rivers",
]

TYPOLOGY_COLORS = {
    "Tidal systems": "#0072B2",
    "Small deltas": "#D99A00",
    "Lagoons": "#009E73",
    "Fjords and fjaerds": "#7E858D",
    "Arheic": "#C85A17",
    "Karst": "#B86B9D",
    "Large Rivers": "#4BA7D8",
}


def style_map(ax) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(*MAP_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(OCEAN)
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.tick_params(labelsize=6.8, colors=MUTED, length=2.4, pad=1)
    ax.grid(color="#E4E8EC", lw=0.42, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#AEB5BC")
        spine.set_linewidth(0.55)


def style_chart(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#BFC5CB")
    ax.tick_params(axis="y", length=0, colors=INK, labelsize=7.7)
    ax.tick_params(axis="x", colors=MUTED, labelsize=7.2)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_axisbelow(True)


def read_shape_records(path: Path):
    reader = shapefile.Reader(str(path))
    fields = [field.name for field in reader.fields[1:]]
    for shape_record in reader.iterShapeRecords():
        yield dict(zip(fields, shape_record.record)), shape_record.shape


def split_parts(shape: shapefile.Shape) -> list[np.ndarray]:
    points = np.asarray(shape.points, dtype=float)
    if len(points) == 0:
        return []
    starts = list(shape.parts) + [len(points)]
    parts = []
    for start, end in zip(starts[:-1], starts[1:]):
        part = points[start:end]
        if len(part) < 2:
            continue
        jumps = np.where(np.abs(np.diff(part[:, 0])) > 180)[0]
        sub_starts = [0] + (jumps + 1).tolist()
        sub_ends = (jumps + 1).tolist() + [len(part)]
        for sub_start, sub_end in zip(sub_starts, sub_ends):
            sub = part[sub_start:sub_end].copy()
            if len(sub) >= 2 and np.nanmax(sub[:, 1]) >= MAP_YLIM[0] and np.nanmin(sub[:, 1]) <= MAP_YLIM[1]:
                sub[:, 1] = np.clip(sub[:, 1], *MAP_YLIM)
                parts.append(sub)
    return parts


def draw_land(ax) -> None:
    if not LAND.exists():
        return
    data = json.loads(LAND.read_text(encoding="utf-8"))
    patches = []

    def add_ring(coords):
        arr = np.asarray(coords, dtype=float)
        if len(arr) < 3:
            return
        jumps = np.where(np.abs(np.diff(arr[:, 0])) > 180)[0]
        starts = [0] + (jumps + 1).tolist()
        ends = (jumps + 1).tolist() + [len(arr)]
        for start, end in zip(starts, ends):
            seg = arr[start:end].copy()
            if len(seg) < 3 or np.nanmax(seg[:, 1]) < MAP_YLIM[0] or np.nanmin(seg[:, 1]) > MAP_YLIM[1]:
                continue
            seg[:, 1] = np.clip(seg[:, 1], *MAP_YLIM)
            patches.append(Polygon(seg[:, :2], closed=True))

    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "Polygon":
            for ring in geom.get("coordinates", []):
                add_ring(ring)
        elif geom.get("type") == "MultiPolygon":
            for polygon in geom.get("coordinates", []):
                for ring in polygon:
                    add_ring(ring)
    if patches:
        ax.add_collection(
            PatchCollection(
                patches,
                facecolor=LAND_FILL,
                edgecolor=LAND_EDGE,
                linewidth=0.25,
                zorder=1,
            )
        )


def draw_ipcc_map(ax, summary: pd.DataFrame):
    counts = dict(zip(summary["ipcc_ar6_Name"], summary["paper_count"]))
    patches = []
    values = []
    boundaries = []
    for record, shape in read_shape_records(IPCC_SHP):
        value = counts.get(record["Name"])
        for part in split_parts(shape):
            if len(part) >= 3:
                patches.append(Polygon(part[:, :2], closed=True))
                values.append(value if value is not None and value > 0 else np.nan)
                boundaries.append(part[:, :2])

    draw_land(ax)
    norm = LogNorm(vmin=max(1, np.nanmin(values)), vmax=np.nanmax(values))
    collection = PatchCollection(
        patches,
        cmap="YlGnBu",
        norm=norm,
        edgecolor="#FFFFFF",
        linewidth=0.42,
        alpha=0.68,
        zorder=2,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)
    ax.add_collection(LineCollection(boundaries, colors="#38424A", linewidths=0.22, alpha=0.62, zorder=3))
    style_map(ax)
    return collection


def draw_attention_density(ax) -> None:
    df = pd.read_csv(ASSIGNMENTS, usecols=["lon_norm", "lat", "typology_distance_deg", "typology_label"])
    df = df[np.isfinite(df["lon_norm"]) & np.isfinite(df["lat"])]
    df = df[df["lat"].between(MAP_YLIM[0], MAP_YLIM[1])]
    df = df[df["typology_label"].notna() & df["typology_distance_deg"].between(0, 1)]
    lon_edges = np.arange(-180, 181, 2)
    lat_edges = np.arange(MAP_YLIM[0], MAP_YLIM[1] + 1, 2)
    hist, _, _ = np.histogram2d(df["lon_norm"], df["lat"], bins=[lon_edges, lat_edges])
    masked = np.ma.masked_where(hist.T <= 0, hist.T)
    cmap = LinearSegmentedColormap.from_list(
        "attention_density",
        ["#FFFFFF", "#D8D8D8", "#8C8C8C", "#252525"],
    )
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        masked,
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(10, float(hist.max()))),
        shading="auto",
        alpha=0.64,
        zorder=2,
    )
    return mesh


def draw_typology_map(ax) -> None:
    draw_land(ax)
    mesh = draw_attention_density(ax)
    lines_by_type = {label: [] for label in TYPOLOGY_COLORS}
    for record, shape in read_shape_records(TYPOLOGY_SHP):
        label = TYPOLOGY_LABELS.get(int(record["FIN_TYP"]))
        if label not in lines_by_type:
            continue
        for part in split_parts(shape):
            lines_by_type[label].append(part[:, :2])

    for label in TYPOLOGY_ORDER:
        lines = lines_by_type.get(label, [])
        if not lines:
            continue
        ax.add_collection(
            LineCollection(
                lines,
                colors=TYPOLOGY_COLORS[label],
                linewidths=0.62 if label != "Large Rivers" else 0.78,
                alpha=0.95,
                zorder=4,
            )
        )
    style_map(ax)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="none",
            markerfacecolor=TYPOLOGY_COLORS[label],
            markeredgecolor=TYPOLOGY_COLORS[label],
            markersize=5.3,
            label=label,
        )
        for label in TYPOLOGY_ORDER
    ]
    return mesh, handles


def draw_ipcc_stats(ax, summary: pd.DataFrame) -> None:
    top = summary.head(12).copy()
    short_names = {
        "West&Central-Europe": "W&C Europe",
        "S.E.South-America": "S.E. South America",
        "S.W.South-America": "S.W. South America",
        "N.Central-America": "N. Central America",
        "C.North-America": "C. North America",
        "E.North-America": "E. North America",
        "W.North-America": "W. North America",
    }
    top["display_name"] = top["ipcc_ar6_Name"].replace(short_names)
    top = top.iloc[::-1]
    colors = plt.cm.YlGnBu(np.linspace(0.48, 0.93, len(top)))
    ax.barh(top["display_name"], top["paper_count"], color=colors, height=0.68)
    for y, (_, row) in enumerate(top.iterrows()):
        value = row["paper_count"]
        inside = value > 7_500
        ax.text(
            value - 900 if inside else value + 650,
            y,
            f"{int(value):,}",
            ha="right" if inside else "left",
            va="center",
            fontsize=7.4,
            color="white" if inside else INK,
        )
    ax.set_xlim(0, max(top["paper_count"]) * 1.17)
    ax.set_xlabel("Located literature records", fontsize=7.8, color=MUTED)
    style_chart(ax)


def draw_typology_stats(ax, summary: pd.DataFrame) -> None:
    data = summary.set_index("typology_label").loc[TYPOLOGY_ORDER].reset_index()
    data = data.iloc[::-1]
    ax.barh(
        data["typology_label"],
        data["paper_count"],
        color=[TYPOLOGY_COLORS[label] for label in data["typology_label"]],
        height=0.68,
    )
    total = summary["paper_count"].sum()
    for y, (_, row) in enumerate(data.iterrows()):
        pct = row["paper_count"] / total * 100
        ax.text(
            row["paper_count"] + 950,
            y,
            f"{int(row['paper_count']):,} ({pct:.1f}%)",
            va="center",
            fontsize=7.4,
            color=INK,
        )
    ax.set_xlim(0, max(data["paper_count"]) * 1.23)
    ax.set_xlabel("Located literature records assigned to typology", fontsize=7.8, color=MUTED)
    style_chart(ax)


def draw_figure() -> Path:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlepad": 5,
        }
    )
    ipcc = pd.read_csv(STATS / "literature_ipcc_ar6_region_summary.csv")
    typology = pd.read_csv(STATS / "literature_typology_summary.csv")

    fig = plt.figure(figsize=(11.8, 7.25), dpi=450)
    fig.patch.set_facecolor("white")

    left = 0.085
    col_w = 0.395
    gap = 0.105
    right = left + col_w + gap
    map_y = 0.575
    map_h = 0.275
    cbar_y = 0.515
    cbar_h = 0.020
    chart_y = 0.115
    chart_h = 0.300

    axes = [
        fig.add_axes([left, map_y, col_w, map_h]),
        fig.add_axes([right, map_y, col_w, map_h]),
        fig.add_axes([left, chart_y, col_w, chart_h]),
        fig.add_axes([right, chart_y, col_w, chart_h]),
    ]

    ipcc_mappable = draw_ipcc_map(axes[0], ipcc)
    typology_mappable, typology_handles = draw_typology_map(axes[1])
    draw_ipcc_stats(axes[2], ipcc)
    draw_typology_stats(axes[3], typology)

    cax_a = fig.add_axes([left, cbar_y, col_w, cbar_h])
    cb_a = fig.colorbar(ipcc_mappable, cax=cax_a, orientation="horizontal")
    cb_a.outline.set_visible(False)
    cb_a.ax.tick_params(labelsize=6.6, colors=INK, length=2)
    cb_a.set_label("Located literature records per IPCC AR6 region", fontsize=7.0, color=INK, labelpad=2)
    cb_a.formatter = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    cb_a.update_ticks()

    legend = fig.legend(
        handles=typology_handles,
        loc="upper left",
        bbox_to_anchor=(right, cbar_y + 0.043),
        ncol=2,
        frameon=False,
        facecolor="white",
        edgecolor="#D8DDE2",
        framealpha=0.94,
        fontsize=6.4,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    legend.set_zorder(10)

    cax_b = fig.add_axes([right + col_w * 0.47, cbar_y, col_w * 0.53, cbar_h])
    cb_b = fig.colorbar(typology_mappable, cax=cax_b, orientation="horizontal")
    cb_b.outline.set_visible(False)
    cb_b.ax.tick_params(labelsize=6.6, colors=INK, length=2)
    cb_b.set_label("Located literature records per 2° grid cell", fontsize=7.0, color=INK, labelpad=2)
    cb_b.formatter = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    cb_b.update_ticks()

    for ax in axes:
        ax.set_title("")

    label_dx = 0.036
    title_y_top = map_y + map_h + 0.018
    title_y_bottom = chart_y + chart_h + 0.018
    title_style = {"fontsize": 8.9, "weight": "bold", "color": INK, "va": "bottom"}
    label_style = {"fontsize": 10.2, "weight": "bold", "color": INK, "va": "bottom", "ha": "right"}
    fig.text(left - label_dx, title_y_top, "a", **label_style)
    fig.text(left, title_y_top, "IPCC AR6 regional partition", **title_style)
    fig.text(right - label_dx, title_y_top, "b", **label_style)
    fig.text(right, title_y_top, "Coastal typology", **title_style)
    fig.text(left - label_dx, title_y_bottom, "c", **label_style)
    fig.text(left, title_y_bottom, "Most represented IPCC AR6 regions", **title_style)
    fig.text(right - label_dx, title_y_bottom, "d", **label_style)
    fig.text(right, title_y_bottom, "Attention by coastal typology", **title_style)

    output = OUT / "figure2_ipcc_ar6_coastal_typology.png"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    summary = [
        "Figure 2 IPCC AR6 and coastal typology partition summary",
        f"IPCC records matched: {int(ipcc['paper_count'].sum()):,}",
        f"Top IPCC region: {ipcc.iloc[0]['ipcc_ar6_Name']} ({int(ipcc.iloc[0]['paper_count']):,})",
        f"Typology records matched: {int(typology['paper_count'].sum()):,}",
        f"Top typology: {typology.iloc[0]['typology_label']} ({int(typology.iloc[0]['paper_count']):,})",
    ]
    output.with_suffix(".summary.txt").write_text("\n".join(summary), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(draw_figure())
