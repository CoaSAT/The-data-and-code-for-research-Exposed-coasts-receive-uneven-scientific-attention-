from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "categorical_blindspot_burden_map"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "_matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize


ATTENTION = ROOT / "data" / "gcc_attention_2deg" / "gcc_attention_from_2deg_buffer_window.csv"
FIG4_PROCESSED = ROOT / "data" / "figure4_new_risk_attention" / "figure4_new_risk_attention_processed.csv"
EUT_LATEST = (
    ROOT
    / "data"
    / "eutrophication_recomputed_from_monthly_chl_r12_mk"
    / "gcc_eutrophication_risk_recomputed_from_monthly_chl.csv"
)
LAND = ROOT / "data" / "ne_110m_land.geojson"

EXPOSURE_WEIGHT = {"Low": 0.0, "Moderate": 0.5, "High": 1.0}
ATTENTION_WEIGHT = {"Cold": 0.0, "Intermediate": 0.5, "Hot": 1.0}


def classify_attention(records: pd.Series) -> pd.Series:
    values = records.fillna(0).clip(lower=0)
    out = pd.Series("Intermediate", index=records.index, dtype=object)
    out[values < 150] = "Cold"
    out[values > 3000] = "Hot"
    return out


def load_data() -> pd.DataFrame:
    attention = pd.read_csv(ATTENTION, usecols=["id", "lon", "lat", "All_attention_research_count"])
    esl = pd.read_csv(FIG4_PROCESSED, usecols=["id", "ESL_exposure_class"])
    eut = pd.read_csv(EUT_LATEST, usecols=["id", "Eutrophication_exposure_recomputed"])
    df = attention.merge(esl, on="id", how="left").merge(eut, on="id", how="left")
    df["attention_class"] = classify_attention(df["All_attention_research_count"])
    df["A_weight"] = df["attention_class"].map(ATTENTION_WEIGHT)
    df["ESL_E_weight"] = df["ESL_exposure_class"].map(EXPOSURE_WEIGHT)
    df["Eutrophication_E_weight"] = df["Eutrophication_exposure_recomputed"].map(EXPOSURE_WEIGHT)
    df["ESL_burden"] = (df["ESL_E_weight"] * (1 - df["A_weight"])).fillna(0)
    df["Eutrophication_burden"] = (df["Eutrophication_E_weight"] * (1 - df["A_weight"])).fillna(0)
    df["total_burden"] = df["ESL_burden"] + df["Eutrophication_burden"]
    return df


def burden_class(values: pd.Series) -> pd.Series:
    bins = [-0.001, 0.0, 0.5, 1.0, 1.5, 2.0]
    labels = ["No blind-spot burden", "Low (>0-0.5)", "Moderate (0.5-1.0)", "High (1.0-1.5)", "Very high (>1.5)"]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True)


def gaussian_smooth(values: np.ndarray, sigma: float = 1.25) -> np.ndarray:
    radius = int(np.ceil(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(values.astype(float), radius, mode="wrap")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def draw_land(ax: plt.Axes) -> None:
    geo = json.loads(LAND.read_text(encoding="utf-8"))
    for feature in geo["features"]:
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polygons:
            outer = np.asarray(poly[0])
            if outer.ndim == 2 and outer.shape[0] > 2:
                ax.fill(outer[:, 0], outer[:, 1], facecolor="#E2E6E9", edgecolor="#C5CCD2", linewidth=0.35, zorder=0)


def draw_map(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "axes.edgecolor": "#8C98A4",
        }
    )
    df = df.copy()
    df["burden_class"] = burden_class(df["total_burden"]).astype(str)

    fig = plt.figure(figsize=(14.8, 9.8), dpi=360)
    ax = fig.add_axes([0.065, 0.405, 0.84, 0.535])
    draw_land(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 82)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 81, 30))
    ax.grid(color="#E7ECEF", linewidth=0.7, zorder=-1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Evidence-risk blind-spot burden along the evaluated global coastline", loc="left", pad=10)

    zero = df["total_burden"] <= 0
    ax.scatter(
        df.loc[zero, "lon"],
        df.loc[zero, "lat"],
        s=0.55,
        c="#CFE8F1",
        alpha=0.42,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    cmap = LinearSegmentedColormap.from_list(
        "blindspot_burden",
        ["#2A6FBB", "#50B4A4", "#F2D16B", "#E47B36", "#8B1E3F"],
        N=256,
    )
    nonzero = df["total_burden"] > 0
    norm = Normalize(vmin=0, vmax=2)
    sc = ax.scatter(
        df.loc[nonzero, "lon"],
        df.loc[nonzero, "lat"],
        s=2.35,
        c=df.loc[nonzero, "total_burden"],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    zero_handle = Line2D([0], [0], marker="o", linestyle="", color="#CFE8F1", markersize=6, label="No burden")
    ax.legend(
        handles=[zero_handle],
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE5",
        loc="lower left",
        fontsize=11.5,
    )
    cax = fig.add_axes([0.925, 0.405, 0.018, 0.535])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Blind-spot burden per GCC point")
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cbar.ax.tick_params(labelsize=15)

    # Panel b: longitude-aligned burden profiles.
    axb = fig.add_axes([0.065, 0.080, 0.84, 0.195], sharex=ax)
    lon_bins = np.arange(-180, 181, 1)
    lon_mid = (lon_bins[:-1] + lon_bins[1:]) / 2
    df["_lon_bin"] = pd.cut(df["lon"], lon_bins, labels=False, include_lowest=True)
    prof = (
        df.groupby("_lon_bin", dropna=True)[["ESL_burden", "Eutrophication_burden"]]
        .sum()
        .reindex(np.arange(len(lon_mid)), fill_value=0)
    )
    esl_smooth = gaussian_smooth(prof["ESL_burden"].to_numpy(), sigma=1.35)
    eut_smooth = gaussian_smooth(prof["Eutrophication_burden"].to_numpy(), sigma=1.35)
    profile_out = pd.DataFrame(
        {
            "lon_mid": lon_mid,
            "ESL_burden_1deg_raw": prof["ESL_burden"].to_numpy(),
            "Eutrophication_burden_1deg_raw": prof["Eutrophication_burden"].to_numpy(),
            "ESL_burden_1deg_gaussian_smoothed_sigma1p35": esl_smooth,
            "Eutrophication_burden_1deg_gaussian_smoothed_sigma1p35": eut_smooth,
        }
    )
    profile_out.to_csv(OUT / "categorical_blindspot_burden_longitude_profile_1deg.csv", index=False, encoding="utf-8-sig")
    axb.plot(lon_mid, esl_smooth, color="#2A6FBB", linewidth=2.5, label="Extreme sea-level exposure")
    axb.plot(lon_mid, eut_smooth, color="#43AA8B", linewidth=2.5, label="Eutrophication exposure")
    axb.fill_between(lon_mid, esl_smooth, color="#2A6FBB", alpha=0.12)
    axb.fill_between(lon_mid, eut_smooth, color="#43AA8B", alpha=0.12)
    axb.set_xlim(-180, 180)
    axb.set_xticks(np.arange(-180, 181, 60))
    axb.set_ylabel("Burden\nper 1 deg lon")
    axb.set_xlabel("Longitude")
    axb.set_title("b  Longitudinal burden profiles by exposure domain", loc="left", fontsize=19, fontweight="bold", pad=8)
    axb.grid(color="#E8EDF2", linewidth=0.8)
    axb.spines[["top", "right"]].set_visible(False)
    axb.spines[["left", "bottom"]].set_color("#AAB4BF")
    axb.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE5",
        framealpha=0.92,
        loc="upper right",
        ncol=2,
        fontsize=14.2,
        borderaxespad=0.7,
    )
    ax.text(-179.5, 80.5, "a", fontsize=21, fontweight="bold", ha="left", va="top")
    fig.savefig(OUT / "categorical_blindspot_burden_global_map.png", bbox_inches="tight")
    fig.savefig(OUT / "categorical_blindspot_burden_global_map_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "categorical_blindspot_burden_global_map.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_map_only(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "axes.edgecolor": "#8C98A4",
        }
    )
    fig = plt.figure(figsize=(14.8, 7.25), dpi=360)
    ax = fig.add_axes([0.065, 0.305, 0.88, 0.615])
    draw_land(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 82)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 81, 30))
    ax.grid(color="#E7ECEF", linewidth=0.7, zorder=-1)
    ax.set_xlabel("Longitude", labelpad=12)
    ax.set_ylabel("Latitude")
    ax.set_title("Evidence-risk blind-spot burden along the evaluated global coastline", loc="left", pad=10)

    zero = df["total_burden"] <= 0
    ax.scatter(
        df.loc[zero, "lon"],
        df.loc[zero, "lat"],
        s=0.55,
        c="#CFE8F1",
        alpha=0.42,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    cmap = LinearSegmentedColormap.from_list(
        "blindspot_burden",
        ["#2A6FBB", "#50B4A4", "#F2D16B", "#E47B36", "#8B1E3F"],
        N=256,
    )
    nonzero = df["total_burden"] > 0
    norm = Normalize(vmin=0, vmax=2)
    sc = ax.scatter(
        df.loc[nonzero, "lon"],
        df.loc[nonzero, "lat"],
        s=2.35,
        c=df.loc[nonzero, "total_burden"],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    cax = fig.add_axes([0.245, 0.055, 0.56, 0.035])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Blind-spot burden per GCC point")
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cbar.ax.tick_params(labelsize=15)
    ax.text(-179.5, 80.5, "a", fontsize=21, fontweight="bold", ha="left", va="top")

    fig.savefig(OUT / "categorical_blindspot_burden_global_map_only.png", bbox_inches="tight")
    fig.savefig(OUT / "categorical_blindspot_burden_global_map_only_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "categorical_blindspot_burden_global_map_only.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    df = df.copy()
    df["burden_class"] = burden_class(df["total_burden"]).astype(str)
    summary = (
        df.groupby("burden_class", dropna=False)
        .agg(
            gcc_points=("id", "count"),
            coastline_share_pct=("id", lambda x: len(x) / len(df) * 100),
            ESL_burden=("ESL_burden", "sum"),
            Eutrophication_burden=("Eutrophication_burden", "sum"),
            total_burden=("total_burden", "sum"),
        )
        .reset_index()
    )
    domain = pd.DataFrame(
        [
            {
                "domain": "Extreme sea levels",
                "exposure_source": str(FIG4_PROCESSED),
                "exposure_field": "ESL_exposure_class",
                "missing_exposure_points": int(df["ESL_exposure_class"].isna().sum()),
                "total_burden": float(df["ESL_burden"].sum()),
            },
            {
                "domain": "Eutrophication",
                "exposure_source": str(EUT_LATEST),
                "exposure_field": "Eutrophication_exposure_recomputed",
                "missing_exposure_points": int(df["Eutrophication_exposure_recomputed"].isna().sum()),
                "total_burden": float(df["Eutrophication_burden"].sum()),
            },
        ]
    )
    df.to_csv(OUT / "categorical_blindspot_burden_global_points_latest_eutrophication.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "categorical_blindspot_burden_global_class_summary.csv", index=False, encoding="utf-8-sig")
    domain.to_csv(OUT / "categorical_blindspot_burden_domain_source_check.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    write_summary(df)
    draw_map(df)
    draw_map_only(df)
    print(OUT)


if __name__ == "__main__":
    main()

