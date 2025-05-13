# autosatai-backend/app/analyze.py
"""Single-dataset analysis endpoint – Night-time lights v5
────────────────────────────────────────────────────────────
POST  /api/analyse
Body ───
{
  "dataset_id": "gee_viirs_20231201",
  "analysis":    "night_lights",
  "assets":      { "data": "<signed-url>.tif" }
}

Response ──
{
  "dataset_id": str,
  "metrics":    { … },
  "plots":      { … },    # static PNG fall-backs
  "series":     { … }     # arrays for interactive charts
}
"""
from __future__ import annotations

import os
import uuid
import tempfile
import logging
from pathlib import Path

import numpy as np
import rasterio
import requests

# ← Force non-GUI backend so no Tk errors
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.cluster import DBSCAN
from scipy import stats
from flask import Blueprint, jsonify, request, current_app

# ───────────────────────── Flask blueprint ────────────────────────────
analyse_bp = Blueprint("analyse", __name__, url_prefix="/api")

# ─────────────────────────── constants ────────────────────────────────
THRESH_RADIANCE = 5                     # nW cm⁻² sr⁻¹ mask
PIXEL_M        = 463.83                # metres per pixel
PIXEL_AREA_KM2 = (PIXEL_M / 1_000) ** 2
STATIC_DIR     = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ───────────────────────── helper utils ────────────────────────────────
def _download_geotiff(url: str) -> str:
    """Download remote GeoTIFF to a temp file and return its path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    for chunk in resp.iter_content(8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name

def _save_fig(fig, stem: str) -> str:
    """Save fig to static folder and return '/static/…png' URL."""
    fname = f"{stem}_{uuid.uuid4().hex}.png"
    fig.savefig(STATIC_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    return f"/static/{fname}"

# ───────────────────────── core analysis ───────────────────────────────
def _analyse_night_lights(tif_path: str) -> tuple[dict, dict, dict]:
    """Compute metrics, static PNGs and series arrays for interactive charts."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1, masked=True)

    data = np.nan_to_num(arr, nan=0.0)
    mask = data > THRESH_RADIANCE
    vals = data[mask].astype(float)
    ys, xs = np.where(mask)

    # ── METRICS ─────────────────────────────────────────────────────────
    metrics = {
        "min":            float(vals.min())     if vals.size else 0.0,
        "max":            float(vals.max())     if vals.size else 0.0,
        "mean":           float(vals.mean())    if vals.size else 0.0,
        "median":         float(np.median(vals))if vals.size else 0.0,
        "std":            float(vals.std())     if vals.size else 0.0,
        "skewness":       float(stats.skew(vals))     if vals.size else 0.0,
        "kurtosis":       float(stats.kurtosis(vals)) if vals.size else 0.0,
        "total_radiance": float(vals.sum()),
        "lit_area_km2":   float(mask.sum() * PIXEL_AREA_KM2),
    }

    # ── CLUSTER BRIGHT PIXELS ──────────────────────────────────────────
    cluster_sizes = []
    centers_x, centers_y = [], []
    if vals.size:
        db = DBSCAN(eps=3, min_samples=20).fit(np.c_[ys, xs])
        labels = db.labels_
        nclus = len(set(labels)) - (1 if -1 in labels else 0)
        for lab in set(labels):
            if lab == -1:
                continue
            pts = labels == lab
            cluster_sizes.append(int(pts.sum()))
            centers_y.append(float(ys[pts].mean()))
            centers_x.append(float(xs[pts].mean()))
    else:
        nclus = 0
    metrics["bright_clusters"] = nclus

    # ── STATIC PNG FALLBACKS ───────────────────────────────────────────
    plots: dict[str, str] = {}

    # 1) Histogram
    fig, ax = plt.subplots()
    counts, bins, _ = ax.hist(vals, bins=50, color="steelblue", ec="black")
    ax.set(title="Radiance Histogram", xlabel="Radiance", ylabel="Count")
    plots["histogram"] = _save_fig(fig, "hist")

    # 2) CDF
    fig, ax = plt.subplots()
    sv = np.sort(vals)
    cdf = np.linspace(0, 1, sv.size)
    ax.plot(sv, cdf, lw=2)
    ax.set(title="Cumulative Distribution", xlabel="Radiance", ylabel="Cum. Prob.")
    plots["cdf"] = _save_fig(fig, "cdf")

    # 3) Boxplot
    fig, ax = plt.subplots()
    ax.boxplot(vals, vert=False)
    ax.set(title="Radiance Boxplot", xlabel="Radiance")
    plots["boxplot"] = _save_fig(fig, "boxplot")

    # 4) Heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, cmap="hot", vmin=0, vmax=data.max())
    ax.axis("off"); ax.set_title("Radiance Heatmap")
    fig.colorbar(im, ax=ax, orientation="vertical", label="Radiance")
    plots["heatmap"] = _save_fig(fig, "heatmap")

    # 5) Scatter
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(xs, ys, s=0.5, c="lime")
    ax.invert_yaxis(); ax.axis("off"); ax.set_title("Bright Pixels")
    plots["scatter"] = _save_fig(fig, "scatter")

    # 6) Cluster size distribution
    if cluster_sizes:
        fig, ax = plt.subplots()
        ax.hist(cluster_sizes, bins=20, color="orchid", ec="black")
        ax.set(title="Cluster Size Distribution", xlabel="Pixels", ylabel="Count")
        plots["cluster_sizes"] = _save_fig(fig, "cluster_sizes")

    # ── INTERACTIVE SERIES PAYLOAD ──────────────────────────────────────
    # deciles
    quantiles = {f"{p}%": float(np.percentile(vals, p)) for p in range(10, 100, 10)} if vals.size else {}
    # area fraction above thresholds
    thresholds = [10, 25, 50, 100]
    area_frac = {str(t): float((vals > t).sum() / vals.size * 100) for t in thresholds} if vals.size else {}
    # top 10 clusters
    top10 = sorted(cluster_sizes, reverse=True)[:10]
    # outliers
    q1, med, q3 = np.percentile(vals, [25, 50, 75]) if vals.size else (0, 0, 0)
    iqr = q3 - q1
    outliers = vals[(vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)].tolist()
    # time‐series stub
    times = [current_app.config.get("ANALYSIS_DATE", "2025-01-01")]
    avg_r = [metrics["mean"]]

    series = {
        "radiance_values":    vals.tolist(),
        "histogram":          {"bins": bins[:-1].tolist(), "counts": counts.tolist()},
        "cdf":                {"x": sv.tolist(), "y": cdf.tolist()},
        "box_whisker":        {"min":metrics["min"],"q1":q1,"median":med,"q3":q3,"max":metrics["max"],"outliers":outliers},
        "bright_pixels_xy":   {"x": xs.tolist(), "y": ys.tolist()},
        "cluster_sizes":      cluster_sizes,
        "cluster_centers":    {"x": centers_x, "y": centers_y},
        "quantiles":          quantiles,
        "area_fraction":      area_frac,
        "top10_clusters":     top10,
        "time_series":        {"times": times, "avg_radiance": avg_r},
    }

    return metrics, plots, series

# ───────────────────────────── route ──────────────────────────────────
@analyse_bp.route("/analyse", methods=["POST"])
def analyse_single():
    payload = request.get_json(silent=True) or {}
    ds_id   = payload.get("dataset_id")
    an      = payload.get("analysis")
    tif_url = payload.get("assets", {}).get("data")

    if not (ds_id and an and tif_url):
        return jsonify(error="dataset_id, analysis and assets.data required"), 400
    if an != "night_lights":
        return jsonify(error=f"analysis '{an}' not supported"), 400

    logging.info("Night-light analysis → %s", ds_id)
    tif_path = None
    try:
        tif_path = _download_geotiff(tif_url)
        metrics, plots, series = _analyse_night_lights(tif_path)
    except Exception as exc:
        current_app.logger.exception("Analysis failed")
        return jsonify(error=str(exc)), 500
    finally:
        if tif_path and os.path.exists(tif_path):
            try: os.remove(tif_path)
            except: pass

    return jsonify(dataset_id=ds_id,
                   metrics=metrics,
                   plots=plots,
                   series=series), 200
