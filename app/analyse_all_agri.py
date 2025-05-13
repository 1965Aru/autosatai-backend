import os
import tempfile
from datetime import datetime

import numpy as np
import rasterio
import requests
from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin

analyse_all_agri_bp = Blueprint("analyse_all_agri", __name__, url_prefix="/api/agri")


def _download_geotiff(url: str) -> str:
    """Download a GeoTIFF from the given URL to a temp file, return its path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    for chunk in resp.iter_content(8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


@analyse_all_agri_bp.route("/analyse-all", methods=["OPTIONS", "POST"])
@cross_origin()  # keep it simple like night-lights
def analyse_all_agri():
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return "", 200

    payload = request.get_json(silent=True) or {}
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return jsonify(error="jobs list required"), 400

    # Prepare accumulators
    results = []
    dates = []
    means, maxs, mins, stds = [], [], [], []

    for job in jobs:
        ds_id    = job.get("dataset_id")
        analysis = job.get("analysis")
        assets   = job.get("assets", {})

        # Front-end may use `.ndvi` or older `.data`
        url   = assets.get("ndvi") or assets.get("data")
        thumb = assets.get("thumb")  # FIXED: do not construct broken :getMap fallback

        if analysis != "agriculture_hotspot":
            continue
        if not ds_id or not url:
            results.append({
                "dataset_id": ds_id,
                "error":      "Missing dataset_id or data URL"
            })
            continue

        try:
            # 1) download the NDVI GeoTIFF
            path = _download_geotiff(url)

            # 2) read the single band as masked array, extract valid values
            with rasterio.open(path) as src:
                arr = src.read(1, masked=True).compressed()

            # cleanup
            os.remove(path)

            # 3) compute stats (or None if empty)
            if arr.size == 0:
                stats = {
                    "NDVI_mean":   None,
                    "NDVI_max":    None,
                    "NDVI_min":    None,
                    "NDVI_stdDev": None,
                }
                distribution = {"histogram": [], "bucketMeans": []}
                percentiles = {"p25": None, "p50": None, "p75": None}
            else:
                stats = {
                    "NDVI_mean":   float(arr.mean()),
                    "NDVI_max":    float(arr.max()),
                    "NDVI_min":    float(arr.min()),
                    "NDVI_stdDev": float(arr.std()),
                }
                # Accumulate for batch summary & series
                means.append(stats["NDVI_mean"])
                maxs.append(stats["NDVI_max"])
                mins.append(stats["NDVI_min"])
                stds.append(stats["NDVI_stdDev"])

                # ─── normalize NDVI to [0, 1] before histogram ─────────────
                arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
                arr_norm = np.clip(arr_norm, 0, 1)

                # ─── compute histogram & bucket means ─────────────────────
                hist, bin_edges = np.histogram(arr_norm, bins=10, range=(0.0, 1.0))
                bucket_means = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
                distribution = {
                    "histogram": hist.tolist(),
                    "bucketMeans": bucket_means
                }

                # ─── compute percentiles on original arr ──────────────────
                p25 = float(np.percentile(arr, 25))
                p50 = float(np.percentile(arr, 50))
                p75 = float(np.percentile(arr, 75))
                percentiles = {"p25": p25, "p50": p50, "p75": p75}
                # ──────────────────────────────────────────────────────────

            # 4) timestamp for this tile
            ts = datetime.utcnow().isoformat() + "Z"
            dates.append(ts)

            # 5) build the per-tile result
            results.append({
                "dataset_id":   ds_id,
                "analysis":     analysis,
                "stats":        stats,
                "distribution": distribution,
                "percentiles":  percentiles,
                "assets":       {"ndvi": url, "thumb": thumb},
                "timestamp":    ts
            })

        except Exception as exc:
            current_app.logger.error(
                "Batch agriculture analysis failed for %s: %s", ds_id, exc
            )
            results.append({
                "dataset_id": ds_id,
                "error":      str(exc)
            })

    # If nothing succeeded, return empty results
    if not results:
        return jsonify(results=[]), 200

    # Compute batch summary (averages)
    summary = {}
    count = len(means)
    if count > 0:
        summary = {
            "NDVI_mean_avg":   sum(means)  / count,
            "NDVI_max_avg":    sum(maxs)   / count,
            "NDVI_min_avg":    sum(mins)   / count,
            "NDVI_stdDev_avg": sum(stds)   / count,
        }

    # Build time-series for front-end
    series = {
        "dates":   dates,
        "mean":    means,
        "max":     maxs,
        "min":     mins,
        "stdDev":  stds,
    }

    # ─── Simple anomaly & breakpoint detection ────────────────────────
    anomalies = []
    breakpoints = []
    if count > 1:
        arr = np.array(means, dtype=float)
        mu, sigma = arr.mean(), arr.std(ddof=0)
        # anomalies: |x - μ| > 2σ
        anomalies = [int(i) for i, x in enumerate(arr) if sigma and abs(x - mu) > 2 * sigma]
        # breakpoints: jumps between consecutive points > σ
        for i in range(1, len(arr)):
            if sigma and abs(arr[i] - arr[i-1]) > sigma:
                breakpoints.append(i)
    # ──────────────────────────────────────────────────────────────────

    # Return everything in one payload
    return jsonify(
        results=results,
        summary=summary,
        series=series,
        anomalies=anomalies,
        breakpoints=breakpoints,
    ), 200
