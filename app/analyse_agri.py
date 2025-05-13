from flask import Blueprint, request, jsonify, current_app
import ee
from datetime import datetime
from flask_cors import cross_origin

analyse_agri_bp = Blueprint("analyse_agri", __name__, url_prefix="/api/agri")

@analyse_agri_bp.route("/analyse", methods=["POST", "OPTIONS"])
@cross_origin()
def analyse_agri():
    # handle CORS preflight
    if request.method == "OPTIONS":
        return "", 200

    req        = request.get_json(silent=True) or {}
    dataset_id = req.get("dataset_id")
    analysis   = req.get("analysis")

    if not dataset_id or analysis != "agriculture_hotspot":
        return jsonify(error="Missing or unsupported analysis"), 400

    try:
        # ── 1) load & mask Sentinel-2 SR, add NDVI band ─────────────────────
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR")
              .filter(ee.Filter.eq("system:index", dataset_id))
              .map(lambda img: img.updateMask(
                  img.select("QA60").bitwiseAnd(1 << 10).eq(0)
                  .And(img.select("QA60").bitwiseAnd(1 << 11).eq(0))
              ))
              .map(lambda img: img
                  .divide(10000)
                  .addBands(
                      img.divide(10000)
                         .normalizedDifference(["B8", "B4"])
                         .rename("NDVI")
                  )
              )
              .select("NDVI")
        )
        img = coll.first()
        if img is None:
            return jsonify(error="Image not found"), 404

        # ── 2) get EE geometry and its client‐side dict ──────────────────────
        region_geom = img.geometry().bounds()
        region      = region_geom.getInfo()

        # ── 3) basic stats: mean, max, min, stdDev ──────────────────────────
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.max(),   "", True)
                .combine(ee.Reducer.min(),   "", True)
                .combine(ee.Reducer.stdDev(), "", True),
            geometry=region_geom,
            scale=10,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()

        # ── 4) full NDVI histogram ─────────────────────────────────────────
        hist_raw = img.reduceRegion(
            reducer=ee.Reducer.histogram(),
            geometry=region_geom,
            scale=10,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo().get("NDVI", {})

        if "histogram" in hist_raw:
            counts = hist_raw["histogram"]
            bucket_means = hist_raw.get("bucketMeans", [])

            # Fallback to compute bucketMeans if missing
            if not bucket_means and "bucketWidth" in hist_raw and "min" in hist_raw:
                min_ = hist_raw["min"]
                width = hist_raw["bucketWidth"]
                edges = [min_ + i * width for i in range(len(counts) + 1)]
                bucket_means = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

            hist = {
                "histogram": counts,
                "bucketMeans": bucket_means
            }
        else:
            hist = {"histogram": [], "bucketMeans": []}

        # ── 5) key percentiles: 25th, 50th, 75th ────────────────────────────
        pct = img.reduceRegion(
            reducer=ee.Reducer.percentile([25, 50, 75]),
            geometry=region_geom,
            scale=10,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()
        percentiles = {
            "p25": pct.get("NDVI_p25"),
            "p50": pct.get("NDVI_p50"),
            "p75": pct.get("NDVI_p75"),
        }

        # ── 6) download & thumbnail URLs ────────────────────────────────────
        download_url = img.getDownloadURL({
            "scale": 10,
            "crs": "EPSG:4326",
            "region": region,
            "format": "GEO_TIFF",
            "name": f"ndvi_{dataset_id}"
        })
        thumb_url = img.getThumbURL({
            "bands": ["NDVI"],
            "min": 0,
            "max": 1,
            "palette": ["FFFFFF", "006400"],
            "dimensions": 256,
            "region": region  # ✅ FIXED LINE
        })

        # ── 7) return full payload ────────────────────────────────────────
        return jsonify({
            "dataset_id":   dataset_id,
            "analysis":     analysis,
            "stats":        stats,
            "distribution": hist,
            "percentiles":  percentiles,
            "geometry":     region,
            "assets":       {"ndvi": download_url, "thumb": thumb_url},
            "timestamp":    datetime.utcnow().isoformat() + "Z"
        }), 200

    except Exception as exc:
        current_app.logger.error("Analysis failed for %s: %s", dataset_id, exc)
        return jsonify(error=str(exc)), 502
