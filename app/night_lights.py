# app/night_lights.py  ────────────────────────────────────────────────
import os
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import numpy as np
from sentinelhub import (
    SHConfig, BBox, CRS,
    SentinelHubRequest, DataCollection
)

load_dotenv()

night_lights_bp = Blueprint(
    "night_lights",
    __name__,
    url_prefix="/night-lights"
)

@night_lights_bp.route("/info", methods=["GET", "POST", "OPTIONS"])
@cross_origin(origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
def get_night_lights():
    if request.method == "OPTIONS":
        return "", 204

    # ── 1. Parse input ───────────────────────────────────────────────
    payload = request.get_json() if request.method == "POST" else request.args
    location = (payload or {}).get("location", "").strip()
    days     = int((payload or {}).get("days", 30))  # how many days back
    if not location:
        return jsonify({"error": "Missing 'location' field"}), 400

    # ── 2. Geocode to lat/lon ────────────────────────────────────────
    geolocator = Nominatim(user_agent="autosatai-night-lights")
    try:
        geo = geolocator.geocode(location)
        if not geo:
            return jsonify({"error": "Location not found"}), 404
        lon, lat = geo.longitude, geo.latitude
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        return jsonify({"error": f"Geocoding failed: {e}"}), 500

    # 3 × 3 km box (VIIRS is coarse, 750 m/px) 
    delta = 0.03
    bbox  = BBox((lon - delta, lat - delta, lon + delta, lat + delta), crs=CRS.WGS84)

    # ── 3. Sentinel-Hub credentials ──────────────────────────────────
    cfg = SHConfig()
    cfg.instance_id      = os.getenv("SH_INSTANCE_ID")
    cfg.sh_client_id     = os.getenv("SH_CLIENT_ID")
    cfg.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    if not all([cfg.instance_id, cfg.sh_client_id, cfg.sh_client_secret]):
        raise RuntimeError("Sentinel-Hub credentials missing")

    # ── 4. Build Process API request (VIIRS DNB) ─────────────────────
    end   = datetime.utcnow().date()
    start = end - timedelta(days=days)

    evalscript = """//VERSION=3
function setup() {
  return {input:["DNB_BRDF_Corrected_NTL"], output:{bands:1,sampleType:"FLOAT32"}};
}
function evaluatePixel(s) {
  // Radiance already in nW/cm2/sr in composite
  return [s.DNB_BRDF_Corrected_NTL];
}"""

    req = SentinelHubRequest(
        evalscript      = evalscript,
        input_data      = [SentinelHubRequest.input_data(
                              data_collection=DataCollection.VIIRS_DAY_NIGHT_BAND,
                              time_interval=(str(start), str(end)),
                              mosaicking_order="mostRecent"
                           )],
        responses       = [{"identifier":"default","format":{"type":"image/tiff"}}],
        bbox            = bbox,
        size            = (256, 256),
        config          = cfg,
    )

    # ── 5. Execute & analyse ─────────────────────────────────────────
    try:
        img = req.get_data()[0]  # shape (1, H, W)
    except Exception as e:
        return jsonify({"error": f"Sentinel-Hub VIIRS request failed: {e}"}), 500

    radiance = img.reshape(-1).astype(float)
    valid    = ~np.isnan(radiance)

    if not valid.any():
        return jsonify({"error": "No VIIRS data for this area/date"}), 404

    stats = {
        "mean_radiance_nW":   round(float(radiance[valid].mean()), 3),
        "max_radiance_nW":    round(float(radiance[valid].max()), 3),
        "bright_pixel_pct":   round(float((radiance[valid] > 20).sum()) / valid.sum() * 100, 1),
        "very_bright_pct":    round(float((radiance[valid] > 50).sum()) / valid.sum() * 100, 1),
        "threshold_nW":       20,
    }

    result = {
        **stats,
        "location":   location,
        "latitude":   lat,
        "longitude":  lon,
        "bbox":       bbox.get_coordinates(),
        "days_range": days,
        "timestamp":  datetime.utcnow().isoformat() + "Z"
    }

    return jsonify(result), 200
