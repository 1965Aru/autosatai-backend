# app/search_datasets.py
"""
Unified catalogue endpoint
──────────────────────────
• Agriculture Hotspot   → Sentinel-2 via Google Earth Engine
• Night-time lights     → NOAA VIIRS DNB Monthly V1 (Google EE)

Both return a JSON structure of the form:
{
  "datasets": [
    {
      "id":       str,
      "datetime": str,   # ISO-UTC
      "lat":      float,
      "lon":      float,
      "assets": {
        "ndvi":  str,    # GeoTIFF URL for NDVI
        "thumb": str     # PNG thumbnail
      }
    }, …
  ]
}
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

import ee
from dotenv import load_dotenv
from flask import Blueprint, jsonify, request, current_app
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from sentinelhub import (BBox, CRS, SHConfig, SentinelHubCatalog)

# ─────────────────── env / initialise ──────────────────────────────────
load_dotenv()

# Sentinel-Hub creds (unchanged)
_sh = SHConfig()
_sh.instance_id      = os.getenv("SH_INSTANCE_ID")
_sh.sh_client_id     = os.getenv("SH_CLIENT_ID")
_sh.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
if not all([_sh.instance_id, _sh.sh_client_id, _sh.sh_client_secret]):
    raise RuntimeError("❌ Sentinel-Hub credentials missing in .env")
sh_catalog = SentinelHubCatalog(config=_sh)

# VIIRS constants (unchanged)
EE_NTL_COLL     = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
EE_NTL_FALLBACK = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
VIIRS_BAND      = "avg_rad"

# Geocoder (unchanged)
geolocator = Nominatim(user_agent="autosatai-search")

# ─────────────────── Blueprint ────────────────────────────────────────
search_datasets_bp = Blueprint("search_datasets", __name__, url_prefix="/api/agri")


@search_datasets_bp.route("/datasets", methods=["POST"])
def search_datasets():
    """POST /api/agri/datasets → agriculture OR night-lights dispatch."""
    req       = request.get_json(silent=True) or {}
    location  = (req.get("location")  or "").strip()
    date_from = (req.get("date_from") or "").strip()
    date_to   = (req.get("date_to")   or "").strip()
    category  = (req.get("category")  or "").strip()

    if not all([location, date_from, date_to, category]):
        return jsonify(error="Missing required fields"), 400

    # parse & validate dates
    try:
        d0 = datetime.strptime(date_from, "%Y-%m-%d").date()
        d1 = datetime.strptime(date_to,   "%Y-%m-%d").date()
        if d0 > d1:
            raise ValueError
    except ValueError:
        return jsonify(error="Invalid date range (YYYY-MM-DD)"), 400

    # geocode
    try:
        g = geolocator.geocode(location, timeout=10)
        if g is None:
            return jsonify(error="Location not found"), 404
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        return jsonify(error=f"Geocoding failed: {exc}"), 502

    lat, lon = g.latitude, g.longitude

    # dispatch on category exactly like before
    if category == "Agriculture Hotspot":
        return _gee_sentinel2(lat, lon, d0, d1)
    if category == "Night Time Light Data":
        return _viirs_gee(lat, lon, d0, d1)

    return jsonify(error="Unsupported category"), 400


def _gee_sentinel2(lat: float, lon: float, d0, d1):
    """Query COPERNICUS/S2_SR via Earth Engine for agriculture hotspots."""
    point = ee.Geometry.Point([lon, lat])
    coll  = (
        ee.ImageCollection("COPERNICUS/S2_SR")
          .filterDate(d0.isoformat(), d1.isoformat())
          .filterBounds(point)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    if coll.size().getInfo() == 0:
        return jsonify(datasets=[]), 200

    indices = coll.aggregate_array("system:index").getInfo()
    times   = coll.aggregate_array("system:time_start").getInfo()

    datasets: List[Dict[str, Any]] = []
    for idx, tms in zip(indices, times):
        img    = coll.filter(ee.Filter.eq("system:index", idx)).first()
        dt_iso = datetime.utcfromtimestamp(tms / 1000).isoformat() + "Z"

        # true-color GeoTIFF (B4,B3,B2)
        try:
            data_url = img.getDownloadURL({
                "scale": 10,
                "crs":   "EPSG:4326",
                "region": point.buffer(5000).bounds(),
                "format": "GEO_TIFF",
                "bands": ["B4", "B3", "B2"],
                "name":   idx,
            })
        except Exception:
            continue

        # thumbnail (256px)
        try:
            thumb_url = img.select(["B4", "B3", "B2"]).getThumbURL({
                "dimensions": 256,
                "min":        [0, 0, 0],
                "max":        [3000, 3000, 3000],
            })
        except Exception:
            thumb_url = None

        entry = {
            "id":       idx,
            "datetime": dt_iso,
            "lat":      lat,
            "lon":      lon,
            "assets":   {
                "ndvi":  data_url,
                **({"thumb": thumb_url} if thumb_url else {})
            },
        }
        datasets.append(entry)

    datasets.sort(key=lambda x: x["datetime"], reverse=True)
    return jsonify(datasets=datasets), 200


def _viirs_gee(lat: float, lon: float, d0, d1):
    """Query VIIRS DNB monthly via Earth Engine for night-lights."""
    point = ee.Geometry.Point([lon, lat])

    def _query(coll_id):
        return ee.ImageCollection(coll_id).filterDate(d0.isoformat(), d1.isoformat()).filterBounds(point)

    icoll = _query(EE_NTL_COLL)
    if icoll.size().getInfo() == 0:
        icoll = _query(EE_NTL_FALLBACK)
        if icoll.size().getInfo() == 0:
            return jsonify(datasets=[]), 200

    indices = icoll.aggregate_array("system:index").getInfo()
    times   = icoll.aggregate_array("system:time_start").getInfo()

    datasets = []
    for idx, tms in zip(indices, times):
        img    = icoll.filter(ee.Filter.eq("system:index", idx)).first()
        dt_iso = datetime.utcfromtimestamp(tms / 1000).isoformat() + "Z"

        data_url = img.getDownloadURL({
            "scale": 463.83,
            "crs":   "EPSG:4326",
            "region": point.buffer(50_000).bounds(),
            "format": "GEO_TIFF",
            "name":   f"viirs_{idx}",
            "bands": [VIIRS_BAND],
        })

        try:
            thumb_url = img.select(VIIRS_BAND).getThumbURL({
                "dimensions": 256,
                "min":        0,
                "max":        65,
                "palette":    ["000000","ffffb2","fdae61","d7191c"],
            })
        except ee.ee_exception.EEException as exc:
            current_app.logger.warning("Thumbnail failed for %s: %s", idx, exc)
            thumb_url = None

        entry = {
            "id":       f"gee_viirs_{idx}",
            "datetime": dt_iso,
            "lat":      lat,
            "lon":      lon,
            "assets":   {"data": data_url, **({"thumb": thumb_url} if thumb_url else {})},
        }
        datasets.append(entry)

    datasets.sort(key=lambda d: d["datetime"], reverse=True)
    return jsonify(datasets=datasets), 200
