#!/usr/bin/env python3
import os
import re
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify, send_from_directory, abort
from flask_cors import cross_origin
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import numpy as np

from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    SentinelHubRequest,
    DataCollection,
)

# ─── for segmentation output ─────────────────────────────────────────────
from scipy.ndimage import binary_opening, binary_closing
import rasterio
from rasterio.transform import from_bounds
from imageio import imwrite

# ─── USER CONFIG ──────────────────────────────────────────────────────────
load_dotenv()

natural_resources_bp = Blueprint(
    "natural_resources",
    __name__,
    url_prefix="/natural-resources",
)

# where we’ll dump our fast outputs
FAST_NR_DIR = "fast_nr"
os.makedirs(FAST_NR_DIR, exist_ok=True)

# ─── HELPERS ───────────────────────────────────────────────────────────────
def slugify(text):
    """Turn arbitrary string into filesystem‐safe slug."""
    safe = re.sub(r'[^0-9a-zA-Z]+', '_', text).strip('_')
    return safe.lower()

def classify_indices(img_data):
    """
    img_data: H×W×5 array of [ndvi, ndwi, ndbi, ndmi, mndwi]
    returns: H×W uint8 mask {1=forest,2=water,3=built-up,4=soil}
    """
    ndvi, ndwi, ndbi, ndmi, mndwi = img_data.transpose(2, 0, 1)
    mask = np.zeros(ndvi.shape, np.uint8)
    mask[ndvi  >= 0.30]               = 1  # forest
    mask[(mask == 0) & (ndwi >= 0.20)] = 2  # water
    mask[(mask == 0) & (ndbi >= 0.10)] = 3  # built-up/mineral
    mask[mask == 0]                   = 4  # soil/other

    # tiny speckle cleanup
    struct = np.ones((3, 3), bool)
    for cls in (1, 2, 3, 4):
        m = (mask == cls)
        m = binary_opening(m, structure=struct)
        m = binary_closing(m, structure=struct)
        mask[m] = cls

    return mask

# ─── MAIN INFO ROUTE ──────────────────────────────────────────────────────
@natural_resources_bp.route("/info", methods=["GET", "POST", "OPTIONS"])
@cross_origin(origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
def get_natural_resources_info():
    if request.method == "OPTIONS":
        return "", 204

    # — parse & slugify location —
    location = (
        (request.get_json() or {}).get("location", "").strip()
        if request.method == "POST"
        else request.args.get("location", "").strip()
    )
    if not location:
        return jsonify({"error": "Missing 'location' field"}), 400
    slug = slugify(location)
    out_dir = os.path.join(FAST_NR_DIR, slug)
    os.makedirs(out_dir, exist_ok=True)

    # — geocode —
    geolocator = Nominatim(user_agent="autosatai-natural-resources")
    try:
        geo = geolocator.geocode(location, timeout=10)
        if not geo:
            return jsonify({"error": "Location not found"}), 404
        lon, lat = geo.longitude, geo.latitude
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        return jsonify({"error": f"Geocoding failed: {e}"}), 500

    # — build bbox & time span —
    delta = 0.05
    bbox = BBox((lon - delta, lat - delta, lon + delta, lat + delta), crs=CRS.WGS84)
    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=7)
    time_interval = (str(start_date), str(end_date))

    # — configure Sentinel-Hub —
    cfg = SHConfig()
    cfg.instance_id      = os.getenv("SH_INSTANCE_ID")
    cfg.sh_client_id     = os.getenv("SH_CLIENT_ID")
    cfg.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    if not all([cfg.instance_id, cfg.sh_client_id, cfg.sh_client_secret]):
        raise RuntimeError("Sentinel-Hub credentials missing")

    # — main evalscript for indices —
    evalscript = """//VERSION=3
    function setup() {
      return {input:["B03","B04","B08","B8A","B11"],output:{bands:5,sampleType:"FLOAT32"}};
    }
    function evaluatePixel(s) {
      let ndvi  = (s.B08 - s.B04)/(s.B08 + s.B04);
      let ndwi  = (s.B03 - s.B08)/(s.B03 + s.B08);
      let ndbi  = (s.B11 - s.B08)/(s.B11 + s.B08);
      let ndmi  = (s.B8A - s.B11)/(s.B8A + s.B11);
      let mndwi = (s.B03 - s.B11)/(s.B03 + s.B11);
      return [ndvi, ndwi, ndbi, ndmi, mndwi];
    }
    """

    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            mosaicking_order="mostRecent",
            other_args={"dataFilter": {"maxCloudCoverage": 20}}
        )],
        responses=[{"identifier":"default","format":{"type":"image/tiff"}}],
        bbox=bbox,
        size=(512, 512),
        config=cfg,
    )

    try:
        img_data = req.get_data()[0]  # shape (512,512,5)
    except Exception as e:
        return jsonify({"error": f"Sentinel-Hub request failed: {e}"}), 500

    # — generate raw true-color PNG preview —
    evalscript_rgb = """//VERSION=3
    function setup() { return {input:["B04","B03","B02"],output:{bands:3}}; }
    function evaluatePixel(s) { return [s.B04, s.B03, s.B02]; }
    """
    req_rgb = SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            mosaicking_order="mostRecent",
            other_args={"dataFilter": {"maxCloudCoverage": 20}}
        )],
        responses=[{"identifier":"default","format":{"type":"image/png"}}],
        bbox=bbox,
        size=(512, 512),
        config=cfg,
    )
    try:
        rgb_img = req_rgb.get_data()[0]  # numpy array (H×W×3)
        raw_preview_path = os.path.join(out_dir, "raw_preview.png")
        imwrite(raw_preview_path, rgb_img)
    except Exception:
        raw_preview_path = None

    # — flat bands → basic index stats —
    bands = img_data.reshape(-1, 5).T
    ndvi, ndwi, ndbi, ndmi, mndwi = bands
    def pct(arr, th):
        m = ~np.isnan(arr)
        return round(float((arr[m] > th).sum()) / m.sum() * 100, 1) if m.any() else 0.0

    result = {
        "slug":                           slug,
        "location":                       location,
        "latitude":                       lat,
        "longitude":                      lon,
        "timestamp":                      datetime.utcnow().isoformat() + "Z",
        "forest_area_percentage":         pct(ndvi, 0.30),
        "water_body_percentage":          pct(ndwi, 0.20),
        "mineral_richness_index":         pct(ndbi, 0.10),
        "soil_moisture_index":            pct(ndmi, 0.00),
        "enhanced_water_body_percentage": pct(mndwi, 0.20),
    }

    # — segmentation mask & extra stats —
    mask = classify_indices(img_data)  # H×W uint8
    total_px = mask.size
    seg_pcts = {
        "forest_segmentation_percentage":   round((mask == 1).sum() / total_px * 100, 1),
        "water_segmentation_percentage":    round((mask == 2).sum() / total_px * 100, 1),
        "builtup_segmentation_percentage":  round((mask == 3).sum() / total_px * 100, 1),
        "soil_segmentation_percentage":     round((mask == 4).sum() / total_px * 100, 1),
    }
    result.update(seg_pcts)

    # — persist preview, mask, CSV into slug folder —
    h, w = mask.shape

    # 1) color‐coded PNG preview
    cmap = {
        1: (34, 139,  34),   # forest → green
        2: (  0, 191, 255),  # water  → blue
        3: (169, 169, 169),  # built-up → gray
        4: (210, 180, 140),  # soil → tan
    }
    rgb = np.zeros((h, w, 3), np.uint8)
    for cls, col in cmap.items():
        rgb[mask == cls] = col
    preview_path = os.path.join(out_dir, "preview.png")
    imwrite(preview_path, rgb)

    # 2) GeoTIFF mask
    mask_path = os.path.join(out_dir, "mask.tif")
    transform = from_bounds(
        lon - delta, lat - delta, lon + delta, lat + delta,
        w, h
    )
    with rasterio.open(
        mask_path, "w", driver="GTiff",
        height=h, width=w, count=1,
        dtype="uint8", crs="EPSG:4326",
        transform=transform
    ) as dst:
        dst.write(mask, 1)

    # 3) summary.csv of pixel counts
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        f.write("class,pixels\n")
        for cls in (1, 2, 3, 4):
            f.write(f"{cls},{int((mask == cls).sum())}\n")

    # ─── add download URLs for front-end schema ───────────────────────────
    base = request.host_url.rstrip('/')
    urls = {
        "preview_png": f"{base}/natural-resources/preview/{slug}",
        "mask_tif":    f"{base}/natural-resources/download/{slug}/mask.tif",
        "summary_csv": f"{base}/natural-resources/download/{slug}/summary.csv",
    }
    if raw_preview_path:
        urls["raw_png"] = f"{base}/natural-resources/download/{slug}/raw_preview.png"
    result.update(urls)

    return jsonify(result), 200

# ─── SERVE THE PREVIEW PNG ────────────────────────────────────────────────
@natural_resources_bp.route("/preview/<slug>")
def serve_preview(slug):
    folder = os.path.join(FAST_NR_DIR, slug)
    preview = os.path.join(folder, "preview.png")
    if not os.path.exists(preview):
        abort(404)
    return send_from_directory(folder, "preview.png", mimetype="image/png")

# ─── SUMMARY JSON (pixel counts) ─────────────────────────────────────────
@natural_resources_bp.route("/summary/<slug>")
def serve_summary(slug):
    folder = os.path.join(FAST_NR_DIR, slug)
    csv_file = os.path.join(folder, "summary.csv")
    if not os.path.exists(csv_file):
        abort(404)
    resp = {}
    with open(csv_file) as f:
        next(f)  # skip header
        for line in f:
            cls, px = line.strip().split(",")
            resp[f"class_{cls}"] = int(px)
    return jsonify(resp), 200

# ─── GENERIC DOWNLOAD ENDPOINT ───────────────────────────────────────────
@natural_resources_bp.route("/download/<slug>/<filename>")
def download_file(slug, filename):
    folder = os.path.join(FAST_NR_DIR, slug)
    safe_name = filename  # our code only writes preview.png, mask.tif, summary.csv, raw_preview.png
    file_path = os.path.join(folder, safe_name)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(folder, safe_name, as_attachment=True)
