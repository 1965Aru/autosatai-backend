import os
import json
import time
import re
import requests
from datetime import datetime
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin

load_dotenv()

# Blueprint for agriculture reports
# strict_slashes=False prevents Flask from redirecting /report/agri ↔ /report/agri/
report_agri_bp = Blueprint("report_agri", __name__, url_prefix="/report/agri")

# Load Gemini configuration from environment
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT")

if not GEMINI_API_KEY or not GEMINI_ENDPOINT:
    raise RuntimeError("GEMINI_API_KEY and GEMINI_ENDPOINT must be set in .env")

# retry settings for overloaded model
MAX_RETRIES = 3
BACKOFF_SEC = 2

@report_agri_bp.route("/", methods=["POST", "OPTIONS"], strict_slashes=False)
@cross_origin(
    origins="*",
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)
def generate_agri_report():
    """
    POST /report/agri
    Accepts either:
      • { "datasets": [...], "analysisResults": [...] }
      • or legacy single‐item payload with dataset_id, stats, distribution, percentiles
    Returns JSON with:
      title, generated_at, metadata, metrics, timeseries, histogram, anomalies,
      spatial_summary, and report_sections (with executive_summary, key_findings,
      methodology, spatial_summary, recommendations).
    """
    # CORS preflight
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}

    # 1) Extract datasets & analysis results
    datasets_list     = payload.get("datasets", [])
    analysis_results  = payload.get("analysisResults") or payload.get("singleResults") or []
    # If no array, perhaps caller sent a single‐item shape directly
    if not isinstance(analysis_results, list):
        analysis_results = [payload]

    # 2) Pick the latest analysis entry
    latest = analysis_results[-1] if analysis_results else {}
    dataset_id   = latest.get("dataset_id", "")
    stats        = latest.get("stats", {})
    distribution = latest.get("distribution", {})
    percentiles  = latest.get("percentiles", {})

    # 3) Optional time‐series fields (if someone passed them)
    series      = payload.get("series", {})
    anomalies   = payload.get("anomalies", [])
    breakpoints = payload.get("breakpoints", [])

    # 4) Basic validation
    if not dataset_id or not stats:
        return jsonify(error="Missing dataset_id or stats in payload"), 400

    # 5) Build metadata
    metadata = {
        "location": payload.get("location", "Unknown"),
        "date_range": {"from": "", "to": ""},
        "datasets_count": len(datasets_list)
    }
    # derive date_range from the datasets array if available
    datetimes = [d.get("datetime") for d in datasets_list if d.get("datetime")]
    if datetimes:
        metadata["date_range"]["from"] = min(datetimes)
        metadata["date_range"]["to"]   = max(datetimes)

    # 6) Build detailed JSON‐structured prompt for Gemini
    prompt = json.dumps({
        "prompt_type": "agriculture_report",
        "dataset_id": dataset_id,
        "stats": {
            "mean": stats.get("NDVI_mean"),
            "min":  stats.get("NDVI_min"),
            "max":  stats.get("NDVI_max"),
            "stdDev": stats.get("NDVI_stdDev")
        },
        "timeseries": series,
        "distribution": distribution,
        "percentiles": percentiles,
        "anomalies": anomalies,
        "breakpoints": breakpoints,
        "instructions": {
            "structure": {
                "executive_summary": "2–3 sentence overview",
                "key_findings":      "bullet list of top 5 insights",
                "methodology":       "brief text data sources & NDVI calculation",
                "spatial_summary":   "narrative of hotspot locations/clusters",
                "recommendations":   "next steps"
            },
            "return_format": "JSON only, no extra prose"
        }
    })

    # 7) Call the Gemini API with retry/back-off
    url  = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    body = {"contents":[{"parts":[{"text":prompt}]}]}
    resp = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=body, timeout=60)
        except Exception as e:
            current_app.logger.error("Gemini network error on attempt %d: %s", attempt, e)
            if attempt == MAX_RETRIES:
                return jsonify(error="Failed to reach Gemini API"), 502
            time.sleep(BACKOFF_SEC * attempt)
            continue

        if resp.status_code == 200:
            break
        elif resp.status_code == 503 and attempt < MAX_RETRIES:
            current_app.logger.warning("Gemini busy (503), retrying attempt %d/%d...", attempt, MAX_RETRIES)
            time.sleep(BACKOFF_SEC * attempt)
            continue
        else:
            # non-200 or exhausted retries on 503
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            current_app.logger.error("Gemini API error (%d): %s", resp.status_code, err)
            return jsonify(error="Gemini returned an error", detail=err), 502

    gemini_out = resp.json()

    # --- DEBUG LOG: display raw Gemini response in VSCode console ---
    current_app.logger.debug("🛰️ Gemini raw response:\n%s", json.dumps(gemini_out, indent=2))

    # 8) Extract, strip any Markdown fences, and parse the JSON report sections
    text_out = ""
    if gemini_out.get("candidates"):
        cand = gemini_out["candidates"][0]
        content_obj = cand.get("content", {}) or {}
        parts = content_obj.get("parts", [])
        if parts and isinstance(parts, list):
            text_out = parts[0].get("text", "")
    else:
        text_out = (
            gemini_out.get("choices", [{}])[0]
                     .get("message", {})
                     .get("content", "")
            or gemini_out.get("text", "")
        )

    # strip triple-backtick fences around JSON, if present
    stripped = text_out.strip()
    m = re.match(r"^```(?:json)?\s*(\{.*\})\s*```$", stripped, flags=re.DOTALL)
    if m:
        stripped = m.group(1)

    try:
        report_sections = json.loads(stripped)
    except Exception:
        # Fallback: wrap plain text
        report_sections = {"executive_summary": stripped}

    # 9) Assemble structured JSON response
    response = {
        "title":        f"Agriculture Report — {dataset_id}",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metadata":     metadata,
        "metrics": [
            {"label": "Mean NDVI",   "value": stats.get("NDVI_mean")},
            {"label": "Min NDVI",    "value": stats.get("NDVI_min")},
            {"label": "Max NDVI",    "value": stats.get("NDVI_max")},
            {"label": "StdDev NDVI", "value": stats.get("NDVI_stdDev")}
        ],
        "timeseries": {
            "dates":    series.get("dates", []),
            "avg_ndvi": series.get("avg_ndvi", series.get("mean", [])),
            "max_ndvi": series.get("max_ndvi", series.get("max", []))
        },
        "histogram": {
            "bins":   distribution.get("bucketMeans", []),
            "counts": distribution.get("histogram",   [])
        },
        "anomalies": [
            {"date": d, "description": "Anomaly detected"} for d in anomalies
        ],
        # fill spatial_summary from the Gemini output if provided
        "spatial_summary": report_sections.get("spatial_summary", ""),
        # put the full free-form sections under "report"
        "report": report_sections
    }

    return jsonify(response), 200
