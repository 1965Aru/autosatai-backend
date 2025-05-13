import os
import json
import re
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin

load_dotenv()

report_nl_bp = Blueprint("report_night_lights", __name__, url_prefix="/report/night-lights")

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT")
if not GEMINI_API_KEY or not GEMINI_ENDPOINT:
    raise RuntimeError("GEMINI_API_KEY and GEMINI_ENDPOINT must be set in .env")

# Retry settings
MAX_RETRIES = 3
BACKOFF_SEC = 2

@report_nl_bp.route("", methods=["POST", "OPTIONS"])
@report_nl_bp.route("/", methods=["POST", "OPTIONS"])
@cross_origin(
    origins="*",
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)
def generate_night_lights_report():
    if request.method == "OPTIONS":
        return "", 204

    payload         = request.get_json(silent=True) or {}
    datasets        = payload.get("datasets", [])
    analysis_results = payload.get("analysisResults") or []

    if not analysis_results:
        return jsonify(error="Missing analysisResults in payload"), 400

    reports = []
    for ds in analysis_results:
        dataset_id = ds.get("dataset_id")
        series     = ds.get("series", {})

        dates      = series.get("dates", [])
        avg_rad    = series.get("avg_radiance", [])
        pct_bright = series.get("pct_bright", [])
        lit_area   = series.get("lit_area_km2", [])
        forecast   = series.get("forecast", {})
        residual   = series.get("residual", [])

        # derive top-3 anomalies
        anomalies = [
            {"date": dates[i], "type": "bright" if r>0 else "dim", "value": r}
            for i, r in enumerate(residual)
        ]
        anomalies = sorted(anomalies, key=lambda a: abs(a["value"]), reverse=True)[:3]

        # build Gemini prompt
        prompt = f"""
You are an expert in remote sensing and night-time lights analysis. Prepare a detailed report for dataset '{dataset_id}' covering:
  • Temporal trends in average radiance: {avg_rad}
  • Percentage of bright pixels over time: {pct_bright}
  • Lit area (km²) evolution: {lit_area}
  • Notable anomalies detected on dates: {[a['date'] for a in anomalies]}
  • Short-term forecast for radiance: dates {forecast.get('dates')} with {forecast.get('avg_radiance')}

Return a JSON object with these keys:
{{
  "executive_summary": "…",
  "key_findings": [ … ],
  "methodology": "…",
  "spatial_summary": "…",
  "recommendations": "…"
}}
Only return the JSON—no extra prose or code fences.
""".strip()

        # call Gemini with retry/backoff
        url  = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
        body = {"contents":[{"parts":[{"text": prompt}]}]}
        gemini_out = None
        for attempt in range(1, MAX_RETRIES+1):
            try:
                resp = requests.post(url, json=body, timeout=60)
                if resp.status_code == 200:
                    gemini_out = resp.json()
                    break
                elif resp.status_code == 503 and attempt < MAX_RETRIES:
                    current_app.logger.warning("Gemini busy, retry %d/%d", attempt, MAX_RETRIES)
                    time.sleep(BACKOFF_SEC * attempt)
                    continue
                else:
                    resp.raise_for_status()
            except Exception as exc:
                current_app.logger.error("Gemini request failed for %s on attempt %d: %s", dataset_id, attempt, exc)
                if attempt == MAX_RETRIES:
                    gemini_out = None
                else:
                    time.sleep(BACKOFF_SEC * attempt)
        # parse Gemini output
        if not gemini_out:
            report_sections = {"executive_summary": "Failed to generate report."}
        else:
            # extract text
            text_out = ""
            if gemini_out.get("candidates"):
                parts = gemini_out["candidates"][0].get("content", {}).get("parts", [])
                text_out = parts[0].get("text", "") if parts else ""
            else:
                # fallback legacy
                choices = gemini_out.get("choices", [])
                if choices:
                    text_out = choices[0].get("message", {}).get("content", "") or gemini_out.get("text", "")
                else:
                    text_out = gemini_out.get("text", "")

            stripped = text_out.strip()
            m = re.match(r"^```(?:json)?\s*(\{.*\})\s*```$", stripped, flags=re.DOTALL)
            if m:
                stripped = m.group(1)
            try:
                report_sections = json.loads(stripped)
            except Exception:
                report_sections = {"executive_summary": stripped}

        # build KPIs
        kpis = []
        if avg_rad:
            kpis.append({"name":"Average Radiance","value":round(avg_rad[-1],2)})
        if lit_area:
            kpis.append({"name":"Lit Area (km²)","value":round(lit_area[-1],1)})
        if pct_bright:
            kpis.append({"name":"Bright Pixels (%)","value":round(pct_bright[-1],1),"suffix":"%"})

        # metadata location
        location = dataset_id
        for d in datasets:
            if d.get("id")==dataset_id and d.get("lat") is not None and d.get("lon") is not None:
                location = f"{d['lat']:.4f}, {d['lon']:.4f}"
                break

        date_from = dates[0] if dates else None
        date_to   = dates[-1] if dates else None

        reports.append({
            "title":        f"Night-Lights Report for {dataset_id}",
            "generated_at": datetime.utcnow().isoformat()+"Z",
            "metadata": {
                "location":       location,
                "date_range":     {"from":date_from,"to":date_to},
                "datasets_count": len(datasets)
            },
            "kpis":     kpis,
            "series": {
                "dates":       dates,
                "avg_radiance":avg_rad,
                "pct_bright":  pct_bright,
                "lit_area_km2":lit_area,
                "forecast":    forecast
            },
            "anomalies": anomalies,
            "report":    report_sections
        })

    return jsonify(reports=reports), 200
