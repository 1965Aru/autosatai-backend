# app/analyse_all_nights.py

from __future__ import annotations
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import rasterio
import requests
from flask import Blueprint, request, jsonify, current_app
from flask_cors import CORS, cross_origin
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Optional imports; if missing, these sections are skipped.
try:
    from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except ImportError:
    seasonal_decompose = None  # type: ignore
    ARIMA = None               # type: ignore

# ───────────────── constants ────────────────────────────────────────────
THRESH_RADIANCE = 5                     # nW cm⁻² sr⁻¹
PIXEL_M        = 463.83                 # metres per pixel
PIXEL_AREA_KM2 = (PIXEL_M / 1_000) ** 2  # km² per pixel

# ───────────────── blueprint ────────────────────────────────────────────
analyse_all_bp = Blueprint("analyse_all", __name__, url_prefix="/api")
CORS(analyse_all_bp, origins="*")  # allow CORS

def _download_geotiff(url: str) -> str:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    for chunk in resp.iter_content(8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name

@analyse_all_bp.route("/analyse-all-nights", methods=["OPTIONS", "POST"])
@cross_origin()  # fallback
def analyse_all():
    if request.method == "OPTIONS":
        return "", 200

    payload = request.get_json(silent=True) or {}
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return jsonify(error="jobs list required"), 400

    # 1) Download & read
    tif_paths = []
    for job in jobs:
        if job.get("analysis") != "night_lights":
            continue
        ds_id = job.get("dataset_id")
        url   = job.get("assets", {}).get("data")
        if not url:
            current_app.logger.warning("No URL for %s", ds_id)
            continue
        try:
            path = _download_geotiff(url)
            tif_paths.append((ds_id, path))
        except Exception:
            current_app.logger.exception("Download failed %s", ds_id)

    if not tif_paths:
        return jsonify(results=[]), 200

    arrays, dates = [], []
    for ds_id, path in tif_paths:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1, masked=True).filled(np.nan)
        except Exception:
            current_app.logger.exception("Read failed %s", ds_id)
            continue
        arrays.append(arr)
        # parse date
        suf = ds_id.rsplit("_",1)[-1]
        try:
            if suf.isdigit() and len(suf)==8:
                dt = datetime.strptime(suf, "%Y%m%d").date().isoformat()
            else:
                dt = datetime.fromisoformat(suf.rstrip("Z")).date().isoformat()
        except:
            dt = ""
        dates.append(dt)

    # cleanup temp files
    for _, p in tif_paths:
        try: os.remove(p)
        except: pass

    stack = np.stack(arrays, axis=0)  # (T, H, W)
    T, H, W = stack.shape

    # global histogram bins
    all_vals = np.concatenate([a[a>THRESH_RADIANCE].ravel() for a in arrays])
    hist_bins = np.histogram_bin_edges(all_vals, bins=50) if all_vals.size else np.linspace(0,1,51)

    # prepare outputs
    metrics_over_time = []
    series = {
        "dates": dates,
        "avg_radiance": [],
        "max_radiance": [],
        "pct_bright": [],
        "lit_area_km2": [],
        "new_lit_km2": [],
        "bright_clusters": [],
        "cluster_sizes": [],
        "hist_counts": [],
    }

    prev_lit = 0.0
    # 2) per-time metrics
    for t in range(T):
        arr = stack[t]
        data = np.nan_to_num(arr, nan=0.0)
        mask = data > THRESH_RADIANCE
        vals = data[mask]

        # basic stats
        mean  = float(vals.mean())   if vals.size else 0.0
        mx    = float(vals.max())    if vals.size else 0.0
        total = float(vals.sum())
        lit_area = mask.sum() * PIXEL_AREA_KM2
        new_lit  = max(0.0, lit_area - prev_lit)
        prev_lit = lit_area

        # histogram
        counts, _ = np.histogram(vals, bins=hist_bins)

        # cluster morphology
        sizes = []
        if vals.size:
            ys, xs = np.where(mask)
            lbls = DBSCAN(eps=3, min_samples=20).fit_predict(np.c_[ys,xs])
            for lab in set(lbls):
                if lab == -1: continue
                count = int((lbls==lab).sum())
                sizes.append(count)
        cluster_cnt = len(sizes)
        cluster_areas = [s * PIXEL_AREA_KM2 for s in sizes]

        # record
        metrics_over_time.append({
            "date": dates[t],
            "mean": mean,
            "max": mx,
            "total_radiance": total,
            "lit_area_km2": lit_area,
            "new_lit_km2": new_lit,
            "bright_clusters": cluster_cnt,
            "cluster_count": cluster_cnt,
            "avg_cluster_km2": float(np.mean(cluster_areas)) if cluster_areas else 0.0,
            "max_cluster_km2": float(np.max(cluster_areas))  if cluster_areas else 0.0,
        })

        series["avg_radiance"].append(mean)
        series["max_radiance"].append(mx)
        series["lit_area_km2"].append(lit_area)
        series["new_lit_km2"].append(new_lit)
        series["pct_bright"].append(float(mask.sum()/data.size * 100))
        series["bright_clusters"].append(cluster_cnt)
        series["cluster_sizes"].append(sizes)
        series["hist_counts"].append(counts.tolist())

    # 3) Decompose seasonality
    if seasonal_decompose:
        try:
            period = max(2, T//4)
            dec = seasonal_decompose(series["avg_radiance"], period=period,
                                     model="additive", two_sided=False,
                                     extrapolate_trend="freq")
            series["trend"]    = [float(x) if not np.isnan(x) else 0 for x in dec.trend]
            series["seasonal"]= [float(x) for x in dec.seasonal]
            series["residual"]= [float(x) if not np.isnan(x) else 0 for x in dec.resid]
        except:
            series.update({k:[0]*T for k in ("trend","seasonal","residual")})
    else:
        series.update({k:[0]*T for k in ("trend","seasonal","residual")})

    # 4) Anomaly detection on residuals
    try:
        iso = IsolationForest(contamination=0.1, random_state=0)
        resid = np.array(series["residual"]).reshape(-1,1)
        iso.fit(resid)
        preds = iso.predict(resid)
        series["anomalies"] = [dates[i] for i,p in enumerate(preds) if p==-1]
    except:
        series["anomalies"] = []

    # 5) Forecast average radiance
    if ARIMA:
        try:
            model = ARIMA(series["avg_radiance"], order=(1,1,1))
            fit = model.fit()
            steps = min(3, T)
            fc = fit.forecast(steps=steps)
            last = datetime.fromisoformat(dates[-1])
            delta = (last - datetime.fromisoformat(dates[-2])) if T>1 else timedelta(days=30)
            f_dates = [(last + delta*(i+1)).date().isoformat() for i in range(steps)]
            series["forecast"] = {"dates": f_dates, "avg_radiance": [float(x) for x in fc]}
        except:
            series["forecast"] = {"dates": [], "avg_radiance": []}
    else:
        series["forecast"] = {"dates": [], "avg_radiance": []}

    # 6) 3D spatio-temporal clustering
    pts = []
    for t in range(T):
        mask = (np.nan_to_num(stack[t], nan=0.0) > THRESH_RADIANCE)
        ys,xs = np.where(mask)
        pts.extend([(t,y,x) for y,x in zip(ys,xs)])
    if pts:
        try:
            pts_arr = np.array(pts)
            st_lbl = DBSCAN(eps=2, min_samples=20).fit_predict(pts_arr)
            series["st_cluster_count"] = len(set(st_lbl) - {-1})
        except:
            series["st_cluster_count"] = 0
    else:
        series["st_cluster_count"] = 0

    # 7) PCA on per-pixel time-series (only lit pixels)
    # build matrix: N_lit x T
    mask_any = np.any(stack > THRESH_RADIANCE, axis=0)
    coords = np.column_stack(np.where(mask_any))
    if coords.size:
        mat = np.stack([stack[:,y,x] for y,x in coords], axis=1)  # (T, N)
        try:
            pca = PCA(n_components=2)
            comp = pca.fit_transform(mat)  # (T,2)? Actually PCA on features -> we want transpose: run on mat.T
            comp = PCA(n_components=2).fit_transform(mat.T)
            series["pca"] = {
                "x": comp[:,0].tolist(),
                "y": comp[:,1].tolist(),
                "coords": coords.tolist()
            }
        except:
            series["pca"] = {"x":[], "y":[], "coords":[]}
    else:
        series["pca"] = {"x":[], "y":[], "coords":[]}

    # assemble batch
    batch_id = f"night_lights_batch_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    result = {
        "dataset_id": batch_id,
        "metrics_over_time": metrics_over_time,
        "series": series
    }

    return jsonify(results=[result]), 200
