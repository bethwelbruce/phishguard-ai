"""
PhishGuard AI - FastAPI backend.

Endpoints
---------
GET  /api/health              Liveness check
POST /api/predict             Single-URL prediction + confidence + features
POST /api/predict/batch      Batch prediction (list of URLs)
GET  /api/metrics             Held-out test metrics of the deployment model
GET  /api/feature-importance  Model feature importances (sorted)
POST /api/explain             SHAP + LIME explanations for one URL

Label convention (PhiUSIIL): 1 = Legitimate, 0 = Phishing.

Run:  uvicorn app.main:app --reload --port 8000   (from backend/)
"""

import os
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .features import FEATURE_ORDER, extract_features, feature_vector

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUTS = os.path.join(_HERE, "..", "..", "outputs")

app = FastAPI(
    title="PhishGuard AI",
    description="Phishing URL detection with Decision Tree + SHAP/LIME XAI",
    version="1.0.0",
)

# CORS: allow the React dev server and any deployed frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model + explainers, loaded once at startup
# ---------------------------------------------------------------------------
MODEL = joblib.load(os.path.join(_OUTPUTS, "deployment_model.joblib"))
METRICS = joblib.load(os.path.join(_OUTPUTS, "deployment_metrics.joblib"))
LIME_BG = np.load(os.path.join(_OUTPUTS, "lime_background.npy"))

_shap_explainer = None
_lime_explainer = None


def get_shap_explainer():
    """Lazy-load SHAP TreeExplainer (fast + exact for tree models)."""
    global _shap_explainer
    if _shap_explainer is None:
        import shap

        _shap_explainer = shap.TreeExplainer(MODEL)
    return _shap_explainer


def get_lime_explainer():
    """Lazy-load LIME tabular explainer with training background data."""
    global _lime_explainer
    if _lime_explainer is None:
        from lime.lime_tabular import LimeTabularExplainer

        _lime_explainer = LimeTabularExplainer(
            LIME_BG,
            feature_names=FEATURE_ORDER,
            class_names=["Phishing", "Legitimate"],
            mode="classification",
            random_state=42,
        )
    return _lime_explainer


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class URLRequest(BaseModel):
    url: str = Field(..., min_length=4, examples=["https://example.com"])


class BatchRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, max_length=100)


def _predict_one(url: str) -> dict:
    vec = feature_vector(url)
    label = int(MODEL.predict(vec)[0])
    proba = MODEL.predict_proba(vec)[0]
    return {
        "url": url,
        "prediction": "legitimate" if label == 1 else "phishing",
        "label": label,
        "confidence": round(float(max(proba)), 4),
        "probabilities": {
            "phishing": round(float(proba[0]), 4),
            "legitimate": round(float(proba[1]), 4),
        },
        "features": extract_features(url),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/api/predict")
def predict(req: URLRequest):
    try:
        return _predict_one(req.url)
    except Exception as exc:  # malformed URL edge cases
        raise HTTPException(status_code=422, detail=f"Could not process URL: {exc}")


@app.post("/api/predict/batch")
def predict_batch(req: BatchRequest):
    results, errors = [], []
    for u in req.urls:
        try:
            results.append(_predict_one(u))
        except Exception as exc:
            errors.append({"url": u, "error": str(exc)})
    return {"count": len(results), "results": results, "errors": errors}


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.get("/api/feature-importance")
def feature_importance():
    pairs = sorted(
        zip(FEATURE_ORDER, MODEL.feature_importances_.tolist()),
        key=lambda p: p[1],
        reverse=True,
    )
    return {
        "features": [
            {"name": n, "importance": round(v, 5)} for n, v in pairs
        ]
    }


@app.post("/api/explain")
def explain(req: URLRequest):
    """SHAP + LIME explanations for a single URL prediction."""
    try:
        vec = feature_vector(req.url)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process URL: {exc}")

    base = _predict_one(req.url)

    # --- SHAP (exact, tree-based) ---
    shap_exp = get_shap_explainer()
    shap_vals = shap_exp.shap_values(vec)
    # sklearn trees: shap returns (n, features, classes) or list per class
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:  # (samples, features, classes) -> class 1 (legitimate)
        contrib = arr[0, :, 1]
    else:  # list [class0, class1]
        contrib = np.asarray(shap_vals[1])[0]
    shap_out = sorted(
        (
            {"feature": f, "shap_value": round(float(v), 5)}
            for f, v in zip(FEATURE_ORDER, contrib)
        ),
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )[:10]

    # --- LIME (local surrogate) ---
    lime_exp = get_lime_explainer()
    explanation = lime_exp.explain_instance(
        vec[0], MODEL.predict_proba, num_features=10
    )
    lime_out = [
        {"rule": rule, "weight": round(float(w), 5)}
        for rule, w in explanation.as_list()
    ]

    return {
        "url": req.url,
        "prediction": base["prediction"],
        "confidence": base["confidence"],
        "shap": {
            "note": "Positive values push toward LEGITIMATE; negative toward PHISHING.",
            "top_features": shap_out,
        },
        "lime": {
            "note": "Positive weights support LEGITIMATE; negative support PHISHING.",
            "top_rules": lime_out,
        },
    }
