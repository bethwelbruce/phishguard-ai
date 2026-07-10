"""API tests for PhishGuard AI backend (run with pytest from backend/)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

LEGIT_URL = "https://www.google.com"
SUSPICIOUS_URL = "http://paypa1-secure-login.win/verify.php?id=aGVsbG8=&acct=443"


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_single():
    r = client.post("/api/predict", json={"url": LEGIT_URL})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("legitimate", "phishing")
    assert 0.0 <= body["confidence"] <= 1.0
    assert "URLLength" in body["features"]


def test_predict_batch():
    r = client.post(
        "/api/predict/batch", json={"urls": [LEGIT_URL, SUSPICIOUS_URL]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2


def test_predict_rejects_empty():
    r = client.post("/api/predict", json={"url": ""})
    assert r.status_code == 422


def test_metrics():
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert "accuracy" in r.json()


def test_feature_importance():
    r = client.get("/api/feature-importance")
    assert r.status_code == 200
    feats = r.json()["features"]
    assert len(feats) == 20
    # sorted descending
    assert feats[0]["importance"] >= feats[-1]["importance"]


def test_explain_contains_shap_and_lime():
    r = client.post("/api/explain", json={"url": SUSPICIOUS_URL})
    assert r.status_code == 200
    body = r.json()
    assert len(body["shap"]["top_features"]) > 0
    assert len(body["lime"]["top_rules"]) > 0
