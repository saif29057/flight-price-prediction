"""API tests. A small model is trained on data/sample_flights_3000.csv so CI does not need the full dataset."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_flights_3000.csv"

FLIGHT = {
    "distance_km": 1454.0, "duration_hours": 2.5, "stops": 0, "departure_hour": 9, "month": 7,
    "seats_available": 30, "airline_code": "TU", "origin_country": "France", "dest_country": "Tunisia",
    "cabin_class": "Economy", "fare_type": "Flexible", "flight_category": "Moyen-courrier (500-2000 km)",
}


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{models_dir / 'mlflow.db'}"}
    subprocess.run([sys.executable, "-m", "flyprice.train", "--data", str(SAMPLE), "--out", str(models_dir)],
                   cwd=ROOT, env=env, check=True)
    os.environ["MODELS_DIR"] = str(models_dir)
    os.environ["DATA_PATH"] = str(SAMPLE)
    os.environ["RAG_EMBEDDINGS"] = "tfidf"
    import flyprice.api as api

    importlib.reload(api)
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_single(client):
    r = client.post("/predict/single", json=FLIGHT)
    assert r.status_code == 200
    assert r.json()["predicted_price_usd"] > 0


def test_business_costs_more_than_economy(client):
    eco = client.post("/predict/single", json=FLIGHT).json()["predicted_price_usd"]
    biz = client.post("/predict/single", json={**FLIGHT, "cabin_class": "Business"}).json()["predicted_price_usd"]
    assert biz > eco


def test_predict_validation(client):
    assert client.post("/predict/single", json={**FLIGHT, "month": 13}).status_code == 422
    assert client.post("/predict/single", json={**FLIGHT, "cabin_class": "Luxury"}).status_code == 422


def test_batch(client):
    r = client.post("/predict", json=[FLIGHT, {**FLIGHT, "distance_km": 6000, "duration_hours": 8}])
    assert len(r.json()["predictions"]) == 2


def test_model_info(client):
    info = client.get("/predict/model/info").json()
    assert info["best_model"] in {"linear_regression", "random_forest", "xgboost"}
    assert set(info["metrics"]) == {"linear_regression", "random_forest", "xgboost"}


def test_recommend_sorted_by_price(client):
    routes = client.get("/recommend/popular-routes", params={"limit": 1}).json()
    o, d = routes[0]["origin_city"], routes[0]["dest_city"]
    flights = client.get("/recommend", params={"origin": o, "destination": d}).json()
    prices = [f["price_usd"] for f in flights]
    assert prices and prices == sorted(prices)


def test_chat_retrieves_relevant_section(client):
    r = client.post("/chat", json={"question": "How much baggage can I check in economy?"}).json()
    assert r["sources"][0]["source"] == "baggage.md"
    assert r["confidence"] > 0


def test_chat_unknown_topic(client):
    r = client.post("/chat", json={"question": "zzzz qqqq xxxx"}).json()
    assert r["mode"] == "none"
