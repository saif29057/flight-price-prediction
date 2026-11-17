"""FastAPI backend: price prediction, recommendations and the RAG travel assistant.

Run:  uvicorn flyprice.api:app --reload     (docs at http://localhost:8000/docs)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from flyprice.features import load_dataset, to_frame
from flyprice.rag import TravelAssistant
from flyprice.recommend import FlightCatalog

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT / "data" / "flights_dataset_100000.csv.xz"))

app = FastAPI(title="FlyPrice API", version="1.0.0",
              description="Flight price prediction, recommendations and a RAG travel assistant.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Flight(BaseModel):
    distance_km: float = Field(..., gt=0, examples=[1454.0])
    duration_hours: float = Field(..., gt=0, examples=[2.5])
    stops: int = Field(0, ge=0, le=3)
    departure_hour: int = Field(..., ge=0, le=23, examples=[9])
    month: int = Field(..., ge=1, le=12, examples=[7])
    seats_available: int = Field(30, ge=0)
    airline_code: str = Field(..., examples=["TU"])
    origin_country: str = Field(..., examples=["Tunisia"])
    dest_country: str = Field(..., examples=["France"])
    cabin_class: Literal["Economy", "Premium Economy", "Business", "First"] = "Economy"
    fare_type: str = Field("Flexible", examples=["Flexible"])
    flight_category: str = Field(..., examples=["Moyen-courrier (500-2000 km)"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["When is the cheapest time to book a long-haul flight?"])
    k: int = Field(3, ge=1, le=10)


@lru_cache
def model():
    path = MODELS_DIR / "best_model.joblib"
    if not path.exists():
        raise HTTPException(503, "Model not trained yet: run `python -m flyprice.train`.")
    return joblib.load(path)


@lru_cache
def metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


@lru_cache
def catalog() -> FlightCatalog:
    return FlightCatalog(load_dataset(str(DATA_PATH)))


@lru_cache
def assistant() -> TravelAssistant:
    return TravelAssistant()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (MODELS_DIR / "best_model.joblib").exists()}


@app.post("/predict/single")
def predict_single(flight: Flight):
    price = float(model().predict(to_frame([flight.model_dump()]))[0])
    return {"predicted_price_usd": round(price, 2), "model": metadata().get("best_model")}


@app.post("/predict")
def predict_batch(flights: list[Flight]):
    if not flights:
        raise HTTPException(422, "Send at least one flight.")
    prices = model().predict(to_frame([f.model_dump() for f in flights]))
    return {"predictions": [round(float(p), 2) for p in prices], "model": metadata().get("best_model")}


@app.get("/predict/model/info")
def model_info():
    meta = metadata()
    if not meta:
        raise HTTPException(503, "No trained model metadata found.")
    return meta


@app.get("/predict/sample-flights")
def sample_flights(n: int = Query(5, ge=1, le=50)):
    return catalog().sample_flights(n)


@app.get("/recommend")
def recommend(origin: str, destination: str, cabin_class: str | None = None,
              month: int | None = Query(None, ge=1, le=12), max_price: float | None = None,
              limit: int = Query(5, ge=1, le=50)):
    return catalog().recommend(origin, destination, cabin_class, month, max_price, limit)


@app.get("/recommend/popular-routes")
def popular_routes(limit: int = Query(10, ge=1, le=50)):
    return catalog().popular_routes(limit)


@app.get("/recommend/airline-stats")
def airline_stats(limit: int = Query(10, ge=1, le=50)):
    return catalog().airline_stats(limit)


@app.post("/chat")
def chat(req: ChatRequest):
    return assistant().answer(req.question, req.k)


@app.get("/chat/knowledge/search")
def knowledge_search(q: str = Query(..., min_length=2), k: int = Query(3, ge=1, le=10)):
    return assistant().search(q, k)


@app.get("/chat/knowledge/categories")
def knowledge_categories():
    return assistant().categories()
