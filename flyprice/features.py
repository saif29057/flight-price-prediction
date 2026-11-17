"""Feature definitions shared by training, the API and the tests."""

from __future__ import annotations

import pandas as pd

TARGET = "price_usd"

NUMERIC = ["distance_km", "duration_hours", "stops", "departure_hour", "month", "seats_available"]
CATEGORICAL = ["airline_code", "origin_country", "dest_country", "cabin_class", "fare_type", "flight_category"]
FEATURES = NUMERIC + CATEGORICAL

CABIN_CLASSES = ["Economy", "Premium Economy", "Business", "First"]


def load_dataset(path: str) -> pd.DataFrame:
    """Read the flights CSV and keep the columns used by the models."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=FEATURES + [TARGET])
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Turn API payloads into a DataFrame with the training column order."""
    df = pd.DataFrame(records)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df[FEATURES]
