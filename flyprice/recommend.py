"""Flight recommendations and statistics computed from the flights dataset."""

from __future__ import annotations

import pandas as pd


class FlightCatalog:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def recommend(self, origin: str, destination: str, cabin_class: str | None = None,
                  month: int | None = None, max_price: float | None = None, limit: int = 5) -> list[dict]:
        """Cheapest flights between two cities or airports, with optional filters.

        `origin` and `destination` match an IATA airport code or a city name (case-insensitive).
        """
        df = self.df
        o, d = origin.strip().lower(), destination.strip().lower()
        mask = ((df["origin_airport"].str.lower() == o) | (df["origin_city"].str.lower() == o)) & \
               ((df["dest_airport"].str.lower() == d) | (df["dest_city"].str.lower() == d))
        if cabin_class:
            mask &= df["cabin_class"] == cabin_class
        if month:
            mask &= df["month"] == month
        if max_price:
            mask &= df["price_usd"] <= max_price
        cols = ["flight_number", "airline", "origin_airport", "origin_city", "dest_airport", "dest_city",
                "flight_date", "departure_time", "duration", "stops", "cabin_class", "fare_type", "price_usd"]
        best = df.loc[mask, cols].sort_values("price_usd").head(limit)
        return best.to_dict(orient="records")

    def popular_routes(self, limit: int = 10) -> list[dict]:
        g = (self.df.groupby(["origin_city", "dest_city"])
             .agg(flights=("price_usd", "size"), avg_price=("price_usd", "mean"))
             .sort_values("flights", ascending=False).head(limit).reset_index())
        g["avg_price"] = g["avg_price"].round(2)
        return g.to_dict(orient="records")

    def airline_stats(self, limit: int = 10) -> list[dict]:
        g = (self.df.groupby("airline")
             .agg(flights=("price_usd", "size"), avg_price=("price_usd", "mean"),
                  avg_distance_km=("distance_km", "mean"))
             .sort_values("flights", ascending=False).head(limit).reset_index())
        return g.round(2).to_dict(orient="records")

    def sample_flights(self, n: int = 5, seed: int = 0) -> list[dict]:
        return self.df.sample(n=n, random_state=seed).to_dict(orient="records")
