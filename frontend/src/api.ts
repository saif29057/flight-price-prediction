const BASE = import.meta.env.VITE_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export type Flight = {
  distance_km: number;
  duration_hours: number;
  stops: number;
  departure_hour: number;
  month: number;
  seats_available: number;
  airline_code: string;
  origin_country: string;
  dest_country: string;
  cabin_class: string;
  fare_type: string;
  flight_category: string;
};

export type Metrics = { mae: number; rmse: number; mape: number; r2: number; train_seconds: number };
export type ModelInfo = { best_model: string; n_rows: number; metrics: Record<string, Metrics> };

export type Offer = {
  flight_number: string;
  airline: string;
  origin_airport: string;
  dest_airport: string;
  flight_date: string;
  departure_time: string;
  duration: string;
  cabin_class: string;
  fare_type: string;
  price_usd: number;
};

export type ChatAnswer = {
  answer: string;
  confidence: number;
  mode: string;
  sources: { source: string; title: string; score: number }[];
};

export const api = {
  predict: (f: Flight) =>
    request<{ predicted_price_usd: number; model: string }>("/predict/single", {
      method: "POST",
      body: JSON.stringify(f),
    }),
  modelInfo: () => request<ModelInfo>("/predict/model/info"),
  recommend: (params: Record<string, string>) =>
    request<Offer[]>(`/recommend?${new URLSearchParams(params)}`),
  chat: (question: string) =>
    request<ChatAnswer>("/chat", { method: "POST", body: JSON.stringify({ question }) }),
};
