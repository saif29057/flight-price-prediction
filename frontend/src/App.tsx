import { FormEvent, useEffect, useState } from "react";
import { api, ChatAnswer, Flight, ModelInfo, Offer } from "./api";

type Tab = "predict" | "recommend" | "chat" | "models";

const CABINS = ["Economy", "Premium Economy", "Business", "First"];
const FARES = ["Non-Refundable", "Semi-Flexible", "Flexible", "Full-Fare"];

function category(km: number): string {
  if (km < 500) return "Court-courrier (< 500 km)";
  if (km < 2000) return "Moyen-courrier (500-2000 km)";
  if (km < 6000) return "Long-courrier (2000-6000 km)";
  return "Ultra long-courrier (> 6000 km)";
}

export default function App() {
  const [tab, setTab] = useState<Tab>("predict");
  return (
    <div className="page">
      <header>
        <h1>✈️ FlyPrice</h1>
        <p>Flight price prediction, cheapest-flight recommendations and a RAG travel assistant.</p>
        <nav>
          {(["predict", "recommend", "chat", "models"] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
              {{ predict: "Predict a price", recommend: "Find flights", chat: "Travel assistant", models: "Models" }[t]}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {tab === "predict" && <Predict />}
        {tab === "recommend" && <Recommend />}
        {tab === "chat" && <Chat />}
        {tab === "models" && <Models />}
      </main>
    </div>
  );
}

function Predict() {
  const [f, setF] = useState({
    distance_km: 1454, duration_hours: 2.5, stops: 0, departure_hour: 9, month: 7, seats_available: 30,
    airline_code: "TU", origin_country: "France", dest_country: "Tunisia",
    cabin_class: "Economy", fare_type: "Flexible",
  });
  const [price, setPrice] = useState<number | null>(null);
  const [error, setError] = useState("");

  const set = (k: keyof typeof f, v: string) =>
    setF({ ...f, [k]: typeof f[k] === "number" ? Number(v) : v });

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    try {
      const flight: Flight = { ...f, flight_category: category(f.distance_km) };
      setPrice((await api.predict(flight)).predicted_price_usd);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <form className="card grid" onSubmit={submit}>
      <Field label="Distance (km)"><input type="number" value={f.distance_km} onChange={(e) => set("distance_km", e.target.value)} /></Field>
      <Field label="Duration (hours)"><input type="number" step="0.1" value={f.duration_hours} onChange={(e) => set("duration_hours", e.target.value)} /></Field>
      <Field label="Stops"><input type="number" min={0} max={3} value={f.stops} onChange={(e) => set("stops", e.target.value)} /></Field>
      <Field label="Month"><input type="number" min={1} max={12} value={f.month} onChange={(e) => set("month", e.target.value)} /></Field>
      <Field label="Departure hour"><input type="number" min={0} max={23} value={f.departure_hour} onChange={(e) => set("departure_hour", e.target.value)} /></Field>
      <Field label="Seats available"><input type="number" min={0} value={f.seats_available} onChange={(e) => set("seats_available", e.target.value)} /></Field>
      <Field label="Airline code"><input value={f.airline_code} onChange={(e) => set("airline_code", e.target.value)} /></Field>
      <Field label="Origin country"><input value={f.origin_country} onChange={(e) => set("origin_country", e.target.value)} /></Field>
      <Field label="Destination country"><input value={f.dest_country} onChange={(e) => set("dest_country", e.target.value)} /></Field>
      <Field label="Cabin class">
        <select value={f.cabin_class} onChange={(e) => set("cabin_class", e.target.value)}>
          {CABINS.map((c) => <option key={c}>{c}</option>)}
        </select>
      </Field>
      <Field label="Fare type">
        <select value={f.fare_type} onChange={(e) => set("fare_type", e.target.value)}>
          {FARES.map((c) => <option key={c}>{c}</option>)}
        </select>
      </Field>
      <div className="actions">
        <button type="submit">Predict price</button>
        {price !== null && <span className="result">${price.toFixed(2)}</span>}
        {error && <span className="error">{error}</span>}
      </div>
    </form>
  );
}

function Recommend() {
  const [q, setQ] = useState({ origin: "Paris", destination: "Tunis", cabin_class: "", month: "", max_price: "" });
  const [offers, setOffers] = useState<Offer[] | null>(null);
  const [error, setError] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    const params = Object.fromEntries(Object.entries(q).filter(([, v]) => v !== ""));
    try {
      setOffers(await api.recommend(params));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <form className="card grid" onSubmit={submit}>
        <Field label="From (city or IATA)"><input value={q.origin} onChange={(e) => setQ({ ...q, origin: e.target.value })} /></Field>
        <Field label="To (city or IATA)"><input value={q.destination} onChange={(e) => setQ({ ...q, destination: e.target.value })} /></Field>
        <Field label="Cabin">
          <select value={q.cabin_class} onChange={(e) => setQ({ ...q, cabin_class: e.target.value })}>
            <option value="">Any</option>
            {CABINS.map((c) => <option key={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Month"><input type="number" min={1} max={12} value={q.month} onChange={(e) => setQ({ ...q, month: e.target.value })} /></Field>
        <Field label="Max price ($)"><input type="number" value={q.max_price} onChange={(e) => setQ({ ...q, max_price: e.target.value })} /></Field>
        <div className="actions"><button type="submit">Search</button>{error && <span className="error">{error}</span>}</div>
      </form>
      {offers && (
        <div className="card">
          {offers.length === 0 ? <p>No flight found for this route.</p> : (
            <table>
              <thead><tr><th>Flight</th><th>Airline</th><th>Route</th><th>Date</th><th>Duration</th><th>Cabin</th><th>Fare</th><th>Price</th></tr></thead>
              <tbody>
                {offers.map((o) => (
                  <tr key={o.flight_number + o.flight_date}>
                    <td>{o.flight_number}</td><td>{o.airline}</td><td>{o.origin_airport} → {o.dest_airport}</td>
                    <td>{o.flight_date} {o.departure_time}</td><td>{o.duration}</td><td>{o.cabin_class}</td>
                    <td>{o.fare_type}</td><td className="num">${o.price_usd.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </>
  );
}

function Chat() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<{ q: string; a: ChatAnswer }[]>([]);
  const [error, setError] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setError("");
    try {
      const a = await api.chat(question);
      setHistory([...history, { q: question, a }]);
      setQuestion("");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="card chat">
      {history.length === 0 && <p className="muted">Ask about booking, baggage, airports, visas or delays.</p>}
      {history.map((h, i) => (
        <div key={i} className="turn">
          <p className="q">{h.q}</p>
          <p className="a">{h.a.answer}</p>
          {h.a.sources.length > 0 && (
            <p className="muted">
              Sources: {h.a.sources.map((s) => `${s.title} (${s.source})`).join(" · ")} — confidence {h.a.confidence}
            </p>
          )}
        </div>
      ))}
      <form onSubmit={submit} className="ask">
        <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="When should I book a long-haul flight?" />
        <button type="submit">Ask</button>
      </form>
      {error && <span className="error">{error}</span>}
    </div>
  );
}

function Models() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.modelInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);
  if (error) return <p className="error">{error}</p>;
  if (!info) return <p>Loading…</p>;
  return (
    <div className="card">
      <p>Trained on {info.n_rows.toLocaleString()} flights (80/20 split). Best model: <b>{info.best_model}</b>.</p>
      <table>
        <thead><tr><th>Model</th><th>MAE ($)</th><th>RMSE ($)</th><th>MAPE</th><th>R²</th><th>Training (s)</th></tr></thead>
        <tbody>
          {Object.entries(info.metrics).map(([name, m]) => (
            <tr key={name} className={name === info.best_model ? "best" : ""}>
              <td>{name}</td><td className="num">{m.mae.toFixed(1)}</td><td className="num">{m.rmse.toFixed(1)}</td>
              <td className="num">{(m.mape * 100).toFixed(1)}%</td><td className="num">{m.r2.toFixed(3)}</td>
              <td className="num">{m.train_seconds}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="field"><span>{label}</span>{children}</label>;
}
