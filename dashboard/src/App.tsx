import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { Sparkles, RefreshCw, TrendingUp, ArrowRightLeft, Gauge } from "lucide-react";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { Tabs } from "./components/ui/tabs";
import { ForecastRow, MetricsRecord, PredictionRow, loadForecast, loadMetrics, loadPredictions } from "./lib/data";

type LoadState = "idle" | "loading" | "error" | "ready";

function formatNumber(value: number, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function buildResidualBins(rows: PredictionRow[]) {
  const residuals = rows.map((r) => r.predicted - r.actual);
  const bins = new Array(12).fill(0).map((_, idx) => ({
    name: `${idx - 6}`,
    count: 0
  }));
  const std = Math.max(1e-9, Math.sqrt(residuals.reduce((acc, v) => acc + v * v, 0) / residuals.length));
  residuals.forEach((r) => {
    const z = Math.max(-6, Math.min(5.99, r / std));
    const bin = Math.floor(z) + 6;
    bins[bin].count += 1;
  });
  return bins;
}

export default function App() {
  const [state, setState] = useState<LoadState>("idle");
  const [metrics, setMetrics] = useState<MetricsRecord[]>([]);
  const [predictions, setPredictions] = useState<PredictionRow[]>([]);
  const [forecast, setForecast] = useState<ForecastRow[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"perf" | "residuals" | "forecast">("perf");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    setState("loading");
    Promise.all([loadMetrics(), loadPredictions(), loadForecast()])
      .then(([m, p, f]) => {
        setMetrics(m);
        setPredictions(p);
        setForecast(f);
        setSelectedModel(m[0]?.model ?? null);
        setErrorMessage(null);
        setState("ready");
      })
      .catch((err: any) => {
        console.error(err);
        setErrorMessage(err?.message ?? "Failed to load reports");
        setState("error");
      });
  }, []);

  const bestModel = useMemo(() => {
    if (!metrics.length) return null;
    return metrics.reduce((best, current) => (current.RMSE < best.RMSE ? current : best), metrics[0]);
  }, [metrics]);

  const currentMetrics = metrics.find((m) => m.model === selectedModel);
  const filtered = useMemo(
    () => predictions.filter((row) => row.model === selectedModel),
    [predictions, selectedModel]
  );
  const residualBins = useMemo(() => buildResidualBins(filtered), [filtered]);
  const forecastSubset = useMemo(
    () => forecast.filter((row) => row.model === selectedModel),
    [forecast, selectedModel]
  );

  const cards = [
    {
      label: "RMSE",
      value: formatNumber(currentMetrics?.RMSE ?? NaN),
      icon: <Gauge size={18} className="text-accent" />
    },
    {
      label: "MAE",
      value: formatNumber(currentMetrics?.MAE ?? NaN),
      icon: <TrendingUp size={18} className="text-success" />
    },
    {
      label: "Directional Hit",
      value: `${formatNumber((currentMetrics?.DirectionalAccuracy ?? 0) * 100)}%`,
      icon: <ArrowRightLeft size={18} className="text-accentSoft" />
    },
    {
      label: "Top Performer",
      value: bestModel ? bestModel.model.toUpperCase() : "—",
      icon: <Sparkles size={18} className="text-warning" />
    }
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">KOSPI Predictor</p>
          <h1 className="font-display text-3xl font-semibold text-white">
            December 1st–5th KOSPI Index Predictions
          </h1>
          <p className="mt-1 text-slate-400">
            Beautiful, component-driven view of training quality, residuals, and the Dec 2025 outlook.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {bestModel ? <Badge tone="success">Best RMSE: {formatNumber(bestModel.RMSE)}</Badge> : null}
          <Button
            variant="outline"
            onClick={() => {
              setState("loading");
              Promise.all([loadMetrics(), loadPredictions(), loadForecast()])
                .then(([m, p, f]) => {
                  setMetrics(m);
                  setPredictions(p);
                  setForecast(f);
                  setSelectedModel((prev) => prev ?? m[0]?.model ?? null);
                  setErrorMessage(null);
                  setState("ready");
                })
                .catch((err) => {
                  console.error(err);
                  setErrorMessage(err?.message ?? "Failed to load reports");
                  setState("error");
                });
            }}
          >
            <RefreshCw size={16} />
            Reload
          </Button>
        </div>
      </header>

      {state === "loading" && (
        <Card title="Loading">
          <p className="text-slate-300">Pulling reports from /reports…</p>
        </Card>
      )}

      {state === "error" && (
        <Card title="Error" className="border border-red-500/40">
          <p className="text-red-200">
            Unable to load dashboard data. Ensure reports are generated via python main.py.
            <br />
            {errorMessage ? <span className="text-red-300">Details: {errorMessage}</span> : null}
          </p>
        </Card>
      )}

      {state === "ready" && selectedModel && (
        <>
          <div className="glass mb-6 rounded-3xl border border-white/5 p-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-sm text-slate-400">Focus model</p>
                <div className="mt-1 flex flex-wrap gap-2">
                  {metrics.map((m) => (
                    <Button
                      key={m.model}
                      variant={m.model === selectedModel ? "solid" : "outline"}
                      onClick={() => setSelectedModel(m.model)}
                    >
                      {m.model.toUpperCase()}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Badge tone="info">Reports source: {import.meta.env.VITE_REPORTS_BASE || "../reports"}</Badge>
              </div>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {cards.map((card) => (
              <Card key={card.label} className="bg-surface">
                <div className="flex items-center justify-between">
                  <span className="card-title">{card.label}</span>
                  {card.icon}
                </div>
                <p className="mt-3 text-2xl font-semibold text-white">{card.value}</p>
              </Card>
            ))}
          </div>

          <div className="mt-5 space-y-4">
            <Tabs
              active={activeTab}
              onChange={(id) => setActiveTab(id as typeof activeTab)}
              items={[
                {
                  id: "perf",
                  label: "Performance",
                  content: (
                    <Card title="Actual vs Predicted">
                      <div className="w-full" style={{ height: 360 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={filtered} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                            <XAxis dataKey="date" stroke="#94A3B8" hide />
                            <YAxis stroke="#94A3B8" />
                            <Tooltip
                              contentStyle={{ background: "#0F172A", borderColor: "#1E293B" }}
                              labelFormatter={(v) => new Date(v).toISOString().slice(0, 10)}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="actual" stroke="#38BDF8" dot={false} />
                            <Line type="monotone" dataKey="predicted" stroke="#A855F7" dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </Card>
                  )
                },
                {
                  id: "residuals",
                  label: "Residuals",
                  content: (
                    <div className="grid gap-4 lg:grid-cols-2">
                      <Card title="Residual Histogram">
                        <div className="w-full" style={{ height: 320 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={residualBins}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                              <XAxis dataKey="name" stroke="#94A3B8" />
                              <YAxis stroke="#94A3B8" />
                              <Tooltip contentStyle={{ background: "#0F172A", borderColor: "#1E293B" }} />
                              <Bar dataKey="count" fill="#A855F7" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </Card>
                      <Card title="Recent Residuals">
                        <div className="space-y-3">
                          {filtered
                            .slice(-6)
                            .reverse()
                            .map((row) => {
                              const residual = row.predicted - row.actual;
                              return (
                                <div
                                  key={`${row.date}-${row.model}`}
                                  className="flex items-center justify-between rounded-xl bg-surface/70 px-4 py-3"
                                >
                                  <div>
                                    <p className="text-sm text-slate-400">{row.date}</p>
                                    <p className="text-lg font-semibold text-white">
                                      Residual {formatNumber(residual)}
                                    </p>
                                  </div>
                                  <Badge tone={Math.abs(residual) < 1 ? "success" : "warning"}>
                                    {residual > 0 ? "Over" : "Under"}
                                  </Badge>
                                </div>
                              );
                            })}
                        </div>
                      </Card>
                    </div>
                  )
                },
                {
                  id: "forecast",
                  label: "Dec 2025 Outlook",
                  content: (
                    <Card
                      title="Five-day projection"
                      action={<Badge tone="info">{forecastSubset.length} days</Badge>}
                    >
                      <div className="w-full" style={{ height: 320 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={forecastSubset}>
                            <defs>
                              <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#A855F7" stopOpacity={0.6} />
                                <stop offset="95%" stopColor="#A855F7" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                            <XAxis dataKey="date" stroke="#94A3B8" />
                            <YAxis stroke="#94A3B8" />
                            <Tooltip contentStyle={{ background: "#0F172A", borderColor: "#1E293B" }} />
                            <Area
                              type="monotone"
                              dataKey="predicted_close"
                              stroke="#A855F7"
                              fillOpacity={1}
                              fill="url(#forecastFill)"
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </Card>
                  )
                }
              ]}
            />
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            <Card title="Latest predictions" className="lg:col-span-2">
              <div className="grid grid-cols-2 gap-2 text-xs uppercase tracking-wide text-slate-400 sm:grid-cols-5">
                <span>Date</span>
                <span>Actual</span>
                <span>Predicted</span>
                <span>Residual</span>
                <span>Split</span>
              </div>
              <div className="mt-2 divide-y divide-slate-800/80">
                {filtered.slice(-15).map((row) => {
                  const residual = row.predicted - row.actual;
                  return (
                    <div
                      key={`${row.date}-${row.predicted}`}
                      className="grid grid-cols-2 gap-2 py-3 text-sm sm:grid-cols-5"
                    >
                      <span className="text-slate-300">{row.date}</span>
                      <span className="font-semibold text-slate-50">{formatNumber(row.actual)}</span>
                      <span className="font-semibold text-slate-50">{formatNumber(row.predicted)}</span>
                      <span className={residual >= 0 ? "text-success" : "text-warning"}>
                        {formatNumber(residual)}
                      </span>
                      <span className="text-slate-400">{row.split}</span>
                    </div>
                  );
                })}
              </div>
            </Card>
            <Card title="Data sources">
              <ul className="space-y-2 text-sm text-slate-300">
                <li>
                  <strong className="text-white">Metrics:</strong> evaluation_metrics.json
                </li>
                <li>
                  <strong className="text-white">Predictions:</strong> predictions.csv
                </li>
                <li>
                  <strong className="text-white">Forecast:</strong> forecast_dec_2025.csv
                </li>
                <li className="text-slate-400">
                  Update by re-running <code className="text-accent">python main.py</code>.
                </li>
              </ul>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
