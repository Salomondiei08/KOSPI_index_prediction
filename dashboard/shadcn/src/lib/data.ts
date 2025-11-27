import { csvParse } from "d3-dsv";

const REPORTS_BASE = import.meta.env.VITE_REPORTS_BASE || "../reports";

export type MetricsRecord = {
  model: string;
  RMSE: number;
  MAE: number;
  DirectionalAccuracy: number;
};

export type PredictionRow = {
  date: string;
  model: string;
  actual: number;
  predicted: number;
  split: string;
};

export type ForecastRow = {
  date: string;
  model: string;
  predicted_close: number;
  predicted_log_return: number;
};

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}: ${res.status}`);
  }
  return (await res.json()) as T;
}

async function fetchCsv<T>(path: string, parse: (row: any) => T): Promise<T[]> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}: ${res.status}`);
  }
  const text = await res.text();
  return csvParse(text).map(parse);
}

export async function loadMetrics(): Promise<MetricsRecord[]> {
  const payload = await fetchJson<Record<string, { RMSE: number; MAE: number; DirectionalAccuracy: number }>>(
    `${REPORTS_BASE}/evaluation_metrics.json`
  );
  return Object.entries(payload).map(([model, metrics]) => ({
    model,
    RMSE: metrics.RMSE,
    MAE: metrics.MAE,
    DirectionalAccuracy: metrics.DirectionalAccuracy
  }));
}

export async function loadPredictions(): Promise<PredictionRow[]> {
  return fetchCsv<PredictionRow>(`${REPORTS_BASE}/predictions.csv`, (row) => ({
    date: row.date,
    model: row.model,
    actual: Number(row.actual),
    predicted: Number(row.predicted),
    split: row.split
  }));
}

export async function loadForecast(): Promise<ForecastRow[]> {
  return fetchCsv<ForecastRow>(`${REPORTS_BASE}/forecast_dec_2025.csv`, (row) => ({
    date: row.date,
    model: row.model,
    predicted_close: Number(row.predicted_close),
    predicted_log_return: Number(row.predicted_log_return)
  }));
}
