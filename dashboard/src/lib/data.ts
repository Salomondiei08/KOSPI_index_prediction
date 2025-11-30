import { csvParse } from "d3-dsv";

const REPORTS_BASES = [
  import.meta.env.VITE_REPORTS_BASE,
  "../reports",
  "/reports"
].filter(Boolean) as string[];

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

async function fetchWithFallback<T>(paths: string[], loader: (p: string) => Promise<T>): Promise<T> {
  const errors: string[] = [];
  for (const path of paths) {
    try {
      return await loader(path);
    } catch (err: any) {
      errors.push(`${path}: ${err?.message ?? err}`);
    }
  }
  throw new Error(errors.join(" | "));
}

export async function loadMetrics(): Promise<MetricsRecord[]> {
  return fetchWithFallback(
    REPORTS_BASES.map((base) => `${base}/evaluation_metrics.json`),
    async (path) => {
      const payload = await fetchJson<
        Record<string, { RMSE: number; MAE: number; DirectionalAccuracy: number }>
      >(path);
      return Object.entries(payload).map(([model, metrics]) => ({
        model,
        RMSE: metrics.RMSE,
        MAE: metrics.MAE,
        DirectionalAccuracy: metrics.DirectionalAccuracy
      }));
    }
  );
}

export async function loadPredictions(): Promise<PredictionRow[]> {
  return fetchWithFallback(
    REPORTS_BASES.map((base) => `${base}/predictions.csv`),
    (path) =>
      fetchCsv<PredictionRow>(path, (row) => ({
        date: row.date,
        model: row.model,
        actual: Number(row.actual),
        predicted: Number(row.predicted),
        split: row.split
      }))
  );
}

export async function loadForecast(): Promise<ForecastRow[]> {
  return fetchWithFallback(
    REPORTS_BASES.map((base) => `${base}/forecast_dec_2025.csv`),
    (path) =>
      fetchCsv<ForecastRow>(path, (row) => ({
        date: row.date,
        model: row.model,
        predicted_close: Number(row.predicted_close),
        predicted_log_return: Number(row.predicted_log_return)
      }))
  );
}
