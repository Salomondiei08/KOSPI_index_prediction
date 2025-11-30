# KOSPI Predictor

End-to-end deep-learning pipeline for short-term KOSPI forecasts using LSTM/RNN, Transformer, and Hybrid architectures. The project covers data preprocessing, model training/evaluation, a React dashboard, and Dockerized CLI runs.

## Project layout
```
kospi_predictor/
├── data/                # Raw and processed datasets
├── src/                 # Data pipeline, models, training & evaluation
├── dashboard/           # shadcn React dashboard
├── models/              # Saved PyTorch checkpoints
├── reports/             # Metrics, predictions, and plots
├── main.py              # Orchestrates preprocess → train → evaluate
├── requirements.txt
└── Dockerfile
```

## Quick start
1. **Create environment & install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Run the full pipeline**
   ```bash
   python main.py
   ```
   By default the preprocessing step pulls the full KOSPI history from Yahoo Finance (`^KS11`, 1983→today) and caches it in `data/kospi_data.csv`. If the API call fails (e.g., offline) it will fall back to any existing CSV in that path. After the download it trains the three models and writes metrics/predictions to `reports/`.
3. **Launch the dashboard**
   ```bash
   # shadcn React (requires npm; serves Dec 1–5 2025 view)
   cd dashboard/shadcn
   npm install
   ln -snf ../../reports public/reports   # or copy the files into public/reports
   VITE_REPORTS_BASE=/reports npm run dev
   ```
   Inspect metrics, charts, upload custom prediction files, and review the generated Dec 1–5 2025 outlook (fed by `reports/forecast_dec_2025.csv`) in the redesigned dark UI.

## Docker
```bash
docker build -t kospi-predictor .
docker run kospi-predictor
```
The container installs requirements and runs `python main.py` to produce fresh artifacts in `reports/`.

## Key modules
- `src/data_fetcher.py`: Downloads historical KOSPI OHLCV data from Yahoo Finance and stores it locally for repeatable runs.
- `src/data_preprocessing.py`: Cleans the cached data, engineers rich features (normalized price/volume, MA ratios, volatility, momentum), standardizes the log-return targets, and creates sliding windows saved to `data/processed.npz`.
- `src/models/*`: LSTM/GRU, Transformer (with positional encoding + attention export), and Hybrid encoders.
- `src/train.py`: Config-driven training loop with AdamW, ReduceLROnPlateau scheduler, gradient clipping, and early stopping.
- `src/evaluate.py`: Loads checkpoints, scores on the test split, saves metrics JSON/CSV, and exports plots/attention heatmaps.
- `src/forecast.py`: Rolls each trained model forward to produce Dec 1–5 2025 predictions saved under `reports/forecast_dec_2025.csv`.
- `dashboard/shadcn`: React dashboard for metrics, residuals, and the Dec 1–5 2025 outlook.

## Configuration
Adjust hyperparameters via `PreprocessingConfig`, `TrainingConfig`, and `ForecastConfig` in `main.py` / `src/train.py` (window size, Yahoo Finance ticker/range, hidden dims, learning rate, forecast window, etc.). Set `prefer_api=False` if you want to skip the live download and exclusively rely on a local CSV.

## Enhancements
- Swap in live KRX/Yahoo Finance ingestion.
- Add macro indicators or quantile regression heads.
- Integrate SHAP/explainability hooks.
- Schedule retraining via cron or CI and deploy the Docker image to cloud (AWS/GCP/Azure).
