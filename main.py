from __future__ import annotations

from src.data_preprocessing import PreprocessingConfig, preprocess_and_save
from src.evaluate import evaluate_models
from src.forecast import ForecastConfig, generate_forecast
from src.train import TrainingConfig, run_training_pipeline


def main() -> None:
    pre_cfg = PreprocessingConfig(
        end_date="2025-11-30",
        start_date="1980-01-01",
        window_size=30,
        rolling_window=60,
        train_ratio=0.8,
        val_ratio=0.1,
        force_refresh=True,
    )
    train_cfg = TrainingConfig(
        batch_size=128,
        epochs=60,
        learning_rate=4e-4,
        patience=8,
        hidden_size=160,
        num_layers=2,
        dropout=0.1,
        transformer_dim=160,
        nhead=8,
        weight_decay=1e-4,
    )

    print("📦 Preprocessing data...")
    preprocess_and_save(pre_cfg)

    print("🧠 Training models...")
    run_training_pipeline(pre_cfg, train_cfg)

    print("📈 Evaluating models...")
    metrics = evaluate_models(pre_cfg, train_cfg)
    print("Evaluation summary:", metrics)

    print("🔮 Generating Dec 1–5 2025 forecasts...")
    generate_forecast(pre_cfg, train_cfg, ForecastConfig())
    print("Forecast written to reports/forecast_dec_2025.csv")


if __name__ == "__main__":
    main()
