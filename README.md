# F1 Strategy Prediction

Machine learning project for Formula 1 pit stop strategy prediction and anomaly detection.

## Data

To get the data used for this project (`.parquet` files), please download them from DVL.

## Project Structure

- `main.py` - Main pipeline orchestrator
- `data_loader.py` - F1 telemetry data collection using FastF1
- `feature_engineering.py` - Feature extraction and processing
- `anomaly_model.py` - Autoencoder-based anomaly detection
- `strategy_model.py` - XGBoost strategy prediction models
- `decision_fusion.py` - Fusion of anomaly and strategy signals
- `dashboard.py` - Streamlit interactive dashboard

## Usage

If you already have the `.parquet` data files, simply run:

```bash
python main.py
streamlit run dashboard.py
```

To regenerate data from scratch:

```bash
python main.py --years 2018 2019 2020 --regenerate
```

## Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```
