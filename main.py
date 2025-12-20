"""
main orchestrator script for the f1 prediction project
description: this script runs the entire pipeline from data collection to
             model training and anomaly detection.
"""
import os
import argparse  # Import argparse
import data_loader
import strategy_model
import anomaly_model
import decision_fusion

# --- configuration ---
# we use seasons from 2018 to 2024 by default
# you can change this with command line arguments
YEARS = list(range(2018, 2025))


def _processed_filename(years):
    return f"f1_processed_data_{min(years)}-{max(years)}.parquet"


# this file stores all our processed data
PROCESSED_DATA_FILE = _processed_filename(YEARS)
FINAL_DATA_FILE = 'f1_data_with_anomalies.parquet'  # output file with anomaly scores

# --- main pipeline ---
def run_pipeline(force_regenerate=False, years=YEARS, jobs=1):
    """
    runs the full data processing and modeling pipeline.
    
    args:
        force_regenerate (bool): if true, forces data regeneration.
        years (list[int]): seasons to process.
        jobs (int): number of parallel workers for race processing.
    
    note on overfitting/underfitting:
        - we use train/validation splits to detect overfitting
        - if validation loss is much higher than train loss, model is overfitting
        - if both losses are high, model is underfitting (needs more features or complexity)
    """
    processed_file = _processed_filename(years)

    # === step 1: data collection and processing ===
    # this step fetches data, engineers features, and saves everything
    # skips if file already exists (unless force_regenerate is true)
    if force_regenerate or not os.path.exists(processed_file):
        if force_regenerate:
            print("--- forced regeneration: running step 1 ---")
        else:
            print("--- running step 1: full data collection and processing ---")
        # this function does all the heavy lifting and saves the result
        data_loader.fetch_and_process_data(years=years, output_file=processed_file, n_jobs=jobs)
    else:
        print(f"--- skipping step 1: file '{processed_file}' already exists ---")

    # === step 2 & 3: model training ===
    # load the processed data for training our models
    df_for_models = data_loader.load_processed_data(processed_file)
    
    if df_for_models is None:
        print("stopping pipeline: could not load processed data")
        return

    # --- strategy model training ---
    # we pass a copy to avoid changing the original dataframe
    print("\n--- running step 2: strategy model training ---")
    strategy_outputs = strategy_model.train_strategy_model(df_for_models.copy(), years=years)

    # --- anomaly model training ---
    # again we use a copy here
    print("\n--- running step 3: anomaly model training ---")
    df_with_anomalies = anomaly_model.train_anomaly_model(df_for_models.copy())

    # --- fusion layer ---
    # combines strategy predictions and anomaly scores into one decision
    print("\n--- running step 4: decision fusion ---")
    df_final = decision_fusion.combine_predictions(df_with_anomalies.copy(), strategy_outputs)
    
    # save the final dataset with all scores and predictions
    df_final.to_parquet(FINAL_DATA_FILE, index=False)
    print(f"\n[done] final dataset saved to '{FINAL_DATA_FILE}'")
    
    print("\n" + "="*50)
    print("pipeline complete - now run: streamlit run dashboard.py")
    print("="*50)

if __name__ == '__main__':
    # --- argument parsing ---
    # lets user customize the pipeline from command line
    parser = argparse.ArgumentParser(description="run the f1 data processing and modeling pipeline")
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help="Force regeneration of the processed data file, ignoring any existing cached file."
    )
    parser.add_argument(
        '--years',
        type=str,
        help="Years to process, e.g., '2018-2023' or '2022,2023'. Defaults to 2018-2024."
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help="Parallel workers per race (>=1). Use 1 if issues with FastF1 cache."
    )
    args = parser.parse_args()

    chosen_years = YEARS
    if args.years:
        if '-' in args.years:
            start, end = args.years.split('-')
            chosen_years = list(range(int(start), int(end) + 1))
        else:
            chosen_years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
        print(f"using custom years: {chosen_years}")

    # before running, make sure you have the required packages installed:
    # pip install pandas fastf1 tqdm pyarrow xgboost tensorflow matplotlib seaborn scikit-learn
    run_pipeline(force_regenerate=args.regenerate, years=chosen_years, jobs=args.jobs)
