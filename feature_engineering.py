"""
module: feature_engineering.py
description: enriches a laps dataframe with advanced features from telemetry.
this module extracts useful info like speed, throttle, brake usage from raw telemetry.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

def _get_lap_telemetry_features(lap):
    """
    extracts features from a single lap's telemetry.
    returns a dict with speed stats, throttle/brake usage, etc.
    """
    try:
        telemetry = lap.get_telemetry().copy()
        if telemetry.empty:
            return None

        features = {}
        features['lap_mean_speed'] = telemetry['Speed'].mean()
        features['lap_max_speed'] = telemetry['Speed'].max()
        features['lap_min_speed'] = telemetry['Speed'].min()
        features['lap_std_speed'] = telemetry['Speed'].std()
        
        total_points = len(telemetry)
        if total_points > 0:
            features['lap_perc_throttle_full'] = (telemetry['Throttle'] >= 98).sum() / total_points * 100
            features['lap_perc_brake_on'] = telemetry['Brake'].sum() / total_points * 100
        else:
            features['lap_perc_throttle_full'] = 0
            features['lap_perc_brake_on'] = 0
            
        features['lap_drs_enabled'] = 1 if (telemetry['DRS'] >= 8).any() else 0

        return features
        
    except Exception:
        return None

def add_telemetry_features(laps_df, sample_n=None):
    """
    applies telemetry feature extraction to each lap in the dataframe.

    args:
        laps_df (pd.dataframe): the input dataframe with lap data.
        sample_n (int, optional): process only n laps for quick testing.

    returns:
        pd.dataframe: the dataframe with new telemetry features added.
    """
    print("\n" + "="*50)
    print("starting advanced feature engineering...")
    print("="*50)
    
    if sample_n:
        print(f"running on a sample of {sample_n} laps for testing")
        processing_df = laps_df.sample(n=sample_n, random_state=42)
    else:
        processing_df = laps_df

    tqdm.pandas(desc="Extracting Telemetry Features")
    
    # apply our feature extraction function to each row
    feature_list = processing_df.progress_apply(_get_lap_telemetry_features, axis=1)
    
    # create a dataframe from the list of feature dicts
    features_df = pd.DataFrame(feature_list.dropna().tolist(), index=feature_list.dropna().index)
    
    # join the new features back to the original dataframe
    enriched_df = laps_df.join(features_df)

    print(f"\n[done] feature engineering complete. added {len(features_df.columns)} new features.")
    return enriched_df

if __name__ == '__main__':
    # this block lets us run the script alone for testing
    print("running feature_engineering.py standalone for testing")
    
    # normally you would load your data here
    try:
        df = pd.read_parquet('f1_data_2023-2022.parquet')
        # to run faster, test on a small sample from one race
        df_sample = df[df['EventName'] == 'Bahrain Grand Prix'].head(20)
        
        # run the feature engineering
        df_enriched = add_telemetry_features(df_sample)
        
        print("\n--- enriched dataframe sample ---")
        print(df_enriched[['Driver', 'LapNumber', 'lap_mean_speed', 'lap_perc_throttle_full']].tail())

    except FileNotFoundError:
        print("\n[error] test could not run. 'f1_data_2023-2022.parquet' not found.")
    except Exception as e:
        print(f"\nan error occurred during testing: {e}")
