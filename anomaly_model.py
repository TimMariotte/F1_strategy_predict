"""
module: anomaly_model.py
description: handles training and evaluation of the anomaly detection model.
we use an autoencoder that learns to reconstruct normal laps.
laps with high reconstruction error are flagged as anomalies.

overfitting/underfitting notes:
- we train only on normal laps so the model learns what normal looks like
- validation_split=0.1 lets us check if the model generalizes
- if val loss >> train loss, model is overfitting (memorizing training data)
- if both losses are high, model is underfitting (too simple or not enough epochs)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

def _prepare_anomaly_data(df):
    """
    prepares data for the anomaly model.
    we filter out pit laps and very slow laps since those are not anomalies,
    they are just expected slow laps.
    """

    # make sure we have lap time in seconds
    if 'LapTime_sec' not in df.columns:
        if 'LapTime' in df.columns:
            df['LapTime_sec'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
        else:
            df['LapTime_sec'] = np.nan
            print("[warning] missing 'LapTime' column, slow-lap filtering will be skipped")
    
    requested_features = [
        'LapTime_sec', 'TyreLife', 'Stint',
        'lap_mean_speed', 'lap_max_speed', 'lap_min_speed',
        'lap_std_speed', 'lap_perc_throttle_full', 'lap_perc_brake_on',
        'lap_drs_enabled', 'lap_mean_rpm', 'lap_accel_mean_abs', 'lap_accel_max',
        'lap_distance', 'lap_speed_drop_ratio', 'DeltaToDriverMedian',
        'DeltaToEventMedian', 'StintDegradationSlope', 'Weather_TrackTemp',
        'Weather_AirTemp', 'Weather_WindSpeed'
    ]

    # only keep features that actually exist in the dataframe
    feature_cols = [col for col in requested_features if col in df.columns]
    missing_cols = [col for col in requested_features if col not in df.columns]
    if missing_cols:
        print(f"[warning] skipping missing features: {missing_cols}")
    if not feature_cols:
        raise ValueError("no usable features found. please regenerate the processed data.")
    
    # fill missing values with median (simple approach)
    df_clean = df.copy()
    for col in feature_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # filter out slow laps (more than 15% slower than median)
    # slow laps are not anomalies, they are expected (traffic, mistakes, etc)
    if 'LapTime_sec' in feature_cols and {'Year', 'EventName'}.issubset(df_clean.columns):
        median_lap_times = df_clean.groupby(['Year', 'EventName'])['LapTime_sec'].transform('median')
        is_slow_lap = df_clean['LapTime_sec'] > (median_lap_times * 1.15)
    else:
        is_slow_lap = pd.Series(False, index=df_clean.index)
    
    # also exclude pit laps and lap 1 (formation lap effects)
    pit_col = df_clean['IsPitLap'] if 'IsPitLap' in df_clean.columns else pd.Series(0, index=df_clean.index)
    lap_number_col = df_clean['LapNumber'] if 'LapNumber' in df_clean.columns else pd.Series(2, index=df_clean.index)

    is_normal_lap = (pit_col.fillna(0) == 0) & \
                    (lap_number_col > 1) & \
                    (~is_slow_lap)

    df_clean['IsNormalLap'] = is_normal_lap
    
    X_normal = df_clean[df_clean['IsNormalLap']][feature_cols]
    return df_clean, X_normal, feature_cols

class Autoencoder(Model):
    """
    a simple autoencoder for anomaly detection.
    it compresses the input to a small latent space then reconstructs it.
    if reconstruction error is high, the input is probably an anomaly.
    
    architecture: input -> 16 -> 8 -> 4 (latent) -> 8 -> 16 -> output
    we use relu activation and sigmoid at the end since data is scaled 0-1.
    """
    def __init__(self, input_shape, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu", input_shape=(input_shape,)),
            layers.Dense(8, activation="relu"),
            layers.Dense(encoding_dim, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(input_shape, activation="sigmoid")
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))

def train_anomaly_model(df):
    """
    trains and evaluates the anomaly detection model.
    
    how it works:
    1. we train an autoencoder only on normal laps
    2. we also train isolation forest and one-class svm for comparison
    3. we combine all three models for a more robust anomaly score
    
    overfitting prevention:
    - we use validation_split=0.1 to monitor generalization
    - isolation forest uses contamination=0.005 (conservative)
    - final threshold is set using 99.5th percentile per event
    
    returns:
        pd.dataframe: the input df with anomaly scores and flags added.
    """
    print("\n" + "="*50)
    print("training anomaly detection model (autoencoder)...")
    print("="*50)
    
    df_prepared, X_normal_unscaled, features = _prepare_anomaly_data(df)
    
    # scale the data between 0 and 1 for the autoencoder
    scaler = MinMaxScaler().fit(X_normal_unscaled)
    X_normal_scaled = scaler.transform(X_normal_unscaled)
    
    # build and train the autoencoder
    # encoding_dim=4 means we compress all features into just 4 numbers
    autoencoder = Autoencoder(input_shape=len(features), encoding_dim=4)
    autoencoder.compile(optimizer='adam', loss='mae')
    
    # validation_split=0.1 means 10% of data is used to check for overfitting
    # if val loss stays close to train loss, we are not overfitting
    autoencoder.fit(
        X_normal_scaled, X_normal_scaled,
        epochs=50, batch_size=256,
        validation_split=0.1, shuffle=True, verbose=0
    )
    print("[done] model training complete")
    
    # calculate reconstruction error for all laps
    # high error = lap is different from what the model learned as normal
    all_data_scaled = scaler.transform(df_prepared[features])
    reconstructions = autoencoder.predict(all_data_scaled, verbose=0)
    error = losses.mae(all_data_scaled, reconstructions).numpy()
    df_prepared['AE_ReconstructionError'] = error

    # set a threshold based on training data (mean + 3 standard deviations)
    train_reconstructions = autoencoder.predict(X_normal_scaled, verbose=0)
    train_error = losses.mae(X_normal_scaled, train_reconstructions).numpy()
    global_threshold = np.mean(train_error) + 3 * np.std(train_error)
    print(f"global anomaly threshold (autoencoder): {global_threshold:.4f}")

    # --- additional detectors for ensemble ---
    # isolation forest: isolates anomalies by random splits
    # contamination=0.005 means we expect 0.5% of data to be anomalies
    iso_forest = IsolationForest(
        n_estimators=200, contamination=0.005, random_state=42, n_jobs=-1
    )
    iso_forest.fit(X_normal_scaled)
    iso_score = -iso_forest.score_samples(all_data_scaled)  # higher = more weird

    # one-class svm: finds a boundary around normal data
    # nu=0.01 controls how tight the boundary is
    ocs = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    ocs.fit(X_normal_scaled)
    ocs_score = -ocs.decision_function(all_data_scaled)

    # normalize all scores to 0-1 range so we can combine them
    def _normalize(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    ae_norm = _normalize(df_prepared['AE_ReconstructionError'])
    iso_norm = _normalize(iso_score)
    ocs_norm = _normalize(ocs_score)

    df_prepared['IsoScore'] = iso_score
    df_prepared['OcsScore'] = ocs_score

    # combine all three models: 50% autoencoder, 30% isolation forest, 20% svm
    # this ensemble approach is more robust than using just one model
    df_prepared['AnomalyScore'] = 0.5 * ae_norm + 0.3 * iso_norm + 0.2 * ocs_norm

    # use adaptive threshold per event (99.5th percentile)
    # different tracks have different characteristics
    df_prepared['AnomalyThresholdEvent'] = df_prepared.groupby(['Year', 'EventName'])['AnomalyScore'].transform(
        lambda s: s.quantile(0.995)
    )
    df_prepared['IsAnomaly'] = (df_prepared['AnomalyScore'] > df_prepared['AnomalyThresholdEvent']).astype(int)

    print(f"found {df_prepared['IsAnomaly'].sum()} anomalies across all events")

    # try to classify what type of anomaly it is based on sensor values
    # this is a rough classification using simple rules
    def _anomaly_type(row):
        brake = row.get('lap_perc_brake_on', np.nan)
        speed_drop = row.get('lap_speed_drop_ratio', 1)
        rpm = row.get('lap_mean_rpm', np.nan)
        throttle = row.get('lap_perc_throttle_full', np.nan)
        track_temp = row.get('Weather_TrackTemp', np.nan)

        # lots of braking and big speed drops might mean brake problems
        if brake and brake > 70 and speed_drop and speed_drop < 0.6:
            return 'Brakes/drag'
        # low rpm with high throttle could be engine issue
        if rpm and rpm < 8000 and throttle and throttle > 60:
            return 'Powertrain'
        # hot track with lots of braking might be thermal issue
        if track_temp and track_temp > 45 and brake and brake > 50:
            return 'Thermal'
        return 'Other'

    df_prepared['AnomalyType'] = df_prepared.apply(_anomaly_type, axis=1)

    # save a plot of the anomaly score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df_prepared['AnomalyScore'], bins=100, kde=True)
    plt.axvline(global_threshold, color='r', linestyle='--', label='global threshold')
    plt.title('anomaly score distribution (combined)')
    plt.xlabel('anomaly score (normalized)')
    plt.legend()
    plt.savefig('reconstruction_error_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[done] anomaly score distribution plot saved")

    # Return the original data with anomaly columns, aligned by index
    df['AE_ReconstructionError'] = df_prepared['AE_ReconstructionError']
    df['IsoScore'] = df_prepared['IsoScore']
    df['OcsScore'] = df_prepared['OcsScore']
    df['AnomalyScore'] = df_prepared['AnomalyScore']
    df['AnomalyThresholdEvent'] = df_prepared['AnomalyThresholdEvent']
    df['AnomalyType'] = df_prepared['AnomalyType']
    df['IsAnomaly'] = df_prepared['IsAnomaly']
    
    return df
