"""
module: data_loader.py
description: handles fetching, processing, and feature engineering of f1 data.
this is where we get all the raw data from fastf1 and transform it into features
that our models can use for predictions.
"""

import fastf1
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm

CACHE_DIR = 'f1_cache'

def _get_lap_telemetry_features(lap):
    """
    extracts features from a single lap's telemetry and weather data.
    returns a dict of features or none if something fails.
    
    these features help detect anomalies and predict pit strategy.
    we extract speed, acceleration, brake usage, throttle, drs, etc.
    """
    try:
        telemetry = lap.get_telemetry()
        if telemetry.empty:
            return None

        telemetry = telemetry.copy()
        features = {}

        # --- speed and dynamics ---
        # basic stats about how fast the car went during this lap
        speed = telemetry['Speed']
        features['lap_mean_speed'] = speed.mean()
        features['lap_max_speed'] = speed.max()
        features['lap_min_speed'] = speed.min()
        features['lap_std_speed'] = speed.std()
        features['lap_speed_drop_ratio'] = (speed.min() / speed.max()) if speed.max() else np.nan

        # add distance if we can get it
        try:
            telemetry = telemetry.add_distance()
            features['lap_distance'] = telemetry['Distance'].max() - telemetry['Distance'].min()
        except Exception:
            features['lap_distance'] = np.nan

        # acceleration calculation (we compute it from speed changes over time)
        time_col = telemetry['SessionTime'] if 'SessionTime' in telemetry else telemetry.get('Time')
        if time_col is not None:
            time_seconds = time_col.dt.total_seconds()
            # convert km/h to m/s for proper acceleration units
            speed_mps = speed * (1000 / 3600)
            time_diff = np.diff(time_seconds)
            speed_diff = np.diff(speed_mps)
            valid_mask = time_diff != 0
            accel = np.divide(speed_diff[valid_mask], time_diff[valid_mask]) if len(speed_diff) == len(time_diff) else np.array([])
            if accel.size > 0:
                features['lap_accel_mean_abs'] = np.abs(accel).mean()
                features['lap_accel_max'] = accel.max()
                features['lap_accel_std'] = accel.std()
            else:
                features['lap_accel_mean_abs'] = np.nan
                features['lap_accel_max'] = np.nan
                features['lap_accel_std'] = np.nan
        else:
            features['lap_accel_mean_abs'] = np.nan
            features['lap_accel_max'] = np.nan
            features['lap_accel_std'] = np.nan

        # --- engine and controls ---
        # rpm tells us about engine usage
        if 'RPM' in telemetry:
            features['lap_mean_rpm'] = telemetry['RPM'].mean()
            features['lap_max_rpm'] = telemetry['RPM'].max()
        else:
            features['lap_mean_rpm'] = np.nan
            features['lap_max_rpm'] = np.nan

        # throttle and brake percentages
        # these are super useful for detecting driving style and problems
        throttle = telemetry['Throttle'] if 'Throttle' in telemetry else None
        brake = telemetry['Brake'] if 'Brake' in telemetry else None
        total_points = len(telemetry)
        if throttle is not None and total_points > 0:
            features['lap_perc_throttle_full'] = (throttle >= 98).sum() / total_points * 100
            features['lap_throttle_std'] = throttle.std()
        else:
            features['lap_perc_throttle_full'] = np.nan
            features['lap_throttle_std'] = np.nan

        if brake is not None and total_points > 0:
            # brake is boolean in fastf1 (on or off)
            features['lap_perc_brake_on'] = brake.sum() / total_points * 100
        else:
            features['lap_perc_brake_on'] = np.nan

        # gear changes can indicate driver stress or track characteristics
        if 'nGear' in telemetry:
            gear_changes = telemetry['nGear'].diff().abs().fillna(0)
            features['lap_gear_changes'] = (gear_changes > 0).sum()
            features['lap_max_gear'] = telemetry['nGear'].max()
        else:
            features['lap_gear_changes'] = np.nan
            features['lap_max_gear'] = np.nan

        # --- drs usage ---
        # drs value >= 8 means drs is active
        if 'DRS' in telemetry:
            drs_mask = telemetry['DRS'] >= 8
            features['lap_drs_enabled'] = 1 if drs_mask.any() else 0
            features['lap_drs_usage_pct'] = drs_mask.mean() * 100
        else:
            features['lap_drs_enabled'] = 0
            features['lap_drs_usage_pct'] = np.nan

        # --- position on track (x, y coordinates) ---
        if {'X', 'Y'}.issubset(telemetry.columns):
            features['lap_pos_x_std'] = telemetry['X'].std()
            features['lap_pos_y_std'] = telemetry['Y'].std()
        else:
            features['lap_pos_x_std'] = np.nan
            features['lap_pos_y_std'] = np.nan

        # --- gap to car ahead ---
        if 'DistanceToDriverAhead' in telemetry:
            dist_ahead = telemetry['DistanceToDriverAhead'].replace(0, np.nan)
            features['lap_mean_gap_ahead_m'] = dist_ahead.mean()
        else:
            features['lap_mean_gap_ahead_m'] = np.nan

        # --- weather data ---
        # weather affects grip and tire degradation a lot
        try:
            weather = lap.get_weather_data()
            if weather is not None and not weather.empty:
                features['Weather_AirTemp'] = weather.get('AirTemp', np.nan)
                features['Weather_TrackTemp'] = weather.get('TrackTemp', np.nan)
                features['Weather_Humidity'] = weather.get('Humidity', np.nan)
                features['Weather_Pressure'] = weather.get('Pressure', np.nan)
                features['Weather_WindSpeed'] = weather.get('WindSpeed', np.nan)
                features['Weather_WindDirection'] = weather.get('WindDirection', np.nan)
                features['Weather_Rainfall'] = weather.get('Rainfall', np.nan)
            else:
                features['Weather_AirTemp'] = np.nan
                features['Weather_TrackTemp'] = np.nan
                features['Weather_Humidity'] = np.nan
                features['Weather_Pressure'] = np.nan
                features['Weather_WindSpeed'] = np.nan
                features['Weather_WindDirection'] = np.nan
                features['Weather_Rainfall'] = np.nan
        except Exception:
            features['Weather_AirTemp'] = np.nan
            features['Weather_TrackTemp'] = np.nan
            features['Weather_Humidity'] = np.nan
            features['Weather_Pressure'] = np.nan
            features['Weather_WindSpeed'] = np.nan
            features['Weather_WindDirection'] = np.nan
            features['Weather_Rainfall'] = np.nan

        return features

    except Exception:
        # if something fails, we just skip this lap
        # the lap will be dropped later when we clean the data
        return None

def _process_single_event(task):
    """
    processes one race event (year + race name) and returns dataframe with features.
    designed to work in parallel processing without shared state.
    """
    year = task['year']
    event = task['event']
    meta = task['meta']
    try:
        fastf1.Cache.enable_cache(CACHE_DIR, force_renew=False, ignore_version=True)
        session = fastf1.get_session(year, event, 'R')
        session.load()  # load all data: laps, telemetry, etc

        laps = session.laps.copy()
        laps['LapTime_sec'] = laps['LapTime'].dt.total_seconds()

        # --- create pit lap flag ---
        # a lap is a pit lap if the car entered the pits
        laps['IsPitLap'] = laps['PitInTime'].notna()

        # --- compare each lap to the fastest lap ---
        fastest = laps['LapTime_sec'].min()
        laps['GapToEventFastest'] = laps['LapTime_sec'] - fastest

        tqdm.pandas(desc=f"  - Features for {event}", disable=True)
        feature_list = laps.progress_apply(_get_lap_telemetry_features, axis=1)
        features_df = pd.DataFrame(feature_list.dropna().tolist(), index=feature_list.dropna().index)

        session_laps_with_features = laps.join(features_df)

        session_laps_with_features['Year'] = year
        session_laps_with_features['EventName'] = event
        session_laps_with_features['RoundNumber'] = meta.get('RoundNumber', np.nan)
        session_laps_with_features['Country'] = meta.get('Country', None)
        session_laps_with_features['Location'] = meta.get('Location', None)
        session_laps_with_features['OfficialEventName'] = meta.get('OfficialEventName', None)
        session_laps_with_features['SessionName'] = 'Race'
        return session_laps_with_features
    except Exception as e:
        print(f"    [error] could not process session {year} - {event}: {e}")
        return None


def fetch_and_process_data(years, output_file, n_jobs=1):
    """
    fetches f1 data for given years, processes telemetry features
    for each lap, and saves the final dataframe to a parquet file.
    uses local cache if available so we dont re-download everything.
    """
    print("--- running full data collection and feature engineering ---")
    fastf1.Cache.enable_cache(CACHE_DIR, force_renew=False, ignore_version=True)
    
    tasks = []
    for year in years:
        print(f"\n" + "="*50)
        print(f"processing year: {year}")
        print("="*50)
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = schedule[schedule['RoundNumber'] > 0]
        for _, event in races.iterrows():
            tasks.append({
                'year': year,
                'event': event['EventName'],
                'meta': {
                    'RoundNumber': event.get('RoundNumber', np.nan),
                    'Country': event.get('Country', None),
                    'Location': event.get('Location', None),
                    'OfficialEventName': event.get('OfficialEventName', None),
                }
            })

    all_processed_laps = []
    if n_jobs > 1:
        # parallel processing is faster but can be buggy with fastf1 cache
        print(f"parallel processing enabled with {n_jobs} workers")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            for res in tqdm(pool.imap_unordered(_process_single_event, tasks), total=len(tasks), desc="Races"):
                if res is not None:
                    all_processed_laps.append(res)
    else:
        for task in tqdm(tasks, desc="Races"):
            res = _process_single_event(task)
            if res is not None:
                all_processed_laps.append(res)

    if not all_processed_laps:
        print("\nno data was collected. something went wrong.")
        return False

    def _add_group_level_features(df):
        """
        adds features that need info from multiple laps:
        - degradation slope (how fast tires are losing grip)
        - delta to median lap time
        - pit stop info
        """
        df = df.copy()

        # compare each lap to session and driver averages
        # this helps identify unusually slow or fast laps
        if 'LapTime_sec' in df.columns:
            df['DeltaToEventMedian'] = df['LapTime_sec'] - df.groupby(['Year', 'EventName'])['LapTime_sec'].transform('median')
            df['DeltaToDriverMedian'] = df['LapTime_sec'] - df.groupby(['Year', 'EventName', 'Driver'])['LapTime_sec'].transform('median')
            df['DeltaToPrevLap'] = df.groupby(['Year', 'EventName', 'Driver'])['LapTime_sec'].diff()

        # sector times in seconds if we have them
        for col in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
            if col in df.columns:
                df[f'{col}_sec'] = pd.to_timedelta(df[col]).dt.total_seconds()
                df[f'{col}_delta_event'] = df[f'{col}_sec'] - df.groupby(['Year', 'EventName'])[f'{col}_sec'].transform('median')

        # stint progression (how many laps on current set of tires)
        if {'Year', 'EventName', 'Driver', 'Stint', 'LapNumber'}.issubset(df.columns):
            df['StintLapIndex'] = df.groupby(['Year', 'EventName', 'Driver', 'Stint'])['LapNumber'].rank(method='first')

            def _add_slope(group):
                # degradation slope shows how much slower the car gets per lap
                # positive slope = tires are degrading
                if group['LapNumber'].nunique() < 2 or group['LapTime_sec'].isnull().all():
                    group['StintDegradationSlope'] = np.nan
                else:
                    try:
                        coef = np.polyfit(group['LapNumber'], group['LapTime_sec'], 1)
                        group['StintDegradationSlope'] = coef[0]
                    except Exception:
                        group['StintDegradationSlope'] = np.nan
                return group

            df = df.groupby(['Year', 'EventName', 'Driver', 'Stint'], group_keys=False).apply(_add_slope)

            # stint length helps predict when next pit will happen
            df['StintLength'] = df.groupby(['Year', 'EventName', 'Driver', 'Stint'])['LapNumber'].transform('count')
            if 'Compound' in df.columns:
                # how long do similar tire compounds usually last?
                df['CompoundMedianStintLen_event'] = df.groupby(['Year', 'EventName', 'Compound'])['StintLength'].transform('median')
            else:
                df['CompoundMedianStintLen_event'] = np.nan
            df['StintProgressRatio'] = (df['StintLapIndex'] / df['CompoundMedianStintLen_event']).clip(lower=0, upper=2)

        # count how many pit stops each driver made
        if 'IsPitLap' in df.columns:
            pit_counts = df.groupby(['Year', 'EventName', 'Driver'])['IsPitLap'].sum().rename('PitStopCount')
            df = df.merge(pit_counts, on=['Year', 'EventName', 'Driver'], how='left')
            df['PitStopClass'] = df['PitStopCount'].apply(lambda c: 1 if c <= 1 else 2 if c == 2 else 3)
        else:
            df['PitStopCount'] = np.nan
            df['PitStopClass'] = np.nan

        return df

    # --- combine all races and add extra features ---
    print("\n" + "="*50)
    print("combining all processed data...")
    print("="*50)
    final_df = pd.concat(all_processed_laps, ignore_index=True)

    final_df = _add_group_level_features(final_df)

    def _add_causal_features(df):
        """
        adds features that only look at past laps (no future peeking!).
        this is important to avoid data leakage which causes overfitting.
        
        we build lag features (previous lap values) and rolling averages.
        for example: lap_mean_speed_lag1 is the speed from the previous lap.
        """
        df = df.copy()
        group_cols = ['Year', 'EventName', 'Driver']
        lag_cols = [
            'LapTime_sec', 'lap_mean_speed', 'lap_std_speed', 'lap_perc_throttle_full',
            'lap_perc_brake_on', 'lap_mean_rpm', 'lap_accel_mean_abs', 'lap_accel_max',
            'lap_throttle_std', 'lap_distance', 'lap_drs_usage_pct', 'TyreLife'
        ]
        roll_windows = [3, 5, 8]

        def _per_group(group):
            group = group.sort_values('LapNumber')
            pit_prev = group['IsPitLap'].fillna(False).shift(1).fillna(False)
            # how many laps since the last pit stop
            segment = pit_prev.cumsum()
            group['LapsSincePrevPit'] = group.groupby(segment).cumcount()
            # how many pits in the last 5 laps (helps model urgency)
            group['pit_in_last5'] = pit_prev.rolling(window=5, min_periods=1).sum()

            # create lag and rolling features for each metric
            for col in lag_cols:
                if col not in group.columns:
                    continue
                col_shifted = group[col].shift(1)
                group[f'{col}_lag1'] = col_shifted
                for w in roll_windows:
                    group[f'{col}_roll{w}'] = col_shifted.rolling(window=w, min_periods=1).mean()
            return group

        return df.groupby(group_cols, group_keys=False).apply(_per_group)

    final_df = _add_causal_features(final_df)
    
    try:
        final_df.to_parquet(output_file, index=False)
        print(f"\n[done] saved {len(final_df)} total laps to '{output_file}'")
        print(f"shape of the final dataset: {final_df.shape}")
        return True
    except Exception as e:
        print(f"\n[error] failed to save parquet file: {e}")
        return False

def load_processed_data(file_path):
    """
    loads a processed parquet file into a dataframe.
    """
    print(f"loading data from '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"[error] input file not found at '{file_path}'")
        print("please run the data collection step first.")
        return None
    
    df = pd.read_parquet(file_path)
    print(f"[done] loaded {len(df)} rows and {len(df.columns)} columns.")
    return df
