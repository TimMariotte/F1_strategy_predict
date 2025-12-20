"""
module: strategy_model.py
description: handles training and evaluation of pit stop strategy prediction models.

we train multiple models:
1. pit window classifier: when will the next pit happen (imminent/soon/later/no_stop)
2. pit gap regressor: how many laps until next pit
3. pit next lap classifier: will there be a pit on the next lap
4. total pit class: will this be a 1, 2, or 3+ stop strategy

overfitting/underfitting notes:
- we use hold-out year validation (train on old years, test on most recent)
- class weighting helps with imbalanced classes (imminent pits are rare)
- we do small grid search to find best hyperparameters
- if train f1 >> test f1, model is overfitting
- if both f1 scores are low, model might need more features or different architecture
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def _compute_next_pit_targets(df):
    """
    computes target variables for each lap:
    - how many laps until next pit stop
    - what category is the pit window (imminent/soon/later/no_stop)
    """
    df = df.copy()
    group_cols = ['Year', 'EventName', 'Driver']

    gaps = pd.Series(index=df.index, dtype=float)
    classes = pd.Series(index=df.index, dtype=object)

    # for each driver in each race, find when their next pit is
    for _, group in df.groupby(group_cols):
        pit_laps = group.loc[group['IsPitLap'] == True, 'LapNumber'].sort_values().tolist()
        for idx, lap_num in group['LapNumber'].items():
            # find the next pit after this lap
            next_pits = [p for p in pit_laps if p > lap_num]
            gap = (next_pits[0] - lap_num) if next_pits else np.nan
            gaps.loc[idx] = gap
            # classify into categories
            if np.isnan(gap):
                classes.loc[idx] = 'no_stop'
            elif gap <= 3:
                classes.loc[idx] = 'imminent'  # pit coming in 1-3 laps
            elif gap <= 8:
                classes.loc[idx] = 'soon'      # pit coming in 4-8 laps
            else:
                classes.loc[idx] = 'later'     # pit more than 8 laps away

    df['NextPitGapLaps'] = gaps
    df['PitWindowClass'] = classes
    df['NextPitGapLaps_filled'] = df['NextPitGapLaps'].fillna(30)  # if no pit ahead, assume 30 laps

    return df


def _feature_columns(df):
    """
    returns the list of features we use for strategy prediction.
    we only use past-looking features to avoid data leakage.
    """
    candidates = [
        'LapNumber', 'Stint', 'StintLapIndex', 'StintLength', 'StintProgressRatio',
        'CompoundMedianStintLen_event',
        'TyreLife', 'Compound_encoded',
        'LapsSincePrevPit', 'pit_in_last5',
        'LapTime_sec_lag1', 'LapTime_sec_roll3', 'LapTime_sec_roll5',
        'LapTime_sec_roll8',
        'lap_mean_speed_lag1', 'lap_mean_speed_roll3', 'lap_mean_speed_roll5',
        'lap_mean_speed_roll8',
        'lap_std_speed_lag1',
        'lap_perc_throttle_full_lag1', 'lap_perc_brake_on_lag1',
        'lap_drs_usage_pct_lag1', 'lap_mean_rpm_lag1',
        'lap_accel_mean_abs_lag1', 'lap_accel_mean_abs_roll3', 'lap_accel_mean_abs_roll5', 'lap_accel_mean_abs_roll8',
        'lap_accel_max_lag1',
        'lap_throttle_std_lag1',
        'lap_distance_lag1', 'lap_distance_roll3', 'lap_distance_roll5', 'lap_distance_roll8',
        'Weather_AirTemp', 'Weather_TrackTemp', 'Weather_WindSpeed'
    ]
    return [c for c in candidates if c in df.columns]


def _small_grid_search_clf(base_params, X_train, y_train, X_val, y_val, num_class, sample_weight=None):
    """
    simple grid search for classification.
    we only try 2 configurations to keep runtime reasonable.
    returns the best model based on f1 score.
    """
    grid = [
        {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 300},
        {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 500},
    ]
    best_f1 = -np.inf
    best_model = None
    best_params = None

    for params in grid:
        clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_class,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            **base_params,
            **params,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
        preds = clf.predict(X_val)
        score = f1_score(y_val, preds, average='macro')
        if score > best_f1:
            best_f1 = score
            best_model = clf
            best_params = params

    return best_model, best_params, best_f1


def _small_grid_search_bin(base_params, X_train, y_train, X_val, y_val, sample_weight=None):
    grid = [
        {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200},
        {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300},
    ]
    best_f1 = -np.inf
    best_model = None
    best_params = None

    for params in grid:
        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            **base_params,
            **params,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
        preds = clf.predict(X_val)
        score = f1_score(y_val, preds, average='binary')
        if score > best_f1:
            best_f1 = score
            best_model = clf
            best_params = params

    return best_model, best_params, best_f1


def _small_grid_search_reg(base_params, X_train, y_train, X_val, y_val):
    """
    simple grid search for regression.
    returns the best model based on mean absolute error.
    """
    grid = [
        {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 300},
        {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500},
    ]
    best_mae = np.inf
    best_model = None
    best_params = None

    for params in grid:
        reg = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            **base_params,
            **params,
        )
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = reg.predict(X_val)
        score = mean_absolute_error(y_val, preds)
        if score < best_mae:
            best_mae = score
            best_model = reg
            best_params = params

    return best_model, best_params, best_mae
def _prepare_strategy_data(df):
    """
    prepares data for strategy models.
    encodes categorical variables and creates target labels.
    """

    df = df.copy()
    if 'LapTime_sec' not in df.columns and 'LapTime' in df.columns:
        df['LapTime_sec'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

    df = _compute_next_pit_targets(df)

    # encode tire compound as a number
    compound_le = LabelEncoder()
    df['Compound_encoded'] = compound_le.fit_transform(df['Compound'].astype(str))
    
    df = df.sort_values(by=['Year', 'EventName', 'Driver', 'LapNumber'])
    # create target for next lap pit prediction (shift the pit flag forward)
    df['NextLapIsPit'] = df.groupby(['Year', 'EventName', 'Driver'])['IsPitLap'].shift(-1)

    feature_cols = _feature_columns(df)
    # drop rows without target labels, but keep as many as possible
    df_clean = df.dropna(subset=['PitWindowClass', 'NextPitGapLaps_filled', 'PitStopClass']).copy()

    # fill missing features with median instead of dropping rows
    # this helps keep more training data
    for col in feature_cols:
        if col not in df_clean.columns:
            continue
        if df_clean[col].notna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna(0)

    # encode window class for classification
    window_le = LabelEncoder()
    df_clean['PitWindowClass_encoded'] = window_le.fit_transform(df_clean['PitWindowClass'])

    # pit stop class: 1, 2, or 3+ stops
    if 'PitStopClass' in df_clean.columns:
        df_clean['PitStopClass'] = df_clean['PitStopClass'].fillna(0).astype(int).clip(1, 3)
    else:
        df_clean['PitStopClass'] = 3

    return df_clean, feature_cols, compound_le, window_le

def train_strategy_model(df, years, test_year=None):
    """
    trains and evaluates multiple strategy prediction models.
    
    we train 4 models:
    1. multi-class pit window (when is next pit)
    2. regression for exact gap to next pit
    3. multi-class total pit strategy (1/2/3+ stops)
    4. binary classifier for pit on next lap
    
    validation strategy:
    - we use temporal split: train on older years, test on most recent
    - this simulates real-world use where we predict future races
    - class weighting helps with imbalanced data (imminent pits are rare)
    
    returns:
        dict: trained models and dataframe with predictions.
    """
    print("\n" + "="*50)
    print("training strategy prediction models (multi-task)...")
    print("="*50)

    df_model, features, compound_le, window_le = _prepare_strategy_data(df)

    # temporal split: train on old years, test on most recent
    # this is more realistic than random split for time series data
    available_years = sorted(df_model['Year'].unique())
    test_year = test_year or (max(years) if years else available_years[-1])
    if test_year not in available_years:
        test_year = available_years[-1]

    train_years = [y for y in available_years if y != test_year]
    if not train_years:
        train_years = available_years[:-1] or [available_years[0]]

    train_df = df_model[df_model['Year'].isin(train_years)]
    test_df = df_model[df_model['Year'] == test_year]

    X = train_df[features]
    y_window = train_df['PitWindowClass_encoded']
    y_gap = train_df['NextPitGapLaps_filled']
    y_total = train_df['PitStopClass'] - 1  # shift to 0..2 for XGBoost

    X_train_win, X_val_win, y_win_train, y_win_val = train_test_split(
        X, y_window, test_size=0.2, random_state=42, stratify=y_window
    )
    X_train_gap, X_val_gap, y_gap_train, y_gap_val = train_test_split(
        X, y_gap, test_size=0.2, random_state=42
    )
    X_train_total, X_val_total, y_total_train, y_total_val = train_test_split(
        X, y_total, test_size=0.2, random_state=42, stratify=y_total
    )

    # base xgboost params to prevent overfitting
    # subsample and colsample add randomness (like dropout)
    base_params = {'subsample': 0.8, 'colsample_bytree': 0.8}

    # class weights to handle imbalanced data
    # imminent pits are rare so we give them more weight
    class_counts = y_window.value_counts()
    class_weights = {cls: len(y_window) / (len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weights = y_window.map(class_weights)

    # same weighting for next-lap pit prediction
    y_nextpit = train_df['NextLapIsPit'].fillna(0).astype(int)
    next_class_counts = y_nextpit.value_counts()
    next_weights = {cls: len(y_nextpit) / (len(next_class_counts) * cnt) for cls, cnt in next_class_counts.items()}
    next_sample_weights = y_nextpit.map(next_weights)

    # --- train pit window classifier ---
    pit_win_model, win_params, win_f1 = _small_grid_search_clf(
        base_params,
        X_train_win, y_win_train,
        X_val_win, y_win_val,
        num_class=len(window_le.classes_),
        sample_weight=sample_weights.loc[X_train_win.index] if isinstance(X_train_win, pd.DataFrame) else None
    )
    print(f"pit window best params: {win_params}, val macro-f1={win_f1:.3f}")

    # --- train next lap pit classifier ---
    X_train_next, X_val_next, y_next_train, y_next_val = train_test_split(
        X, y_nextpit, test_size=0.2, random_state=42, stratify=y_nextpit
    )
    next_model, next_params, next_f1 = _small_grid_search_bin(
        base_params,
        X_train_next, y_next_train,
        X_val_next, y_next_val,
        sample_weight=next_sample_weights.loc[X_train_next.index] if isinstance(X_train_next, pd.DataFrame) else None
    )
    print(f"next-lap pit best params: {next_params}, val f1={next_f1:.3f}")

    # --- train next pit gap regressor ---
    pit_gap_model, gap_params, gap_mae = _small_grid_search_reg(
        base_params, X_train_gap, y_gap_train, X_val_gap, y_gap_val
    )
    print(f"next pit gap best params: {gap_params}, val mae={gap_mae:.3f}")

    # --- train total pit class classifier (1/2/3+ stops) ---
    total_model, total_params, total_f1 = _small_grid_search_clf(
        base_params, X_train_total, y_total_train, X_val_total, y_total_val, num_class=3
    )
    print(f"total pit class best params: {total_params}, val macro-f1={total_f1:.3f}")

    # --- evaluate on hold-out year ---
    # this tells us how well the model generalizes to unseen data
    eval_results = {}
    if not test_df.empty:
        X_test = test_df[features]
        y_test_window = test_df['PitWindowClass_encoded']
        y_test_gap = test_df['NextPitGapLaps_filled']
        y_test_total = test_df['PitStopClass'] - 1
        y_test_next = test_df['NextLapIsPit'].fillna(0).astype(int)

        win_preds = pit_win_model.predict(X_test)
        gap_preds = pit_gap_model.predict(X_test)
        total_preds = total_model.predict(X_test)
        next_preds = next_model.predict(X_test)
        next_proba_test = next_model.predict_proba(X_test)[:, 1]

        eval_results['pit_window_f1'] = f1_score(y_test_window, win_preds, average='macro')
        eval_results['pit_gap_mae'] = mean_absolute_error(y_test_gap, gap_preds)
        eval_results['pit_total_f1'] = f1_score(y_test_total, total_preds, average='macro')
        eval_results['pit_next_f1'] = f1_score(y_test_next, next_preds, average='binary')
        eval_results['pit_next_prec'] = ( (y_test_next & (next_preds == 1)).sum() / max((next_preds == 1).sum(), 1) )
        eval_results['pit_next_rec'] = ( (y_test_next & (next_preds == 1)).sum() / max(y_test_next.sum(), 1) )

        print("\n--- pit window evaluation ---")
        print(classification_report(y_test_window, win_preds, target_names=window_le.classes_))
        
        # save confusion matrix to see where the model makes mistakes
        cm = confusion_matrix(y_test_window, win_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=window_le.classes_, yticklabels=window_le.classes_)
        plt.title(f'pit window confusion matrix - test year {test_year}')
        plt.savefig('confusion_matrix_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[done] confusion matrix saved as 'confusion_matrix_strategy.png'")

        # feature importance shows what the model uses most for predictions
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(pit_win_model, max_num_features=15, height=0.8)
        plt.title('strategy model feature importance (pit window)')
        plt.tight_layout()
        plt.savefig('feature_importance_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[done] feature importance plot saved as 'feature_importance_strategy.png'")
    else:
        print("[warning] no hold-out year available for evaluation")

    # --- apply models to entire dataset ---
    # we need predictions for the decision fusion layer
    df_model['PredPitWindowClass'] = window_le.inverse_transform(pit_win_model.predict(df_model[features]))
    win_proba = pit_win_model.predict_proba(df_model[features])
    for idx, cls in enumerate(window_le.classes_):
        df_model[f'PredPitWindowProb_{cls}'] = win_proba[:, idx]

    df_model['PredNextPitGapLaps'] = pd.Series(pit_gap_model.predict(df_model[features])).clip(lower=0)
    df_model['RecommendedPitLap'] = df_model['LapNumber'] + df_model['PredNextPitGapLaps']

    total_preds_full = total_model.predict(df_model[features]) + 1  # shift back to 1..3
    df_model['PredPitStopClass'] = total_preds_full

    # Next-lap pit probability and class
    next_proba = pit_next_probs = next_model.predict_proba(df_model[features])[:, 1]
    df_model['PredPitNextLapProb'] = next_proba
    df_model['PredPitNextLap'] = (next_proba >= 0.5).astype(int)

    artifacts = {
        'pit_window_model': pit_win_model,
        'pit_gap_model': pit_gap_model,
        'pit_total_model': total_model,
        'pit_next_model': next_model,
        'window_encoder': window_le,
        'compound_encoder': compound_le,
        'features': features,
        'eval': eval_results,
        'df_with_predictions': df_model
    }

    return artifacts
