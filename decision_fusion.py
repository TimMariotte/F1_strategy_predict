"""
decision fusion layer combining strategy and anomaly signals into actionable guidance.
this module takes outputs from both models and produces a unified decision score.
"""

import numpy as np
import pandas as pd


def _get_strategy_predictions(df, strategy_outputs):
    """
    copies strategy model predictions into the main dataframe.
    """
    if not strategy_outputs or 'df_with_predictions' not in strategy_outputs:
        return df

    strat_df = strategy_outputs['df_with_predictions']
    # align on index to keep ordering identical to the base dataframe
    strat_df = strat_df.reindex(df.index)

    # columns we want to copy from strategy model output
    cols_to_copy = [
        'PredPitWindowClass', 'PredNextPitGapLaps', 'RecommendedPitLap',
        'PredPitStopClass', 'PredPitNextLapProb', 'PredPitNextLap'
    ] + [c for c in strat_df.columns if c.startswith('PredPitWindowProb_')]

    for col in cols_to_copy:
        if col in strat_df.columns:
            df[col] = strat_df[col]
        else:
            if col.startswith('PredPitWindowProb_'):
                df[col] = 0.0
            else:
                df[col] = np.nan
    return df


def _build_message(row, pit_signal, anomaly_signal):
    """
    builds a simple message explaining what the model thinks is happening.
    """
    if pd.notna(row.get('IsAnomaly')) and row.get('IsAnomaly') == 1:
        return f"alert {row.get('AnomalyType', 'anomaly')}: score={anomaly_signal:.2f}"
    next_prob = row.get('PredPitNextLapProb', 0)
    if next_prob > 0.5:
        return f"pit likely next lap (p={next_prob:.2f})"
    if pit_signal > 0.75:
        return f"optimal pit imminent (lap ~{row.get('RecommendedPitLap', np.nan):.1f})"
    if pit_signal > 0.5:
        return f"prepare pit stop in {row.get('PredNextPitGapLaps', np.nan):.1f} laps"
    return "continue, monitoring"


def combine_predictions(df, strategy_outputs):
    """
    combines strategy and anomaly model outputs into a unified score.
    
    the decision score goes from 0 to 100:
    - high score = something needs attention (pit soon or anomaly detected)
    - low score = everything is normal, keep going
    
    we weight pit signals at 60% and anomaly signals at 40%.
    
    returns:
        pd.dataframe: dataframe with decisionscore, piturge, anomalyseverity, decisionmessage.
    """
    df = df.copy()
    df = _get_strategy_predictions(df, strategy_outputs)

    # --- strategy signal ---
    # combine imminent and soon probabilities
    pit_imminent = df.get('PredPitWindowProb_imminent', pd.Series(0, index=df.index)).fillna(0)
    pit_soon = df.get('PredPitWindowProb_soon', pd.Series(0, index=df.index)).fillna(0)
    pit_window_signal = (0.7 * pit_imminent + 0.3 * pit_soon).clip(0, 1)

    # probability of pit on next lap (more precise timing)
    pit_next_prob = df.get('PredPitNextLapProb', pd.Series(0, index=df.index)).fillna(0)
    # blend both signals: 50% window, 50% next-lap
    pit_signal = (0.5 * pit_window_signal + 0.5 * pit_next_prob).clip(0, 1)

    # --- anomaly signal ---
    # normalize anomaly score to 0-1 range
    anomaly_score = df.get('AnomalyScore', df.get('AE_ReconstructionError', pd.Series(0, index=df.index))).fillna(0)
    if anomaly_score.max() - anomaly_score.min() > 0:
        anomaly_signal = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min())
    else:
        anomaly_signal = pd.Series(0, index=df.index)

    # final decision score: 60% pit, 40% anomaly
    decision_score = 100 * (0.6 * pit_signal + 0.4 * anomaly_signal)

    # save all the components
    df['PitUrgency'] = pit_signal
    df['PitNextLapProb'] = pit_next_prob
    df['AnomalySeverity'] = anomaly_signal
    df['DecisionScore'] = decision_score.clip(0, 100)
    # build a human-readable message for each lap
    df['DecisionMessage'] = [
        _build_message(row, pit_signal[i], anomaly_signal[i]) for i, row in df.iterrows()
    ]

    return df
