"""
f1 predictive analysis dashboard
description: an interactive web app built with streamlit to visualize
             the results of our f1 data analysis and ml models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- page configuration ---
st.set_page_config(
    page_title="F1 Predictive Analysis Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- custom css for better styling ---
st.markdown("""
<style>
    /* main background - light grey for better readability */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 50%, #d9dfe5 100%);
    }
    
    /* sidebar styling - f1 red theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e10600 0%, #b80500 100%);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    [data-testid="metric-container"] label {
        color: #e10600 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
    }
    
    /* headers */
    h1, h2, h3 {
        color: #1a1a2e !important;
    }
    
    /* text */
    p, .stMarkdown {
        color: #333333 !important;
    }
    
    /* dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 8px;
        color: #333;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e10600 !important;
        color: white !important;
    }
    
    /* info boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* expanders */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- load data ---
@st.cache_data
def load_data(file_path):
    """loads the parquet file with all our predictions."""
    try:
        df = pd.read_parquet(file_path)
        return df
    except FileNotFoundError:
        return None


def create_gauge_chart(value, title, max_val=100):
    """creates a nice gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': '#333333'}},
        number={'font': {'color': '#333333'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#333333'},
            'bar': {'color': '#e10600'},
            'bgcolor': 'rgba(240,240,240,1)',
            'bordercolor': 'rgba(200,200,200,1)',
            'steps': [
                {'range': [0, max_val*0.33], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [max_val*0.33, max_val*0.66], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [max_val*0.66, max_val], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_lap_time_chart(filtered_df, selected_driver):
    """creates an interactive lap time chart with anomalies highlighted."""
    fig = go.Figure()
    
    # normal laps
    normal_laps = filtered_df[filtered_df.get('IsAnomaly', 0) == 0]
    anomaly_laps = filtered_df[filtered_df.get('IsAnomaly', 0) == 1]
    
    fig.add_trace(go.Scatter(
        x=normal_laps['LapNumber'],
        y=normal_laps['LapTime_sec'],
        mode='markers+lines',
        name='normal laps',
        marker=dict(color='#00d4ff', size=8),
        line=dict(color='#00d4ff', width=2)
    ))
    
    if not anomaly_laps.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_laps['LapNumber'],
            y=anomaly_laps['LapTime_sec'],
            mode='markers',
            name='anomalies',
            marker=dict(color='#e94560', size=14, symbol='x')
        ))
    
    # add pit stops as vertical lines
    if 'IsPitLap' in filtered_df.columns:
        pit_laps = filtered_df[filtered_df['IsPitLap'] == True]['LapNumber'].tolist()
        for i, lap in enumerate(pit_laps):
            fig.add_vline(
                x=lap, 
                line_dash="dash", 
                line_color="#ffd700",
                annotation_text="PIT"
            )
        
        # add invisible trace for legend
        if pit_laps:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name='pitstop performed by team',
                line=dict(color='#ffd700', dash='dash', width=2),
                showlegend=True
            ))
    
    fig.update_layout(
        title=f'lap times for {selected_driver}',
        xaxis_title='lap number',
        yaxis_title='lap time (seconds)',
        template='plotly_white',
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(color='#333333')
    )
    
    return fig


def create_strategy_chart(filtered_df):
    """creates a dual-axis chart for strategy visualization."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if 'DecisionScore' in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df['LapNumber'],
                y=filtered_df['DecisionScore'],
                name='decision score',
                line=dict(color='#00ff88', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 136, 0.2)'
            ),
            secondary_y=False
        )
    
    if 'PitUrgency' in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df['LapNumber'],
                y=filtered_df['PitUrgency'] * 100,
                name='pit urgency',
                line=dict(color='#ff6b6b', width=3)
            ),
            secondary_y=True
        )
    
    # add pit stops
    if 'IsPitLap' in filtered_df.columns:
        pit_laps = filtered_df[filtered_df['IsPitLap'] == True]['LapNumber'].tolist()
        for lap in pit_laps:
            fig.add_vline(x=lap, line_dash="dash", line_color="#ffd700", line_width=2)
        
        # add invisible trace for legend
        if pit_laps:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name='pitstop performed by team',
                line=dict(color='#ffd700', dash='dash', width=2),
                showlegend=True
            ), secondary_y=False)
    
    fig.update_layout(
        title='strategy signals over race',
        template='plotly_white',
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(color='#333333')
    )
    fig.update_xaxes(title_text='lap number')
    fig.update_yaxes(title_text='decision score', secondary_y=False)
    fig.update_yaxes(title_text='pit urgency (%)', secondary_y=True)
    
    return fig


def create_telemetry_chart(filtered_df):
    """creates a multi-line chart for telemetry data."""
    fig = go.Figure()
    
    if 'lap_mean_speed' in filtered_df.columns:
        # normalize speed to 0-100 scale for comparison
        speed_norm = (filtered_df['lap_mean_speed'] - filtered_df['lap_mean_speed'].min()) / \
                     (filtered_df['lap_mean_speed'].max() - filtered_df['lap_mean_speed'].min() + 0.001) * 100
        fig.add_trace(go.Scatter(
            x=filtered_df['LapNumber'],
            y=speed_norm,
            name='speed (normalized)',
            line=dict(color='#00d4ff', width=2)
        ))
    
    if 'lap_perc_throttle_full' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df['LapNumber'],
            y=filtered_df['lap_perc_throttle_full'],
            name='% full throttle',
            line=dict(color='#00ff88', width=2)
        ))
    
    if 'lap_perc_brake_on' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df['LapNumber'],
            y=filtered_df['lap_perc_brake_on'],
            name='% brake',
            line=dict(color='#ff6b6b', width=2)
        ))
    
    fig.update_layout(
        title='telemetry overview',
        xaxis_title='lap number',
        yaxis_title='percentage / normalized',
        template='plotly_white',
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        height=300,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(color='#333333')
    )
    
    return fig


def create_anomaly_radar(row):
    """creates a radar chart for anomaly analysis."""
    categories = ['speed', 'throttle', 'brake', 'tyre life', 'lap time']
    
    # get values, normalized
    values = [
        row.get('lap_mean_speed', 0) / 300 * 100 if row.get('lap_mean_speed', 0) else 0,
        row.get('lap_perc_throttle_full', 0) if row.get('lap_perc_throttle_full', 0) else 0,
        row.get('lap_perc_brake_on', 0) if row.get('lap_perc_brake_on', 0) else 0,
        min(row.get('TyreLife', 0) / 30 * 100, 100) if row.get('TyreLife', 0) else 0,
        (1 - (row.get('LapTime_sec', 90) - 70) / 50) * 100 if row.get('LapTime_sec', 0) else 0
    ]
    values = [max(0, min(100, v)) for v in values]  # clamp to 0-100
    values.append(values[0])  # close the radar
    categories.append(categories[0])
    
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(233, 69, 96, 0.3)',
        line=dict(color='#e94560', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(255,255,255,0.9)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(0,0,0,0.15)'),
            angularaxis=dict(gridcolor='rgba(0,0,0,0.15)')
        ),
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=300,
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(color='#333333')
    )
    
    return fig


# --- main app ---
def main():
    # header with logo-like styling
    st.markdown("""
        <h1 style='text-align: center; font-size: 3em; margin-bottom: 0; color: #1a1a2e;'>
            🏎️ F1 Predictive Analysis
        </h1>
        <p style='text-align: center; font-size: 1.2em; color: #e10600; margin-top: 0;'>
            strategy prediction & anomaly detection
        </p>
        <hr style='border: 1px solid rgba(0,0,0,0.1); margin: 20px 0;'>
    """, unsafe_allow_html=True)

    df = load_data('f1_data_with_anomalies.parquet')

    if df is None:
        st.error("⚠️ data file 'f1_data_with_anomalies.parquet' not found. please run: `python main.py`")
        st.info("💡 this will process the f1 data and train the models. it may take a few minutes.")
        return

    # --- sidebar filters ---
    with st.sidebar:
        st.markdown("<h2>🔧 Filters</h2>", unsafe_allow_html=True)
        
        selected_year = st.selectbox("📅 Year", sorted(df['Year'].unique(), reverse=True))
        
        events_in_year = df[df['Year'] == selected_year]['EventName'].unique()
        selected_event = st.selectbox("🏁 Grand Prix", events_in_year)
        
        drivers_in_event = df[(df['Year'] == selected_year) & (df['EventName'] == selected_event)]['Driver'].unique()
        selected_driver = st.selectbox("👤 Driver", sorted(drivers_in_event))
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("<h3>📊 Quick Stats</h3>", unsafe_allow_html=True)
        total_races = df.groupby(['Year', 'EventName']).ngroups
        total_laps = len(df)
        st.markdown(f"**Total races:** {total_races}")
        st.markdown(f"**Total laps:** {total_laps:,}")

    # filter the dataframe
    filtered_df = df[
        (df['Year'] == selected_year) &
        (df['EventName'] == selected_event) &
        (df['Driver'] == selected_driver)
    ].copy().sort_values('LapNumber')

    if filtered_df.empty:
        st.error("no data for this selection.")
        return

    # --- main content ---
    st.markdown(f"""
        <h2 style='color: #e10600;'>
            {selected_driver} — {selected_event} {selected_year}
        </h2>
    """, unsafe_allow_html=True)

    # metrics row with gauges
    anomalies = filtered_df[filtered_df.get('IsAnomaly', 0) == 1]
    avg_decision = filtered_df.get('DecisionScore', pd.Series([0])).mean()
    avg_urgency = filtered_df.get('PitUrgency', pd.Series([0])).mean() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 Anomalies", len(anomalies), delta=None)
    with col2:
        st.metric("📊 Avg Decision Score", f"{avg_decision:.1f}")
    with col3:
        st.metric("⏱️ Pit Urgency", f"{avg_urgency:.1f}%")
    with col4:
        total_laps_driver = len(filtered_df)
        st.metric("🏁 Total Laps", total_laps_driver)

    st.markdown("<br>", unsafe_allow_html=True)

    # tabs for different views - strategy first to be default
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategy", "📈 Lap Analysis", "⚠️ Anomalies", "📡 Telemetry"])
    
    with tab1:
        st.plotly_chart(create_strategy_chart(filtered_df), use_container_width=True)
        
        # explanation
        st.info("📊 **Interpretation:** When the 'pit urgency' curve (red) is high, the model indicates it's a good time to perform a pit stop.")
        
        # pit stops info
        if 'IsPitLap' in filtered_df.columns:
            pit_laps = filtered_df[filtered_df['IsPitLap'] == True]['LapNumber'].tolist()
            if pit_laps:
                st.success(f"🏁 Pit stops at laps: {', '.join(map(str, map(int, pit_laps)))}")
            else:
                st.info("ℹ️ No pit stops in this stint")
        
        # strategy predictions table
        strat_cols = ['LapNumber', 'PredPitWindowClass', 'PredNextPitGapLaps', 'DecisionMessage']
        available_cols = [c for c in strat_cols if c in filtered_df.columns]
        if available_cols:
            with st.expander("📋 Strategy predictions"):
                st.dataframe(filtered_df[available_cols], use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_lap_time_chart(filtered_df, selected_driver), use_container_width=True)
        
        # lap details table
        with st.expander("📋 View lap details"):
            display_cols = ['LapNumber', 'LapTime_sec', 'Compound', 'TyreLife', 'Stint']
            display_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(
                filtered_df[display_cols].style.format({'LapTime_sec': '{:.3f}'}),
                use_container_width=True
            )
    
    with tab3:
        if anomalies.empty:
            st.info("✅ No anomalies detected for this driver in this race!")
        else:
            st.warning(f"⚠️ Detected {len(anomalies)} anomalous laps")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # anomaly table
                anomaly_display = ['LapNumber', 'AnomalyScore', 'AnomalyType', 'LapTime_sec']
                anomaly_display = [c for c in anomaly_display if c in anomalies.columns]
                st.dataframe(
                    anomalies[anomaly_display].style.background_gradient(subset=['AnomalyScore'], cmap='Reds'),
                    use_container_width=True
                )
            
            with col2:
                # radar chart for most anomalous lap
                if not anomalies.empty:
                    worst_anomaly = anomalies.iloc[0]
                    st.markdown(f"**Worst anomaly: Lap {int(worst_anomaly['LapNumber'])}**")
                    st.plotly_chart(create_anomaly_radar(worst_anomaly), use_container_width=True)
    
    with tab4:
        st.plotly_chart(create_telemetry_chart(filtered_df), use_container_width=True)
        
        # telemetry details
        telem_cols = ['LapNumber', 'lap_mean_speed', 'lap_max_speed', 'lap_perc_throttle_full', 
                      'lap_perc_brake_on', 'lap_gear_changes']
        available_telem = [c for c in telem_cols if c in filtered_df.columns]
        if available_telem:
            with st.expander("📋 Telemetry data"):
                st.dataframe(filtered_df[available_telem], use_container_width=True)

    # footer
    st.markdown("""
        <hr style='border: 1px solid rgba(0,0,0,0.1); margin-top: 40px;'>
        <p style='text-align: center; color: rgba(0,0,0,0.5); font-size: 0.9em;'>
            f1 predictive analysis project — machine learning for pit stop strategy and anomaly detection
        </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
