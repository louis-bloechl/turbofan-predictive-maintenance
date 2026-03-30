import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import xgboost as xgb
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.features.temporal import build_features
from src.models.predict import load_inference_model, get_engine_analytics

st.set_page_config(page_title="TPMS: Predictive Maintenance", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]

@st.cache_resource
def load_model_and_data():
    model = load_inference_model()

    # Load raw data
    data_path = os.getenv("DATA_PATH", BASE_DIR / "data" / "raw" / "test_FD001.txt")
    df_raw = pd.read_csv(data_path, sep=r'\s+', header=None)
    columns = ['engine_id', 'time_cycle', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df_raw.columns = columns

    # Build features based on raw data
    df_features_history = build_features(df_raw)
    current_state_df = df_features_history.groupby('engine_id').last().reset_index()

    # Drop columns not needed for inference
    drop_cols = ['engine_id', 'time_cycle', 'rul']
    X_fleet = current_state_df.drop(columns=drop_cols, errors='ignore')
    current_state_df['predicted_rul'] = model.predict(X_fleet).astype(int)

    # Categorize engines based on estimated RUL
    def categorize_status(rul):
        if rul <= 25: return "SEVERE"
        elif rul <= 50: return "DEGRADED"
        return "HEALTHY"
    current_state_df['status'] = current_state_df['predicted_rul'].apply(categorize_status)

    return model, df_raw, df_features_history, current_state_df

try:
    model, df_raw, df_features_history, current_state_df = load_model_and_data()
except Exception as e:
    st.error(f"Failed to load model or telemetry data: {e}")
    st.stop()

# Sidebar
st.sidebar.title("TPMS")
st.sidebar.markdown("Turbofan Predictive Maintenance System")
page = st.sidebar.radio("Navigation", ["Fleet Overview", "Engine Health"])

st.sidebar.divider()

# Page 1: Fleet Overview
if page == "Fleet Overview":
    st.title("Fleet Health & Availability")
    st.markdown("Aggregated RUL predictions for active turbofan engines to optimize maintenance scheduling and fleet-wide uptime")

    total_engines = len(current_state_df)
    severe_count = len(current_state_df[current_state_df['status'] == 'SEVERE'])
    degraded_count = len(current_state_df[current_state_df['status'] == 'DEGRADED'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Engines", total_engines)
    col2.metric("Degraded Engines", degraded_count)
    col3.metric("Severe Engines", severe_count, delta_color="inverse")

    st.divider()

    st.subheader("RUL Distribution Across Fleet")
    fleet_hist = px.histogram(current_state_df, x="predicted_rul", nbins=30, 
                             color="status",
                             color_discrete_map={"HEALTHY": "mediumspringgreen", "DEGRADED": "orange", "SEVERE": "tomato"},
                             labels={'predicted_rul': 'Predicted RUL (Cycles)', 'status': 'Engine Status'})
    fleet_hist.add_vline(x=25, line_dash="dash", line_color="tomato", annotation_text="Critical Threshold", annotation_position="top left")
    fleet_hist.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Warning Threshold", annotation_position="top right")
    fleet_hist.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Number of Engines")
    st.plotly_chart(fleet_hist, use_container_width=True)

    st.divider()

    st.subheader("Active Fleet Roster")
    st.dataframe(current_state_df[['engine_id', 'time_cycle', 'predicted_rul', 'status']].sort_values('predicted_rul'), 
                 hide_index=True, use_container_width=True)

# Page 2: Engine Health
elif page == "Engine Health":
    st.sidebar.subheader("Unit Selection")
    selected_engine = st.sidebar.selectbox("Select Engine ID", current_state_df['engine_id'].unique())

    st.title(f"Analytics: Engine {selected_engine}")
    st.markdown("Deep-dive telemetry analysis, RUL trajectory modeling, and XAI feature contribution breakdown")

    engine_data = current_state_df[current_state_df['engine_id'] == selected_engine]
    current_cycle = int(engine_data['time_cycle'].values[0])
    predicted_rul = int(engine_data['predicted_rul'].values[0])
    status = engine_data['status'].values[0]

    # Current flight cycle and health status
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Current Flight Cycle", value=current_cycle)
        if status == "SEVERE": st.error("Status: SEVERE")
        elif status == "DEGRADED": st.warning("Status: DEGRADED")
        else: st.success("Status: HEALTHY")

    # Predicted RUL against gauge
    with col2:
        engine_rul_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = predicted_rul, domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Remaining Useful Life (Cycles)", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'steps': [{'range': [0, 25], 'color': "tomato"}, {'range': [25, 50], 'color': "orange"}, {'range': [50, 150], 'color': "mediumspringgreen"}],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': predicted_rul}
            }))
        engine_rul_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=70, b=10))
        st.plotly_chart(engine_rul_gauge, use_container_width=True)

    st.divider()

    # RUL projections by flight cycle graph
    st.subheader("RUL Projections")

    engine_history = df_features_history[df_features_history['engine_id'] == selected_engine].copy()
    X_history = engine_history.drop(columns=['engine_id', 'time_cycle', 'rul', 'status'], errors='ignore')
    engine_history['historical_prediction'] = model.predict(X_history)

    # Shade and add line to graph based on health thresholds
    projection_graph = go.Figure()
    projection_graph.add_hrect(y0=50, y1=150, fillcolor="mediumspringgreen", opacity=0.15, layer="below", line_width=0)
    projection_graph.add_hrect(y0=25, y1=50, fillcolor="orange", opacity=0.15, layer="below", line_width=0)
    projection_graph.add_hrect(y0=-10, y1=25, fillcolor="tomato", opacity=0.15, layer="below", line_width=0)
    projection_graph.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="Warning Threshold", annotation_position="bottom left")
    projection_graph.add_hline(y=25, line_dash="dash", line_color="tomato", annotation_text="Critical Threshold", annotation_position="bottom left")

    projection_graph.add_trace(go.Scatter(
        x=engine_history['time_cycle'], 
        y=engine_history['historical_prediction'], 
        mode='lines', 
        name='RUL Projection', 
        line=dict(color='silver', width=3)
    ))

    projection_graph.update_layout(
        yaxis_title="Predicted Flight Cycles Remaining", xaxis_title="Flight Cycle", 
        hovermode="x unified", height=350, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(projection_graph, use_container_width=True)

    st.divider()

    # Execute feature attribution logic
    predicted_rul, df_contributions, contributions = get_engine_analytics(engine_data, model)

    # Telemetry plots
    st.subheader("Telemetry and Signal Analytics")

    # Map sensor names to their real-world meaning
    sensor_mapping = {
        'sensor_1': 'Sensor 1: Total Temp at Fan Inlet', 'sensor_2': 'Sensor 2: Total Temp at LPC Outlet',
        'sensor_3': 'Sensor 3: Total Temp at HPC Outlet', 'sensor_4': 'Sensor 4: Total Temp at LPT Outlet',
        'sensor_5': 'Sensor 5: Pressure at Fan Inlet', 'sensor_6': 'Sensor 6: Total Pressure in Bypass Duct',
        'sensor_7': 'Sensor 7: Total Pressure at HPC Outlet', 'sensor_8': 'Sensor 8: Physical Fan Speed',
        'sensor_9': 'Sensor 9: Physical Core Speed', 'sensor_10': 'Sensor 10: Engine Pressure Ratio',
        'sensor_11': 'Sensor 11: Static Pressure at HPC Outlet', 'sensor_12': 'Sensor 12: Ratio of Fuel Flow to Ps30',
        'sensor_13': 'Sensor 13: Corrected Fan Speed', 'sensor_14': 'Sensor 14: Corrected Core Speed',
        'sensor_15': 'Sensor 15: Bypass Ratio', 'sensor_16': 'Sensor 16: Burner Fuel-Air Ratio',
        'sensor_17': 'Sensor 17: Bleed Enthalpy', 'sensor_18': 'Sensor 18: Demanded Fan Speed',
        'sensor_19': 'Sensor 19: Demanded Corrected Fan Speed', 'sensor_20': 'Sensor 20: HPT Coolant Bleed',
        'sensor_21': 'Sensor 21: LPT Coolant Bleed'
    }

    # Format titles for drop down
    def format_feature_title(feature_col):
        base_sensor = feature_col.split('_roll')[0]
        sensor_name = sensor_mapping.get(base_sensor, base_sensor.capitalize())
        if "_roll_" in feature_col:
            _, stat_info = feature_col.split("_roll_")
            stat_parts = stat_info.split("_")
            stat = "Mean" if stat_parts[0] == "mean" else "Std Dev"
            window = stat_parts[1]
            return f"{sensor_name} ({window}-Cycle {stat})"
        return f"{sensor_name} (Raw Telemetry)"

    all_features_ordered = []
    for i in range(1, 22):
        base = f'sensor_{i}'
        all_features_ordered.extend([base, f'{base}_roll_mean_10', f'{base}_roll_std_10', f'{base}_roll_mean_20', f'{base}_roll_std_20'])

    chart_col1, chart_col2 = st.columns(2)

    # Extraction of primary degradation driver
    with chart_col1:
        top_feature_name = df_contributions.iloc[0]['feature']
        base_sensor = top_feature_name.split('_roll')[0]

        st.markdown("**Primary Degradation Indicator**")
        indicator_fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Plot raw telemetry vs top engineered feature
        indicator_fig.add_trace(go.Scatter(x=engine_history['time_cycle'], y=engine_history[base_sensor], mode='lines', name='Raw Sensor', line=dict(color='lightblue', width=1)), secondary_y=False)

        if top_feature_name != base_sensor:
            indicator_fig.add_trace(go.Scatter(
                x=engine_history['time_cycle'],
                y=engine_history[top_feature_name],
                mode='lines',
                name='Model Feature',
                line=dict(color='mediumspringgreen',
                width=3)
            ), secondary_y=True)
            indicator_fig.update_yaxes(title_text="Engineered Value", secondary_y=True, showgrid=False)

        indicator_fig.update_layout(
            title=format_feature_title(top_feature_name),
            hovermode="x unified",
            legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        indicator_fig.update_yaxes(title_text="Raw Value", secondary_y=False)
        st.plotly_chart(indicator_fig, use_container_width=True)

    # Interactive telemetry inspection
    with chart_col2:
        st.markdown("**Feature Inspector**")
        valid_features = [f for f in all_features_ordered if f in engine_history.columns]
        selected_feature = st.selectbox(
            "Select Feature for Inspection:",
            options=valid_features,
            format_func=format_feature_title
        )

        explore_base_sensor = selected_feature.split('_roll')[0]
        inspection_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Render selected sensor stream with secondary-axis feature overlay
        inspection_fig.add_trace(go.Scatter(x=engine_history['time_cycle'], y=engine_history[explore_base_sensor], mode='lines', name='Raw Sensor', line=dict(color='lightblue', width=1)), secondary_y=False)

        if selected_feature != explore_base_sensor:
            inspection_fig.add_trace(go.Scatter(x=engine_history['time_cycle'], y=engine_history[selected_feature], mode='lines', name='Model Feature', line=dict(color='cyan', width=3)), secondary_y=True)
            inspection_fig.update_yaxes(title_text="Engineered Value", secondary_y=True, showgrid=False)

        inspection_fig.update_layout(title=format_feature_title(selected_feature), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=0, r=0, t=40, b=0))
        inspection_fig.update_yaxes(title_text="Raw Value", secondary_y=False)
        st.plotly_chart(inspection_fig, use_container_width=True)

    st.divider()

    st.subheader("Explainable AI (XAI): Primary Prediction Drivers")

    # Pull contributions of top 5 features and sum the rest
    top_n = 5
    df_top = df_contributions.head(top_n)
    other_contributions = df_contributions.iloc[top_n:]['contribution'].sum()

    # Define bar behavior - baseline, top 5 contributions, and final prediction
    measure = ['absolute'] + ['relative'] * top_n + ['relative', 'total']
    x_labels = ['Fleet Baseline'] + [format_feature_title(f) for f in df_top['feature']] + ['All Other Sensors', 'Final Prediction']
    y_values = [contributions[-1]] + df_top['contribution'].tolist() + [other_contributions, predicted_rul]

    # Decompose RUL prediction to show how local sensor deviations pull the unit away from fleet-wide baseline
    contributions_waterfall = go.Figure(go.Waterfall(
        orientation="v", measure=measure, x=x_labels, textposition="outside",
        # Format strings to show polarity for deltas (+/-) and absolute for endpoints
        text=[f"{v:+.1f}" if i != 0 and i != len(y_values)-1 else f"{v:.1f}" for i, v in enumerate(y_values)],
        y=y_values, connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "tomato"}}, increasing={"marker": {"color": "mediumspringgreen"}}, totals={"marker": {"color": "silver"}}      
    ))
    contributions_waterfall.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), waterfallgap=0.3, showlegend=False)
    st.plotly_chart(contributions_waterfall, use_container_width=True)

    # Raw data display
    st.write(" ") 
    expander = st.expander("Raw Telemetry Log (Last 5 Cycles)")
    with expander:
        # Filter raw telemetry for the specific unit and display last 5 cycles
        raw_history = df_raw[df_raw['engine_id'] == selected_engine]
        st.dataframe(raw_history.tail(5), use_container_width=True)