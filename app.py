import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Soil Safety Prediction",
    page_icon="🏗️",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("xgboost_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# =========================
# TITLE
# =========================

st.title("🏗️ Soil Safety Prediction System")
st.markdown("### XGBoost Based Construction Risk Prediction")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("Input Parameters")

soil_type = st.sidebar.selectbox(
    "Soil Type",
    label_encoders['Soil_Type'].classes_
)

soil_moisture = st.sidebar.slider(
    "Soil Moisture (%)",
    0.0,
    100.0,
    25.0
)

shear_strength = st.sidebar.slider(
    "Shear Strength (kPa)",
    0.0,
    300.0,
    100.0
)

bearing_capacity = st.sidebar.slider(
    "Bearing Capacity (kPa)",
    0.0,
    1000.0,
    500.0
)

excavation_depth = st.sidebar.slider(
    "Excavation Depth (m)",
    0.0,
    50.0,
    10.0
)

retaining_wall = st.sidebar.selectbox(
    "Retaining Wall Type",
    label_encoders['Retaining_Wall_Type'].classes_
)

support_system = st.sidebar.selectbox(
    "Support System",
    label_encoders['Support_System'].classes_
)

deformation = st.sidebar.slider(
    "Deformation (mm)",
    0.0,
    100.0,
    20.0
)

rainfall = st.sidebar.slider(
    "Rainfall (mm/day)",
    0.0,
    300.0,
    50.0
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    -10.0,
    50.0,
    25.0
)

groundwater = st.sidebar.slider(
    "Groundwater Level (m)",
    0.0,
    20.0,
    5.0
)

seismic_activity = st.sidebar.selectbox(
    "Seismic Activity",
    [0, 1]
)

ground_settlement = st.sidebar.slider(
    "Ground Settlement (mm)",
    0.0,
    100.0,
    30.0
)

wall_displacement = st.sidebar.slider(
    "Wall Displacement (mm)",
    0.0,
    100.0,
    20.0
)

pore_water = st.sidebar.slider(
    "Pore Water Pressure (kPa)",
    0.0,
    500.0,
    100.0
)

strain_gauge = st.sidebar.slider(
    "Strain Gauge",
    0.0,
    100.0,
    40.0
)

# =========================
# ENCODE INPUTS
# =========================

soil_type_encoded = label_encoders['Soil_Type'].transform([soil_type])[0]

retaining_wall_encoded = label_encoders[
    'Retaining_Wall_Type'
].transform([retaining_wall])[0]

support_system_encoded = label_encoders[
    'Support_System'
].transform([support_system])[0]

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame({
    'Soil_Type': [soil_type_encoded],
    'Soil_Moisture_%': [soil_moisture],
    'Shear_Strength_kPa': [shear_strength],
    'Bearing_Capacity_kPa': [bearing_capacity],
    'Excavation_Depth_m': [excavation_depth],
    'Retaining_Wall_Type': [retaining_wall_encoded],
    'Support_System': [support_system_encoded],
    'Deformation_mm': [deformation],
    'Rainfall_mm_day': [rainfall],
    'Temperature_C': [temperature],
    'Groundwater_Level_m': [groundwater],
    'Seismic_Activity': [seismic_activity],
    'Ground_Settlement_mm': [ground_settlement],
    'Wall_Displacement_mm': [wall_displacement],
    'Pore_Water_Pressure_kPa': [pore_water],
    'Strain_Gauge': [strain_gauge]
})

# =========================
# PREDICTION
# =========================

if st.button("Predict Risk Level"):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    if prediction == 0:
        st.success("Prediction: LOW RISK")

    elif prediction == 1:
        st.warning("Prediction: MEDIUM RISK")

    else:
        st.error("Prediction: HIGH RISK")

    # =========================
    # PROBABILITY CHART
    # =========================

    prob_df = pd.DataFrame({
        'Risk Level': ['Low', 'Medium', 'High'],
        'Probability': probabilities
    })

    fig = px.bar(
        prob_df,
        x='Risk Level',
        y='Probability',
        title='Prediction Probability'
    )

    st.plotly_chart(fig, use_container_width=True)