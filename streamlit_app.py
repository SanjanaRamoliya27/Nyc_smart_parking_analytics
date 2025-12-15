# =====================================================
# SMART PARKING ANALYTICS - STREAMLIT APP (FINAL)
# =====================================================

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from haversine import haversine, Unit
import folium
from streamlit_folium import st_folium

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
BASE_DIR = Path("F:/Projects/NYC_TAXI")

DATA_PATH = BASE_DIR / "parking_preprocessed_final_streamlit.csv"
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
PREPROCESS_PATH = BASE_DIR / "ml_preprocess_pipeline.pkl"
LABEL_PATH = BASE_DIR / "label_encoder.pkl"

# -----------------------------------------------------
# Load Data & Models
# -----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    return (
        pickle.load(open(PREPROCESS_PATH, "rb")),
        pickle.load(open(MODEL_PATH, "rb")),
        pickle.load(open(LABEL_PATH, "rb"))
    )

df = load_data()
preprocess, model, le = load_models()

st.title("🚗 Smart Parking Analytics Dashboard")

# -----------------------------------------------------
# Select Parking Meter
# -----------------------------------------------------
meter = st.sidebar.selectbox(
    "Select Parking Meter ID",
    sorted(df["Parking_Meter_ID"].unique())
)

row = df[df["Parking_Meter_ID"] == meter].iloc[0]

# -----------------------------------------------------
# Meter Details
# -----------------------------------------------------
st.subheader("📌 Parking Meter Details")
st.json(row.to_dict())

# -----------------------------------------------------
# Nearest Meters
# -----------------------------------------------------
def nearest(df, lat, lon, exclude_id, k=5):
    df2 = df.copy()
    df2["Distance_km"] = df2.apply(
        lambda r: haversine(
            (lat, lon),
            (r["Latitude"], r["Longitude"]),
            unit=Unit.KILOMETERS
        ),
        axis=1
    )
    return df2[df2["Parking_Meter_ID"] != exclude_id].nsmallest(k, "Distance_km")

recs = nearest(
    df,
    row["Latitude"],
    row["Longitude"],
    meter
)

st.subheader("📍 Nearest Parking Meters")
st.dataframe(recs[["Parking_Meter_ID", "Distance_km"]])

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
st.subheader("⚙️ Predicted Meter Status")

features = [
    "Max_Parking_Hours",
    "Rule_Count",
    "Enforcement_Duration_Hours",
    "Start_Time_Hours",
    "End_Time_Hours",
    "Latitude",
    "Longitude",
    "DBSCAN_Cluster",
    "KMeans_Cluster"
]

X_input = pd.DataFrame([row[features]])
X_processed = preprocess.transform(X_input)

pred = model.predict(X_processed)[0]
status = le.inverse_transform([pred])[0]

st.success(f"✅ Predicted Meter Status: **{status}**")

# -----------------------------------------------------
# Map
# -----------------------------------------------------
m = folium.Map(
    location=[row["Latitude"], row["Longitude"]],
    zoom_start=15
)

folium.Marker(
    [row["Latitude"], row["Longitude"]],
    popup=f"Meter {meter}",
    icon=folium.Icon(color="red")
).add_to(m)

for _, r in recs.iterrows():
    folium.CircleMarker(
        [r["Latitude"], r["Longitude"]],
        radius=6,
        popup=f"{r['Parking_Meter_ID']} ({r['Distance_km']:.2f} km)",
        color="blue",
        fill=True
    ).add_to(m)

st_folium(m, width=700, height=500)
cluster_features = [
    "Max_Parking_Hours",
    "Rule_Count",
    "Enforcement_Duration_Hours",
    "Start_Time_Hours",
    "End_Time_Hours",
    "Latitude",
    "Longitude"
]