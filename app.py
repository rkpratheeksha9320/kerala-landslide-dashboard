import json
import os
from pathlib import Path

import ee
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "kerala_landslide_dashboard_data.csv"

# Streamlit page configuration must be set before any other Streamlit commands.
st.set_page_config(
    page_title="Kerala Landslide Early Warning System",
    page_icon="🌧️",
    layout="wide"
)

# =========================================================
# LIVE IMERG UPDATE
# =========================================================

def update_live_imerg_risk():
    """Fetch latest IMERG rainfall and update dashboard risk."""

    # Render: use the Earth Engine service-account secret file
    ee_key_path = "/etc/secrets/earthengine-key.json"

    if os.path.exists(ee_key_path):
        with open(ee_key_path, "r", encoding="utf-8") as f:
            ee_key = json.load(f)

        credentials = ee.ServiceAccountCredentials(
            ee_key["client_email"],
            key_file=ee_key_path
        )

        ee.Initialize(
            credentials=credentials,
            project="alien-span-510505-a4"
        )

    else:
        # Local/Colab fallback
        ee.Initialize(project="alien-span-510505-a4")

    # Load current dashboard data from the same folder as app.py
    current_df = pd.read_csv(DATA_FILE)

    # Latest IMERG observation
    imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")

    latest = imerg.sort("system:time_start", False).first()
    latest_date = ee.Date(latest.get("system:time_start"))

    end_date = latest_date.advance(30, "minute")

    # Rainfall accumulation windows
    rainfall_1day = (
        imerg
        .filterDate(end_date.advance(-1, "day"), end_date)
        .select("precipitation")
        .sum()
        .rename("Rainfall_1Day")
    )

    rainfall_3day = (
        imerg
        .filterDate(end_date.advance(-3, "day"), end_date)
        .select("precipitation")
        .sum()
        .rename("Rainfall_3Day")
    )

    rainfall_5day = (
        imerg
        .filterDate(end_date.advance(-5, "day"), end_date)
        .select("precipitation")
        .sum()
        .rename("Rainfall_5Day")
    )

    rainfall_7day = (
        imerg
        .filterDate(end_date.advance(-7, "day"), end_date)
        .select("precipitation")
        .sum()
        .rename("Rainfall_7Day")
    )

    dynamic_rainfall = (
        rainfall_1day
        .addBands(rainfall_3day)
        .addBands(rainfall_5day)
        .addBands(rainfall_7day)
    )

    # Create EE points from dashboard locations
    features = []

    for _, row in current_df.iterrows():
        features.append(
            ee.Feature(
                ee.Geometry.Point([
                    float(row["Longitude"]),
                    float(row["Latitude"])
                ])
            )
        )

    grid_ee = ee.FeatureCollection(features)

    # Sample IMERG
    samples = dynamic_rainfall.sampleRegions(
        collection=grid_ee,
        scale=10000,
        geometries=True,
        tileScale=4
    )

    features_data = samples.getInfo()["features"]

    rainfall_rows = []

    for feature in features_data:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]

        rainfall_rows.append({
            "Longitude": coords[0],
            "Latitude": coords[1],
            "Rainfall_1Day": props.get("Rainfall_1Day"),
            "Rainfall_3Day": props.get("Rainfall_3Day"),
            "Rainfall_5Day": props.get("Rainfall_5Day"),
            "Rainfall_7Day": props.get("Rainfall_7Day")
        })

    rainfall_df = pd.DataFrame(rainfall_rows)

    # Match nearest IMERG point
    from sklearn.neighbors import NearestNeighbors

    dashboard_coords = current_df[
        ["Latitude", "Longitude"]
    ].to_numpy()

    rainfall_coords = rainfall_df[
        ["Latitude", "Longitude"]
    ].to_numpy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(rainfall_coords)

    _, indices = nn.kneighbors(dashboard_coords)

    matched = rainfall_df.iloc[
        indices[:, 0]
    ].reset_index(drop=True)

    updated_df = current_df.reset_index(drop=True).copy()

    for col in [
        "Rainfall_1Day",
        "Rainfall_3Day",
        "Rainfall_5Day",
        "Rainfall_7Day"
    ]:
        updated_df[col] = matched[col].values

    # Rainfall scores
    updated_df["Rainfall_1Day_Score"] = (
        updated_df["Rainfall_1Day"] / 100
    ).clip(0, 1)

    updated_df["Rainfall_3Day_Score"] = (
        updated_df["Rainfall_3Day"] / 150
    ).clip(0, 1)

    updated_df["Rainfall_5Day_Score"] = (
        updated_df["Rainfall_5Day"] / 300
    ).clip(0, 1)

    updated_df["Rainfall_7Day_Score"] = (
        updated_df["Rainfall_7Day"] / 400
    ).clip(0, 1)

    rainfall_trigger = (
        0.25 * updated_df["Rainfall_1Day_Score"]
        + 0.20 * updated_df["Rainfall_3Day_Score"]
        + 0.30 * updated_df["Rainfall_5Day_Score"]
        + 0.25 * updated_df["Rainfall_7Day_Score"]
    )

    # Soil moisture
    updated_df["Soil_Moisture_Score"] = (
        updated_df["Soil_Moisture"] / 0.30
    ).clip(0, 1)

    # Dynamic trigger
    updated_df["Dynamic_Trigger_Score"] = (
        0.80 * rainfall_trigger
        + 0.20 * updated_df["Soil_Moisture_Score"]
    )

    # Dynamic risk
    updated_df["Dynamic_Risk_Score"] = (
        0.75 * updated_df["Dynamic_Trigger_Score"]
        + 0.25 * updated_df["Landslide_Probability"]
    )

    # Warning level
    updated_df["Warning_Level"] = pd.cut(
        updated_df["Dynamic_Risk_Score"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Critical"
        ],
        include_lowest=True,
        right=False
    ).astype(str)

    # Actual latest IMERG timestamp
    timestamp = latest_date.format(
        "YYYY-MM-dd HH:mm:ss"
    ).getInfo()

    updated_df["IMERG_Last_Updated_UTC"] = timestamp

    # Save updated dashboard data beside app.py
    updated_df.to_csv(DATA_FILE, index=False)

    return timestamp, updated_df


# =========================================================
# LIVE IMERG UPDATE BUTTON
# =========================================================

st.markdown("### 🔄 Live Risk Update")

st.caption(
    "Fetch the latest available NASA IMERG rainfall data "
    "and recalculate the model-based landslide risk."
)

if st.button(
    "🔄 Update Latest IMERG & Risk",
    key="live_imerg_update"
):

    with st.spinner(
        "Fetching latest IMERG data and recalculating risk..."
    ):

        try:
            timestamp, updated_data = update_live_imerg_risk()

            st.success(
                f"✅ Risk updated successfully using IMERG data "
                f"from **{timestamp} UTC**."
            )

            st.rerun()

        except Exception as e:

            st.error(
                "❌ Unable to update the latest IMERG data."
            )

            st.exception(e)


# =========================================================
# DATA SOURCE TIMESTAMPS
# =========================================================

st.markdown("### 🕒 Data Source Information")

col1, col2 = st.columns(2)

with col1:
    st.info("🌧️ **IMERG Rainfall Data**\n\n"
            "Last updated: **6 Oct 2026, 03:30 UTC**")

with col2:
    st.info("🌡️ **ERA5-Land Weather Data**\n\n"
            "Last updated: **1 Oct 2026, 23:00 UTC**")

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    margin-bottom: 15px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 10px;
}

.warning-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #fff3cd;
    border: 1px solid #ffe69c;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.error(
        "Dashboard data file not found. "
        "Please make sure kerala_landslide_dashboard_data.csv "
        "is in the same folder as app.py."
    )
    st.stop()

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🌧️ Kerala Landslide Early Warning System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based Landslide Risk Monitoring and Dynamic Early Warning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# SYSTEM OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📊 System Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

total_locations = len(df)

high_risk_count = len(
    df[df["Warning_Level"].isin(["High", "Critical"])]
)

very_high_count = len(
    df[df["Warning_Level"] == "Critical"]
)

maximum_risk = df["Dynamic_Risk_Score"].max()

with col1:
    st.metric(
        "📍 Locations Monitored",
        f"{total_locations:,}"
    )

with col2:
    st.metric(
        "⚠️ High Risk Locations",
        f"{high_risk_count:,}"
    )

with col3:
    st.metric(
        "🚨 Critical Risk",
        f"{very_high_count:,}"
    )

with col4:
    st.metric(
        "🔴 Maximum Risk",
        f"{maximum_risk:.3f}"
    )

# =========================================================
# WARNING DISTRIBUTION
# =========================================================

st.markdown(
    '<div class="section-title">🚦 Warning Level Distribution</div>',
    unsafe_allow_html=True
)

warning_counts = (
    df["Warning_Level"]
    .value_counts()
    .reindex(
        ["Low", "Moderate", "High", "Critical"],
        fill_value=0
    )
)

st.bar_chart(warning_counts)

# =========================================================
# LOCATION SELECTION
# =========================================================

st.markdown(
    '<div class="section-title">📍 Check Current Risk</div>',
    unsafe_allow_html=True
)

st.write(
    "Search and select a Kerala district to view its nearest "
    "predicted landslide-risk location."
)

kerala_locations = {
    "Alappuzha": (9.4981, 76.3388),
    "Ernakulam": (9.9816, 76.2999),
    "Idukki": (9.9189, 77.1025),
    "Kannur": (11.8745, 75.3704),
    "Kasaragod": (12.5102, 74.9852),
    "Kollam": (8.8932, 76.6141),
    "Kottayam": (9.5916, 76.5222),
    "Kozhikode": (11.2588, 75.7804),
    "Malappuram": (11.0510, 76.0711),
    "Palakkad": (10.7867, 76.6548),
    "Pathanamthitta": (9.2648, 76.7870),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Thrissur": (10.5276, 76.2144),
    "Wayanad": (11.6854, 76.1320)
}

selected_location = st.selectbox(
    "🔎 Search your location",
    options=list(kerala_locations.keys())
)

selected_lat, selected_lon = kerala_locations[selected_location]

st.write(
    f"📍 **{selected_location}** "
    f"({selected_lat:.4f}, {selected_lon:.4f})"
)

# =========================================================
# FIND NEAREST MODELLED LOCATION
# =========================================================

distances = (
    (df["Latitude"] - selected_lat) ** 2
    +
    (df["Longitude"] - selected_lon) ** 2
)

nearest_index = distances.idxmin()

location_result = df.loc[nearest_index]

# =========================================================
# SELECTED LOCATION RESULTS
# =========================================================

st.markdown(
    f'<div class="section-title">'
    f'📍 Risk Information — {selected_location}'
    f'</div>',
    unsafe_allow_html=True
)

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

with risk_col1:
    st.metric(
        "Landslide Probability",
        f"{location_result['Landslide_Probability']:.3f}"
    )

with risk_col2:
    st.metric(
        "1-Day Rainfall",
        f"{location_result['Rainfall_1Day']:.1f} mm"
    )

with risk_col3:
    st.metric(
        "7-Day Rainfall",
        f"{location_result['Rainfall_7Day']:.1f} mm"
    )

with risk_col4:
    st.metric(
        "Dynamic Risk",
        f"{location_result['Dynamic_Risk_Score']:.3f}"
    )

warning_level = location_result["Warning_Level"]

if warning_level == "Critical":
    st.error(
        f"🚨 VERY HIGH RISK — {selected_location}"
    )

elif warning_level == "High":
    st.warning(
        f"⚠️ HIGH RISK — {selected_location}"
    )

elif warning_level == "Moderate":
    st.info(
        f"🟡 MODERATE RISK — {selected_location}"
    )

else:
    st.success(
        f"🟢 LOW RISK — {selected_location}"
    )


# =========================================================
# MODEL ALERT POPUP
# =========================================================

if warning_level == "Critical":
    st.markdown(
        f"""
        <div style="
            background-color:#ffebee;
            border:3px solid #d32f2f;
            border-radius:15px;
            padding:25px;
            margin:20px 0;
            text-align:center;
            box-shadow:0 4px 12px rgba(0,0,0,0.2);
        ">
            <h2 style="color:#b71c1c;">🚨 MODEL ALERT — CRITICAL LANDSLIDE RISK</h2>
            <h3>{selected_location}</h3>
            <p style="font-size:18px;">
                The model indicates a <b>critical landslide risk</b>
                for the selected location.
            </p>
            <p style="font-size:16px;">
                <b>Dynamic Risk Score:</b>
                {location_result["Dynamic_Risk_Score"]:.3f}
            </p>
            <p style="color:#555;">
                ⚠️ This is a model-based research alert and
                not an official government warning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif warning_level == "High":
    st.markdown(
        f"""
        <div style="
            background-color:#fff3e0;
            border:3px solid #f57c00;
            border-radius:15px;
            padding:25px;
            margin:20px 0;
            text-align:center;
            box-shadow:0 4px 12px rgba(0,0,0,0.15);
        ">
            <h2 style="color:#e65100;">🚨 MODEL ALERT — HIGH LANDSLIDE RISK</h2>
            <h3>{selected_location}</h3>
            <p style="font-size:18px;">
                The model indicates an <b>elevated landslide risk</b>
                for the selected location.
            </p>
            <p style="font-size:16px;">
                <b>Dynamic Risk Score:</b>
                {location_result["Dynamic_Risk_Score"]:.3f}
            </p>
            <p style="color:#555;">
                ⚠️ This is a model-based research alert and
                not an official government warning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )




# =========================================================
# TEST MODEL ALERT POPUPS
# =========================================================

if "test_alert" not in st.session_state:
    st.session_state.test_alert = None

st.markdown("### 🧪 Test Alert Popup")

test_col1, test_col2 = st.columns(2)

with test_col1:
    if st.button("🟠 Test High Risk Popup", key="test_high_popup"):
        st.session_state.test_alert = "High"

with test_col2:
    if st.button("🔴 Test Critical Risk Popup", key="test_critical_popup"):
        st.session_state.test_alert = "Critical"


# ---------------------------------------------------------
# HIGH RISK TEST ALERT
# ---------------------------------------------------------

if st.session_state.test_alert == "High":

    st.warning("🚨 MODEL ALERT — HIGH LANDSLIDE RISK")

    st.markdown(
        f"""
        ### 📍 TEST LOCATION

        **The model indicates an elevated landslide risk for this location.**

        **Dynamic Risk Score:** `0.68`

        ⚠️ **This is a model-based research alert and not an official government warning.**
        """
    )

    if st.button("✖️ Close Alert", key="close_high_popup"):
        st.session_state.test_alert = None
        st.rerun()


# ---------------------------------------------------------
# CRITICAL RISK TEST ALERT
# ---------------------------------------------------------

elif st.session_state.test_alert == "Critical":

    st.error("🚨 MODEL ALERT — CRITICAL LANDSLIDE RISK")

    st.markdown(
        f"""
        ### 📍 TEST LOCATION

        **The model indicates a critical landslide risk for this location.**

        **Dynamic Risk Score:** `0.89`

        ⚠️ **This is a model-based research alert and not an official government warning.**
        """
    )

    if st.button("✖️ Close Alert", key="close_critical_popup"):
        st.session_state.test_alert = None
        st.rerun()

# =========================================================
# RAINFALL DETAILS
# =========================================================

st.markdown(
    '<div class="section-title">🌧️ Recent Rainfall</div>',
    unsafe_allow_html=True
)

rainfall_data = pd.DataFrame({
    "Period": [
        "1 Day",
        "3 Days",
        "7 Days"
    ],
    "Rainfall (mm)": [
        location_result["Rainfall_1Day"],
        location_result["Rainfall_3Day"],
        location_result["Rainfall_7Day"]
    ]
})

st.bar_chart(
    rainfall_data.set_index("Period")
)

# =========================================================
# DYNAMIC RISK MAP
# =========================================================

st.markdown(
    '<div class="section-title">🗺️ Dynamic Landslide Risk Map</div>',
    unsafe_allow_html=True
)

risk_map = folium.Map(
    location=[10.5, 76.2],
    zoom_start=7,
    tiles=None
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/World_Street_Map/"
        "MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
    name="Esri World Street Map",
    overlay=False,
    control=True
).add_to(risk_map)

# Risk colours
risk_colors = {
    "Low": "green",
    "Moderate": "blue",
    "High": "orange",
    "Critical": "red"
}

# Add risk points
for _, row in df.iterrows():

    level = row["Warning_Level"]

    color = risk_colors.get(
        level,
        "gray"
    )

    popup_text = f"""
    <b>Warning:</b> {level}<br>
    <b>Dynamic Risk:</b> {row['Dynamic_Risk_Score']:.3f}<br>
    <b>Landslide Probability:</b> {row['Landslide_Probability']:.3f}<br>
    <b>1-Day Rainfall:</b> {row['Rainfall_1Day']:.1f} mm<br>
    <b>3-Day Rainfall:</b> {row['Rainfall_3Day']:.1f} mm<br>
    <b>7-Day Rainfall:</b> {row['Rainfall_7Day']:.1f} mm
    """

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            popup_text,
            max_width=300
        )
    ).add_to(risk_map)

# Selected location marker
folium.Marker(
    [selected_lat, selected_lon],
    popup=f"Selected Location: {selected_location}",
    tooltip=f"{selected_location}",
    icon=folium.Icon(
        color="black",
        icon="home"
    )
).add_to(risk_map)

st_folium(
    risk_map,
    width=None,
    height=650
)

# =========================================================
# HIGHEST RISK LOCATION
# =========================================================

st.markdown(
    '<div class="section-title">🚨 Highest-Risk Location</div>',
    unsafe_allow_html=True
)

highest_risk = df.loc[
    df["Dynamic_Risk_Score"].idxmax()
]

high_col1, high_col2, high_col3, high_col4 = st.columns(4)

with high_col1:
    st.write("**Latitude**")
    st.write(
        f"{highest_risk['Latitude']:.4f}"
    )

with high_col2:
    st.write("**Longitude**")
    st.write(
        f"{highest_risk['Longitude']:.4f}"
    )

with high_col3:
    st.write("**Warning Level**")
    st.write(
        highest_risk["Warning_Level"]
    )

with high_col4:
    st.write("**Dynamic Risk**")
    st.write(
        f"{highest_risk['Dynamic_Risk_Score']:.3f}"
    )

# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 About the System</div>',
    unsafe_allow_html=True
)

st.write(
    """
    This prototype combines machine-learning-based landslide
    susceptibility with recent rainfall conditions to estimate
    dynamic landslide risk.

    **Machine Learning Model:** XGBoost

    **Static factors:** Elevation, slope, aspect, curvature,
    rainfall, NDVI and land-use/land-cover.

    **Dynamic rainfall:** 1-day, 3-day and 7-day rainfall.

    **Warning levels:** Low, Moderate, High and Critical.

    The current warning thresholds are research/prototype
    thresholds and require further validation before operational
    emergency use.
    """
)

st.caption(
    "Kerala Landslide Early Warning System — Research Prototype"
)



# =========================================================
