import math
import struct
import io
import wave
from urllib.parse import quote
# ============================================================
# GEOSAFE
# Women Safety Intelligence & Geographical Analysis Platform
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GeoSafe | Women Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
# Support both the recommended data/ folder and a flat folder layout.
if not os.path.isdir(DATA_DIR):
    DATA_DIR = BASE_DIR

INCIDENTS_PATH = os.path.join(DATA_DIR, "geosafe_clustered_incidents.csv")
HOTSPOTS_PATH = os.path.join(DATA_DIR, "geosafe_hotspots.csv")
METRICS_PATH = os.path.join(DATA_DIR, "geosafe_model_metrics.csv")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        background-color: #f5f7fb;
        border-right: 1px solid #e6e9ef;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #5f6673;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        margin-top: 24px;
        margin-bottom: 10px;
    }

    .section-subtitle {
        color: #697386;
        font-size: 15px;
        margin-bottom: 18px;
    }

    .info-card {
        background: #f7f8fb;
        border: 1px solid #e5e8ef;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
    }

    .info-card-title {
        font-size: 14px;
        color: #697386;
        margin-bottom: 8px;
    }

    .info-card-value {
        font-size: 31px;
        font-weight: 750;
    }

    .info-card-caption {
        font-size: 13px;
        color: #7a8494;
        margin-top: 5px;
    }

    .highlight-box {
        background: #f1f7ff;
        border-left: 5px solid #3478f6;
        border-radius: 10px;
        padding: 18px 20px;
        margin: 18px 0;
    }

    .impact-box {
        background: #f7f5ff;
        border-left: 5px solid #7657d9;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }

    .model-card {
        background: #f7f8fb;
        border: 1px solid #e1e5ec;
        border-radius: 16px;
        padding: 22px;
        min-height: 180px;
        margin-bottom: 15px;
    }

    .model-name {
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .model-description {
        color: #697386;
        font-size: 14px;
        line-height: 1.6;
    }

    .accuracy-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 10px;
    }

    .feature-card {
        background: #ffffff;
        border: 1px solid #e5e8ef;
        border-radius: 14px;
        padding: 18px;
        min-height: 145px;
    }

    .feature-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #697386;
        font-size: 14px;
        line-height: 1.6;
    }

    .footer {
        text-align: center;
        color: #7b8492;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    incidents = pd.read_csv(INCIDENTS_PATH)
    hotspots = pd.read_csv(HOTSPOTS_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    return incidents, hotspots, metrics


# ============================================================
# LOAD DATA SAFELY
# ============================================================

try:

    incidents, hotspots, model_metrics = load_data()

except Exception as e:

    st.error("GeoSafe could not load the project data.")

    st.code(str(e))

    st.info(
        """
        Please make sure your project has this structure:

        GeoSafe/
        ├── app.py
        │
        └── data/
            ├── geosafe_clustered_incidents.csv
            ├── geosafe_hotspots.csv
            └── geosafe_model_metrics.csv
        """
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

# Convert numerical metric columns when available

for column in [
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1-Score (%)",
]:

    if column in model_metrics.columns:

        model_metrics[column] = pd.to_numeric(
            model_metrics[column],
            errors="coerce"
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_display_name(column):

    return (
        str(column)
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


def get_random_forest_accuracy():

    if "Model" not in model_metrics.columns:

        return None

    rows = model_metrics[
        model_metrics["Model"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "random forest"
    ]

    if rows.empty:

        return None

    if "Accuracy (%)" not in rows.columns:

        return None

    value = pd.to_numeric(
        rows["Accuracy (%)"].iloc[0],
        errors="coerce"
    )

    if pd.isna(value):

        return None

    return float(value)


def get_model_row(model_name):

    if "Model" not in model_metrics.columns:

        return None

    rows = model_metrics[
        model_metrics["Model"]
        .astype(str)
        .str.strip()
        .str.lower()
        == model_name.lower()
    ]

    if rows.empty:

        return None

    return rows.iloc[0]


def metric_card(title, value, caption):
    html = (
        '<div class="info-card">'
        f'<div class="info-card-title">{title}</div>'
        f'<div class="info-card-value">{value}</div>'
        f'<div class="info-card-caption">{caption}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:28px;
            font-weight:800;
            margin-bottom:3px;
        ">
        🛡️ GeoSafe
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Women Safety Intelligence Platform"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🤖 Risk Prediction",
            "🗺️ Hotspot Intelligence",
            "🧭 Smart Route Analysis",
            "🚨 Alert Center",
            "🆘 SOS Alarm",
            "📊 Model Analysis",
            "📖 Methodology",
        ],
    )

    st.markdown("---")

    st.markdown(
        """
        **GeoSafe integrates**

        • Machine learning risk classification

        • DBSCAN geographical clustering

        • Crime-intensity analysis

        • Interactive hotspot visualization
        """
    )

    st.markdown("---")

    st.caption(
        "Academic Project • Women Safety Analysis"
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="hero-title">🛡️ GeoSafe</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-subtitle">
        Women Safety Intelligence & Geographical Analysis Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="highlight-box">
        <b>GeoSafe</b> combines four machine learning classification
        approaches with geographical hotspot detection to study
        patterns in women-safety incidents.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # KEY METRICS
    # --------------------------------------------------------

    total_incidents = len(incidents)

    total_hotspots = len(hotspots)

    rf_accuracy = get_random_forest_accuracy()

    rf_text = (
        f"{rf_accuracy:.2f}%"
        if rf_accuracy is not None
        else "N/A"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "ANALYZED INCIDENTS",
            f"{total_incidents:,}",
            "Records in the analysis dataset",
        )

    with col2:
        metric_card(
            "GEOGRAPHICAL HOTSPOTS",
            f"{total_hotspots:,}",
            "DBSCAN-detected clusters",
        )

    with col3:
        metric_card(
            "CLASSIFICATION MODELS",
            "4",
            "DT • RF • NB • KNN",
        )

    with col4:
        metric_card(
            "RF TEST ACCURACY",
            rf_text,
            "Test-set accuracy",
        )

    st.markdown("")

    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    left, right = st.columns([1.35, 1])

    with left:

        st.markdown(
            '<div class="section-title">The GeoSafe Approach</div>',
            unsafe_allow_html=True,
        )

        st.write(
            """
            Women-safety analysis involves two complementary
            questions:

            **1. What level of risk is associated with an incident?**

            GeoSafe investigates this through four classification
            algorithms:

            • Decision Tree

            • Random Forest

            • Naive Bayes

            • KNN

            **2. Where are incidents geographically concentrated?**

            DBSCAN clustering is used to identify dense geographical
            groups of incidents and characterize them as hotspots.
            """
        )

    with right:

        st.markdown(
            '<div class="section-title">Project Components</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **Machine Learning**

            Risk-level classification using four algorithms.

            **Geographical Intelligence**

            DBSCAN-based detection of incident concentrations.

            **Hotspot Analysis**

            Incident count, crime count, safety score and
            High/Critical distribution.

            **Interactive Visualization**

            Maps and analytical dashboards for exploration.
            """
        )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Classification Model Performance</div>',
        unsafe_allow_html=True,
    )

    if (
        "Model" in model_metrics.columns
        and "Accuracy (%)" in model_metrics.columns
    ):

        fig = px.bar(
            model_metrics,
            x="Model",
            y="Accuracy (%)",
            text="Accuracy (%)",
            title="Test Accuracy of the Four Classification Models",
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        fig.update_layout(
            yaxis_title="Accuracy (%)",
            xaxis_title="",
            yaxis_range=[0, 100],
            height=450,
            margin=dict(
                t=60,
                l=20,
                r=20,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            width='stretch',
        )

    else:

        st.warning(
            "Accuracy data is not available in the metrics file."
        )

    st.markdown(
        """
        <div class="impact-box">
        <b>Social relevance:</b> GeoSafe is designed as an
        analytical system to help explore women-safety patterns
        and geographically concentrated incident areas.
        It does not replace official crime statistics, emergency
        services, or professional safety assessments.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RISK PREDICTION
# ============================================================

elif page == "🤖 Risk Prediction":

    st.title("🤖 Risk Classification")

    st.write(
        """
        GeoSafe evaluates four supervised machine learning
        algorithms for women-safety risk classification.
        The models were trained and evaluated as part of the
        machine learning analysis.
        """
    )

    st.markdown(
        """
        <div class="highlight-box">
        <h3>How the classification system works</h3>
        <p>
        The dataset is first preprocessed and prepared for
        machine learning. Four classification algorithms are then
        trained to classify the risk level associated with an
        incident.
        </p>
        <p>
        The evaluated models are:
        <b>Decision Tree • Random Forest • Naive Bayes • KNN</b>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # MODEL CARDS
    # --------------------------------------------------------

    model_descriptions = {

        "Decision Tree":
            "A tree-based supervised learning algorithm that "
            "makes decisions through a sequence of feature-based splits.",

        "Random Forest":
            "An ensemble of multiple decision trees. The combined "
            "predictions of the trees provide a more robust classification model.",

        "Naive Bayes":
            "A probabilistic classifier based on Bayes' theorem "
            "with the assumption of conditional independence between features.",

        "KNN":
            "A distance-based algorithm that classifies an observation "
            "using the labels of nearby training observations.",
    }

    st.markdown(
        '<div class="section-title">Evaluated Models</div>',
        unsafe_allow_html=True,
    )

    model_names = [
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
        "KNN",
    ]

    for start in range(0, len(model_names), 2):

        cols = st.columns(2)

        for index in range(2):

            model_index = start + index

            if model_index >= len(model_names):

                continue

            model_name = model_names[model_index]

            row = get_model_row(model_name)

            accuracy = None

            if row is not None:

                if "Accuracy (%)" in row.index:

                    accuracy = pd.to_numeric(
                        row["Accuracy (%)"],
                        errors="coerce",
                    )

            accuracy_text = (
                f"{float(accuracy):.2f}%"
                if accuracy is not None
                and not pd.isna(accuracy)
                else "N/A"
            )

            with cols[index]:

                model_html = (
                    '<div class="model-card">'
                    f'<div class="model-name">{model_name}</div>'
                    f'<div class="model-description">'
                    f'{model_descriptions[model_name]}'
                    f'</div>'
                    f'<div class="accuracy-value">{accuracy_text}</div>'
                    '<div class="info-card-caption">Test accuracy</div>'
                    '</div>'
                )

                st.markdown(
                    model_html,
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Classification Results</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        model_metrics,
        width='stretch',
        hide_index=True,
    )

    st.info(
        """
        These values represent the measured performance of the
        trained classification models on the project's evaluation
        data. They should be interpreted as model-performance
        metrics rather than guarantees about real-world safety.
        """
    )


# ============================================================
# HOTSPOT INTELLIGENCE
# ============================================================

elif page == "🗺️ Hotspot Intelligence":

    st.title("🗺️ Geographical Hotspot Intelligence")

    st.write(
        """
        DBSCAN identifies geographically dense groups of incidents.
        The heatmap represents incident intensity, while the hotspot
        markers represent detected cluster centres.
        """
    )

    # --------------------------------------------------------
    # CITY FILTER
    # --------------------------------------------------------

    city_values = []

    if "city" in incidents.columns:

        city_values = sorted(
            incidents["city"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    selected_city = st.selectbox(
        "Filter by City",
        ["All Cities"] + city_values,
    )

    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    if selected_city == "All Cities":

        map_incidents = incidents.copy()
        map_hotspots = hotspots.copy()

    else:

        map_incidents = incidents[
            incidents["city"].astype(str)
            == selected_city
        ].copy()

        map_hotspots = hotspots[
            hotspots["city"].astype(str)
            == selected_city
        ].copy()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Incidents",
            f"{len(map_incidents):,}",
        )

    with c2:

        st.metric(
            "Hotspots",
            f"{len(map_hotspots):,}",
        )

    if "crime_count" in map_incidents.columns:

        total_crimes = pd.to_numeric(
            map_incidents["crime_count"],
            errors="coerce",
        ).fillna(0).sum()

    else:

        total_crimes = 0

    with c3:

        st.metric(
            "Total Crime Count",
            f"{int(total_crimes):,}",
        )

    if (
        "high_critical_pct" in map_hotspots.columns
        and len(map_hotspots) > 0
    ):

        avg_high_critical = pd.to_numeric(
            map_hotspots["high_critical_pct"],
            errors="coerce",
        ).mean()

    else:

        avg_high_critical = 0

    if pd.isna(avg_high_critical):

        avg_high_critical = 0

    with c4:

        st.metric(
            "Avg High/Critical %",
            f"{avg_high_critical:.1f}%",
        )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Interactive Safety Map</div>',
        unsafe_allow_html=True,
    )

    required_columns = [
        "latitude",
        "longitude",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in map_incidents.columns
    ]

    if missing_columns:

        st.error(
            "Missing geographical columns: "
            + ", ".join(missing_columns)
        )

        st.stop()

    valid_points = map_incidents[
        ["latitude", "longitude"]
    ].copy()

    valid_points["latitude"] = pd.to_numeric(
        valid_points["latitude"],
        errors="coerce",
    )

    valid_points["longitude"] = pd.to_numeric(
        valid_points["longitude"],
        errors="coerce",
    )

    valid_points = valid_points.dropna()

    if valid_points.empty:

        st.warning(
            "No geographical records available for this filter."
        )

    else:

        center_lat = valid_points["latitude"].mean()
        center_lon = valid_points["longitude"].mean()

        m = folium.Map(
            location=[
                center_lat,
                center_lon,
            ],
            zoom_start=5,
            tiles=None,
            control_scale=True,
        )

        # ----------------------------------------------------
        # ESRI BASEMAP
        # ----------------------------------------------------

        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/"
                "ArcGIS/rest/services/World_Street_Map/"
                "MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Tiles © Esri",
            name="Street Map",
            overlay=False,
            control=True,
        ).add_to(m)

        # ----------------------------------------------------
        # HEATMAP
        # ----------------------------------------------------

        if "crime_count" in map_incidents.columns:

            heat_df = map_incidents[
                [
                    "latitude",
                    "longitude",
                    "crime_count",
                ]
            ].copy()

            heat_df["latitude"] = pd.to_numeric(
                heat_df["latitude"],
                errors="coerce",
            )

            heat_df["longitude"] = pd.to_numeric(
                heat_df["longitude"],
                errors="coerce",
            )

            heat_df["crime_count"] = pd.to_numeric(
                heat_df["crime_count"],
                errors="coerce",
            ).fillna(0)

            heat_df = heat_df.dropna(
                subset=[
                    "latitude",
                    "longitude",
                ]
            )

            heat_df["crime_count"] = (
                heat_df["crime_count"]
                .clip(lower=0)
            )

            # Keep map responsive
            if len(heat_df) > 12000:

                heat_df = heat_df.sample(
                    12000,
                    random_state=42,
                )

            heat_data = heat_df[
                [
                    "latitude",
                    "longitude",
                    "crime_count",
                ]
            ].values.tolist()

        else:

            heat_data = valid_points.values.tolist()

        if heat_data:

            HeatMap(
                heat_data,
                radius=15,
                blur=20,
                min_opacity=0.25,
                max_zoom=12,
                name="Crime Intensity",
            ).add_to(m)

        # ----------------------------------------------------
        # HOTSPOT MARKERS
        # ----------------------------------------------------

        hotspot_layer = folium.FeatureGroup(
            name="DBSCAN Hotspots",
            show=True,
        )

        for _, row in map_hotspots.iterrows():

            try:

                lat = float(
                    row["center_latitude"]
                )

                lon = float(
                    row["center_longitude"]
                )

            except Exception:

                continue

            popup_parts = []

            if "hotspot_rank" in row.index:

                try:

                    rank = int(
                        float(row["hotspot_rank"])
                    )

                    popup_parts.append(
                        f"<b>Hotspot #{rank}</b>"
                    )

                except Exception:

                    popup_parts.append(
                        "<b>DBSCAN Hotspot</b>"
                    )

            else:

                popup_parts.append(
                    "<b>DBSCAN Hotspot</b>"
                )

            if "city" in row.index:

                popup_parts.append(
                    f"<b>City:</b> {row['city']}"
                )

            if "area" in row.index:

                popup_parts.append(
                    f"<b>Area:</b> {row['area']}"
                )

            if "incident_count" in row.index:

                try:

                    popup_parts.append(
                        f"<b>Incidents:</b> "
                        f"{int(float(row['incident_count'])):,}"
                    )

                except Exception:

                    pass

            if "total_crime_count" in row.index:

                try:

                    popup_parts.append(
                        f"<b>Total crime count:</b> "
                        f"{int(float(row['total_crime_count'])):,}"
                    )

                except Exception:

                    pass

            if "average_safety_score" in row.index:

                try:

                    popup_parts.append(
                        f"<b>Average safety score:</b> "
                        f"{float(row['average_safety_score']):.2f}"
                    )

                except Exception:

                    pass

            if "high_critical_pct" in row.index:

                try:

                    popup_parts.append(
                        f"<b>High/Critical:</b> "
                        f"{float(row['high_critical_pct']):.2f}%"
                    )

                except Exception:

                    pass

            if "dominant_crime_type" in row.index:

                popup_parts.append(
                    f"<b>Dominant crime:</b> "
                    f"{row['dominant_crime_type']}"
                )

            popup_html = "<br>".join(
                popup_parts
            )

            tooltip_text = "DBSCAN Hotspot"

            if "hotspot_rank" in row.index:

                try:

                    tooltip_text = (
                        f"Hotspot #"
                        f"{int(float(row['hotspot_rank']))}"
                    )

                except Exception:

                    pass

            folium.CircleMarker(
                location=[
                    lat,
                    lon,
                ],
                radius=8,
                popup=folium.Popup(
                    popup_html,
                    max_width=350,
                ),
                tooltip=tooltip_text,
                fill=True,
                fill_opacity=0.85,
                weight=2,
            ).add_to(hotspot_layer)

        hotspot_layer.add_to(m)

        folium.LayerControl().add_to(m)

        st_folium(
            m,
            width='stretch',
            height=680,
            returned_objects=[],
        )

    # --------------------------------------------------------
    # HOTSPOT TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Hotspot Characteristics</div>',
        unsafe_allow_html=True,
    )

    table_columns = [
        "hotspot_rank",
        "city",
        "area",
        "incident_count",
        "total_crime_count",
        "average_safety_score",
        "high_critical_pct",
        "dominant_crime_type",
    ]

    available_columns = [
        col
        for col in table_columns
        if col in map_hotspots.columns
    ]

    if available_columns:

        st.dataframe(
            map_hotspots[
                available_columns
            ],
            width='stretch',
            hide_index=True,
        )

    else:

        st.info(
            "No hotspot summary columns are available."
        )



# ============================================================
# SMART ROUTE ANALYSIS
# ============================================================

elif page == "🧭 Smart Route Analysis":
    st.title("🧭 GeoSafe Route Intelligence")
    st.markdown(
        "Compare available driving alternatives using **historical incident exposure** and "
        "DBSCAN hotspot proximity. GeoSafe highlights the route with the lowest estimated "
        "historical-risk proxy—not a guaranteed safe route."
    )

    st.info(
        "Prototype mode: this version uses your current CSV files and public OpenStreetMap-based "
        "geocoding/routing services. It does not have live incident feeds. A route with fewer "
        "recorded incidents may simply have less data coverage."
    )

    import requests

    def geocode_location(query):
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "GeoSafeAcademicProject/1.0 (route analysis)"}
        response = requests.get(
            url,
            params={"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "in"},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return None
        return {
            "latitude": float(results[0]["lat"]),
            "longitude": float(results[0]["lon"]),
            "display_name": results[0].get("display_name", query),
        }

    def get_driving_routes(start, end):
        """Ask OSRM for alternatives; availability depends on the public routing service."""
        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{start['longitude']},{start['latitude']};"
            f"{end['longitude']},{end['latitude']}"
        )
        response = requests.get(
            url,
            params={"overview": "full", "geometries": "geojson", "steps": "false", "alternatives": "3"},
            headers={"User-Agent": "GeoSafeAcademicProject/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok" or not payload.get("routes"):
            return []
        routes = []
        for idx, item in enumerate(payload["routes"], start=1):
            coords = item["geometry"]["coordinates"]
            routes.append({
                "route_id": idx,
                "coordinates": [[float(lat), float(lon)] for lon, lat in coords],
                "distance_km": item["distance"] / 1000.0,
                "duration_min": item["duration"] / 60.0,
            })
        return routes

    def haversine_km(lat1, lon1, lat2, lon2):
        lat1 = np.radians(np.asarray(lat1, dtype=float))
        lon1 = np.radians(np.asarray(lon1, dtype=float))
        lat2 = np.radians(np.asarray(lat2, dtype=float))
        lon2 = np.radians(np.asarray(lon2, dtype=float))
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def resample_route(route, spacing_km=0.75):
        """Resample route geometry at approximately even distance intervals."""
        coords = np.asarray(route["coordinates"], dtype=float)
        if len(coords) < 2:
            return coords
        segment_lengths = haversine_km(
            coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1]
        )
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total = float(cumulative[-1])
        if total <= 0:
            return coords[:1]
        distances = np.arange(0.0, total, spacing_km)
        if len(distances) == 0 or distances[-1] < total:
            distances = np.append(distances, total)
        latitudes = np.interp(distances, cumulative, coords[:, 0])
        longitudes = np.interp(distances, cumulative, coords[:, 1])
        return np.column_stack((latitudes, longitudes))

    def distance_matrix_km(points, samples):
        """Pairwise haversine distances for [lat, lon] arrays."""
        p_lat = np.radians(points[:, 0])[:, None]
        p_lon = np.radians(points[:, 1])[:, None]
        s_lat = np.radians(samples[:, 0])[None, :]
        s_lon = np.radians(samples[:, 1])[None, :]
        dlat, dlon = s_lat - p_lat, s_lon - p_lon
        a = np.sin(dlat / 2.0) ** 2 + np.cos(p_lat) * np.cos(s_lat) * np.sin(dlon / 2.0) ** 2
        return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def records_near_route(route, source, radius_km):
        """Find unique records near a densely resampled route and estimate spatial coverage."""
        valid = source.copy()
        valid["latitude"] = pd.to_numeric(valid["latitude"], errors="coerce")
        valid["longitude"] = pd.to_numeric(valid["longitude"], errors="coerce")
        valid = valid.dropna(subset=["latitude", "longitude"])
        valid = valid[valid["latitude"].between(-90, 90) & valid["longitude"].between(-180, 180)]
        samples = resample_route(route, spacing_km=0.75)
        if valid.empty or len(samples) == 0:
            return valid.iloc[0:0].copy(), 0.0

        points = valid[["latitude", "longitude"]].to_numpy(dtype=float)
        min_distances = np.full(len(points), np.inf, dtype=float)
        # Chunk records to keep memory bounded on long cross-country routes.
        for offset in range(0, len(points), 400):
            stop = min(offset + 400, len(points))
            distances = distance_matrix_km(points[offset:stop], samples)
            min_distances[offset:stop] = distances.min(axis=1)
        nearby_mask = min_distances <= radius_km
        nearby = valid.loc[nearby_mask].copy()

        # Estimate how much of the sampled route has records nearby. This is a dataset
        # coverage indicator, not a measure of street safety or incident reporting quality.
        covered = np.zeros(len(samples), dtype=bool)
        nearby_points = points[nearby_mask]
        for offset in range(0, len(samples), 80):
            stop = min(offset + 80, len(samples))
            sample_chunk = samples[offset:stop]
            if len(nearby_points):
                dists = distance_matrix_km(nearby_points, sample_chunk)
                covered[offset:stop] = (dists <= radius_km).any(axis=0)
        coverage_pct = float(covered.mean() * 100.0) if len(covered) else 0.0
        return nearby, coverage_pct

    def score_route(route, records, hotspot_data):
        """Transparent comparative proxy; not a calibrated probability of danger."""
        risk = records.get("risk_level", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
        critical = int((risk == "critical").sum())
        high = int((risk == "high").sum())
        medium = int((risk == "medium").sum())
        low = int((risk == "low").sum())
        # Severity-weighted historical-record density per 10 km reduces the length bias.
        weighted = critical * 4 + high * 3 + medium * 2 + low
        per10km = weighted / max(route["distance_km"], 1.0) * 10

        hotspot_count = 0
        if hotspot_data is not None and not hotspot_data.empty and {"center_latitude", "center_longitude"}.issubset(hotspot_data.columns):
            hs = hotspot_data.copy()
            hs["center_latitude"] = pd.to_numeric(hs["center_latitude"], errors="coerce")
            hs["center_longitude"] = pd.to_numeric(hs["center_longitude"], errors="coerce")
            hs = hs.dropna(subset=["center_latitude", "center_longitude"])
            samples = resample_route(route, spacing_km=0.75)
            hlat = hs["center_latitude"].to_numpy(dtype=float)
            hlon = hs["center_longitude"].to_numpy(dtype=float)
            seen = np.zeros(len(hs), dtype=bool)
            for plat, plon in samples:
                seen |= haversine_km(hlat, hlon, plat, plon) <= 1.0
            hotspot_count = int(seen.sum())
        # Scoring formula is a prioritization aid only. Normalize incident severity by distance.
        score = per10km + hotspot_count * 2.0
        return {
            "records": records, "critical": critical, "high": high, "medium": medium,
            "low": low, "weighted_density": per10km, "hotspots": hotspot_count,
            "score": score, "total_records": len(records),
        }

    with st.form("geosafe_route_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            start_query = st.text_input("Starting point", placeholder="e.g., Kollam Railway Station, Kerala")
        with col_b:
            end_query = st.text_input("Destination", placeholder="e.g., Technopark, Thiruvananthapuram")
        radius_km = st.slider(
            "Incident proximity radius (km)", 0.25, 3.0, 1.0, 0.25,
            help="Historical records within this distance of sampled route points are counted.",
        )
        submitted = st.form_submit_button("Compare routes with GeoSafe", type="primary")

    if submitted:
        if not start_query.strip() or not end_query.strip():
            st.warning("Enter both a starting point and a destination.")
        else:
            try:
                with st.spinner("Finding locations and comparing available route alternatives..."):
                    origin = geocode_location(start_query.strip())
                    destination = geocode_location(end_query.strip())
                    if origin is None:
                        st.error("Could not find the starting point. Try a more specific place name.")
                    elif destination is None:
                        st.error("Could not find the destination. Try a more specific place name.")
                    else:
                        routes = get_driving_routes(origin, destination)
                        if not routes:
                            st.error("No driving route was returned for these locations.")
                        else:
                            if len(routes) == 1:
                                st.warning(
                                    "The routing service returned only one driving route for these locations. "
                                    "GeoSafe can analyze this route, but cannot claim it is the safest alternative. "
                                    "Try nearby endpoints or a different route service if you need a comparison."
                                )
                            hotspot_source = hotspots if "hotspots" in globals() and isinstance(hotspots, pd.DataFrame) else pd.DataFrame()
                            comparisons = []
                            for route in routes:
                                nearby, coverage_pct = records_near_route(route, incidents, radius_km)
                                comparison = score_route(route, nearby, hotspot_source)
                                comparison["coverage_pct"] = coverage_pct
                                route.update(comparison)
                                comparisons.append(route)
                            # Stable tie-break: lower score, then shorter duration.
                            comparisons.sort(key=lambda x: (x["score"], x["duration_min"]))
                            for rank, route in enumerate(comparisons, start=1):
                                route["rank"] = rank
                            st.session_state["geosafe_route_comparison"] = {
                                "start": origin, "end": destination, "routes": comparisons,
                                "radius_km": radius_km,
                            }
            except requests.RequestException as exc:
                st.error("A geocoding or routing service could not be reached. Check your connection and try again.")
                st.caption(f"Technical detail: {exc}")
            except (ValueError, KeyError, TypeError) as exc:
                st.error("The route response could not be processed.")
                st.caption(f"Technical detail: {exc}")

    result = st.session_state.get("geosafe_route_comparison")
    if result:
        origin, destination = result["start"], result["end"]
        routes = result["routes"]
        active_radius = result["radius_km"]
        best = routes[0]
        st.subheader("GeoSafe comparison")
        if len(routes) > 1:
            st.success(
                f"Lowest estimated historical-risk proxy among {len(routes)} available alternatives: "
                f"Route {best['route_id']}. This is based on the available dataset and experimental formula, "
                "not a safety guarantee. Check data coverage before interpreting the ranking."
            )
        else:
            st.info(
                "Only one route alternative was returned. GeoSafe can describe its historical incident exposure, "
                "but cannot compare it against another route or conclude that it is the lowest-risk option."
            )
        rows = []
        for route in routes:
            rows.append({
                "Recommendation rank": route["rank"],
                "Route option": f"Route {route['route_id']}",
                "Estimated distance (km)": round(route["distance_km"], 1),
                "Estimated time (min)": round(route["duration_min"]),
                "Nearby records": route["total_records"],
                "High + Critical": route["high"] + route["critical"],
                "DBSCAN hotspots nearby": route["hotspots"],
                "Route data coverage (%)": round(route.get("coverage_pct", 0.0), 1),
                "Comparison score ↓": round(route["score"], 2),
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        st.caption(
            "Scoring method: (4 × Critical + 3 × High + 2 × Medium + Low records) per 10 km, "
            "+ 2 points for each detected hotspot within approximately 1 km of sampled route points. "
            "Lower scores rank better. Route data coverage estimates the share of sampled route points "
            "with at least one dataset record within the chosen radius; it is not a safety score. "
            "This is an experimental, unvalidated comparison—not a probability of harm."
        )

        m = folium.Map(
            location=best["coordinates"][len(best["coordinates"]) // 2],
            zoom_start=7, control_scale=True, tiles="OpenStreetMap",
        )
        palette = ["#14804a", "#e09f1f", "#6b7280", "#8b5cf6"]
        for idx, route in enumerate(routes):
            color = palette[min(idx, len(palette) - 1)]
            label = f"Rank {route['rank']} · Route {route['route_id']} · score {route['score']:.1f}"
            folium.PolyLine(
                route["coordinates"], color=color, weight=7 if route["rank"] == 1 else 5,
                opacity=0.9 if route["rank"] == 1 else 0.65, tooltip=label,
            ).add_to(m)
        folium.Marker(
            [origin["latitude"], origin["longitude"]], popup=origin["display_name"],
            tooltip="Starting point", icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)
        folium.Marker(
            [destination["latitude"], destination["longitude"]], popup=destination["display_name"],
            tooltip="Destination", icon=folium.Icon(color="red", icon="flag"),
        ).add_to(m)

        # Display historical incident records from the currently top-ranked route.
        for _, incident in best["records"].head(300).iterrows():
            popup = (
                f"City: {incident.get('city', 'Not available')}<br>"
                f"Area: {incident.get('area', 'Not available')}<br>"
                f"Recorded risk: {incident.get('risk_level', 'Not available')}<br>"
                f"Crime type: {incident.get('crime_type', 'Not available')}"
            )
            folium.CircleMarker(
                [incident["latitude"], incident["longitude"]], radius=4,
                color="#8a3ffc", fill=True, fill_opacity=0.55,
                popup=folium.Popup(popup, max_width=300), tooltip="Historical dataset record",
            ).add_to(m)
        st.caption(f"Origin: {origin['display_name']}  |  Destination: {destination['display_name']}")
        st_folium(m, width='stretch', height=600, returned_objects=[])
        st.caption("Map colours indicate relative ranking among alternatives returned by the routing service; green is ranked first, not certified safe.")

        st.subheader("Explore a route")
        chosen_label = st.selectbox(
            "Select a route to inspect",
            options=[f"Rank {r['rank']} — Route {r['route_id']}" for r in routes],
            index=0,
        )
        chosen_rank = next(r["rank"] for r in routes if f"Rank {r['rank']} — Route {r['route_id']}" == chosen_label)
        chosen = next(r for r in routes if r["rank"] == chosen_rank)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance", f"{chosen['distance_km']:.1f} km")
        c2.metric("Estimated time", f"{chosen['duration_min']:.0f} min")
        c3.metric("Nearby records", f"{chosen['total_records']:,}")
        c4.metric("Nearby hotspots", str(chosen["hotspots"]))
        st.metric(
            "Estimated route data coverage",
            f"{chosen.get('coverage_pct', 0.0):.1f}%",
            help="Share of evenly sampled route points with at least one historical dataset record within the selected radius. Low coverage means the route score is especially uncertain; it does not mean the route is unsafe or safe.",
        )
        if chosen.get("coverage_pct", 0.0) < 20:
            st.warning("Low historical-data coverage: treat this route's score as highly uncertain. Few records near the route may reflect missing data rather than lower risk.")
        elif chosen.get("coverage_pct", 0.0) < 50:
            st.info("Partial historical-data coverage: the route score should be interpreted cautiously.")
        breakdown = pd.DataFrame({
            "Historical risk label": ["Critical", "High", "Medium", "Low"],
            "Records": [chosen["critical"], chosen["high"], chosen["medium"], chosen["low"]],
        })
        chart_col, note_col = st.columns([1, 1])
        with chart_col:
            st.plotly_chart(px.bar(breakdown, x="Historical risk label", y="Records", title="Historical records by risk label"), width='stretch')
        with note_col:
            st.markdown("**How to interpret this result**")
            st.write(
                "These are historical records found near sampled points along the route. "
                "The analysis does not know current street conditions, lighting at the time of travel, "
                "whether an incident was reported, or whether records are missing."
            )
            if chosen["total_records"] == 0:
                st.warning("No records were found in the selected radius. This is not evidence that the route is safe.")
        display_cols = [c for c in [
            "city", "area", "crime_type", "risk_level", "incident_timestamp", "crime_count", "latitude", "longitude"
        ] if c in chosen["records"].columns]
        if display_cols:
            with st.expander("View historical records used in this route comparison"):
                st.dataframe(chosen["records"][display_cols].head(500), width='stretch', hide_index=True)
                st.caption("At most 500 matching records are displayed.")

        st.warning(
            "Responsible-use note: GeoSafe currently uses historical CSV records and public routing data. "
            "Records may be incomplete, outdated, biased, or synthetic. The score is a prototype heuristic, "
            "not a validated prediction of personal danger. Use local judgement and official guidance."
        )



# ============================================================
# ALERT CENTER
# ============================================================

elif page == "🚨 Alert Center":

    st.title("🚨 GeoSafe Alert Center")
    st.write(
        "Create configurable alerts from the incident and DBSCAN hotspot records "
        "already available in your project. This is a dataset-driven alert dashboard, "
        "not a live emergency-warning or push-notification service."
    )

    st.info(
        "Alert status is calculated from your current CSV files when the page runs. "
        "Without a live incident feed, alerts cannot confirm that a situation is happening now."
    )

    # Normalize the fields used by the alert rules without changing the original dataset.
    alert_data = incidents.copy()
    if "city" not in alert_data.columns:
        alert_data["city"] = "Unknown city"
    if "area" not in alert_data.columns:
        alert_data["area"] = "Unknown area"
    if "risk_level" not in alert_data.columns:
        alert_data["risk_level"] = "Unknown"

    alert_data["city"] = alert_data["city"].fillna("Unknown city").astype(str)
    alert_data["area"] = alert_data["area"].fillna("Unknown area").astype(str)
    alert_data["_risk_norm"] = alert_data["risk_level"].fillna("Unknown").astype(str).str.strip().str.lower()

    available_risks = [
        label for label in ["Critical", "High", "Medium", "Low"]
        if (alert_data["_risk_norm"] == label.lower()).any()
    ]
    if not available_risks:
        available_risks = sorted(alert_data["risk_level"].dropna().astype(str).unique().tolist())

    city_options = sorted(alert_data["city"].dropna().astype(str).unique().tolist())
    with st.expander("⚙️ Configure alert rules", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            alert_city = st.selectbox("Area scope", ["All cities"] + city_options, key="alert_city_scope")
            watched_risks = st.multiselect(
                "Risk levels to watch",
                options=available_risks,
                default=[r for r in ["Critical", "High"] if r in available_risks],
                key="alert_risk_levels",
                help="Alerts are raised when the selected risk levels meet the threshold below.",
            )
        with f2:
            min_area_records = st.number_input(
                "Minimum records in an area",
                min_value=1, max_value=10000, value=3, step=1,
                key="alert_min_area_records",
            )
            min_watch_records = st.number_input(
                "Minimum watched-risk records",
                min_value=1, max_value=10000, value=1, step=1,
                key="alert_min_watch_records",
            )
            min_hotspot_incidents = st.number_input(
                "Minimum incidents for a hotspot alert",
                min_value=1, max_value=100000, value=10, step=1,
                key="alert_min_hotspot_incidents",
            )

    if alert_city != "All cities":
        filtered_alert_data = alert_data[alert_data["city"] == alert_city].copy()
        filtered_hotspots = hotspots.copy()
        if "city" in filtered_hotspots.columns:
            filtered_hotspots = filtered_hotspots[
                filtered_hotspots["city"].fillna("Unknown city").astype(str) == alert_city
            ]
    else:
        filtered_alert_data = alert_data.copy()
        filtered_hotspots = hotspots.copy()

    generated_alerts = []

    # Area-level incident alerts.
    if watched_risks and not filtered_alert_data.empty:
        for (city_name, area_name), group in filtered_alert_data.groupby(["city", "area"], dropna=False):
            group_risks = group["_risk_norm"]
            watched_mask = group_risks.isin([r.lower() for r in watched_risks])
            watched_count = int(watched_mask.sum())
            total_count = int(len(group))
            if total_count < int(min_area_records) or watched_count < int(min_watch_records):
                continue

            critical_count = int((group_risks == "critical").sum())
            high_count = int((group_risks == "high").sum())
            if critical_count > 0:
                severity = "CRITICAL"
                explanation = f"{critical_count} Critical record(s) in this area"
            elif high_count > 0:
                severity = "HIGH"
                explanation = f"{high_count} High record(s) in this area"
            else:
                severity = "REVIEW"
                explanation = f"{watched_count} record(s) match the selected risk levels"

            generated_alerts.append({
                "Alert type": "Incident concentration",
                "Alert level": severity,
                "City": city_name,
                "Area": area_name,
                "Records": total_count,
                "Watched-risk records": watched_count,
                "Critical records": critical_count,
                "High records": high_count,
                "Details": explanation,
            })

    # DBSCAN hotspot alerts use the precomputed hotspot summary.
    if not filtered_hotspots.empty and "incident_count" in filtered_hotspots.columns:
        hs_work = filtered_hotspots.copy()
        hs_work["incident_count"] = pd.to_numeric(hs_work["incident_count"], errors="coerce").fillna(0)
        for _, hs_row in hs_work[hs_work["incident_count"] >= int(min_hotspot_incidents)].iterrows():
            city_name = str(hs_row.get("city", "Unknown city"))
            area_name = str(hs_row.get("area", "Unknown area"))
            count = int(hs_row["incident_count"])
            high_critical_pct = pd.to_numeric(
                pd.Series([hs_row.get("high_critical_pct", np.nan)]), errors="coerce"
            ).iloc[0]
            if pd.notna(high_critical_pct) and high_critical_pct >= 50:
                severity = "HIGH"
                detail = f"DBSCAN hotspot with {count} incidents and {high_critical_pct:.1f}% High/Critical"
            else:
                severity = "REVIEW"
                detail = f"DBSCAN hotspot with {count} incidents; review its historical pattern"
            generated_alerts.append({
                "Alert type": "DBSCAN hotspot",
                "Alert level": severity,
                "City": city_name,
                "Area": area_name,
                "Records": count,
                "Watched-risk records": np.nan,
                "Critical records": np.nan,
                "High records": np.nan,
                "Details": detail,
            })

    alerts_df = pd.DataFrame(generated_alerts)
    if not alerts_df.empty:
        severity_order = {"CRITICAL": 0, "HIGH": 1, "REVIEW": 2}
        alerts_df["_sort"] = alerts_df["Alert level"].map(severity_order).fillna(3)
        alerts_df = alerts_df.sort_values(
            ["_sort", "Records"], ascending=[True, False]
        ).drop(columns=["_sort"]).reset_index(drop=True)

    st.subheader("Alert summary")
    a1, a2, a3, a4 = st.columns(4)
    total_alerts = len(alerts_df)
    critical_alerts = int((alerts_df["Alert level"] == "CRITICAL").sum()) if total_alerts else 0
    high_alerts = int((alerts_df["Alert level"] == "HIGH").sum()) if total_alerts else 0
    hotspot_alerts = int((alerts_df["Alert type"] == "DBSCAN hotspot").sum()) if total_alerts else 0
    a1.metric("Alerts generated", f"{total_alerts:,}")
    a2.metric("Critical alerts", f"{critical_alerts:,}")
    a3.metric("High alerts", f"{high_alerts:,}")
    a4.metric("Hotspot alerts", f"{hotspot_alerts:,}")

    if alerts_df.empty:
        st.success(
            "No records matched the current alert rules. This does not mean the area is safe; "
            "it means the selected thresholds did not produce a dataset alert."
        )
    else:
        for _, alert in alerts_df.head(8).iterrows():
            heading = f"{alert['Alert level']} · {alert['Alert type']}"
            body = (
                f"**Location:** {alert['Area']}, {alert['City']}  \n"
                f"**Records:** {int(alert['Records']):,}  \n"
                f"**Reason:** {alert['Details']}"
            )
            if alert["Alert level"] == "CRITICAL":
                st.error(f"🚨 {heading}\n\n{body}")
            elif alert["Alert level"] == "HIGH":
                st.warning(f"⚠️ {heading}\n\n{body}")
            else:
                st.info(f"ℹ️ {heading}\n\n{body}")

        if len(alerts_df) > 8:
            st.caption(f"Showing the first 8 of {len(alerts_df)} generated alerts. Download the full list below.")

        st.subheader("Alert register")
        st.dataframe(alerts_df, width='stretch', hide_index=True)

        csv_bytes = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download alert report (CSV)",
            data=csv_bytes,
            file_name="geosafe_alert_report.csv",
            mime="text/csv",
            type="primary",
        )

    st.subheader("How this alert system works")
    st.markdown(
        """
        - **Incident concentration:** flags city/area groups when the selected risk categories and record-count thresholds are met.
        - **DBSCAN hotspot alert:** flags precomputed hotspots that meet the minimum incident threshold; High/Critical percentage adds context when available.
        - **Configurable rules:** change the risk levels and thresholds to explore different alert scenarios.
        - **Exportable report:** download the generated alert register for your project demonstration.

        **Limitations:** these alerts are derived from historical CSV records and precomputed hotspots. They do not refresh from live incident feeds, do not send SMS/push notifications, and should not be interpreted as confirmed current emergencies.
        """
    )



# ============================================================
# SOS ALARM & PERSONAL SAFETY TOOLKIT
# ============================================================

elif page == "🆘 SOS Alarm":

    st.title("🆘 GeoSafe SOS Alarm")
    st.write(
        "A quick-access personal safety toolkit with a sound alarm, emergency contact guidance, "
        "and a ready-to-copy help message."
    )

    st.warning(
        "Safety prototype: emergency shortcuts open your device's calling or messaging app where supported. "
        "GeoSafe does not place calls or send messages in the background. Location is not tracked automatically. "
        "Do not rely on this page as your only way to get help."
    )

    st.subheader("1. Audible alarm")
    st.write(
        "Press Start alarm to activate the in-app alarm player. Your browser may require sound to be "
        "enabled or may restrict automatic playback."
    )

    if "geosafe_alarm_active" not in st.session_state:
        st.session_state["geosafe_alarm_active"] = False

    alarm_col1, alarm_col2 = st.columns([1, 1])
    with alarm_col1:
        if st.button(
            "🔊 Start alarm",
            type="primary",
            width='stretch',
            disabled=st.session_state["geosafe_alarm_active"],
        ):
            st.session_state["geosafe_alarm_active"] = True
            st.rerun()

    with alarm_col2:
        if st.button(
            "⏹️ Stop alarm",
            width='stretch',
            disabled=not st.session_state["geosafe_alarm_active"],
        ):
            st.session_state["geosafe_alarm_active"] = False
            st.rerun()

    if st.session_state["geosafe_alarm_active"]:
        # Generate a short alternating-tone WAV locally. It does not require an audio file.
        sample_rate = 22050
        duration_seconds = 1.2
        frames = bytearray()
        total_samples = int(sample_rate * duration_seconds)
        for i in range(total_samples):
            t = i / sample_rate
            # Two alternating high tones to make the alarm noticeable.
            freq = 1450 if (t % 0.4) < 0.2 else 950
            envelope = min(1.0, i / 400.0, (total_samples - i) / 400.0)
            sample = int(0.45 * envelope * 32767 * math.sin(2 * math.pi * freq * t))
            frames.extend(struct.pack("<h", sample))
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(frames))
        st.audio(wav_buffer.getvalue(), format="audio/wav", loop=True, autoplay=True)
        st.error("ALARM MODE IS ON — use Stop alarm to turn it off.")
    else:
        st.info("Alarm is currently off.")

    st.divider()
    st.subheader("2. Emergency help")
    st.markdown(
        """
        **If you are in immediate danger in India, call 112 if you can do so safely.**
        If calling is unsafe, try to move toward a staffed public place or trusted person and use
        the safest available way to contact someone.
        """
    )
    st.markdown("[📞 Open emergency call shortcut (112)](tel:112)")
    st.caption("This link may open the phone app on a supported device. It does not automatically place a call.")

    st.subheader("3. Prepare a help message")
    msg_col1, msg_col2 = st.columns(2)
    with msg_col1:
        contact_name = st.text_input("Trusted contact name (optional)", placeholder="e.g., Amma")
        help_location = st.text_input("Your location (enter manually)", placeholder="e.g., near a bus stand")
    with msg_col2:
        trusted_phone = st.text_input(
            "Trusted contact phone number (optional)",
            placeholder="Include country code if needed",
            help="Used only to prepare a messaging link; this app does not store or send the message to the contact.",
        )
        help_reason = st.selectbox(
            "Message type",
            [
                "I need help now",
                "I feel unsafe and need a call",
                "Please stay on the phone with me",
                "Please contact emergency services",
            ],
        )
        callback_number = st.text_input("Your callback number (optional)", placeholder="Enter only if safe")

    message_parts = [help_reason + "."]
    if help_location.strip():
        message_parts.append("My location is: " + help_location.strip() + ".")
    if callback_number.strip():
        message_parts.append("You can reach me at: " + callback_number.strip() + ".")
    if contact_name.strip():
        message_parts.append("Please send this to " + contact_name.strip() + ".")
    message_parts.append("Please respond as soon as you can.")
    help_message = " ".join(message_parts)

    st.text_area("Message preview — copy and send using your phone", value=help_message, height=120)
    encoded_message = quote(help_message)
    wa_phone = "".join(ch for ch in trusted_phone if ch.isdigit())
    whatsapp_url = (
        f"https://wa.me/{wa_phone}?text={encoded_message}"
        if wa_phone else f"https://wa.me/?text={encoded_message}"
    )
    sms_url = f"sms:{trusted_phone.strip()}?body={encoded_message}" if trusted_phone.strip() else f"sms:?body={encoded_message}"
    share_col1, share_col2 = st.columns(2)
    with share_col1:
        st.markdown(f"[💬 Open WhatsApp with message]({whatsapp_url})")
    with share_col2:
        st.markdown(f"[✉️ Open SMS with message](<{sms_url}>)")
    st.caption(
        "Review the recipient and message before sending. Messaging links depend on your device/browser. "
        "GeoSafe does not send messages automatically."
    )
    st.caption("For privacy, avoid entering sensitive details on a shared device.")

    st.subheader("4. Quick safety checklist")
    checklist_items = [
        "Move toward a staffed, well-populated place if it is safe to move.",
        "Contact a trusted person and tell them where you are.",
        "If in immediate danger, contact emergency services when safe.",
        "Do not confront or follow a person who may be threatening you.",
    ]
    for item in checklist_items:
        st.checkbox(item, key="sos_check_" + str(checklist_items.index(item)))

    st.caption(
        "The alarm is an in-app sound only. It does not notify nearby people automatically, "
        "record audio, track live location, or alert contacts in the background."
    )


# ============================================================
# MODEL ANALYSIS
# ============================================================

elif page == "📊 Model Analysis":

    st.title("📊 Machine Learning Model Analysis")

    st.write(
        """
        GeoSafe evaluates four classification approaches for
        women-safety risk classification.
        """
    )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    st.subheader("Performance Summary")

    st.dataframe(
        model_metrics,
        width='stretch',
        hide_index=True,
    )

    # --------------------------------------------------------
    # METRIC COMPARISON
    # --------------------------------------------------------

    metric_columns = [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1-Score (%)",
    ]

    available_metrics = [
        col
        for col in metric_columns
        if col in model_metrics.columns
    ]

    if available_metrics:

        melted = model_metrics.melt(
            id_vars="Model",
            value_vars=available_metrics,
            var_name="Metric",
            value_name="Score",
        )

        fig = px.bar(
            melted,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text_auto=".2f",
            title="Classification Performance Comparison",
        )

        fig.update_layout(
            yaxis_title="Score (%)",
            xaxis_title="",
            yaxis_range=[
                0,
                100,
            ],
            height=520,
        )

        st.plotly_chart(
            fig,
            width='stretch',
        )

    # --------------------------------------------------------
    # RADAR VIEW
    # --------------------------------------------------------

    st.subheader("Model Performance Profile")

    if (
        "Model" in model_metrics.columns
        and available_metrics
    ):

        radar_model = st.selectbox(
            "Select a model",
            model_metrics["Model"].tolist(),
        )

        radar_row = model_metrics[
            model_metrics["Model"]
            == radar_model
        ].iloc[0]

        radar_metrics = [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1-Score (%)",
        ]

        radar_labels = []
        radar_values = []

        for metric in radar_metrics:

            if metric in radar_row.index:

                value = pd.to_numeric(
                    radar_row[metric],
                    errors="coerce",
                )

                if not pd.isna(value):

                    radar_labels.append(
                        metric.replace(
                            " (%)",
                            "",
                        )
                    )

                    radar_values.append(
                        float(value)
                    )

        if radar_values:

            radar_labels_closed = (
                radar_labels
                + [radar_labels[0]]
            )

            radar_values_closed = (
                radar_values
                + [radar_values[0]]
            )

            radar_fig = go.Figure()

            radar_fig.add_trace(
                go.Scatterpolar(
                    r=radar_values_closed,
                    theta=radar_labels_closed,
                    fill="toself",
                    name=radar_model,
                )
            )

            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[
                            0,
                            100,
                        ],
                    )
                ),
                showlegend=True,
                height=500,
            )

            st.plotly_chart(
                radar_fig,
                width='stretch',
            )

    st.info(
        """
        Accuracy, precision, recall and F1-score describe
        different aspects of classification performance.
        Because the risk classes are not equally represented,
        class-wise metrics should also be considered when
        interpreting the models.
        """
    )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "📖 Methodology":

    st.title("📖 GeoSafe Methodology")

    st.markdown(
        """
        <div class="highlight-box">
        GeoSafe follows a two-part machine learning framework:
        <b>risk classification</b> and <b>geographical hotspot
        detection</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader("Overall Pipeline")

    st.code(
        """
Women Safety Dataset
        │
        ▼
Data Preprocessing
        │
        ├──────────────────────────┐
        ▼                          ▼
Risk Classification       Geographical Analysis
        │                          │
        ▼                          ▼
 ┌──────┬──────┬──────┬──────┐  Latitude + Longitude
 │  DT  │  RF  │  NB  │ KNN  │          │
 └──────┴──────┴──────┴──────┘          ▼
        │                            DBSCAN
        ▼                               │
 Risk-Level Prediction                 ▼
                              Geographical Hotspots
                                      │
                                      ▼
                              Hotspot Characterization
                                      │
                                      ▼
                              Interactive GeoSafe Map
        """,
        language="text",
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    st.subheader("1. Risk Classification")

    st.write(
        """
        Four supervised learning algorithms are used to classify
        the safety risk level:

        • Decision Tree

        • Random Forest

        • Naive Bayes

        • KNN

        The models are evaluated using accuracy, precision, recall
        and F1-score.
        """
    )

    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    st.subheader("Decision Tree")

    st.write(
        """
        Decision Tree is a supervised learning algorithm that
        makes decisions by recursively splitting the data based
        on feature values.
        """
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    st.subheader("Random Forest")

    st.write(
        """
        Random Forest is an ensemble learning algorithm that
        combines multiple decision trees. The collective
        predictions of the trees are used for classification.
        """
    )

    # --------------------------------------------------------
    # NAIVE BAYES
    # --------------------------------------------------------

    st.subheader("Naive Bayes")

    st.write(
        """
        Naive Bayes is a probabilistic classification algorithm
        based on Bayes' theorem and assumes conditional
        independence between features.
        """
    )

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    st.subheader("KNN")

    st.write(
        """
        KNN, or K-Nearest Neighbors, classifies an observation
        based on the labels of nearby training observations.
        It is called a lazy learner because it does not construct
        an explicit model during the training phase.
        """
    )

    # --------------------------------------------------------
    # CLUSTERING
    # --------------------------------------------------------

    st.subheader("2. Geographical Clustering")

    st.write(
        """
        DBSCAN is used to identify geographically dense groups
        of incidents. Latitude and longitude are used as the
        geographical features.

        DBSCAN groups nearby observations into clusters when
        enough observations occur within the specified
        neighbourhood.

        Observations that do not belong to sufficiently dense
        groups can be identified as noise.
        """
    )

    # --------------------------------------------------------
    # HOTSPOT ANALYSIS
    # --------------------------------------------------------

    st.subheader("3. Hotspot Analysis")

    st.write(
        """
        Each detected hotspot is characterized using available
        incident information such as:

        • Number of incidents

        • Total crime count

        • Average safety score

        • High/Critical percentage

        • Dominant crime type
        """
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    st.subheader("4. Geographical Visualization")

    st.write(
        """
        The interactive map combines two complementary views:

        **Crime intensity heatmap**

        Shows the spatial concentration of incident activity.

        **DBSCAN hotspot markers**

        Shows the centres of detected geographical clusters and
        provides detailed hotspot statistics through interactive
        popups.
        """
    )

    # --------------------------------------------------------
    # RESPONSIBLE INTERPRETATION
    # --------------------------------------------------------

    st.subheader("5. Responsible Interpretation")

    st.warning(
        """
        A detected hotspot represents a concentration of incidents
        in the analyzed dataset. It should not automatically be
        interpreted as proof that an entire city, neighbourhood or
        population is unsafe.

        Likewise, machine-learning results are analytical outputs
        rather than guarantees about an individual's real-world
        safety.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    <b>GeoSafe</b> • Women Safety Intelligence & Geographical Analysis<br>
    Academic Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)