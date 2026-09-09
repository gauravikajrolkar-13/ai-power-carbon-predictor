import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AERIS | AI Energy Requirement Intelligence System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "ai_power_random_forest_model.pkl"
CARBON_PATH = BASE_DIR / "electricity_carbon_intensity_2024.csv"
CONFORMAL_PATH = BASE_DIR / "conformal_prediction_params.csv"
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #fafafa;
        color: #202124;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Remove Streamlit top decoration */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* AERIS header */
    .aeris-header {
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }

    .aeris-title {
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        line-height: 1;
        color: #171717;
        margin-bottom: 0.45rem;
    }

    .aeris-full-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.35rem;
        color: #303030;
        margin-bottom: 0.5rem;
    }

    .aeris-tagline {
        max-width: 760px;
        margin: 0 auto;
        color: #6b6b6b;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Navigation */
    .nav-divider {
        height: 1px;
        background: #dedede;
        margin: 1.5rem 0 1.25rem 0;
    }

    div.stButton > button {
        border: none;
        background: transparent;
        color: #555;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0.45rem 0.2rem;
        border-radius: 0;
    }

    div.stButton > button:hover {
        color: #111;
        background: transparent;
    }

    /* Cards */
    .card {
        background: white;
        border: 1px solid #e2e2e2;
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
    }

    .result-card {
        background: white;
        border: 1px solid #dedede;
        border-radius: 14px;
        padding: 1.5rem;
        min-height: 155px;
    }

    .result-label {
        color: #777;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.55rem;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 600;
        color: #171717;
    }

    .result-subtext {
        color: #777;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    .best-model-card {
        background: #f4f4f2;
        border: 1px solid #dcdcd8;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .best-model-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #777;
        margin-bottom: 0.3rem;
    }

    .best-model-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #171717;
    }

    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        color: #171717;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .section-description {
        color: #6d6d6d;
        line-height: 1.65;
        margin-bottom: 1.25rem;
    }

    .metric-large {
        font-size: 2.4rem;
        font-weight: 600;
        color: #171717;
    }

    .small-label {
        color: #777;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.78rem;
        padding-top: 2.5rem;
        border-top: 1px solid #e5e5e5;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_carbon_data():
    if CARBON_PATH.exists():
        data = pd.read_csv(CARBON_PATH)

        # Handle possible column naming variations
        if "Carbon intensity" in data.columns:
            data = data.rename(columns={"Carbon intensity": "carbon_intensity"})

        return data

    # Fallback values from the project
    return pd.DataFrame({
        "Entity": [
            "India",
            "United States",
            "Germany",
            "France"
        ],
        "Year": [2024, 2024, 2024, 2024],
        "carbon_intensity": [
            705.40,
            383.78,
            336.38,
            40.48
        ]
    })


@st.cache_data
def load_conformal_margin():
    if CONFORMAL_PATH.exists():
        data = pd.read_csv(CONFORMAL_PATH)

        # Try to locate the margin automatically
        possible_columns = [
            "conformal_error_margin",
            "error_margin",
            "margin",
            "conformal_margin"
        ]

        for column in possible_columns:
            if column in data.columns:
                return float(data[column].iloc[0])

        # If the CSV has a single numeric value
        numeric_columns = data.select_dtypes(include=np.number).columns

        if len(numeric_columns) > 0:
            return float(data[numeric_columns[0]].iloc[0])

    # Project's calculated 90% conformal margin
    return 1007.35


model = load_model()
carbon_data = load_carbon_data()
CONFORMAL_MARGIN = load_conformal_margin()


# ============================================================
# NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"


def navigation():
    pages = [
        "Overview",
        "Data & Features",
        "Model Results",
        "Interpretation",
        "About"
    ]

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    cols = st.columns(len(pages))

    for col, page in zip(cols, pages):
        with col:
            if st.button(
                page,
                key=f"nav_{page}",
                use_container_width=True
            ):
                st.session_state.page = page
                st.rerun()

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="aeris-header">
        <div class="aeris-title">AERIS</div>
        <div class="aeris-full-name">
            AI Energy Requirement Intelligence System
        </div>
        <div class="aeris-tagline">
            Predicting AI training power consumption from workload and
            hardware characteristics, with downstream energy and
            carbon-footprint estimation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

navigation()


# ============================================================
# CONSTANTS
# ============================================================

WORKLOADS = [
    "Feature Forecasting",
    "Reinforcement Learning",
    "Image Classification",
    "Text Generation",
    "Image Captioning",
    "Image Generation",
    "LLM"
]

HARDWARE = [
    "RTX3060",
    "H100",
    "B200"
]

COUNTRIES = [
    "India",
    "United States",
    "Germany",
    "France"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_carbon_intensity(country):
    row = carbon_data[
        carbon_data["Entity"].astype(str).str.strip() == country
    ]

    if len(row) == 0:
        fallback = {
            "India": 705.40,
            "United States": 383.78,
            "Germany": 336.38,
            "France": 40.48
        }
        return fallback.get(country, 500.0)

    return float(row.iloc[0]["carbon_intensity"])


def build_prediction_row(
    workload,
    hardware,
    num_gpus,
    batch_size,
    model_size=np.nan,
    image_size=np.nan,
    sequence_length=np.nan,
    layer_size=np.nan,
    num_layers=np.nan,
    embedding_dim=np.nan,
    hidden_units=np.nan,
    num_filters=np.nan,
    optimizer="Not_Applicable",
    input_type="Not_Applicable",
    parallelization="Not_Applicable"
):

    return pd.DataFrame([{
        "workload": workload,
        "hardware": hardware,
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "model_size": model_size,
        "image_size": image_size,
        "sequence_length": sequence_length,
        "layer_size": layer_size,
        "num_layers": num_layers,
        "embedding_dim": embedding_dim,
        "hidden_units": hidden_units,
        "num_filters": num_filters,
        "optimizer": optimizer,
        "input_type": input_type,
        "parallelization": parallelization
    }])


def predict_power(input_data):
    prediction = float(model.predict(input_data)[0])
    return max(prediction, 0)


def get_runtime_energy(power_watts, runtime_minutes):
    runtime_hours = runtime_minutes / 60
    return power_watts * runtime_hours / 1000


def get_carbon(energy_kwh, carbon_intensity):
    return energy_kwh * carbon_intensity / 1000


# ============================================================
# OVERVIEW / HOMEPAGE
# ============================================================

def overview_page():

    st.markdown(
        '<div class="section-title">AI Training Energy Workspace</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Configure an AI training workload and hardware environment.
            AERIS predicts the expected power requirement and converts
            that prediction into estimated energy use and carbon emissions.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### Workload Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        workload = st.selectbox(
            "AI Workload",
            WORKLOADS
        )

    with col2:
        hardware = st.selectbox(
            "Hardware",
            HARDWARE
        )

    with col3:
        if hardware == "RTX3060":
            num_gpus = 1
        else:
            num_gpus = 8

        st.selectbox(
            "Number of GPUs",
            [num_gpus],
            disabled=True
        )

    # --------------------------------------------------------
    # WORKLOAD PARAMETERS
    # --------------------------------------------------------

    st.markdown("#### Workload Parameters")

    batch_size = None

    model_size = np.nan
    image_size = np.nan
    sequence_length = np.nan
    layer_size = np.nan
    num_layers = np.nan
    embedding_dim = np.nan
    hidden_units = np.nan
    num_filters = np.nan

    optimizer = "Not_Applicable"
    input_type = "Not_Applicable"
    parallelization = "Not_Applicable"

    # Feature Forecasting
    if workload == "Feature Forecasting":

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            batch_size = st.selectbox(
                "Batch Size",
                [50, 100, 150]
            )

        with c2:
            model_size = st.selectbox(
                "Model Size",
                [474000, 1600000, 3600000]
            )

        with c3:
            sequence_length = st.selectbox(
                "Sequence Length",
                [96, 192, 672]
            )

        with c4:
            num_layers = st.selectbox(
                "Number of Layers",
                [6, 8, 10]
            )

    # Reinforcement Learning
    elif workload == "Reinforcement Learning":

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            batch_size = st.selectbox(
                "Batch Size",
                [150, 250, 350]
            )

        with c2:
            layer_size = st.selectbox(
                "Layer Size",
                [256, 512, 1024]
            )

        with c3:
            sequence_length = st.selectbox(
                "Input Sequence",
                [150, 250, 350]
            )

        with c4:
            input_type = st.selectbox(
                "Input Type",
                ["Feature", "Sequence"]
            )

    # Image Classification
    elif workload == "Image Classification":

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            batch_size = st.selectbox(
                "Batch Size",
                [750, 1500, 2250]
            )

        with c2:
            image_size = st.selectbox(
                "Image Size",
                [112, 224, 280]
            )

        with c3:
            num_filters = st.selectbox(
                "Number of Filters",
                [8, 16, 32]
            )

        with c4:
            optimizer = st.selectbox(
                "Optimizer",
                ["Adam", "SGD", "RMS"]
            )

    # Text Generation
    elif workload == "Text Generation":

        c1, c2, c3 = st.columns(3)

        with c1:
            batch_size = st.selectbox(
                "Batch Size",
                [32, 128, 512]
            )

        with c2:
            sequence_length = st.selectbox(
                "Sequence Length",
                [100, 250, 500]
            )

        with c3:
            embedding_dim = st.selectbox(
                "Embedding Dimension",
                [100, 300, 1000]
            )

    # Image Captioning
    elif workload == "Image Captioning":

        c1, c2, c3 = st.columns(3)

        with c1:
            batch_size = st.selectbox(
                "Batch Size",
                [128, 256, 1024]
            )

        with c2:
            layer_size = st.selectbox(
                "Layer Size",
                [512, 1024, 2048]
            )

        with c3:
            embedding_dim = st.selectbox(
                "Embedding Dimension",
                [256, 512, 1024]
            )

    # Image Generation
    elif workload == "Image Generation":

        c1, c2, c3 = st.columns(3)

        with c1:
            batch_size = st.selectbox(
                "Batch Size",
                [128, 256, 512]
            )

        with c2:
            image_size = st.selectbox(
                "Image Size",
                [32, 64, 128]
            )

        with c3:
            model_size = st.selectbox(
                "Model Size",
                [107000000, 430000000, 1700000000]
            )

    # LLM
    elif workload == "LLM":

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            batch_size = st.selectbox(
                "Batch Size",
                [2, 16, 32]
            )

        with c2:
            parallelization = st.selectbox(
                "Parallelization",
                ["ZeRO-1", "ZeRO-2", "ZeRO-3"]
            )

        with c3:
            sequence_length = st.selectbox(
                "Sequence Length",
                [1024, 2048, 4096]
            )

        with c4:
            model_size = st.selectbox(
                "Model Size",
                [1000000000, 3000000000, 8000000000]
            )

    st.markdown("#### Estimation Settings")

    c1, c2 = st.columns(2)

    with c1:
        runtime_minutes = st.number_input(
            "Estimated Training Runtime (minutes)",
            min_value=1.0,
            max_value=100000.0,
            value=15.0,
            step=1.0
        )

    with c2:
        country = st.selectbox(
            "Electricity Grid",
            COUNTRIES,
            index=0
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    run_prediction = st.button(
        "Run AERIS",
        type="primary",
        use_container_width=True
    )

    if run_prediction:

        input_data = build_prediction_row(
            workload=workload,
            hardware=hardware,
            num_gpus=num_gpus,
            batch_size=batch_size,
            model_size=model_size,
            image_size=image_size,
            sequence_length=sequence_length,
            layer_size=layer_size,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            num_filters=num_filters,
            optimizer=optimizer,
            input_type=input_type,
            parallelization=parallelization
        )

        predicted_power = predict_power(input_data)

        # 90% conformal prediction interval
        lower_power = max(
            0,
            predicted_power - CONFORMAL_MARGIN
        )

        upper_power = predicted_power + CONFORMAL_MARGIN

        predicted_energy = get_runtime_energy(
            predicted_power,
            runtime_minutes
        )

        carbon_intensity = get_carbon_intensity(country)

        estimated_carbon = get_carbon(
            predicted_energy,
            carbon_intensity
        )

        st.session_state.prediction = {
            "power": predicted_power,
            "lower_power": lower_power,
            "upper_power": upper_power,
            "energy": predicted_energy,
            "carbon": estimated_carbon,
            "country": country,
            "carbon_intensity": carbon_intensity,
            "runtime": runtime_minutes
        }

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "prediction" in st.session_state:

        result = st.session_state.prediction

        st.markdown(
            '<div class="section-title">AERIS Prediction</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Power</div>
                    <div class="result-value">
                        {result["power"]:,.0f} W
                    </div>
                    <div class="result-subtext">
                        Estimated mean power requirement
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated Energy</div>
                    <div class="result-value">
                        {result["energy"]:,.2f} kWh
                    </div>
                    <div class="result-subtext">
                        For {result["runtime"]:,.0f} minutes of operation
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated Carbon</div>
                    <div class="result-value">
                        {result["carbon"]:,.2f} kg
                    </div>
                    <div class="result-subtext">
                        Based on the {result["country"]} electricity grid
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # UNCERTAINTY
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="card">
                <div class="small-label">Prediction Reliability</div>
                <div style="font-size:1.15rem; font-weight:600; margin-top:0.35rem;">
                    90% Prediction Interval
                </div>
                <div style="font-size:1.4rem; margin-top:0.4rem;">
                    {result["lower_power"]:,.0f} W
                    &nbsp;—&nbsp;
                    {result["upper_power"]:,.0f} W
                </div>
                <div style="color:#777; font-size:0.82rem; margin-top:0.35rem;">
                    Based on conformal prediction uncertainty from the evaluated model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="best-model-card">
                <div class="best-model-label">Best Model</div>
                <div class="best-model-name">
                    Random Forest Regressor
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # CARBON SCENARIO NOTE
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="card">
                <div class="small-label">Carbon Scenario</div>
                <div style="margin-top:0.45rem; line-height:1.6;">
                    AERIS estimates <strong>{result["carbon"]:,.2f} kg CO₂e</strong>
                    using an electricity carbon intensity of
                    <strong>{result["carbon_intensity"]:,.2f} gCO₂e/kWh</strong>
                    for {result["country"]} in 2024.
                </div>
                <div style="color:#777; font-size:0.8rem; margin-top:0.5rem;">
                    This is a downstream scenario estimate. The model predicts power,
                    not carbon emissions directly.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RESEARCH SNAPSHOT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Research Snapshot</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Dataset</div>
                <div class="metric-large">6,480</div>
                <div style="color:#777;">
                    cleaned observations
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Training Sessions</div>
                <div class="metric-large">72</div>
                <div style="color:#777;">
                    AI workload sessions
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Prediction Target</div>
                <div class="metric-large">Power</div>
                <div style="color:#777;">
                    mean power consumption
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DATA & FEATURES
# ============================================================

def data_features_page():

    st.markdown(
        '<div class="section-title">Data & Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            AERIS uses workload configuration and hardware characteristics
            to predict mean power consumption. Identifiers, measurement
            metadata and target-derived power statistics are excluded from
            the predictive feature set to prevent leakage.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Clean Observations", "6,480")

    with c2:
        st.metric("Sessions", "72")

    with c3:
        st.metric("Workloads", "7")

    with c4:
        st.metric("Hardware Types", "3")

    st.markdown(
        '<div class="section-title">Predictive Features</div>',
        unsafe_allow_html=True
    )

    features = pd.DataFrame({
        "Feature": [
            "workload",
            "hardware",
            "num_gpus",
            "batch_size",
            "model_size",
            "image_size",
            "sequence_length",
            "layer_size",
            "num_layers",
            "embedding_dim",
            "hidden_units",
            "num_filters",
            "optimizer",
            "input_type",
            "parallelization"
        ],
        "Role": [
            "AI workload category",
            "GPU hardware",
            "Number of GPUs",
            "Training batch size",
            "Model size",
            "Image resolution",
            "Sequence length",
            "Layer size",
            "Number of layers",
            "Embedding dimension",
            "Hidden units",
            "Number of filters",
            "Optimization algorithm",
            "Input representation",
            "Distributed parallelization"
        ]
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Workload Distribution</div>',
        unsafe_allow_html=True
    )

    workload_counts = pd.DataFrame({
        "Workload": [
            "LLM",
            "Image Generation",
            "Feature Forecasting",
            "Image Classification",
            "Reinforcement Learning",
            "Image Captioning",
            "Text Generation"
        ],
        "Observations": [
            2160,
            1620,
            1080,
            1080,
            990,
            810,
            810
        ]
    })

    fig = px.bar(
        workload_counts,
        x="Workload",
        y="Observations"
    )

    fig.update_layout(
        template="simple_white",
        height=450,
        margin=dict(l=20, r=20, t=30, b=80)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            Numerical variables are median-imputed with missingness indicators.
            Categorical variables use a structural "Not Applicable" category
            where a parameter does not belong to a particular workload, followed
            by one-hot encoding. The neural network pipeline additionally applies
            standard scaling to numerical predictors.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL RESULTS
# ============================================================

def model_results_page():

    st.markdown(
        '<div class="section-title">Model Results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Three supervised regression approaches were evaluated using the
            same configuration-based predictors: a linear baseline, Random
            Forest and a deep learning MLP. The final evaluation uses a
            session-grouped train-test split so that observations from the
            same training session do not appear in both sets.
        </div>
        """,
        unsafe_allow_html=True
    )

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest Regressor",
            "MLP Deep Learning"
        ],
        "MAE (W)": [
            564.0397,
            519.9674,
            597.5449
        ],
        "RMSE (W)": [
            872.1531,
            797.7397,
            941.0382
        ],
        "R²": [
            0.8172,
            0.8470,
            0.7872
        ]
    })

    st.markdown(
        '<div class="section-title">Test Set Comparison</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        results.style.format({
            "MAE (W)": "{:.2f}",
            "RMSE (W)": "{:.2f}",
            "R²": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    best = results.loc[results["R²"].idxmax()]

    st.markdown(
        f"""
        <div class="best-model-card">
            <div class="best-model-label">Best Performing Model</div>
            <div class="best-model-name">
                {best["Model"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MODEL COMPARISON IMAGE
    # --------------------------------------------------------

    comparison_path = ASSETS_DIR / "model_comparison.png"

    if comparison_path.exists():
        st.image(
            str(comparison_path),
            use_container_width=True
        )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Grouped 5-Fold Cross-Validation</div>',
        unsafe_allow_html=True
    )

    cv_results = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "MAE (W)": [
            480.96,
            425.75,
            394.60,
            323.29,
            317.87
        ],
        "RMSE (W)": [
            885.60,
            703.10,
            726.70,
            613.79,
            621.86
        ],
        "R²": [
            0.8393,
            0.8719,
            0.8433,
            0.8815,
            0.9032
        ]
    })

    st.dataframe(
        cv_results.style.format({
            "MAE (W)": "{:.2f}",
            "RMSE (W)": "{:.2f}",
            "R²": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Mean CV MAE</div>
                <div class="metric-large">388.49 W</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Mean CV RMSE</div>
                <div class="metric-large">710.21 W</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Mean CV R²</div>
                <div class="metric-large">0.868</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    actual_predicted = ASSETS_DIR / "actual_vs_predicted.png"

    if actual_predicted.exists():

        st.markdown(
            '<div class="section-title">Prediction Performance</div>',
            unsafe_allow_html=True
        )

        st.image(
            str(actual_predicted),
            use_container_width=True
        )

    residual_plot = ASSETS_DIR / "residual_plot.png"

    if residual_plot.exists():

        st.markdown(
            '<div class="section-title">Residual Diagnostics</div>',
            unsafe_allow_html=True
        )

        st.image(
            str(residual_plot),
            use_container_width=True
        )


# ============================================================
# INTERPRETATION
# ============================================================

def interpretation_page():

    st.markdown(
        '<div class="section-title">Interpretation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Model interpretation identifies which predictors contribute most
            strongly to prediction. These are predictive relationships within
            the dataset and should not be interpreted as causal effects.
        </div>
        """,
        unsafe_allow_html=True
    )

    shap_path = ASSETS_DIR / "shap_importance.png"

    if shap_path.exists():

        st.markdown(
            '<div class="section-title">SHAP Feature Importance</div>',
            unsafe_allow_html=True
        )

        st.image(
            str(shap_path),
            use_container_width=True
        )

    else:

        shap_features = pd.DataFrame({
            "Feature": [
                "Hardware",
                "Number of GPUs",
                "Image Size",
                "Batch Size",
                "Workload",
                "Sequence Length",
                "Model Size"
            ],
            "Mean Absolute SHAP": [
                946.19,
                921.11,
                334.33,
                113.56,
                47.64,
                29.54,
                25.75
            ]
        })

        fig = px.bar(
            shap_features.sort_values("Mean Absolute SHAP"),
            x="Mean Absolute SHAP",
            y="Feature",
            orientation="h"
        )

        fig.update_layout(
            template="simple_white",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">Key Finding</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            Hardware and GPU count are the dominant predictive factors in the
            Random Forest model. This reflects the substantial difference in
            power regimes between the RTX3060, H100 and B200 systems represented
            in the dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Important Limitation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            The main prediction model uses workload configuration and hardware
            characteristics available before or at workload setup. It does not
            use real-time GPU utilisation, temperature or power telemetry.
            Therefore, it predicts the expected power regime rather than
            fine-grained fluctuations within an individual training session.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Carbon Interpretation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            Carbon emissions are estimated downstream from predicted energy
            consumption and annual electricity-grid carbon intensity. The same
            computational energy requirement can therefore produce different
            emissions estimates under different electricity-grid scenarios.
            These country-level values should not be interpreted as
            real-time carbon-aware scheduling data.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ABOUT
# ============================================================

def about_page():

    st.markdown(
        '<div class="section-title">About AERIS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            AERIS — AI Energy Requirement Intelligence System — is the
            deployment interface for a predictive modelling study examining
            the relationship between AI workload configurations, computing
            hardware and training power consumption.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="small-label">Research Question</div>
            <p>
                How accurately can machine learning and deep learning models
                predict the power consumption of AI training workloads from
                workload configuration and hardware characteristics?
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="small-label">Prediction Pipeline</div>
            <p>
                Workload configuration → Random Forest prediction →
                estimated power → estimated energy → electricity-grid
                carbon intensity → estimated carbon footprint.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="small-label">Dataset</div>
            <p>
                The study uses high-resolution measurements of AI data-centre
                training workloads across RTX3060, H100 and B200 hardware
                configurations. After removal of exact duplicate observations,
                the modelling dataset contains 6,480 observations across
                72 training sessions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="small-label">Methodology</div>
            <p>
                Linear Regression provides the baseline. Random Forest provides
                the conventional machine-learning model, while an MLP provides
                the deep-learning comparison. Evaluation uses MAE, MSE, RMSE
                and R² with session-grouped validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="small-label">Uncertainty</div>
            <p>
                A 90% conformal prediction interval is used to communicate
                uncertainty around individual power predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="small-label">Project Scope</div>
            <p>
                AERIS is a prediction and scenario-analysis system. It does
                not directly predict carbon emissions, perform causal analysis,
                or function as a real-time carbon-aware scheduling system.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ROUTING
# ============================================================

if st.session_state.page == "Overview":
    overview_page()

elif st.session_state.page == "Data & Features":
    data_features_page()

elif st.session_state.page == "Model Results":
    model_results_page()

elif st.session_state.page == "Interpretation":
    interpretation_page()

elif st.session_state.page == "About":
    about_page()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AERIS · AI Energy Requirement Intelligence System
        <br>
        Predictive modelling for AI training power and downstream carbon estimation
    </div>
    """,
    unsafe_allow_html=True
)
