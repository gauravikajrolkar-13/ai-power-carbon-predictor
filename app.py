import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AERIS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ai_power_random_forest_model.pkl"
CARBON_PATH = BASE_DIR / "electricity_carbon_intensity_2024.csv"
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---------------------------------------------------------
   GLOBAL TYPOGRAPHY
--------------------------------------------------------- */

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #fafafa;
    color: #20252b;
}

.block-container {
    max-width: 1350px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}


/* ---------------------------------------------------------
   HIDE STREAMLIT DEFAULT ELEMENTS
--------------------------------------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ---------------------------------------------------------
   MAIN TITLE
--------------------------------------------------------- */

.aeris-header {
    text-align: center;
    padding: 1.2rem 0 1.5rem 0;
}

.aeris-title {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 600;
    letter-spacing: -1.5px;
    color: #171b20;
    line-height: 1;
}

.aeris-full-name {
    margin-top: 0.65rem;
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: #555e68;
}

.aeris-tagline {
    margin: 0.6rem auto 0 auto;
    max-width: 760px;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #7a828b;
}


/* ---------------------------------------------------------
   NAVIGATION
--------------------------------------------------------- */

.nav-container {
    border-top: 1px solid #e6e8eb;
    border-bottom: 1px solid #e6e8eb;
    padding: 0.75rem 0;
    margin-bottom: 2.5rem;
}

div[data-testid="stHorizontalBlock"] {
    align-items: center;
}


/* ---------------------------------------------------------
   SECTION TITLES
--------------------------------------------------------- */

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 500;
    color: #20252b;
    margin-bottom: 0.35rem;
}

.section-description {
    color: #747c85;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 1.7rem;
}


/* ---------------------------------------------------------
   CARDS
--------------------------------------------------------- */

.card {
    background: #ffffff;
    border: 1px solid #e5e7ea;
    border-radius: 12px;
    padding: 1.35rem;
}

.card-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #69717b;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.card-value {
    font-size: 1.7rem;
    font-weight: 600;
    color: #20252b;
    margin-top: 0.4rem;
}

.card-note {
    font-size: 0.76rem;
    color: #9299a1;
    margin-top: 0.25rem;
}


/* ---------------------------------------------------------
   PREDICTION PANEL
--------------------------------------------------------- */

.prediction-panel {
    background: #ffffff;
    border: 1px solid #e1e4e8;
    border-radius: 15px;
    padding: 1.7rem;
}

.prediction-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.55rem;
    color: #20252b;
    margin-bottom: 0.3rem;
}

.prediction-description {
    font-size: 0.84rem;
    color: #7b838d;
    margin-bottom: 1.4rem;
}


/* ---------------------------------------------------------
   RESULT CARDS
--------------------------------------------------------- */

.result-card {
    background: #20252b;
    color: white;
    border-radius: 13px;
    padding: 1.45rem;
    min-height: 145px;
}

.result-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #aeb5bc;
}

.result-value {
    font-size: 2.2rem;
    font-weight: 600;
    margin-top: 0.35rem;
}

.result-unit {
    font-size: 0.78rem;
    color: #b9c0c7;
    margin-top: 0.15rem;
}


/* ---------------------------------------------------------
   INFO BOX
--------------------------------------------------------- */

.info-box {
    background: #f3f5f7;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    color: #626b75;
    font-size: 0.83rem;
    line-height: 1.65;
}


/* ---------------------------------------------------------
   DIVIDER
--------------------------------------------------------- */

.divider {
    height: 1px;
    background: #e5e7ea;
    margin: 2.4rem 0;
}


/* ---------------------------------------------------------
   BUTTON
--------------------------------------------------------- */

.stButton > button {
    width: 100%;
    background: #20252b;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

.stButton > button:hover {
    background: #353c44;
}


/* ---------------------------------------------------------
   INPUTS
--------------------------------------------------------- */

.stSelectbox label,
.stNumberInput label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #454d56 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 7px;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer {
    border-top: 1px solid #e5e7ea;
    margin-top: 4rem;
    padding-top: 1.1rem;
    text-align: center;
    font-size: 0.74rem;
    color: #9aa1a8;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# LOAD CARBON DATA
# ============================================================

@st.cache_data
def load_carbon_data():

    if CARBON_PATH.exists():
        return pd.read_csv(CARBON_PATH)

    return pd.DataFrame({
        "Entity": [
            "France",
            "Germany",
            "United States",
            "India"
        ],
        "Year": [2024, 2024, 2024, 2024],
        "Carbon intensity": [
            40.48,
            336.38,
            383.78,
            705.40
        ]
    })


carbon_data = load_carbon_data()


# ============================================================
# RESEARCH RESULTS
# ============================================================

MODEL_RESULTS = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
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


CV_RESULTS = pd.DataFrame({
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


WORKLOAD_RESULTS = pd.DataFrame({
    "Workload": [
        "LLM",
        "Image Generation",
        "Feature Forecasting",
        "Image Classification",
        "Image Captioning",
        "Text Generation",
        "Reinforcement Learning"
    ],
    "Mean Power (W)": [
        4214.483,
        2579.815,
        74.093,
        61.313,
        49.529,
        23.963,
        21.131
    ]
})


HARDWARE_RESULTS = pd.DataFrame({
    "Hardware": [
        "B200",
        "H100",
        "RTX3060"
    ],
    "Mean Power (W)": [
        4034.817,
        2963.815,
        47.554
    ]
})


SHAP_RESULTS = pd.DataFrame({
    "Feature": [
        "hardware_RTX3060",
        "num_gpus",
        "image_size",
        "hardware_H100",
        "batch_size",
        "hardware_B200",
        "parallelization_ZeRO-3",
        "workload_Image Generation",
        "missingindicator_sequence_length",
        "workload_LLM",
        "sequence_length",
        "missingindicator_image_size",
        "model_size",
        "parallelization_Not_Applicable",
        "missingindicator_model_size"
    ],
    "Mean Absolute SHAP": [
        946.187484,
        921.105685,
        334.330740,
        168.774638,
        113.562424,
        107.405175,
        49.568367,
        47.636713,
        36.118500,
        31.142442,
        29.538025,
        27.717738,
        25.748512,
        24.359839,
        24.006272
    ]
})


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_DEFINITIONS = pd.DataFrame({
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
    "Description": [
        "AI training workload type",
        "GPU hardware platform",
        "Number of GPUs",
        "Training batch size",
        "Model parameter scale",
        "Input image resolution",
        "Input sequence or cutoff length",
        "Neural network layer size",
        "Number of model layers",
        "Embedding dimensionality",
        "Number of hidden units",
        "Number of convolutional filters",
        "Training optimizer",
        "Input representation",
        "Distributed training strategy"
    ]
})


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="aeris-header">

    <div class="aeris-title">
        AERIS
    </div>

    <div class="aeris-full-name">
        AI Energy Requirement Intelligence System
    </div>

    <div class="aeris-tagline">
        Predicting AI training power consumption from workload and hardware
        characteristics, with downstream energy and carbon-footprint estimation.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"


st.markdown('<div class="nav-container">', unsafe_allow_html=True)

nav_columns = st.columns(5)

navigation = [
    ("Overview", nav_columns[0]),
    ("Data & Features", nav_columns[1]),
    ("Model Results", nav_columns[2]),
    ("Interpretation", nav_columns[3]),
    ("About", nav_columns[4])
]

for name, column in navigation:

    with column:

        if st.button(
            name,
            key=f"nav_{name}",
            use_container_width=True
        ):
            st.session_state.page = name
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page


# ============================================================
# OVERVIEW / PREDICTION
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">AI Workload Estimator</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Configure an AI training workload and estimate its power, energy '
        'requirements and associated carbon emissions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="prediction-panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="prediction-heading">Workload Configuration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="prediction-description">'
        'Select a configuration represented in the training dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        workload = st.selectbox(
            "AI Workload",
            [
                "Feature Forecasting",
                "Reinforcement Learning",
                "Image Classification",
                "Text Generation",
                "Image Captioning",
                "Image Generation",
                "LLM"
            ]
        )

    with col2:

        hardware = st.selectbox(
            "Hardware",
            ["RTX3060", "H100", "B200"]
        )

    with col3:

        num_gpus = 1 if hardware == "RTX3060" else 8

        st.number_input(
            "Number of GPUs",
            value=num_gpus,
            disabled=True
        )

    batch_options = {
        "Feature Forecasting": [50, 100, 150],
        "Reinforcement Learning": [150, 250, 350],
        "Image Classification": [750, 1500, 2250],
        "Text Generation": [32, 128, 512],
        "Image Captioning": [128, 256, 1024],
        "Image Generation": [128, 256, 512],
        "LLM": [2, 16, 32]
    }

    batch_size = st.selectbox(
        "Batch Size",
        batch_options[workload]
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

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

    with col1:

        if workload == "Feature Forecasting":

            model_size = st.selectbox(
                "Model Size",
                [474000, 1600000, 3600000],
                format_func=lambda x: f"{x / 1e6:g}M parameters"
            )

        elif workload == "Image Generation":

            model_size = st.selectbox(
                "Model Size",
                [107000000, 430000000, 1700000000],
                format_func=lambda x: f"{x / 1e6:g}M parameters"
            )

        elif workload == "LLM":

            model_size = st.selectbox(
                "Model Size",
                [1000000000, 3000000000, 8000000000],
                format_func=lambda x: f"{x / 1e9:g}B parameters"
            )

        elif workload == "Reinforcement Learning":

            layer_size = st.selectbox(
                "Layer Size",
                [256, 512, 1024]
            )

        elif workload == "Image Captioning":

            layer_size = st.selectbox(
                "Layer Size",
                [512, 1024, 2048]
            )

        elif workload == "Text Generation":

            sequence_length = st.selectbox(
                "Sequence Length",
                [100, 250, 500]
            )

        elif workload == "Image Classification":

            image_size = st.selectbox(
                "Image Size",
                [112, 224, 280]
            )

    with col2:

        if workload == "Feature Forecasting":

            sequence_length = st.selectbox(
                "Sequence Length",
                [96, 192, 672]
            )

        elif workload == "Reinforcement Learning":

            sequence_length = st.selectbox(
                "Sequence Length",
                [150, 250, 350]
            )

        elif workload == "Image Classification":

            num_filters = st.selectbox(
                "Number of Filters",
                [8, 16, 32]
            )

        elif workload == "Text Generation":

            embedding_dim = st.selectbox(
                "Embedding Dimension",
                [100, 300, 1000]
            )

        elif workload == "Image Captioning":

            embedding_dim = st.selectbox(
                "Embedding Dimension",
                [256, 512, 1024]
            )

        elif workload == "Image Generation":

            image_size = st.selectbox(
                "Image Size",
                [32, 64, 128]
            )

        elif workload == "LLM":

            sequence_length = st.selectbox(
                "Sequence Length",
                [1024, 2048, 4096]
            )

    with col3:

        if workload == "Feature Forecasting":

            num_layers = st.selectbox(
                "Number of Layers",
                [6, 8, 10]
            )

        elif workload == "Reinforcement Learning":

            input_type = st.selectbox(
                "Input Type",
                ["Feature", "Sequence"]
            )

        elif workload == "Image Classification":

            optimizer = st.selectbox(
                "Optimizer",
                ["Adam", "SGD", "RMS"]
            )

        elif workload == "LLM":

            parallelization = st.selectbox(
                "Parallelization",
                ["ZeRO-1", "ZeRO-2", "ZeRO-3"]
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:

        duration_minutes = st.number_input(
            "Estimated Runtime (minutes)",
            min_value=0.1,
            max_value=1440.0,
            value=15.0,
            step=1.0
        )

    with col2:

        st.markdown(
            '<div class="info-box">'
            'AERIS predicts mean power for a 10-second measurement window. '
            'Runtime is used only to derive energy consumption and carbon '
            'emissions. The calculation assumes predicted mean power remains '
            'constant throughout the selected runtime.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    run_prediction = st.button(
        "Run AERIS",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if run_prediction:

        input_data = pd.DataFrame([{
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

        try:

            predicted_power = float(
                model.predict(input_data)[0]
            )

            energy_kwh = (
                predicted_power
                * duration_minutes
                / 60
                / 1000
            )

            # Default carbon scenario
            country = "India"

            country_rows = carbon_data[
                carbon_data["Entity"] == country
            ]

            if len(country_rows) > 0:

                carbon_intensity = float(
                    country_rows.iloc[0]["Carbon intensity"]
                )

            else:

                carbon_intensity = 705.40

            emissions = (
                energy_kwh
                * carbon_intensity
                / 1000
            )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">AERIS Estimation</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Power is predicted by the Random Forest model. Energy and '
                'carbon are calculated from the predicted power.'
                '</div>',
                unsafe_allow_html=True
            )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Predicted Power
                        </div>
                        <div class="result-value">
                            {predicted_power:,.1f}
                        </div>
                        <div class="result-unit">
                            watts · mean power
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r2:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Estimated Energy
                        </div>
                        <div class="result-value">
                            {energy_kwh:,.3f}
                        </div>
                        <div class="result-unit">
                            kWh · {duration_minutes:g} minutes
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r3:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Estimated Carbon
                        </div>
                        <div class="result-value">
                            {emissions:,.3f}
                        </div>
                        <div class="result-unit">
                            kg CO₂e · {country}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ------------------------------------------------
            # CARBON SCENARIO
            # ------------------------------------------------

            st.markdown("### Carbon Scenario")

            carbon_col1, carbon_col2 = st.columns(2)

            with carbon_col1:

                available_countries = sorted(
                    carbon_data["Entity"].dropna().unique().tolist()
                )

                selected_country = st.selectbox(
                    "Electricity Grid",
                    available_countries,
                    index=(
                        available_countries.index("India")
                        if "India" in available_countries
                        else 0
                    )
                )

            with carbon_col2:

                selected_rows = carbon_data[
                    carbon_data["Entity"] == selected_country
                ]

                if len(selected_rows) > 0:

                    selected_intensity = float(
                        selected_rows.iloc[0]["Carbon intensity"]
                    )

                else:

                    selected_intensity = 705.40

                selected_emissions = (
                    energy_kwh
                    * selected_intensity
                    / 1000
                )

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">
                            Estimated CO₂e
                        </div>
                        <div class="card-value">
                            {selected_emissions:.3f} kg
                        </div>
                        <div class="card-note">
                            {selected_intensity:.2f} gCO₂e/kWh
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.caption(
                "Carbon intensity values represent annual 2024 electricity "
                "carbon intensity. They are scenario values, not real-time "
                "grid intensity."
            )

        except Exception as error:

            st.error(
                "AERIS could not generate a prediction for this configuration."
            )

            st.exception(error)

    # --------------------------------------------------------
    # RESEARCH SNAPSHOT
    # --------------------------------------------------------

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Research Snapshot</div>',
        unsafe_allow_html=True
    )

    snapshot1, snapshot2, snapshot3, snapshot4 = st.columns(4)

    with snapshot1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Observations</div>
            <div class="card-value">6,480</div>
            <div class="card-note">Clean observations</div>
        </div>
        """, unsafe_allow_html=True)

    with snapshot2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Sessions</div>
            <div class="card-value">72</div>
            <div class="card-note">Training sessions</div>
        </div>
        """, unsafe_allow_html=True)

    with snapshot3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Best R²</div>
            <div class="card-value">0.847</div>
            <div class="card-note">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    with snapshot4:
        st.markdown("""
        <div class="card">
            <div class="card-title">CV R²</div>
            <div class="card-value">0.868</div>
            <div class="card-note">5-fold grouped CV</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# DATA & FEATURES
# ============================================================

elif page == "Data & Features":

    st.markdown(
        '<div class="section-title">Data & Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'The dataset, predictor structure and exploratory findings behind AERIS.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Original rows", "8,550")

    with c2:
        st.metric("Clean rows", "6,480")

    with c3:
        st.metric("Sessions", "72")

    with c4:
        st.metric("Workloads", "7")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Dataset")

    st.write(
        "The dataset contains high-resolution AI training workload measurements "
        "from RTX3060, H100 and B200 GPU systems. Each session contains 90 "
        "approximately ten-second windows representing a 15-minute training session."
    )

    st.markdown("### Predictor Variables")

    st.dataframe(
        FEATURE_DEFINITIONS,
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Mean Power by Workload")

        fig = px.bar(
            WORKLOAD_RESULTS.sort_values(
                "Mean Power (W)",
                ascending=True
            ),
            x="Mean Power (W)",
            y="Workload",
            orientation="h",
            template="simple_white"
        )

        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Mean Power (W)",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.markdown("### Mean Power by Hardware")

        fig = px.bar(
            HARDWARE_RESULTS.sort_values(
                "Mean Power (W)",
                ascending=True
            ),
            x="Mean Power (W)",
            y="Hardware",
            orientation="h",
            template="simple_white"
        )

        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Mean Power (W)",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Preprocessing")

    preprocessing = pd.DataFrame({
        "Stage": [
            "Duplicate handling",
            "Numerical missing values",
            "Categorical missing values",
            "Categorical encoding",
            "Scaling",
            "Data splitting"
        ],
        "Method": [
            "Exact duplicate removal",
            "Median imputation + indicators",
            "Not Applicable category",
            "One-hot encoding",
            "StandardScaler for MLP",
            "GroupShuffleSplit by session"
        ]
    })

    st.dataframe(
        preprocessing,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL RESULTS
# ============================================================

elif page == "Model Results":

    st.markdown(
        '<div class="section-title">Model Results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Comparative evaluation of conventional machine learning and deep learning.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Performance Comparison")

    st.dataframe(
        MODEL_RESULTS.style.format({
            "MAE (W)": "{:.2f}",
            "RMSE (W)": "{:.2f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="MAE",
                x=MODEL_RESULTS["Model"],
                y=MODEL_RESULTS["MAE (W)"]
            )
        )

        fig.add_trace(
            go.Bar(
                name="RMSE",
                x=MODEL_RESULTS["Model"],
                y=MODEL_RESULTS["RMSE (W)"]
            )
        )

        fig.update_layout(
            barmode="group",
            template="simple_white",
            height=400,
            yaxis_title="Error (W)",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            MODEL_RESULTS,
            x="Model",
            y="R²",
            template="simple_white"
        )

        fig.update_layout(
            height=400,
            yaxis_title="R²",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Grouped Cross-Validation")

    st.write(
        "The Random Forest was evaluated using five grouped folds, with "
        "training sessions kept separate between folds."
    )

    st.dataframe(
        CV_RESULTS.style.format({
            "MAE (W)": "{:.2f}",
            "RMSE (W)": "{:.2f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Mean CV MAE", "388.49 W")

    with c2:
        st.metric("Mean CV RMSE", "710.21 W")

    with c3:
        st.metric("Mean CV R²", "0.8678")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Model Selection")

    st.write(
        "Random Forest achieved the strongest overall test-set performance. "
        "It outperformed both Linear Regression and the MLP deep learning model "
        "on MAE, RMSE and R²."
    )

    st.info(
        "The MLP result is an important finding rather than a failure. "
        "For this structured tabular dataset, the more complex deep learning "
        "model did not outperform the conventional ensemble model."
    )


# ============================================================
# INTERPRETATION
# ============================================================

elif page == "Interpretation":

    st.markdown(
        '<div class="section-title">Interpretation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Understanding the predictors that contribute most strongly to AERIS predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### SHAP Feature Importance")

    shap_image = ASSETS_DIR / "shap_importance.png"

    if shap_image.exists():

        st.image(
            str(shap_image),
            use_container_width=True
        )

    else:

        top_shap = SHAP_RESULTS.head(10).sort_values(
            "Mean Absolute SHAP",
            ascending=True
        )

        fig = px.bar(
            top_shap,
            x="Mean Absolute SHAP",
            y="Feature",
            orientation="h",
            template="simple_white"
        )

        fig.update_layout(
            height=450,
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Main Findings")

    st.write(
        "Hardware type and number of GPUs are the two dominant predictive "
        "features. Image size also has substantial predictive influence, "
        "followed by batch size and several workload and configuration variables."
    )

    st.markdown(
        '<div class="info-box">'
        '<strong>Interpretation caution:</strong> Feature importance represents '
        'predictive usefulness within this dataset. It should not be interpreted '
        'as proof that changing a feature independently will cause a particular '
        'change in power consumption.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### SHAP Results")

    st.dataframe(
        SHAP_RESULTS.style.format({
            "Mean Absolute SHAP": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="section-title">About AERIS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Research methodology, model scope and limitations.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>AERIS — AI Energy Requirement Intelligence System</h3>

    <p>
    AERIS is a predictive modelling framework designed to estimate the
    electricity requirements of AI training workloads from workload
    configuration and hardware characteristics.
    </p>

    <p>
    The project compares conventional machine learning with deep learning,
    evaluates predictive performance using grouped validation, and applies
    SHAP-based interpretation to understand the model's predictions.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Final Model")

    final_model = pd.DataFrame({
        "Property": [
            "Model",
            "Target",
            "Test MAE",
            "Test RMSE",
            "Test R²",
            "Mean CV MAE",
            "Mean CV RMSE",
            "Mean CV R²"
        ],
        "Value": [
            "Random Forest Regressor",
            "mean_power_W",
            "519.97 W",
            "797.74 W",
            "0.8470",
            "388.49 W",
            "710.21 W",
            "0.8678"
        ]
    })

    st.dataframe(
        final_model,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Research Workflow")

    workflow = pd.DataFrame({
        "Stage": [
            "Problem Definition",
            "Data Understanding",
            "Preprocessing",
            "Model Development",
            "Validation",
            "Interpretation",
            "Carbon Analysis"
        ],
        "Approach": [
            "Predict mean AI training power",
            "EDA and statistical analysis",
            "Imputation and encoding",
            "Linear Regression, Random Forest, MLP",
            "Grouped train-test + GroupKFold",
            "SHAP analysis",
            "Electricity carbon-intensity scenarios"
        ]
    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Limitations")

    st.write(
        "AERIS predicts mean power rather than instantaneous power. "
        "Because the main predictors describe workload configuration and "
        "hardware, the model cannot capture all temporal runtime behaviour. "
        "The dataset also does not contain perfectly balanced combinations "
        "of every workload and hardware platform."
    )

    st.write(
        "Carbon estimates are downstream scenario calculations. They depend "
        "on the selected electricity carbon intensity and should not be "
        "interpreted as real-time grid emissions."
    )

    st.markdown("### Data Source")

    st.write(
        "Elsayed, Ahmed Elaziz; Al-Obaidi, Abdullah Azhar; Farag, Hany E.Z. "
        "“Characterization of high-resolution AI data center training workloads "
        "on single and multiple GPU nodes.” Scientific Data, 2026."
    )

    st.write(
        "Dataset DOI: 10.6084/m9.figshare.31654879"
    )

    st.write(
        "Electricity carbon-intensity data: Our World in Data, annual "
        "electricity carbon intensity dataset."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    AERIS · AI Energy Requirement Intelligence System · Research Project
</div>
""", unsafe_allow_html=True)
