import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
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
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f8f9fb;
    color: #17202a;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}


/* ---------- HEADER ---------- */

.aeris-header {
    padding: 1.2rem 0 1.8rem 0;
    border-bottom: 1px solid #e4e7eb;
    margin-bottom: 1.2rem;
}

.aeris-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 600;
    letter-spacing: -1px;
    margin: 0;
    color: #17202a;
}

.aeris-subtitle {
    font-size: 1.05rem;
    color: #667085;
    margin-top: 0.35rem;
}

.aeris-description {
    max-width: 900px;
    color: #667085;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-top: 0.8rem;
}


/* ---------- NAVIGATION ---------- */

div[data-testid="stRadio"] > div {
    gap: 0.35rem;
}

div[data-testid="stRadio"] label {
    background: transparent;
    border-radius: 7px;
    padding: 0.45rem 0.75rem;
    color: #667085;
    font-size: 0.88rem;
    border: 1px solid transparent;
}

div[data-testid="stRadio"] label:hover {
    background: #eef1f4;
    color: #17202a;
}

div[data-testid="stRadio"] label[data-checked="true"] {
    background: #17202a;
    color: white;
}


/* ---------- SECTION HEADINGS ---------- */

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    margin-top: 1.5rem;
    margin-bottom: 0.35rem;
    color: #17202a;
}

.section-subtitle {
    color: #667085;
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
}


/* ---------- METRIC CARDS ---------- */

.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem;
    min-height: 120px;
}

.metric-label {
    color: #667085;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 600;
    margin-top: 0.45rem;
    color: #17202a;
}

.metric-note {
    color: #98a2b3;
    font-size: 0.76rem;
    margin-top: 0.3rem;
}


/* ---------- INFO CARDS ---------- */

.info-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.35rem;
    height: 100%;
}

.info-card h4 {
    margin-top: 0;
    color: #17202a;
}

.info-card p {
    color: #667085;
    line-height: 1.65;
    font-size: 0.9rem;
}


/* ---------- RESULT CARDS ---------- */

.result-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.3rem;
}

.result-model {
    color: #667085;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.result-score {
    font-size: 2rem;
    font-weight: 600;
    margin-top: 0.25rem;
}

.result-best {
    color: #177245;
    font-size: 0.78rem;
    margin-top: 0.25rem;
}


/* ---------- DIVIDERS ---------- */

.soft-divider {
    height: 1px;
    background: #e5e7eb;
    margin: 2rem 0;
}


/* ---------- PREDICTION ---------- */

.prediction-result {
    background: #17202a;
    color: white;
    border-radius: 14px;
    padding: 1.6rem;
}

.prediction-label {
    color: #aeb7c2;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.prediction-value {
    font-size: 2.5rem;
    font-weight: 600;
    margin-top: 0.3rem;
}

.prediction-unit {
    color: #c4cad1;
    font-size: 0.85rem;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    border-radius: 8px;
    border: none;
    background: #17202a;
    color: white;
    padding: 0.65rem 1.2rem;
    font-weight: 500;
}

.stButton > button:hover {
    background: #303b47;
}


/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
}


/* ---------- FOOTER ---------- */

.footer {
    border-top: 1px solid #e5e7eb;
    margin-top: 4rem;
    padding-top: 1.2rem;
    color: #98a2b3;
    font-size: 0.78rem;
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
        data = pd.read_csv(CARBON_PATH)
        return data

    # Fallback values used in the research analysis
    return pd.DataFrame({
        "Entity": ["France", "Germany", "United States", "India"],
        "Year": [2024, 2024, 2024, 2024],
        "Carbon intensity": [40.48, 336.38, 383.78, 705.40]
    })


carbon_data = load_carbon_data()


# ============================================================
# PROJECT RESULTS
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
    "MAE (W)": [480.96, 425.75, 394.60, 323.29, 317.87],
    "RMSE (W)": [885.60, 703.10, 726.70, 613.79, 621.86],
    "R²": [0.8393, 0.8719, 0.8433, 0.8815, 0.9032]
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
        "missingindicator_model_size",
        "parallelization_ZeRO-1",
        "missingindicator_num_layers",
        "parallelization_ZeRO-2",
        "workload_Feature Forecasting",
        "num_layers"
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
        24.006272,
        4.424313,
        4.088312,
        2.507303,
        2.388078,
        1.849083
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
    "Role": ["Model predictor"] * 15,
    "Description": [
        "Type of AI training workload.",
        "GPU hardware platform.",
        "Number of GPUs used.",
        "Number of samples processed in a batch.",
        "Approximate model size / parameter scale.",
        "Input image resolution.",
        "Input sequence or cutoff length.",
        "Neural network layer size.",
        "Number of model layers.",
        "Embedding dimensionality.",
        "Number of hidden units.",
        "Number of convolutional filters.",
        "Training optimizer.",
        "Input representation type.",
        "Distributed training parallelization strategy."
    ]
})


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="aeris-header">
    <div class="aeris-name">AERIS</div>
    <div class="aeris-subtitle">
        AI Energy Requirement Intelligence System
    </div>
    <div class="aeris-description">
        Predicting AI training power consumption from workload configuration
        and hardware characteristics using machine learning and deep learning,
        with downstream carbon-footprint scenario analysis.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

pages = [
    "Overview",
    "Data & Features",
    "Model Results",
    "Interpretation",
    "Prediction",
    "Carbon Analysis",
    "About"
]

page = st.radio(
    "Navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Research Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A predictive framework for estimating the power requirements of AI training workloads.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Clean observations</div>
            <div class="metric-value">6,480</div>
            <div class="metric-note">After duplicate removal</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Training sessions</div>
            <div class="metric-value">72</div>
            <div class="metric-note">Grouped validation units</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best R²</div>
            <div class="metric-value">0.847</div>
            <div class="metric-note">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test-set energy</div>
            <div class="metric-value">7.10 kWh</div>
            <div class="metric-note">1,350 test windows</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("""
        <div class="info-card">
            <h4>Research Question</h4>
            <p>
            How accurately can machine learning and deep learning models
            predict the power consumption of AI training workloads from
            workload configuration and hardware characteristics?
            </p>

            <p>
            The predicted power is subsequently converted into estimated
            energy consumption and evaluated under different electricity
            carbon-intensity scenarios.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="info-card">
            <h4>Prediction Pipeline</h4>
            <p>
            Workload configuration + hardware
            → predictive model
            → mean power (W)
            → energy (kWh)
            → electricity carbon intensity
            → estimated CO₂e.
            </p>

            <p>
            Carbon emissions are a downstream scenario calculation rather
            than the direct prediction target.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Key Findings</div>',
        unsafe_allow_html=True
    )

    findings = [
        "Random Forest achieved the strongest overall performance among the three evaluated models.",
        "Hardware type and number of GPUs were the dominant predictive features.",
        "The model captured major differences between low-power single-GPU workloads and high-power multi-GPU workloads.",
        "Prediction errors were substantially larger for high-power LLM and Image Generation workloads.",
        "The configuration-only model cannot capture fine-grained temporal changes within an individual training session.",
        "The same predicted energy demand can correspond to substantially different estimated emissions depending on electricity carbon intensity."
    ]

    for finding in findings:
        st.markdown(
            f'<div style="margin:0.65rem 0;color:#475467;font-size:0.92rem;">'
            f'• {finding}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# DATA & FEATURES
# ============================================================

elif page == "Data & Features":

    st.markdown(
        '<div class="section-title">Data & Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Structure, predictors, exploratory analysis and preprocessing decisions.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Original observations", "8,550")

    with c2:
        st.metric("Clean observations", "6,480")

    with c3:
        st.metric("Unique sessions", "72")

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Dataset")

    st.write(
        "The dataset contains high-resolution measurements of AI training "
        "workloads across RTX3060, H100 and B200 GPU platforms. Each training "
        "session contains 90 ten-second windows representing approximately "
        "15 minutes of activity."
    )

    dataset_info = pd.DataFrame({
        "Property": [
            "Target",
            "Predictor count",
            "Workloads",
            "Hardware platforms",
            "Observations after cleaning",
            "Sessions",
            "Window duration"
        ],
        "Value": [
            "mean_power_W",
            "15",
            "7",
            "3",
            "6,480",
            "72",
            "10 seconds"
        ]
    })

    st.dataframe(dataset_info, use_container_width=True, hide_index=True)

    st.markdown("### Predictor Dictionary")

    st.dataframe(
        FEATURE_DEFINITIONS,
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Power by Workload")

    fig = px.bar(
        WORKLOAD_RESULTS.sort_values("Mean Power (W)", ascending=True),
        x="Mean Power (W)",
        y="Workload",
        orientation="h",
        template="simple_white"
    )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis_title="Mean Power (W)",
        yaxis_title=""
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Power by Hardware")

    fig = px.bar(
        HARDWARE_RESULTS.sort_values("Mean Power (W)", ascending=True),
        x="Mean Power (W)",
        y="Hardware",
        orientation="h",
        template="simple_white"
    )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis_title="Mean Power (W)",
        yaxis_title=""
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Preprocessing")

    preprocessing = pd.DataFrame({
        "Step": [
            "Exact duplicates",
            "Numerical missing values",
            "Categorical missing values",
            "Categorical encoding",
            "Numerical scaling",
            "Train-test split"
        ],
        "Method": [
            "Removed",
            "Median imputation + missing indicators",
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

    st.info(
        "The model uses configuration-level predictors only. Runtime power, "
        "GPU utilisation and other measured power variables are excluded "
        "from the predictive feature set to avoid target leakage."
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
        '<div class="section-subtitle">'
        'Comparative evaluation of conventional machine learning and deep learning.'
        '</div>',
        unsafe_allow_html=True
    )

    best_model = MODEL_RESULTS.loc[
        MODEL_RESULTS["R²"].idxmax()
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="result-card">
            <div class="result-model">Best Model</div>
            <div class="result-score">Random Forest</div>
            <div class="result-best">Highest test-set R²</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="result-card">
            <div class="result-model">R²</div>
            <div class="result-score">0.8470</div>
            <div class="result-best">Explained variance</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="result-card">
            <div class="result-model">RMSE</div>
            <div class="result-score">797.74 W</div>
            <div class="result-best">Test-set performance</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Model Comparison")

    st.dataframe(
        MODEL_RESULTS.style.format({
            "MAE (W)": "{:.2f}",
            "RMSE (W)": "{:.2f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="MAE",
        x=MODEL_RESULTS["Model"],
        y=MODEL_RESULTS["MAE (W)"]
    ))

    fig.add_trace(go.Bar(
        name="RMSE",
        x=MODEL_RESULTS["Model"],
        y=MODEL_RESULTS["RMSE (W)"]
    ))

    fig.update_layout(
        barmode="group",
        template="simple_white",
        height=420,
        yaxis_title="Error (W)",
        xaxis_title=""
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        MODEL_RESULTS,
        x="Model",
        y="R²",
        template="simple_white"
    )

    fig.update_layout(
        height=350,
        yaxis_title="R²",
        xaxis_title=""
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 5-Fold Group Cross-Validation")

    st.write(
        "Cross-validation was performed by training session rather than by "
        "individual observation. This prevents windows from the same training "
        "session appearing in both training and validation folds."
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

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Model Selection")

    st.write(
        "Random Forest was selected as the final predictive model because it "
        "achieved the best test-set performance across MAE, RMSE and R². "
        "The MLP deep learning model performed below both conventional models, "
        "which is a useful empirical result for this structured tabular dataset."
    )

    st.info(
        "The deep learning model was retained as a genuine comparison model. "
        "Its lower performance does not invalidate the experiment; it demonstrates "
        "that greater model complexity does not automatically produce better "
        "predictions for structured tabular data."
    )


# ============================================================
# INTERPRETATION
# ============================================================

elif page == "Interpretation":

    st.markdown(
        '<div class="section-title">Model Interpretation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Understanding which predictors matter most to the Random Forest model.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### SHAP Feature Importance")

    shap_plot = ASSETS_DIR / "shap_importance.png"

    if shap_plot.exists():
        st.image(str(shap_plot), use_container_width=True)
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
            height=480,
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title=""
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Numerical SHAP Results")

    st.dataframe(
        SHAP_RESULTS.head(15).style.format({
            "Mean Absolute SHAP": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### What the Model Learned")

    interpretation = [
        "Hardware type is the strongest predictive factor.",
        "Number of GPUs is almost equally influential because the dataset contains fundamentally different single-GPU and multi-GPU configurations.",
        "Image size is the next major configuration-level predictor.",
        "Batch size has measurable predictive influence, although its relationship with power should not be interpreted as causal.",
        "Workload type contributes additional information beyond hardware configuration.",
        "The model's feature importance represents predictive usefulness, not causal influence."
    ]

    for item in interpretation:
        st.markdown(
            f'<div style="margin:0.7rem 0;color:#475467;">• {item}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Important Limitation")

    st.warning(
        "Hardware and workload characteristics are not perfectly crossed in "
        "the dataset. Consequently, feature importance should not be interpreted "
        "as evidence that changing one feature alone will cause a specific change "
        "in power consumption."
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.markdown(
        '<div class="section-title">AERIS Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Configure an AI workload and estimate its mean training power requirement.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

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

        hardware = st.selectbox(
            "Hardware",
            ["RTX3060", "H100", "B200"]
        )

        default_gpus = 1 if hardware == "RTX3060" else 8

        num_gpus = st.number_input(
            "Number of GPUs",
            min_value=default_gpus,
            max_value=default_gpus,
            value=default_gpus,
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

    with col2:

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

        if workload == "Feature Forecasting":

            model_size = st.selectbox(
                "Model Size",
                [474000, 1600000, 3600000],
                format_func=lambda x: f"{x / 1e6:g}M parameters"
            )

            sequence_length = st.selectbox(
                "Sequence Length",
                [96, 192, 672]
            )

            num_layers = st.selectbox(
                "Number of Layers",
                [6, 8, 10]
            )

        elif workload == "Reinforcement Learning":

            layer_size = st.selectbox(
                "Layer Size",
                [256, 512, 1024]
            )

            sequence_length = st.selectbox(
                "Sequence Length",
                [150, 250, 350]
            )

            input_type = st.selectbox(
                "Input Type",
                ["Feature", "Sequence"]
            )

        elif workload == "Image Classification":

            image_size = st.selectbox(
                "Image Size",
                [112, 224, 280]
            )

            num_filters = st.selectbox(
                "Number of Filters",
                [8, 16, 32]
            )

            optimizer = st.selectbox(
                "Optimizer",
                ["Adam", "SGD", "RMS"]
            )

        elif workload == "Text Generation":

            sequence_length = st.selectbox(
                "Sequence Length",
                [100, 250, 500]
            )

            embedding_dim = st.selectbox(
                "Embedding Dimension",
                [100, 300, 1000]
            )

        elif workload == "Image Captioning":

            layer_size = st.selectbox(
                "Layer Size",
                [512, 1024, 2048]
            )

            embedding_dim = st.selectbox(
                "Embedding Dimension",
                [256, 512, 1024]
            )

        elif workload == "Image Generation":

            image_size = st.selectbox(
                "Image Size",
                [32, 64, 128]
            )

            model_size = st.selectbox(
                "Model Size",
                [107000000, 430000000, 1700000000],
                format_func=lambda x: f"{x / 1e6:g}M parameters"
            )

        elif workload == "LLM":

            parallelization = st.selectbox(
                "Parallelization",
                ["ZeRO-1", "ZeRO-2", "ZeRO-3"]
            )

            sequence_length = st.selectbox(
                "Sequence Length",
                [1024, 2048, 4096]
            )

            model_size = st.selectbox(
                "Model Size",
                [1000000000, 3000000000, 8000000000],
                format_func=lambda x: f"{x / 1e9:g}B parameters"
            )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Estimation Horizon")

    duration_minutes = st.number_input(
        "Estimated runtime (minutes)",
        min_value=0.1,
        max_value=1440.0,
        value=15.0,
        step=1.0
    )

    st.caption(
        "The trained model predicts mean power for the dataset's 10-second "
        "measurement window. The selected runtime is used only for downstream "
        "energy estimation and assumes predicted mean power remains constant."
    )

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

    if st.button("Run AERIS Prediction", use_container_width=True):

        try:

            predicted_power = float(model.predict(input_data)[0])

            energy_kwh = (
                predicted_power
                * (duration_minutes / 60)
                / 1000
            )

            st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

            r1, r2 = st.columns(2)

            with r1:
                st.markdown(f"""
                <div class="prediction-result">
                    <div class="prediction-label">
                        Predicted Mean Power
                    </div>
                    <div class="prediction-value">
                        {predicted_power:,.1f}
                    </div>
                    <div class="prediction-unit">
                        watts
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="prediction-result">
                    <div class="prediction-label">
                        Estimated Energy
                    </div>
                    <div class="prediction-value">
                        {energy_kwh:,.3f}
                    </div>
                    <div class="prediction-unit">
                        kWh for {duration_minutes:g} minutes
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

            st.markdown("### Configuration Used")

            display_input = input_data.T.reset_index()
            display_input.columns = ["Feature", "Value"]

            display_input = display_input[
                display_input["Value"].notna()
            ]

            st.dataframe(
                display_input,
                use_container_width=True,
                hide_index=True
            )

            st.session_state["latest_power"] = predicted_power
            st.session_state["latest_energy"] = energy_kwh

        except Exception as e:

            st.error(
                "The prediction could not be generated for this configuration."
            )

            st.exception(e)


# ============================================================
# CARBON ANALYSIS
# ============================================================

elif page == "Carbon Analysis":

    st.markdown(
        '<div class="section-title">Carbon Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Translating predicted electricity demand into location-dependent CO₂e scenarios.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h4>How this calculation works</h4>
        <p>
        AERIS predicts electricity demand through mean power. Energy is then
        calculated for a selected runtime. Estimated emissions are obtained by
        multiplying that energy demand by the annual electricity carbon intensity
        of the selected country.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Research Test-Set Scenario")

    test_energy = 7.1027

    country_comparison = carbon_data[
        carbon_data["Entity"].isin(
            ["India", "United States", "France", "Germany"]
        )
    ].copy()

    if "Carbon intensity" not in country_comparison.columns:
        carbon_column = [
            c for c in country_comparison.columns
            if "carbon" in c.lower()
        ][0]
        country_comparison["Carbon intensity"] = (
            country_comparison[carbon_column]
        )

    country_comparison["Estimated emissions (kg CO2e)"] = (
        test_energy
        * country_comparison["Carbon intensity"]
        / 1000
    )

    country_comparison = country_comparison.sort_values(
        "Estimated emissions (kg CO2e)"
    )

    st.write(
        "The following scenario uses the same predicted 7.1027 kWh of electricity "
        "demand for every country. Only the electricity carbon intensity changes."
    )

    st.dataframe(
        country_comparison[
            [
                "Entity",
                "Year",
                "Carbon intensity",
                "Estimated emissions (kg CO2e)"
            ]
        ].style.format({
            "Carbon intensity": "{:.2f}",
            "Estimated emissions (kg CO2e)": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        country_comparison,
        x="Entity",
        y="Estimated emissions (kg CO2e)",
        template="simple_white"
    )

    fig.update_layout(
        height=400,
        xaxis_title="Country",
        yaxis_title="Estimated CO₂e (kg)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Interactive Carbon Scenario")

    p1, p2 = st.columns(2)

    with p1:

        if "latest_energy" in st.session_state:
            selected_energy = st.number_input(
                "Electricity demand (kWh)",
                min_value=0.001,
                value=float(st.session_state["latest_energy"]),
                step=0.1
            )
        else:
            selected_energy = st.number_input(
                "Electricity demand (kWh)",
                min_value=0.001,
                value=1.0,
                step=0.1
            )

    with p2:

        available_countries = sorted(
            country_comparison["Entity"].unique().tolist()
        )

        selected_country = st.selectbox(
            "Country",
            available_countries
        )

    selected_row = country_comparison[
        country_comparison["Entity"] == selected_country
    ].iloc[0]

    intensity = float(selected_row["Carbon intensity"])

    estimated_emissions = (
        selected_energy * intensity / 1000
    )

    st.markdown(
        f"""
        <div class="prediction-result">
            <div class="prediction-label">
                Estimated emissions in {selected_country}
            </div>
            <div class="prediction-value">
                {estimated_emissions:.3f}
            </div>
            <div class="prediction-unit">
                kg CO₂e · electricity intensity: {intensity:.2f} gCO₂e/kWh
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Country values represent annual electricity carbon intensity and should "
        "not be interpreted as hourly grid-intensity or scheduling data."
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
        '<div class="section-subtitle">'
        'Research methodology, scope and reproducibility.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h4>AERIS — AI Energy Requirement Intelligence System</h4>
        <p>
        AERIS is a predictive modelling framework developed to estimate the
        power consumption of AI training workloads from workload configuration
        and hardware characteristics.
        </p>

        <p>
        The project evaluates both conventional machine learning and deep
        learning approaches and uses model interpretation techniques to examine
        which predictors contribute most strongly to predictive performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Methodology")

    methodology = pd.DataFrame({
        "Stage": [
            "Problem definition",
            "Data understanding",
            "Preprocessing",
            "Model development",
            "Validation",
            "Interpretation",
            "Carbon analysis"
        ],
        "Approach": [
            "Predict mean AI training power",
            "EDA and statistical analysis",
            "Imputation, encoding and scaling",
            "Linear Regression, Random Forest, MLP",
            "Grouped train-test + 5-fold Group CV",
            "SHAP + permutation importance",
            "Electricity carbon-intensity scenarios"
        ]
    })

    st.dataframe(
        methodology,
        use_container_width=True,
        hide_index=True
    )

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

    st.markdown("### Scope and Limitations")

    limitations = [
        "The model predicts mean power rather than instantaneous power.",
        "The primary predictors are configuration and hardware characteristics available before or at workload setup.",
        "The configuration-only model cannot capture temporal runtime behaviour within a training session.",
        "The dataset does not provide a perfectly balanced combination of every workload and hardware platform.",
        "Feature importance indicates predictive association rather than causation.",
        "Carbon estimates depend on the electricity carbon-intensity scenario selected.",
        "Annual country-level carbon intensity should not be interpreted as real-time grid intensity."
    ]

    for limitation in limitations:
        st.markdown(
            f'<div style="margin:0.65rem 0;color:#475467;">'
            f'• {limitation}</div>',
            unsafe_allow_html=True
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

    st.markdown("### Carbon Data Source")

    st.write(
        "Electricity carbon-intensity data were obtained from Our World in Data's "
        "annual electricity carbon-intensity dataset."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    AERIS · AI Energy Requirement Intelligence System · Research Prototype
</div>
""", unsafe_allow_html=True)
