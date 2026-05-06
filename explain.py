import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import json
from tensorflow.keras.models import load_model

from data.features import get_feature_columns
from data.dataset import WINDOW_SIZE

MODELS_DIR  = 'saved_models'
RESULTS_DIR = 'results'
FEATURE_COLS = get_feature_columns()


# ── Load SHAP Explainer ───────────────────────────────────────────────────────

@st.cache_resource
def get_explainer(model_name: str, X_background: np.ndarray):
    """
    Create a SHAP GradientExplainer for a given model.

    TODO:
    1. Load model from MODELS_DIR/{model_name}.keras with compile=False
    2. Create shap.GradientExplainer(model, X_background)
       - model here should be the keras model object
       - X_background is a small sample of training data (100 rows)
         used as the reference distribution SHAP compares against
    3. Return (explainer, model)

    Why GradientExplainer?
    - Works directly with TF/Keras models
    - Uses backpropagation to compute exact gradients
    - Much faster than KernelExplainer for neural nets
    - DeepExplainer is alternative but less stable with dual-output models
    """
    pass


def compute_shap_values(explainer, X_sample: np.ndarray):
    """
    Compute SHAP values for a sample of inputs.

    TODO:
    1. Call explainer.shap_values(X_sample)
    2. Result is a list of 2 arrays (one per output head):
       shap_values[0] → price head   shape: (n_samples, 60, 12)
       shap_values[1] → direction head shape: (n_samples, 60, 12)
    3. Return (shap_price, shap_direction)

    Note: this can take 30-60 seconds for 50 samples.
    Wrap in st.spinner() when calling from app.
    """
    pass


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(shap_values: np.ndarray, head_name: str, model_name: str):
    """
    Bar chart of mean absolute SHAP per feature (aggregated across all timesteps).

    TODO:
    1. Aggregate across samples and timesteps:
       importance = np.abs(shap_values).mean(axis=(0, 1))
       → shape: (12,) — one value per feature

    2. Sort features by importance descending

    3. Create horizontal bar chart:
       - y axis: feature names (sorted)
       - x axis: mean absolute SHAP value
       - color: use a gradient (high importance = darker)
       - title: f"{model_name.upper()} — Feature Importance ({head_name} head)"

    4. Return fig (don't call st.pyplot here — caller handles display)

    This answers: "Which features does the model rely on most?"
    """
    pass


def plot_timestep_importance(shap_values: np.ndarray, head_name: str, model_name: str):
    """
    Line chart of mean absolute SHAP per timestep (aggregated across features).

    TODO:
    1. Aggregate across samples and features:
       timestep_importance = np.abs(shap_values).mean(axis=(0, 2))
       → shape: (60,) — one value per timestep

    2. x axis: days ago (60 = oldest, 1 = most recent)
       Hint: x = list(range(60, 0, -1))

    3. Plot as line chart with area fill (plt.fill_between)
       - highlight the last 5 timesteps (most recent days) in different color
       - add vertical line at day 5 with annotation "Last 5 days"

    4. title: f"{model_name.upper()} — Timestep Importance ({head_name} head)"
    5. xlabel: "Days Ago", ylabel: "Mean |SHAP|"

    6. Return fig

    This answers: "Do recent days matter more than older days?"
    (Spoiler: they should for GRU/LSTM, less clear for Transformer)
    """
    pass


def plot_model_comparison(shap_dict: dict, head_name: str):
    """
    Side by side feature importance comparison across GRU, LSTM, Transformer.

    shap_dict format: {'gru': shap_array, 'lstm': shap_array, 'transformer': shap_array}
    Each shap_array shape: (n_samples, 60, 12)

    TODO:
    1. Compute importance per model:
       importance[model] = np.abs(shap_values).mean(axis=(0,1)) → (12,)

    2. Create grouped bar chart:
       - x axis: feature names
       - 3 bars per feature (one per model)
       - different color per model (blue=GRU, orange=LSTM, green=Transformer)
       - title: f"Model Comparison — Feature Importance ({head_name} head)"

    3. Return fig

    This answers: "Do different architectures learn to rely on different features?"
    This is the most interesting plot for your presentation.
    """
    pass


def plot_shap_heatmap(shap_values: np.ndarray, head_name: str, model_name: str):
    """
    2D heatmap of SHAP values: timesteps (rows) x features (cols).
    Averaged across all samples.

    TODO:
    1. mean_shap = np.abs(shap_values).mean(axis=0) → shape (60, 12)

    2. plt.imshow(mean_shap, aspect='auto', cmap='RdYlGn')
       - x axis: feature names (FEATURE_COLS)
       - y axis: timestep 1-60 (1=oldest, 60=most recent)
       - colorbar label: "Mean |SHAP|"

    3. title: f"{model_name.upper()} — SHAP Heatmap ({head_name} head)"

    4. Return fig

    This answers: "At which timestep does each feature matter most?"
    """
    pass


# ── LLM Summary ──────────────────────────────────────────────────────────────

def generate_shap_summary(shap_price: np.ndarray, shap_dir: np.ndarray,
                           model_name: str) -> str:
    """
    Generate a natural language summary of SHAP findings using Claude API.

    TODO:
    1. Compute top 3 features for price head:
       price_importance = np.abs(shap_price).mean(axis=(0,1))
       top_price_features = sorted(zip(FEATURE_COLS, price_importance),
                                   key=lambda x: x[1], reverse=True)[:3]

    2. Compute top 3 features for direction head similarly

    3. Compute whether recent timesteps matter more:
       timestep_imp = np.abs(shap_price).mean(axis=(0,2))  → (60,)
       recent_ratio = timestep_imp[-10:].mean() / timestep_imp[:10].mean()
       # ratio > 1 means recent days more important than older days

    4. Build a prompt with these findings and call Claude API:
       See prompt template below.

    5. Return the generated summary string.
    """

    # Compute stats
    # TODO: compute top features and recent_ratio as described above

    prompt = f"""
    You are explaining SHAP (SHapley Additive exPlanations) results from a 
    stock market prediction model to a non-technical audience.
    
    Model: {model_name.upper()} trained on Nifty 50 Indian stocks
    
    Findings:
    - Top 3 features for PRICE prediction: {top_price_features}
    - Top 3 features for DIRECTION prediction: {top_dir_features}  
    - Recent days vs older days importance ratio: {recent_ratio:.2f}
      (>1 means recent days matter more, <1 means older history matters more)
    
    Write a clear, concise 3-4 sentence explanation of what these findings mean
    for understanding how the model makes predictions. Use plain English.
    Mention what this implies about market behavior and the model's strategy.
    Do not use bullet points. Do not mention SHAP by name — explain the concept
    instead as 'feature contribution analysis'.
    """

    # TODO: call Claude API here
    # Use fetch with api.anthropic.com/v1/messages
    # model: claude-sonnet-4-20250514
    # Return response text

    pass


# ── Main Streamlit Section ────────────────────────────────────────────────────

def render_explanations(model_name: str, X_train: np.ndarray):
    """
    Full XAI section rendered inside Streamlit.
    Called from app.py after prediction.

    TODO:
    1. st.subheader("🔍 Model Explanation (SHAP)")

    2. Sample background and explanation data:
       background = X_train[np.random.choice(len(X_train), 100, replace=False)]
       X_sample   = X_train[np.random.choice(len(X_train), 50, replace=False)]

    3. Get explainer and compute SHAP values:
       explainer, _ = get_explainer(model_name, background)
       shap_price, shap_dir = compute_shap_values(explainer, X_sample)

    4. Two tabs — one per head:
       tab1, tab2 = st.tabs(["💰 Price Head", "📈 Direction Head"])

       In each tab:
       a. Feature importance bar chart
       b. Timestep importance line chart
       c. SHAP heatmap
       d. LLM summary

    5. Third tab for model comparison:
       tab3 = "⚖️ Model Comparison"
       - Compute SHAP for all 3 models
       - Call plot_model_comparison() for both heads

    Note: SHAP computation is slow — wrap in st.spinner()
    and cache results using st.session_state so it doesn't
    recompute on every Streamlit rerun.
    """
    pass