import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from data.fetch import fetch_for_inference
from data.features import add_technical_indicators, add_targets, get_feature_columns
from data.dataset import WINDOW_SIZE

# ── Config ────────────────────────────────────────────────────────────────────

MODELS_DIR  = 'saved_models'
TICKERS_JSON = 'tickers.json'

st.set_page_config(
    page_title="Indian Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Load Resources (cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """
    Load all 3 saved models once and cache them.

    TODO:
    1. Load gru, lstm, transformer from MODELS_DIR using load_model()
    2. Return dict: {'gru': model, 'lstm': model, 'transformer': model}

    Note: @st.cache_resource means this runs only once per session.
    Without caching, models reload on every user interaction — very slow.
    """
    pass


@st.cache_resource
def load_scalers():
    """
    Load saved scalers dict from disk.

    TODO: joblib.load() from MODELS_DIR/scalers.joblib
    """
    pass


@st.cache_data
def load_ticker_map():
    """
    Load company name → NSE ticker mapping from tickers.json.

    TODO:
    1. Open and json.load() TICKERS_JSON
    2. Return the dict

    tickers.json format: {"Reliance Industries": "RELIANCE.NS", ...}
    Build this file manually — Google "Nifty 50 company names and NSE tickers"
    and create the mapping. Takes 10 minutes.
    """
    pass


# ── Inference Pipeline ────────────────────────────────────────────────────────

def prepare_inference_data(ticker: str, scalers: dict):
    """
    Fetch and prepare data for a single stock at inference time.

    TODO:
    1. Call fetch_for_inference(ticker) → raw df
    2. If df is empty or too short → return None, None, None

    3. Call add_technical_indicators(df)
    4. Drop NaNs
    5. Check we have at least WINDOW_SIZE rows → return None if not

    6. Get the scaler for this ticker:
       - If ticker in scalers → use saved scaler (transform only, don't fit)
       - If ticker NOT in scalers → fit a fresh MinMaxScaler on this data
         (this handles stocks outside Nifty 50)

    7. Scale feature columns → scaled_data
    8. Take LAST WINDOW_SIZE rows as input window → shape (1, WINDOW_SIZE, num_features)
       Hint: scaled_data[-WINDOW_SIZE:] then np.expand_dims(..., axis=0)

    9. Get last_close = df['Close'].iloc[-1] (for converting prediction back to ₹)

    10. Return (input_window, last_close, df)
        df is returned for plotting the historical chart
    """
    pass


def predict(model, input_window: np.ndarray, last_close: float):
    """
    Run inference and convert outputs to interpretable values.

    TODO:
    1. model.predict(input_window) → [pred_return, pred_dir_prob]
    2. predicted_price = last_close * (1 + pred_return.flatten()[0])
    3. direction_prob = pred_dir_prob.flatten()[0]
    4. direction_label = 'UP ↑' if direction_prob > 0.5 else 'DOWN ↓'
    5. confidence = direction_prob if UP, else (1 - direction_prob)
    6. Return (predicted_price, direction_label, confidence)
    """
    pass


# ── UI Components ─────────────────────────────────────────────────────────────

def render_price_chart(df: pd.DataFrame, predicted_price: float, ticker: str):
    """
    Render an interactive Plotly candlestick chart with predicted price marker.

    TODO:
    1. Create a go.Figure()
    2. Add go.Candlestick trace using last 60 rows of df:
       - x=df['Date'], open=df['Open'], high=df['High'],
         low=df['Low'], close=df['Close']
    3. Add go.Scatter trace for predicted price:
       - Single point at (last date + 1 trading day, predicted_price)
       - Make it a star marker, different color, size=15
       - Label it 'Predicted Close'
    4. Update layout: title, xaxis_title, yaxis_title, template='plotly_dark'
    5. st.plotly_chart(fig, use_container_width=True)

    Hint for next trading day date:
    from pandas.tseries.offsets import BDay
    next_day = df['Date'].iloc[-1] + BDay(1)
    """
    pass


def render_metrics(predicted_price: float, last_close: float,
                   direction: str, confidence: float, model_name: str):
    """
    Render prediction metrics using Streamlit metric cards.

    TODO using st.metric():
    1. Predicted Close Price → value=f'₹{predicted_price:.2f}'
                               delta=f'{((predicted_price-last_close)/last_close)*100:.2f}%'
    2. Next Open Direction   → value=direction
    3. Confidence            → value=f'{confidence:.1%}'
    4. Model Used            → value=model_name.upper()

    Use st.columns(4) to lay them out side by side.
    """
    pass


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.title("📈 Indian Stock Market Predictor")
    st.caption("Predicts next day closing price and opening direction using GRU, LSTM and Transformer models")

    # Load resources
    models  = load_models()
    scalers = load_scalers()
    ticker_map = load_ticker_map()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    # TODO 1: Search box
    # Use st.sidebar.selectbox() or st.sidebar.text_input()
    # If using selectbox: options = list(ticker_map.keys())
    # If using text_input: do fuzzy matching with rapidfuzz
    # Resolve selection to NSE ticker using ticker_map

    # TODO 2: Model selector
    # st.sidebar.radio() with options ['gru', 'lstm', 'transformer']
    # Let user pick which model to use for prediction

    # TODO 3: Show raw data toggle
    # st.sidebar.checkbox('Show raw data')

    # ── Main Panel ────────────────────────────────────────────────────────────

    # TODO 4: Predict button
    # if st.button('Predict'):
    #     with st.spinner('Fetching data and running prediction...'):
    #         input_window, last_close, df = prepare_inference_data(ticker, scalers)
    #
    #         if input_window is None:
    #             st.error('Not enough data for this stock. Try another.')
    #         else:
    #             predicted_price, direction, confidence = predict(model, input_window, last_close)
    #             render_metrics(predicted_price, last_close, direction, confidence, model_name)
    #             render_price_chart(df, predicted_price, ticker)
    #
    #             if show_raw_data:
    #                 st.subheader('Raw Data')
    #                 st.dataframe(df.tail(10))


if __name__ == "__main__":
    main()