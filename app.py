import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from keras.src.models.functional import Functional

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
def load_models() -> dict[str, Functional]:
    """
    Load all 3 saved models once and cache them.
    """
    from tensorflow.keras.models import load_model      # type: ignore
    models = {}

    model_list = ['gru', 'lstm', 'transformer']
    for model in model_list:
        curr_model = load_model(f"{MODELS_DIR}/{model}.keras", compile=False)
        models[model] = curr_model
    
    return models


@st.cache_resource
def load_scalers() -> dict:
    """
    Load saved scalers dict from disk.
    """
    scalers = joblib.load(f"{MODELS_DIR}/scalers.joblib")
    return scalers

@st.cache_data
def load_ticker_map() -> dict[str, str]:
    """
    Load company name → NSE ticker mapping from tickers.json.
    """
    tickers = {}
    with open("tickers.json", "r") as file:
        tickers = json.load(file)
    
    return tickers


# ── Inference Pipeline ────────────────────────────────────────────────────────

def prepare_inference_data(ticker: str, scalers: dict):
    """
    Fetch and prepare data for a single stock at inference time.
    """
    stock = fetch_for_inference(ticker, "1y")
    if not isinstance(stock, pd.DataFrame) or stock.empty:
        print(f"Invalid Stock: {stock}  |  Empty Results")
        return None, None, None
    
    stock = add_technical_indicators(stock)
    if len(stock) < WINDOW_SIZE:
        st.error("Not enough data after computing indicators.")
        return None, None, None
    
    if ticker in scalers.keys():
        scaler = scalers[ticker]
        scaled_stock = scaler.transform(stock[get_feature_columns()])
    else:
        scaler = MinMaxScaler()
        scaled_stock = scaler.fit_transform(stock[get_feature_columns()])
    
    input_window = scaled_stock[-WINDOW_SIZE:]
    input_window = np.expand_dims(input_window, axis=0)     # shape (1, WINDOW_SIZE, num_features)

    last_close = stock['Close'].iloc[-1]

    return (input_window, last_close, stock)

def predict(model, input_window: np.ndarray, last_close: float):
    """
    Run inference and convert outputs to interpretable values.
    """
    predictions = model.predict(input_window)
    if isinstance(predictions, dict):
        pred_returns = predictions['price'].flatten()
        pred_dir_prob = predictions['direction'].flatten()
    else:
        pred_returns = predictions[0].flatten()
        pred_dir_prob = predictions[1].flatten()
    
    pred_return = predictions[0].flatten()[0]
    predicted_price = float(last_close) * (1 + pred_return)

    direction_prob = pred_dir_prob.flatten()[0]
    direction_label = 'UP ↑' if direction_prob > 0.5 else 'DOWN ↓'
    confidence = direction_prob if direction_prob > 0.5 else (1 - direction_prob)

    return (predicted_price, direction_label, confidence)

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