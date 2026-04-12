import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler

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
def load_models() -> dict:
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
    
    pred_return = pred_returns[0]
    predicted_price = float(last_close) * (1 + pred_return)

    direction_prob = pred_dir_prob[0]
    direction_label = 'UP ↑' if direction_prob > 0.5 else 'DOWN ↓'

    confidence = direction_prob if direction_prob > 0.5 else (1 - direction_prob)

    return (predicted_price, direction_label, confidence)

# ── UI Components ─────────────────────────────────────────────────────────────

def render_price_chart(df: pd.DataFrame, predicted_price: float, ticker: str):
    """
    Render an interactive Plotly candlestick chart with predicted price marker.
    """

    df['Date'] = pd.to_datetime(df['Date'])
    df_last = df.tail(60)
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_last['Date'],
        open=df_last['Open'],
        high=df_last['High'],
        low=df_last['Low'],
        close=df_last['Close'],
        name='Price'
    ))

    last_close = df_last['Close'].iloc[-1]
    color = 'green' if predicted_price > last_close else 'red'
    next_day = df_last['Date'].iloc[-1] + BDay(1)

    fig.add_trace(go.Scatter(
        x=[next_day],
        y=[predicted_price],
        mode='markers+text',
        marker=dict(
            symbol='star',
            size=15,
            color=color
        ),
        name='Predicted Close',
        text=[f"{predicted_price:.2f}"],
        textposition='top center'
    ))

    fig.add_hline(
        y=predicted_price,
        line_dash="dash",
        line_color=color,
        annotation_text="Predicted Price",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"{ticker} Price Chart with Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_metrics(predicted_price: float, last_close: float,
                   direction: str, confidence: float, model_name: str):
    """
    Render prediction metrics using Streamlit metric cards.
    """
    delta_pct = ((predicted_price - last_close) / last_close) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Predicted Close Price",
        value=f"₹{predicted_price:.2f}",
        delta=f"{delta_pct:.2f}%"
    )

    col2.metric(
        label="Next Open Direction",
        value=direction
    )

    col3.metric(
        label="Confidence",
        value=f"{confidence:.1%}"
    )

    col4.metric(
        label="Model Used",
        value=model_name.upper()
    )


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

    # selectbox natively supports typing-to-search in Streamlit
    selected_company = st.sidebar.selectbox(
        "Search Company / Ticker", 
        options=list(ticker_map.keys()),
        help="Type the name of the company to search"
    )
    # Resolve selection to the actual NSE ticker
    ticker = ticker_map[selected_company]

    model_name = st.sidebar.radio(
        "Select Model", 
        options=['gru', 'lstm', 'transformer'],
        format_func=lambda x: x.upper() # Formats the display text to uppercase (GRU, LSTM...)
    )
    # Grab the selected model object from the loaded models dictionary
    model = models[model_name]

    show_raw_data = st.sidebar.checkbox("Show raw data", value=False)

    # ── Main Panel ────────────────────────────────────────────────────────────

    if st.button('Predict Next Day', type='primary', use_container_width=True):
        with st.spinner(f'Fetching live data and running {model_name.upper()} prediction...'):
            
            input_window, last_close, df = prepare_inference_data(ticker, scalers)

            if input_window is None:
                st.error(f'Not enough recent data available for {selected_company} ({ticker}). Please try another stock.')
            else:
                # Run the prediction
                predicted_price, direction, confidence = predict(model, input_window, last_close)
                
                # Render the UI components
                render_metrics(predicted_price, last_close, direction, confidence, model_name)
                st.divider()
                render_price_chart(df, predicted_price, ticker)

                # Show dataframe if toggle is checked
                if show_raw_data:
                    st.divider()
                    st.subheader('Raw Data (Last 10 Trading Days)')
                    st.dataframe(df.tail(10), use_container_width=True)


if __name__ == "__main__":
    main()