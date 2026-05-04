import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json

from rapidfuzz import fuzz
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

@st.cache_data
def load_all_stocks():
    """Load all NSE stocks"""
    with open('tickers_all.json', 'r') as f:
        stocks_dict = json.load(f)
    
    # Convert to list of tuples for easier iteration
    stocks_list = [(name, ticker) for name, ticker in stocks_dict.items()]
    return stocks_list, stocks_dict


# ── Inference Pipeline ────────────────────────────────────────────────────────

def prepare_inference_data(ticker: str):
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

def calculate_match_score(query, company_name, ticker):
    """
    Calculate weighted score for a stock.
    """
    query_upper = query.upper().strip()

    if ticker.startswith(query_upper):
        ticker_score = 100
    else:
        ticker_score = fuzz.ratio(query_upper, ticker)
    
    company_score = fuzz.token_sort_ratio(query_upper, company_name.upper())
    
    final_score = (0.6 * ticker_score) + (0.4 * company_score)

    return final_score

@st.cache_data
def fuzzy_search(query, threshold=85, top_k=5):
    """
    Search for stocks matching the query.
    
    Returns:
    - results: list of formatted strings ["INFY — Infosys Limited", ...]
    - ticker_map: dict to map display string back to ticker
    """
    if not query or len(query.strip()) == 0:
        return [], {}
    
    stocks_list, stocks_dict = load_all_stocks()
    
    # Score all stocks
    scored_results = []
    for company_name, ticker in stocks_list:
        score = calculate_match_score(
            query, company_name, ticker
        )
        # print(f"\033[92mTICKER: {ticker}\033[0m")
        scored_results.append({
            'ticker': ticker,
            'company': company_name,
            'score': score
        })
    
    # Filter: keep results >= threshold, OR top 1 if none above threshold
    above_threshold = [r for r in scored_results if r['score'] >= threshold]
    
    if above_threshold:
        results = above_threshold
    else:
        # No results above threshold: return top 1 suggestion
        results = sorted(scored_results, key=lambda x: x['score'], reverse=True)[:5]
    
    # Sort by score descending and take top 5
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    # Format for display and create reverse mapping
    formatted_results = []
    ticker_map = {}
    
    for r in results:
        display_string = f"{r['ticker']} — {r['company']}"
        formatted_results.append(display_string)
        ticker_map[display_string] = r['ticker']
    
    return formatted_results, ticker_map

def get_all_stocks_for_fallback():
    """Get all stocks as fallback (for initial display)"""
    _, stocks_dict = load_all_stocks()
    formatted = [f"{ticker} — {name}" for ticker, name in stocks_dict.items()]
    return formatted, {f"{ticker} — {name}": ticker for ticker, name in stocks_dict.items()}

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
        annotation_position="top left"
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
    st.title("Indian Stock Market Predictor")
    st.caption("Predicts next day closing price and opening direction using GRU, LSTM and Transformer models")

    # Load resources
    models  = load_models()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    st.sidebar.write("### Search Company / Ticker")

    # Text input for search query
    search_query = st.sidebar.text_input(
        label="Type ticker or company name",
        placeholder="e.g., 'INFY' or 'Infosys'",
        help="Search by ticker or company name"
    )
    
    if search_query and len(search_query) > 0:
        results, ticker_map = fuzzy_search(search_query, threshold=85, top_k=5)
    else:
        # Show all stocks if no search query
        results, ticker_map = get_all_stocks_for_fallback()

    # Selectbox with search results
    if results:
        selected_company = st.sidebar.selectbox(
            "Select from results",
            options=results,
            help="Click to select a stock"
        )
        
        # Resolve selection to ticker
        ticker = ticker_map[selected_company]
    else:
        st.sidebar.warning("No stocks found. Try a different search.")
        ticker = None

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
            
            input_window, last_close, df = prepare_inference_data(ticker)

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
    else:
        st.info("Please select a stock to proceed.")


if __name__ == "__main__":
    main()