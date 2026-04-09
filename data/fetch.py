import yfinance as yf
import pandas as pd
from typing import List

import os
print("fetch.py running from:", os.getcwd())

NIFTY50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS", "INFY.NS", "INDIGO.NS", "JSWSTEEL.NS",
    "JIOFIN.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "MAXHEALTH.NS",
    "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS",
    "SHRIRAMFIN.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TMPV.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

def fetch_single_stock(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch OHLCV data for a single stock using yfinance.
    """
    stock_df = yf.download(tickers=ticker,   period=period, progress=False)
    stock_df = stock_df.dropna().reset_index()
    stock_df = stock_df.drop(columns=['Adj Close'], errors='ignore')

    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.droplevel(1)
    
    stock_df.columns.name = None

    return stock_df


def fetch_all_nifty50(period: str = "5y") -> dict:
    """
    Fetch data for all Nifty 50 stocks.
    """
    nifty_50 = {}
    for stock in NIFTY50_TICKERS:
        stock_df = fetch_single_stock(stock, period)
        if stock_df.empty:
            print(f"WARNING (fetch): No data for {stock}, skipping.")
            continue
        # print(f"Fetched {stock}: {stock_df.shape}")
        nifty_50[stock] = stock_df
    
    return nifty_50

def fetch_for_inference(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch recent data for a stock at inference time (in the Streamlit app).
    """
    stock = fetch_single_stock(ticker, period)
    return stock


if __name__ == "__main__":
    rel = fetch_all_nifty50("3mo")