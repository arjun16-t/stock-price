import pandas as pd
import pandas_ta as ta
from typing import List

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
   """
   Add all technical indicators to a raw OHLCV DataFrame.
   Assumes df has columns: Open, High, Low, Close, Volume
   
   After adding all indicators:
   - Drop rows with NaN (indicators need warmup periods)
   - Return the full DataFrame with all new columns
   """
   df.ta.sma(length=10, append=True)         # SMA 10 (Short Term Trend)
   df.ta.ema(length=26, append=True)         # EMA 26 (Medium Term Trend)
   df.ta.rsi(length=14, append=True)         # RSI 14 (Momentum)
   df.ta.bbands(length=20, append=True)      # Bollinger Bands (Position within volatility bands)
   df.ta.atr(length=14, append=True)         # Average True Range (Volatility Magnitude)

   df['Daily_Return'] = df['Close'].pct_change()
   df['HL_Range'] = (df['High'] - df['Low']) / df['Close']

   df = df.rename(columns={
               'BBL_20_2.0_2.0': 'BBL',
               'BBM_20_2.0_2.0': 'BBM',
               'BBU_20_2.0_2.0': 'BBU',
               'BBB_20_2.0_2.0': 'BBB',
               'BBP_20_2.0_2.0': 'BBP'
               })

   df = df.drop(['BBB', 'BBL', 'BBM', 'BBU'], axis=1)
   df.dropna(inplace=True)
   return df

def get_feature_columns() -> List[str]:
   """
   Return the list of column names that will be used as model input features.
   """
   columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_10', 'EMA_26',
      'RSI_14', 'BBP', 'ATRr_14', 'Daily_Return', 'HL_Range']
   return columns


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
   """
   Add prediction target columns to the DataFrame.
   """
   df['next_close'] = df['Close'].shift(-1)
   df['next_open_dir'] = (df['Open'].shift(-1) > df['Close']).astype(int)
   df.drop(df.index[-1], inplace=True)
   return df

if __name__ == "__main__":
   from fetch import fetch_single_stock

   rel = fetch_single_stock("RELIANCE.NS", "1y")

   # 2. Add indicators
   print(f'Shape before technical indicators: {rel.shape}')
   rel = add_technical_indicators(rel)

   # 3. Check all expected columns exist
   print(rel.columns)

   # 4. Add targets and verify
   rel = add_targets(rel)
   print(rel[['Close', 'next_close', 'next_open_dir']].tail(5))

   # 5. Check no NaNs remain
   print(rel.isnull().sum())