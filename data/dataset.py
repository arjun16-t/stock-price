import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from .features import get_feature_columns

WINDOW_SIZE = 60

def scale_stock_data(df, feature_cols: list) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scale features for a single stock independently.
    """
    df_feat = df[feature_cols]

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_feat)
    return (df_scaled, scaler)


def create_sequences(scaled_data: np.ndarray, targets) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding window sequences for time series input.
    """
    X, y_price, y_dir, last_closes = [], [], [], []

    for i in range(WINDOW_SIZE, len(scaled_data)):
        X.append(scaled_data[i - WINDOW_SIZE : i])

        current_close = targets['Close'].iloc[i]
        next_close = targets['next_close'].iloc[i]
        y_price.append((next_close - current_close) / current_close) 
        last_closes.append(current_close)

        y_dir.append(targets['next_open_dir'].iloc[i])
    
    return np.array(X), np.array(y_price).reshape(-1,1), np.array(y_dir).reshape(-1,1), np.array(last_closes)


def train_val_test_split(X, y_price, y_dir, last_closes, train=0.7, val=0.15):
    """
    Split into train/val/test WITHOUT shuffling (order matters in time series).
    """
    split     = int(len(X) * train)
    val_split = int(len(X) * (train + val))

    X_train, X_val, X_test                   = X[:split], X[split:val_split], X[val_split:]
    y_price_train, y_price_val, y_price_test = y_price[:split], y_price[split:val_split], y_price[val_split:]
    y_dir_train, y_dir_val, y_dir_test       = y_dir[:split], y_dir[split:val_split], y_dir[val_split:]
    lc_train, lc_val, lc_test                = last_closes[:split], last_closes[split:val_split], last_closes[val_split:]

    return (X_train, X_val, X_test,
            y_price_train, y_price_val, y_price_test,
            y_dir_train, y_dir_val, y_dir_test,
            lc_train, lc_val, lc_test)


def build_multi_stock_dataset(stock_data_dict: dict, feature_cols: list):
    all_X, all_y_price, all_y_dir, all_last_closes = [], [], [], []
    scalers = {}

    for ticker, df in stock_data_dict.items():
        if len(df) < WINDOW_SIZE + 10:
            print(f'WARNING: Length of {ticker} is not enough')
            continue

        scaled_data, scaler = scale_stock_data(df, feature_cols)
        scalers[ticker] = scaler

        X, y_price, y_dir, last_closes = create_sequences(scaled_data, df)

        all_X.append(X)
        all_y_price.append(y_price)
        all_y_dir.append(y_dir)
        all_last_closes.append(last_closes)

    X = np.concatenate(all_X, axis=0)
    y_price = np.concatenate(all_y_price, axis=0)
    y_dir = np.concatenate(all_y_dir, axis=0)
    last_closes = np.concatenate(all_last_closes, axis=0)

    splits = train_val_test_split(X, y_price, y_dir, last_closes)
    return splits, scalers





if __name__ == "__main__":
    from fetch import fetch_single_stock, fetch_all_nifty50
    from features import add_technical_indicators, add_targets, get_feature_columns

    # PART 1: Test on single stock first
    rel = fetch_single_stock("RELIANCE.NS", "1y")
    rel = add_technical_indicators(rel)
    rel = add_targets(rel)

    scaled_data, _ = scale_stock_data(rel, get_feature_columns())
    print("Shape of Scaled Data: ", scaled_data.shape)
    print("Min: ", scaled_data.min(), "Max: ", scaled_data.max())

    X, y_price, y_dir = create_sequences(
        scaled_data,
        rel[['next_close', 'next_open_dir']]
    )

    print("X Shape:", X.shape)
    print("y_price Shape:", y_price.shape)
    print("y_dir", y_dir.shape)
    print(np.unique(y_dir))


    # PART 2: Test multi-stock pipeline
    nifty_50 = fetch_all_nifty50("1y")
    for ticker, df in nifty_50.items():
        df = add_technical_indicators(df)
        df = add_targets(df)
        nifty_50[ticker] = df
    

    (X_train, X_val, X_test,
     y_price_train, y_price_val, y_price_test,
     y_dir_train, y_dir_val, y_dir_test), scalers = build_multi_stock_dataset(
        nifty_50,
        get_feature_columns()
    )
    print("X_train: ", X_train.shape)
    print("X_val: ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("y_price_train: ", y_price_train.shape)
    print("y_price_val: ", y_price_val.shape)
    print("y_price_test: ", y_price_test.shape)
    print("y_dir_train: ", y_dir_train.shape)
    print("y_dir_val: ", y_dir_val.shape)
    print("y_dir_test: ", y_dir_test.shape)