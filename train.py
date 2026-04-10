import numpy as np
import joblib
import json
import os
print("Working directory:", os.getcwd())

import warnings
warnings.filterwarnings("ignore")

# 1. IMPORT YFINANCE (via fetch) BEFORE TENSORFLOW
from data.fetch import fetch_all_nifty50
from data.features import add_technical_indicators, add_targets, get_feature_columns
from data.dataset import build_multi_stock_dataset, WINDOW_SIZE

# 2. NOW IMPORT TENSORFLOW
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.models import Model # type: ignore
from models.gru_model import build_gru_model
from models.lstm_model import build_lstm_model
from models.transformer_model import build_transformer_model

# ── Config ────────────────────────────────────────────────────────────────────

EPOCHS     = 50
BATCH_SIZE = 32
LR         = 1e-3
MODELS_DIR = 'saved_models'
RESULTS_DIR = 'results'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data Pipeline ─────────────────────────────────────────────────────────────

def prepare_data():
    """
    Run the full data pipeline and return splits + scalers.
    """ 
    nifty_50 = fetch_all_nifty50("10y")
    processed = {}
    for ticker, df in nifty_50.items():
        df = add_technical_indicators(df)
        df = add_targets(df)
        if df.empty:
            print(f"WARNING(train): No data for {ticker}, skipping.")
            continue

        processed[ticker] = df
    
    splits, scalers = build_multi_stock_dataset(processed, get_feature_columns())
    joblib.dump(scalers, f"{MODELS_DIR}/scalers.joblib")
    return splits


# ── Compile ───────────────────────────────────────────────────────────────────

def compile_model(model: Model, down_weight: float):
    """
    Compile a model with the shared loss, metrics and optimizer.
    """
    def weighted_binary_crossentropy(y_true, y_pred):
        # Standard cross entropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        weight_vector = y_true * 1.0 + (1.0 - y_true) * down_weight
        return tf.reduce_mean(bce * tf.cast(weight_vector, tf.float32))
    
    model.compile(
        optimizer=Adam(LR),
        loss={
            'price': 'mse',
            'direction': weighted_binary_crossentropy
        },
        loss_weights={
            'price': 1.0,
            'direction': 0.5
        },
        metrics={
            'price': ['mae'],
            'direction': ['binary_accuracy']
        }
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str):
    """
    Return list of callbacks for training.
    """
    cb = []
    cb.append(EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ))
    cb.append(ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr= 1e-6
    ))
    cb.append(ModelCheckpoint(
        filepath=f"{MODELS_DIR}/{model_name}.keras",
        monitor="val_loss",
        save_best_only=True
    ))

    return cb


# ── Train One Model ───────────────────────────────────────────────────────────

def train_model(model:Model, model_name: str, splits: tuple):
    """
    Train a single model and save its history.

    """
    (X_train, X_val, X_test,
    y_price_train, y_price_val, y_price_test,
    y_dir_train, y_dir_val, y_dir_test,
    lc_train, lc_val, lc_test) = splits

    y_dir_flat = y_dir_train.flatten()
    num_down = np.sum(y_dir_flat == 0)
    num_up = np.sum(y_dir_flat == 1)
    
    down_weight = num_up / num_down if num_down > 0 else 1.0
    
    print(f"\033[93mApplying Down-Class Weight Factor: {down_weight:.2f}\033[0m")

    compiled_model = compile_model(model, down_weight)

    history = compiled_model.fit(x = X_train,
                                y = {'price': y_price_train, 'direction': y_dir_train},
                                validation_data=(X_val, {'price': y_price_val, 'direction': y_dir_val}),
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                callbacks=get_callbacks(model_name),
                                verbose=2
                            )
    
    with open(f"{RESULTS_DIR}/{model_name}_history.json", "w") as f:
        json.dump(history.history, f)
    
    return history

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    feature_cols = get_feature_columns()
    num_features = len(feature_cols)

    splits = prepare_data()

    lstm = build_lstm_model(WINDOW_SIZE, num_features)
    gru = build_gru_model(WINDOW_SIZE, num_features)
    transformer = build_transformer_model(WINDOW_SIZE, num_features)

    histories = {}
    for name, model in [('gru', gru), ('lstm', lstm), ('transformer', transformer)]:
        histories[name] = train_model(model, name, splits)

    print(f"\n{'Model':<15} {'Val Loss':<15} {'Best Epoch'}")
    print("-" * 40)
    for name, history in histories.items():
        val_losses = history.history['val_loss']
        best_val_loss = min(val_losses)
        best_epoch = val_losses.index(best_val_loss) + 1
        print(f"{name:<15} {best_val_loss:<15.4f} {best_epoch}")


if __name__ == "__main__":
    main()