import tensorflow as tf
from tensorflow.keras.models import Model                           # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input     # type: ignore
import warnings
warnings.filterwarnings("ignore")

def build_lstm_model(window_size: int, num_features: int) -> Model:
    """
    Build a LSTM model with dual output heads.

    Architecture:
    Input → LSTM(128, return_sequences=True) → Dropout
            → LSTM(64) → Dropout
            → Dense(32, relu)
            → [price_output: Dense(1, linear)] + [dir_output: Dense(1, sigmoid)]
    """
    inputs = Input(shape=(window_size, num_features))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    price_output = Dense(1, activation='linear', name='price')(x)
    dir_output = Dense(1, activation='sigmoid', name='direction')(x)

    return Model(
        inputs=inputs, 
        outputs={
            'price': price_output, 
            'direction': dir_output
        }
    )