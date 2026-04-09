import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D
)
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """
    Generate sinusoidal positional encodings.

    Return shape should be (1, length, depth) — the 1 is for batch broadcasting
    """
    pe = np.zeros((length, depth))
    position = np.arange(0, length).reshape(-1, 1)  # (length, 1)
    div_term = np.exp(np.arange(0, depth, 2) * (-np.log(10000.0) / depth))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    pe = pe[np.newaxis, :, :]           # (1, length, depth)

    return tf.cast(pe, dtype=tf.float32)
def transformer_encoder_block(x, d_model: int, num_heads: int, ff_dim: int, dropout: float):
    """
    One transformer encoder block.
    """
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(query=x, value=x, key=x)

    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout)(ffn)

    output = LayerNormalization(epsilon=1e-6)(out1 + ffn)

    return output
def build_transformer_model(window_size: int, num_features: int) -> Model:
    """
    Lightweight transformer for time series.
    
    Architecture:
    Input → Linear projection to d_model=64
            → Positional Encoding
            → TransformerEncoderBlock(d_model=64, heads=4, ff_dim=128)
            → TransformerEncoderBlock(d_model=64, heads=4, ff_dim=128)
            → GlobalAveragePooling1D
            → Dense(32, relu) → Dropout
            → [price_output] + [dir_output]
    """
    d_model = 64

    # 1. Input
    inputs = Input(shape=(window_size, num_features))

    # 2. Linear projection → d_model
    x = Dense(d_model)(inputs)

    # 3. Add positional encoding
    pos_encoding = positional_encoding(window_size, d_model)
    x = x + pos_encoding

    # 4. Transformer blocks
    x = transformer_encoder_block(x, d_model, 4, 128, 0.1)
    x = transformer_encoder_block(x, d_model, 4, 128, 0.1)

    # 5. Pooling
    x = GlobalAveragePooling1D()(x)

    # 6. Dense head
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)

    # 7. Dual outputs
    price_output = Dense(1, activation='linear', name='price')(x)
    dir_output   = Dense(1, activation='sigmoid', name='direction')(x)

    return Model(
        inputs=inputs, 
        outputs={
            'price': price_output, 
            'direction': dir_output
        }
    )

if __name__ == '__main__':
    from gru_model import build_gru_model
    from lstm_model import build_lstm_model

    gru = build_gru_model(60, 12)
    lstm = build_lstm_model(60, 12)
    transformer = build_transformer_model(60, 12)

    print("========== GRU ==========")
    gru.summary()

    print("========== LSTM ==========")
    lstm.summary()

    print("========== TRANSFORMER ==========")
    transformer.summary()