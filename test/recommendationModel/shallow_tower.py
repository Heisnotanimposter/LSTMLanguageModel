# shallow_tower.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model

def build_shallow_tower(input_shape):
    """
    Build a shallow neural network tower.
    """
    inputs = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    # Example input
    input_shape = 128  # Assuming combined embedding size
    model = build_shallow_tower(input_shape)
    model.summary()