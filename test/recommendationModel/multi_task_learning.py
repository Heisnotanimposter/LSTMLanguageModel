# multi_task_learning.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

def build_multi_task_model(input_shape):
    """
    Build a multi-task learning model.
    """
    inputs = Input(shape=(input_shape,))
    
    # Shared layers
    shared_layer = Dense(64, activation='relu')(inputs)
    
    # Task-specific outputs
    click_output = Dense(1, activation='sigmoid', name='click_output')(shared_layer)
    watch_time_output = Dense(1, activation='linear', name='watch_time_output')(shared_layer)
    like_output = Dense(1, activation='sigmoid', name='like_output')(shared_layer)
    
    model = Model(inputs=inputs, outputs=[click_output, watch_time_output, like_output])
    return model

def compile_multi_task_model(model):
    """
    Compile the multi-task model with appropriate loss functions and weights.
    """
    losses = {
        'click_output': 'binary_crossentropy',
        'watch_time_output': 'mse',
        'like_output': 'binary_crossentropy'
    }
    loss_weights = {
        'click_output': 1.0,
        'watch_time_output': 0.5,
        'like_output': 1.0
    }
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
    return model

if __name__ == "__main__":
    input_shape = 32  # Output from shallow tower
    model = build_multi_task_model(input_shape)
    model = compile_multi_task_model(model)
    model.summary()