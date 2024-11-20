# gating_mechanisms.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Multiply, Concatenate
from tensorflow.keras.models import Model

def build_expert_models(input_shape, num_experts=3):
    """
    Create a list of expert models.
    """
    experts = []
    for _ in range(num_experts):
        inputs = Input(shape=(input_shape,))
        x = Dense(32, activation='relu')(inputs)
        experts.append(Model(inputs=inputs, outputs=x))
    return experts

def build_gating_network(input_shape, num_experts=3):
    """
    Build a gating network to produce weights for each expert.
    """
    inputs = Input(shape=(input_shape,))
    x = Dense(num_experts, activation='softmax')(inputs)
    model = Model(inputs=inputs, outputs=x)
    return model

def mixture_of_experts(inputs, experts, gating_weights):
    """
    Combine expert outputs using gating weights.
    """
    expert_outputs = [expert(inputs) for expert in experts]
    weighted_expert_outputs = [Multiply()([gate, expert_output]) for gate, expert_output in zip(tf.unstack(gating_weights, axis=1), expert_outputs)]
    output = tf.add_n(weighted_expert_outputs)
    return output

if __name__ == "__main__":
    input_shape = 32
    num_experts = 3
    inputs = Input(shape=(input_shape,))
    
    experts = build_expert_models(input_shape, num_experts)
    gating_network = build_gating_network(input_shape, num_experts)
    
    gating_weights = gating_network(inputs)
    moe_output = mixture_of_experts(inputs, experts, gating_weights)
    
    model = Model(inputs=inputs, outputs=moe_output)
    model.summary()