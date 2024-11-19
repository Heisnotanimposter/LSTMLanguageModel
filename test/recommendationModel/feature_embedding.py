# feature_embedding.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_text_embeddings(text_data, max_words=1000, embedding_dim=16, max_length=10):
    """
    Convert text data into embeddings using Tokenizer and Embedding layers.
    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    input_text = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length)(input_text)
    flatten_layer = Flatten()(embedding_layer)
    model = Model(inputs=input_text, outputs=flatten_layer)
    
    embeddings = model.predict(padded_sequences)
    return embeddings

def create_categorical_embeddings(df, categorical_columns, embedding_dim=8):
    """
    Create embeddings for categorical variables.
    """
    embeddings = {}
    for col in categorical_columns:
        unique_values = df[col].nunique()
        input_cat = Input(shape=(1,))
        embedding_layer = Embedding(input_dim=unique_values + 1, output_dim=embedding_dim, input_length=1)(input_cat)
        flatten_layer = Flatten()(embedding_layer)
        model = Model(inputs=input_cat, outputs=flatten_layer)
        embeddings[col] = model.predict(df[col].values)
    return embeddings

def scale_numerical_features(df, numerical_columns):
    """
    Standardize numerical features.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_columns])
    return scaled_features

if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        'title': ['Funny Cats', 'Cooking Pasta', 'News Update', 'Gaming Highlights'],
        'category_Pets': [1, 0, 0, 0],
        'category_Food': [0, 1, 0, 0],
        'category_News': [0, 0, 1, 0],
        'category_Gaming': [0, 0, 0, 1],
        'watch_time': [0.5, 0.3, 0.8, 0.6]
    })
    
    text_embeddings = create_text_embeddings(df['title'])
    categorical_embeddings = create_categorical_embeddings(df, ['category_Pets', 'category_Food', 'category_News', 'category_Gaming'])
    numerical_features = scale_numerical_features(df, ['watch_time'])
    
    print("Text Embeddings Shape:", text_embeddings.shape)
    print("Numerical Features Shape:", numerical_features.shape)