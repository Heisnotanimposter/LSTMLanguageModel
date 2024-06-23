from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, LayerNormalization, Conv1D, AveragePooling1D, Input, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import Adam
import numpy as np

# Custom Transformer Block
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation='gelu'), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Sample data
text = "Hello world. This is a sample text for language modeling."

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

# Save the tokenizer to a file in Google Drive
with open('/content/drive/MyDrive/merged_model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Prepare the dataset
vocab_size = len(tokenizer.word_index) + 1
seq_length = 10  # Increase sequence length to handle more tokens
dataset = []
for i in range(seq_length, len(sequences)):
    dataset.append(sequences[i-seq_length:i+1])
dataset = tf.constant(dataset)

# Build the upgraded model
embed_dim = 64  # Increased embedding size for each token
num_heads = 4  # Increased number of attention heads
ff_dim = 64  # Increased hidden layer size in feed forward network inside transformer

inputs = Input(shape=(seq_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_length)(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)(embedding_layer)
conv_layer = Conv1D(64, 3, activation='relu')(transformer_block)
pool_layer = AveragePooling1D()(conv_layer)
lstm_layer = LSTM(100, return_sequences=True)(pool_layer)
lstm_layer = LSTM(50)(lstm_layer)
outputs = Dense(vocab_size, activation='softmax')(lstm_layer)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
model.fit(dataset[:, :-1], dataset[:, -1], epochs=50, verbose=1)  # Increase epochs for better training

# Save the trained model
model.save('/content/drive/MyDrive/merged_model/HINAI_upgraded_with_transformer.h5')

# Load the trained model and tokenizer for text generation
model = load_model('/content/drive/MyDrive/merged_model/HINAI_upgraded_with_transformer.h5', custom_objects={'TransformerBlock': TransformerBlock})

with open('/content/drive/MyDrive/merged_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def generate_text(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = np.pad(token_list, (seq_length-len(token_list), 0), 'constant')
        token_list = np.array([token_list])
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        output_word = tokenizer.index_word.get(predicted_word_index, '')
        seed_text += " " + output_word
    return seed_text

# Test the model
seed_text = "Hello world"
generated_text = generate_text(seed_text, next_words=20)
print(generated_text)