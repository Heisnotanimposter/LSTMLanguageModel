

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample data
text = "Hello world. This is a sample text for language modeling."

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

# Save the tokenizer to a file in Google Drive
with open('/content/drive/MyDrive/merged_model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/content/drive/MyDrive/merged_model/HINAI.tflite', 'wb') as handle:
    tflite.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Prepare the dataset
vocab_size = len(tokenizer.word_index) + 1
seq_length = 3
dataset = []
for i in range(seq_length, len(sequences)):
    dataset.append(sequences[i-seq_length:i+1])
dataset = tf.constant(dataset)

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=10, input_length=seq_length),
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(dataset[:, :-1], dataset[:, -1], epochs=200, verbose=1)

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('/content/drive/MyDrive/merged_model/HINAI.tflite', 'wb') as handle:
    handle.write(tflite_model)

