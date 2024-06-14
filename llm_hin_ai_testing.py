from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import tensorflow as tf
from google.colab import drive
import pickle

# Mount Google Drive
#drive.mount('/content/drive')

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/content/drive/MyDrive/merged_model/HINAI.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the tokenizer from Google Drive
with open('/content/drive/MyDrive/merged_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Sample input text
sample_text = "Hello world"

# Tokenize the input text
input_sequence = tokenizer.texts_to_sequences([sample_text])
input_sequence = np.array(input_sequence, dtype=np.float32)

# Ensure the input is the correct shape (batch_size, input_length)
input_sequence = np.pad(input_sequence, [(0, 0), (0, input_details[0]['shape'][1] - len(input_sequence[0]))], mode='constant')

# Set the tensor to point to the input data to be inferred
interpreter.set_tensor(input_details[0]['index'], input_sequence)

# Run the inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Convert the output indices to words
predicted_indices = np.argmax(output_data, axis=-1)
predicted_words = [tokenizer.index_word[index] for index in predicted_indices if index in tokenizer.index_word]

# Join the words to form the output sentence
predicted_sentence = ' '.join(predicted_words)
print(predicted_sentence)

print("Input sequence:", input_sequence)

# Set the tensor to point to the input data to be inferred
interpreter.set_tensor(input_details[0]['index'], input_sequence)
# Run the inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Debug: Print raw output data
print("Raw output data:", output_data)

print("Tokenizer word index:", tokenizer.word_index)

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

print("Raw output data:", output_data)

# Convert the output indices to words
predicted_indices = np.argmax(output_data, axis=-1)
predicted_words = [tokenizer.index_word[index] for index in predicted_indices if index in tokenizer.index_word]

# Join the words to form the output sentence
predicted_sentence = ' '.join(predicted_words)
print(predicted_sentence)

