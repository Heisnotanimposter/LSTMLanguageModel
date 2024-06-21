from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create a Sequential model
model = Sequential()

# Flatten the input data
model.add(Flatten(input_shape=(28, 28)))

# Add a Dense layer with a linear activation function
model.add(Dense(10, activation='linear'))

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

import matplotlib.pyplot as plt

# Assuming X_train contains the MNIST images and Y_train contains the corresponding labels

# Select an image to visualize
image_index = 0 # You can change this index to visualize different images
selected_image = X_train[image_index].reshape(28, 28) # Reshape the image to 28x28 if needed

# Plot the selected image
plt.imshow(selected_image, cmap='gray')
plt.title(f'Label: {Y_train[image_index]}')
plt.show()

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i in range(9): # Loop over the first 9 images
    # Select the image and label
    image = X_train[i].reshape(28, 28) # Reshape the image to 28x28 if needed
    label = Y_train[i]

    # Plot the image in the ith subplot
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off') # Hide the axis ticks and labels

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()
