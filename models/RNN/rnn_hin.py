# Transformer-like RNN model with enhanced RNN features and visualization

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class TransformerRNN(nn.Module):
    """
    Transformer-like RNN model combining LSTM layers with a multi-head attention mechanism.

    Args:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        num_heads (int): Number of attention heads.
        output_size (int): Number of features in the output.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(TransformerRNN, self).__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size).
        """
        # LSTM forward pass
        lstm_output, _ = self.lstm(x)
        # lstm_output shape: (batch_size, sequence_length, hidden_size)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        # attn_output shape: (batch_size, sequence_length, hidden_size)
        
        # Use the output of the last time step
        last_output = attn_output[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        output = self.fc(last_output)  # Shape: (batch_size, output_size)
        
        return output

def train_transformer_rnn():
    """
    Training function for the TransformerRNN model with loss visualization.
    """
    # Hyperparameters
    input_size = 10        # Number of input features
    hidden_size = 64       # Number of features in hidden state
    num_layers = 2         # Number of LSTM layers
    num_heads = 4          # Number of attention heads
    output_size = 1        # Number of output features
    num_epochs = 20        # Number of training epochs
    learning_rate = 0.001  # Learning rate
    batch_size = 64        # Batch size
    sequence_length = 15   # Sequence length

    # Initialize the model
    model = TransformerRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        output_size=output_size
    )

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy data for demonstration
    inputs = torch.randn(batch_size, sequence_length, input_size)
    targets = torch.randn(batch_size, output_size)

    # List to store loss values for visualization
    loss_values = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_values.append(loss.item())
        
        # Print training progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Visualization of training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_values, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_transformer_rnn()