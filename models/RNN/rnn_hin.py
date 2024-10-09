
# Example code for implementing a Transformer-like model with RNN-based architecture
import torch
import torch.nn as nn
import torch.optim as optim

class MinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(MinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return output

class TransformerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(TransformerRNN, self).__init__()
        
        # Using minimal RNN (minLSTM) as a base layer
        self.rnn = MinLSTM(input_size, hidden_size)
        
        # Adding a multi-head attention layer for Transformer-like attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        
        # Fully connected layer to get the final output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Passing the input through the RNN layer
        rnn_output = self.rnn(x)
        
        # Reshaping for compatibility with multi-head attention
        rnn_output = rnn_output.permute(1, 0, 2)  # Change to (sequence_length, batch_size, hidden_size)
        
        # Passing the RNN output through the attention layer
        attn_output, _ = self.attention(rnn_output, rnn_output, rnn_output)
        
        # Passing the attention output through a fully connected layer
        attn_output = attn_output.permute(1, 0, 2)  # Change back to (batch_size, sequence_length, hidden_size)
        output = self.fc(attn_output[:, -1, :])  # Only take the output of the last time step
        
        return output

# Example training setup
def train_transformer_rnn():
    # Hyperparameters
    input_size = 10
    hidden_size = 32
    num_layers = 2
    num_heads = 4
    output_size = 1
    num_epochs = 5
    learning_rate = 0.001

    # Model, loss function, optimizer
    model = TransformerRNN(input_size, hidden_size, num_layers, num_heads, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy data for demonstration purposes
    inputs = torch.randn(64, 10, input_size)  # (batch_size, sequence_length, input_size)
    targets = torch.randn(64, output_size)  # (batch_size, output_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    train_transformer_rnn()