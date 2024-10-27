import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('daily-min-temperatures.csv', parse_dates=['Date'], index_col='Date')
data = data[['Temp']]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Temp'] = scaler.fit_transform(data)

# Prepare sequences for LSTM
def create_sequences(data, seq_length=30):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

# Create sequences (30 days to predict the next day)
seq_length = 30
X, y = create_sequences(data['Temp'].values, seq_length)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# Initialize model, loss function, and optimizer
model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for seq, labels in zip(X, y):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs} Loss: {single_loss.item():.4f}')


# Predict the next 7 days
model.eval()
future_days = 7
predictions = []
seq = X[-1]  # Last observed sequence

with torch.no_grad():
    for _ in range(future_days):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        pred = model(seq)
        predictions.append(pred.item())
        seq = torch.cat((seq[1:], pred.view(1)))  # Slide window by one

# Inverse transform to get actual values
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Print predictions
print("Next 7 days' predicted temperatures:", predictions)


# Plot the historical temperatures along with predicted temperatures
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(data.index, scaler.inverse_transform(data['Temp'].values.reshape(-1, 1)), label='Historical Temperatures', color='blue')

# Generate future dates for the predicted values
future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1)[1:]  # Exclude the start date to align with predictions

# Plot predictions for the next 7 days
plt.plot(future_dates, predictions, label='Predicted Temperatures (Next 7 Days)', color='orange', linestyle='--', marker='o')

# Enhancing the plot with labels and legend
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Temperature Forecast for the Next 7 Days")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.tight_layout()  # Adjust layout to fit labels

plt.show()
