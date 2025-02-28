import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

# CHANGE: Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CHANGE 1: Read text from file instead of using a hardcoded sample
file_path = '/home/anabil/Development/PhD Courses/ECGR 5106/HW3P1/Dataset/hw_sequence.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

print(f"Loaded text with {len(text)} characters")

# Creating character vocabulary
chars = sorted(list(set(text)))
ix_to_char = {i: ch for i, ch in enumerate(chars)}
char_to_ix = {ch: i for i, ch in enumerate(chars)}

# CHANGE 2: Define a function to prepare dataset with variable sequence length
def prepare_dataset(text, sequence_length):
    X = []
    y = []
    for i in range(len(text) - sequence_length):
        sequence = text[i:i + sequence_length]
        label = text[i + sequence_length]
        X.append([char_to_ix[char] for char in sequence])
        y.append(char_to_ix[label])
    
    X = np.array(X)
    y = np.array(y)
    
    # Splitting the dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Converting to PyTorch tensors and moving them to the selected device
    X_train = torch.tensor(X_train, dtype=torch.long).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    return X_train, y_train, X_val, y_val

# CHANGE 3: Modify the RNN model to support different RNN types
class CharRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm'):
        super(CharRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Choose RNN type based on parameter
        if rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
            
        self.fc = nn.Linear(hidden_size, output_size)
        self.rnn_type = rnn_type.lower()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # Get the output of the last RNN cell
        return output

# CHANGE 4: Create a training function with performance tracking
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    training_history = []
    
    # Start timing for training duration
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            _, predicted = torch.max(val_output, 1)
            val_accuracy = (predicted == y_val).float().mean()
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': loss.item(),
            'val_loss': val_loss.item(),
            'val_accuracy': val_accuracy.item()
        })
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item():.4f}')
    
    # Calculate training duration
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds")
    
    return training_history, training_duration

# CHANGE 5: Create a prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str, sequence_length):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-sequence_length:]], dtype=torch.long).unsqueeze(0).to(device)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

# CHANGE 6: Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# CHANGE 7: Function to plot training history
def plot_training_history(histories, model_names, metric='loss', save_path=None):
    plt.figure(figsize=(12, 8))
    
    for history, name in zip(histories, model_names):
        data = [entry[metric] for entry in history]
        epochs = [entry['epoch'] for entry in history]
        plt.plot(epochs, data, label=name)
    
    plt.title(f'Training {metric} over epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# Hyperparameters
hidden_size = 128
learning_rate = 0.005
epochs = 100

# Configuration for different models
rnn_types = ['rnn', 'lstm', 'gru']
sequence_lengths = [10, 20, 30]

# Create directories for saving models and plots
save_dir = 'saved_models'
plots_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# CHANGE 8: Create a dataframe to store performance metrics
performance_metrics = []

# CHANGE 9: Train and save multiple models with performance metrics
for rnn_type in rnn_types:
    for seq_len in sequence_lengths:
        print(f"\n{'='*50}")
        print(f"Training {rnn_type.upper()} model with sequence length {seq_len}")
        print(f"{'='*50}")
        
        # Prepare dataset with the current sequence length
        X_train, y_train, X_val, y_val = prepare_dataset(text, seq_len)
        
        # Initialize model and move it to the selected device
        model = CharRNNModel(len(chars), hidden_size, len(chars), rnn_type=rnn_type).to(device)
        
        # Count parameters before training
        num_params = count_parameters(model)
        print(f"Model has {num_params:,} trainable parameters")
        
        # Calculate model size in MB
        model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
        print(f"Approximate model size: {model_size_mb:.2f} MB")
        
        # Train model and measure time
        history, training_time = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, learning_rate=learning_rate)
        
        # Get final metrics
        final_train_loss = history[-1]['loss']
        final_val_loss = history[-1]['val_loss']
        final_val_accuracy = history[-1]['val_accuracy']
        
        # Track metrics
        performance_metrics.append({
            'Model Type': rnn_type.upper(),
            'Sequence Length': seq_len,
            'Parameters': num_params,
            'Model Size (MB)': model_size_mb,
            'Training Time (s)': training_time,
            'Final Training Loss': final_train_loss,
            'Final Validation Loss': final_val_loss,
            'Final Validation Accuracy': final_val_accuracy,
            'Device': str(device)
        })
        
        # Save model
        model_filename = f"{save_dir}/{rnn_type}_seq{seq_len}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.Adam(model.parameters(), lr=learning_rate).state_dict(),
            'rnn_type': rnn_type,
            'sequence_length': seq_len,
            'hidden_size': hidden_size,
            'char_to_ix': char_to_ix,
            'ix_to_char': ix_to_char,
            'vocabulary_size': len(chars),
            'training_history': history,
            'num_parameters': num_params,
            'model_size_mb': model_size_mb,
            'training_time': training_time
        }, model_filename)
        
        print(f"Model saved to {model_filename}")
        
        # Test prediction
        test_str = text[:seq_len]  # Use the first seq_len characters from the text
        predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str, seq_len)
        print(f"Test prediction with input '{test_str}': Next character: '{predicted_char}'")

# CHANGE: 10 Convert performance metrics to DataFrame and save to CSV
metrics_df = pd.DataFrame(performance_metrics)
metrics_csv_path = 'model_performance_metrics.csv'
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nPerformance metrics saved to {metrics_csv_path}")

# CHANGE 11: Display performance comparison table
print("\n" + "="*80)
print("PERFORMANCE COMPARISON ACROSS MODELS")
print("="*80)
print(metrics_df.to_string())
print("="*80)

# CHANGE 12: Create visualization for comparative analysis
# Group metrics by model type and sequence length for plotting
grouped_metrics = {}
for rnn_type in rnn_types:
    for seq_len in sequence_lengths:
        model_key = f"{rnn_type.upper()}_seq{seq_len}"
        model_data = next((m for m in performance_metrics if m['Model Type'] == rnn_type.upper() and m['Sequence Length'] == seq_len), None)
        if model_data:
            grouped_metrics[model_key] = model_data

# Plot training time comparison
plt.figure(figsize=(12, 6))
model_names = list(grouped_metrics.keys())
train_times = [grouped_metrics[name]['Training Time (s)'] for name in model_names]

plt.bar(model_names, train_times)
plt.title('Training Time Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{plots_dir}/training_time_comparison.png")

# Plot model size comparison
plt.figure(figsize=(12, 6))
model_sizes = [grouped_metrics[name]['Model Size (MB)'] for name in model_names]

plt.bar(model_names, model_sizes)
plt.title('Model Size Comparison')
plt.xlabel('Model')
plt.ylabel('Model Size (MB)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{plots_dir}/model_size_comparison.png")

# Plot validation accuracy comparison
plt.figure(figsize=(12, 6))
accuracies = [grouped_metrics[name]['Final Validation Accuracy'] for name in model_names]

plt.bar(model_names, accuracies)
plt.title('Validation Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Validation Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{plots_dir}/validation_accuracy_comparison.png")

# CHANGE 13: Plot loss curves for all models
# Collect all training histories
all_histories = []
all_model_names = []

for rnn_type in rnn_types:
    for seq_len in sequence_lengths:
        model_filename = f"{save_dir}/{rnn_type}_seq{seq_len}.pt"
        if os.path.exists(model_filename):
            checkpoint = torch.load(model_filename)
            all_histories.append(checkpoint['training_history'])
            all_model_names.append(f"{rnn_type.upper()}_seq{seq_len}")

# Plot training loss curves
plot_training_history(
    all_histories, 
    all_model_names, 
    metric='loss', 
    save_path=f"{plots_dir}/training_loss_curves.png"
)

# Plot validation loss curves
plot_training_history(
    all_histories, 
    all_model_names, 
    metric='val_loss', 
    save_path=f"{plots_dir}/validation_loss_curves.png"
)

# Plot validation accuracy curves
plot_training_history(
    all_histories, 
    all_model_names, 
    metric='val_accuracy', 
    save_path=f"{plots_dir}/validation_accuracy_curves.png"
)

print("\nAll models trained and saved successfully!")
print(f"Performance visualizations saved to {plots_dir}/")