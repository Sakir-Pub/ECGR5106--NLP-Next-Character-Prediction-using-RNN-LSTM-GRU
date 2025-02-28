import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import requests

# CHANGE 1: Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CHANGE 2: Download the Shakespeare dataset instead of reading local file
def download_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading Shakespeare data from {url}")
    response = requests.get(url)
    text = response.text
    print(f"Downloaded data with {len(text)} characters")
    return text

text = download_shakespeare_data()

# CHANGE 3: Create character mappings
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary size: {len(chars)} unique characters")

# CHANGE 4: Custom dataset class (from provided code)
class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

# CHANGE 5: Function to prepare dataset with variable sequence length
def prepare_dataloaders(text, sequence_length, batch_size=128):
    # Encode the text into integers
    encoded_text = [char_to_int[ch] for ch in text]
    
    # Create sequences and targets
    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length):
        seq = encoded_text[i:i+sequence_length]
        target = encoded_text[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    # Convert lists to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Instantiate the dataset
    dataset = CharDataset(sequences, targets)
    
    # Create train and test splits
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    return train_loader, test_loader

# CHANGE 6: RNN model class (updated for batch processing)
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

# CHANGE 7: Updated training function for DataLoader
def train_model(model, train_loader, test_loader, epochs=100, learning_rate=0.005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    training_history = []
    
    # Start timing for training duration
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for batch_sequences, batch_targets in train_loader:
            # Move data to device
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch_sequences)
            loss = criterion(output, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / total_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                # Move data to device
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                
                # Forward pass
                output = model(batch_sequences)
                loss = criterion(output, batch_targets)
                
                # Calculate metrics
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_correct += (predicted == batch_targets).sum().item()
                val_total += batch_targets.size(0)
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = val_correct / val_total
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    # Calculate training duration
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds")
    
    return training_history, training_duration

# CHANGE 8: Updated prediction function
def predict_next_char(model, initial_str, sequence_length):
    model.eval()
    with torch.no_grad():
        # Convert string to indices
        char_indices = [char_to_int[c] for c in initial_str[-sequence_length:]]
        # Create input tensor
        initial_input = torch.tensor([char_indices], dtype=torch.long).to(device)
        # Get prediction
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return int_to_char[predicted_index]

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to plot training history
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

# CHANGE 9: Updated hyperparameters and sequence lengths
hidden_size = 128
learning_rate = 0.005
epochs = 100
batch_size = 128
sequence_lengths = [20, 30, 50]  # Changed to 20, 30, 50
rnn_types = ['lstm', 'gru']

# Create directories for saving models and plots
save_dir = 'saved_models_shakespear'
plots_dir = 'plots_shakespear'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Create a dataframe to store performance metrics
performance_metrics = []

# Train and save multiple models with performance metrics
for rnn_type in rnn_types:
    for seq_len in sequence_lengths:
        print(f"\n{'='*50}")
        print(f"Training {rnn_type.upper()} model with sequence length {seq_len}")
        print(f"{'='*50}")
        
        # Prepare dataloaders with the current sequence length
        train_loader, test_loader = prepare_dataloaders(text, seq_len, batch_size)
        print(f"Created data loaders with sequence length {seq_len}")
        print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        # Initialize model and move it to the selected device
        model = CharRNNModel(len(chars), hidden_size, len(chars), rnn_type=rnn_type).to(device)
        
        # Count parameters before training
        num_params = count_parameters(model)
        print(f"Model has {num_params:,} trainable parameters")
        
        # Calculate model size in MB
        model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
        print(f"Approximate model size: {model_size_mb:.2f} MB")
        
        # Train model and measure time
        history, training_time = train_model(model, train_loader, test_loader, epochs=epochs, learning_rate=learning_rate)
        
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
            'char_to_int': char_to_int,
            'int_to_char': int_to_char,
            'vocabulary_size': len(chars),
            'training_history': history,
            'num_parameters': num_params,
            'model_size_mb': model_size_mb,
            'training_time': training_time
        }, model_filename)
        
        print(f"Model saved to {model_filename}")
        
        # Test prediction
        test_str = text[:seq_len]  # Use the first seq_len characters from the text
        predicted_char = predict_next_char(model, test_str, seq_len)
        print(f"Test prediction with input '{test_str[:10]}...': Next character: '{predicted_char}'")

# Convert performance metrics to DataFrame and save to CSV
metrics_df = pd.DataFrame(performance_metrics)
metrics_csv_path = 'model_performance_metrics_shakespear.csv'
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nPerformance metrics saved to {metrics_csv_path}")

# Display performance comparison table
print("\n" + "="*80)
print("PERFORMANCE COMPARISON ACROSS MODELS")
print("="*80)
print(metrics_df.to_string())
print("="*80)

# Create visualization for comparative analysis
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

# Plot loss curves for all models
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