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

# CHANGE 6: Updated RNN model class to support multiple layers
class CharRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm', num_layers=1):
        super(CharRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Choose RNN type based on parameter
        if rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
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

# CHANGE 9: Updated hyperparameters and model variations
sequence_lengths = [20, 30, 50]  # Sequence lengths to try
rnn_types = ['lstm', 'gru']      # RNN types to try
hidden_sizes = [64, 128, 256]    # Hidden sizes to try
num_layers_options = [1, 2, 3]   # Number of layers to try

learning_rate = 0.005
epochs = 100
batch_size = 128

# Create directory for saving models and data
save_dir = 'saved_models_shakespear'
data_dir = 'model_data_shakespear'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Create a dataframe to store performance metrics
performance_metrics = []

# Train selected configurations to avoid training all possible combinations
# We'll use a subset of all possible combinations to keep training time reasonable
selected_configurations = [
    # Base configurations from original code
    # {'rnn_type': 'lstm', 'seq_len': 20, 'hidden_size': 128, 'num_layers': 1},
    # {'rnn_type': 'lstm', 'seq_len': 30, 'hidden_size': 128, 'num_layers': 1},
    # {'rnn_type': 'lstm', 'seq_len': 50, 'hidden_size': 128, 'num_layers': 1},
    # {'rnn_type': 'gru', 'seq_len': 20, 'hidden_size': 128, 'num_layers': 1},
    # {'rnn_type': 'gru', 'seq_len': 30, 'hidden_size': 128, 'num_layers': 1},
    # {'rnn_type': 'gru', 'seq_len': 50, 'hidden_size': 128, 'num_layers': 1},
    
    # Additional hidden size variations (with fixed sequence length and num_layers)
    # {'rnn_type': 'lstm', 'seq_len': 30, 'hidden_size': 64, 'num_layers': 1},
    {'rnn_type': 'lstm', 'seq_len': 30, 'hidden_size': 256, 'num_layers': 1},
    # {'rnn_type': 'gru', 'seq_len': 30, 'hidden_size': 64, 'num_layers': 1},
    # {'rnn_type': 'gru', 'seq_len': 30, 'hidden_size': 256, 'num_layers': 1},
    
    # # Additional layer variations (with fixed sequence length and hidden_size)
    # {'rnn_type': 'lstm', 'seq_len': 30, 'hidden_size': 128, 'num_layers': 2},
    # {'rnn_type': 'lstm', 'seq_len': 30, 'hidden_size': 128, 'num_layers': 3},
    # {'rnn_type': 'gru', 'seq_len': 30, 'hidden_size': 128, 'num_layers': 2},
    # {'rnn_type': 'gru', 'seq_len': 30, 'hidden_size': 128, 'num_layers': 3},
]

# Train and save multiple models with performance metrics
for config in selected_configurations:
    rnn_type = config['rnn_type']
    seq_len = config['seq_len']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    
    print(f"\n{'='*50}")
    print(f"Training {rnn_type.upper()} model with:")
    print(f"- Sequence length: {seq_len}")
    print(f"- Hidden size: {hidden_size}")
    print(f"- Number of layers: {num_layers}")
    print(f"{'='*50}")
    
    # Prepare dataloaders with the current sequence length
    train_loader, test_loader = prepare_dataloaders(text, seq_len, batch_size)
    print(f"Created data loaders with sequence length {seq_len}")
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model and move it to the selected device
    model = CharRNNModel(
        input_size=len(chars), 
        hidden_size=hidden_size, 
        output_size=len(chars), 
        rnn_type=rnn_type,
        num_layers=num_layers
    ).to(device)
    
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
        'Hidden Size': hidden_size,
        'Num Layers': num_layers,
        'Parameters': num_params,
        'Model Size (MB)': model_size_mb,
        'Training Time (s)': training_time,
        'Final Training Loss': final_train_loss,
        'Final Validation Loss': final_val_loss,
        'Final Validation Accuracy': final_val_accuracy,
        'Device': str(device)
    })
    
    # Save model
    model_filename = f"{save_dir}/{rnn_type}_seq{seq_len}_h{hidden_size}_l{num_layers}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.Adam(model.parameters(), lr=learning_rate).state_dict(),
        'rnn_type': rnn_type,
        'sequence_length': seq_len,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
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

# Save each model's metrics and training history to its own CSV file
for i, metric in enumerate(performance_metrics):
    # Create individual CSV file for each model
    model_type = metric['Model Type'].lower()
    seq_len = metric['Sequence Length']
    hidden_size = metric['Hidden Size']
    num_layers = metric['Num Layers']
    
    # Create a single-row DataFrame for this model
    model_df = pd.DataFrame([metric])
    
    # Save to individual CSV in the data directory
    csv_filename = f"{data_dir}/{model_type}_seq{seq_len}_h{hidden_size}_l{num_layers}_metrics.csv"
    model_df.to_csv(csv_filename, index=False)
    print(f"Metrics for {model_type.upper()} (seq={seq_len}, h={hidden_size}, l={num_layers}) saved to {csv_filename}")
    
    # Also save the training history for this model
    # Get the filename of the saved model to load the history
    model_filename = f"{save_dir}/{model_type}_seq{seq_len}_h{hidden_size}_l{num_layers}.pt"
    if os.path.exists(model_filename):
        checkpoint = torch.load(model_filename)
        history_df = pd.DataFrame(checkpoint['training_history'])
        history_csv = f"{data_dir}/{model_type}_seq{seq_len}_h{hidden_size}_l{num_layers}_history.csv"
        history_df.to_csv(history_csv, index=False)
        print(f"Training history saved to {history_csv}")

# Display summary of trained models
print("\n" + "="*80)
print(f"TRAINED {len(performance_metrics)} MODELS")
print("="*80)
for i, metric in enumerate(performance_metrics):
    model_type = metric['Model Type']
    seq_len = metric['Sequence Length']
    hidden_size = metric['Hidden Size']
    num_layers = metric['Num Layers']
    accuracy = metric['Final Validation Accuracy']
    
    print(f"{i+1}. {model_type} (seq={seq_len}, h={hidden_size}, l={num_layers}): Accuracy = {accuracy:.4f}")
print("="*80)

# # Create a metadata file with information about all the experiments
# metadata = {
#     'experiment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
#     'total_models_trained': len(performance_metrics),
#     'rnn_types': rnn_types,
#     'sequence_lengths': sequence_lengths,
#     'hidden_sizes': hidden_sizes,
#     'num_layers_options': num_layers_options,
#     'learning_rate': learning_rate,
#     'epochs': epochs,
#     'batch_size': batch_size,
#     'device': str(device),
#     'vocabulary_size': len(chars)
# }

# # Save metadata as JSON
# import json
# with open(f"{data_dir}/experiment_metadata.json", 'w') as f:
#     json.dump(metadata, f, indent=4)

print("\nAll models trained and saved successfully!")
print(f"Model checkpoints saved to: {save_dir}/")
print(f"Performance metrics and histories saved to: {data_dir}/")
print("You can use these saved CSV files for visualization and further analysis.")