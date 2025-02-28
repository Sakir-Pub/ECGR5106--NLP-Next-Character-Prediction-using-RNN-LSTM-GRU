import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
import re

def extract_model_info_from_filename(filename):
    """Extract model information from the filename pattern."""
    # Pattern: {model_type}_seq{seq_len}_h{hidden_size}_l{num_layers}
    pattern = r'([a-z]+)_seq(\d+)_h(\d+)_l(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        return {
            'Model Type': match.group(1).upper(),
            'Sequence Length': int(match.group(2)),
            'Hidden Size': int(match.group(3)),
            'Num Layers': int(match.group(4))
        }
    else:
        return None

def load_all_metrics(data_dir):
    """Load all model metrics from individual CSV files."""
    # Get all metric CSV files
    metric_files = glob.glob(f"{data_dir}/*_metrics.csv")
    
    all_metrics = []
    for file in metric_files:
        # Load metrics
        df = pd.read_csv(file)
        if not df.empty:
            all_metrics.append(df.iloc[0].to_dict())
    
    return all_metrics

def load_all_histories(data_dir):
    """Load all training histories from individual CSV files."""
    # Get all history CSV files
    history_files = glob.glob(f"{data_dir}/*_history.csv")
    
    all_histories = {}
    for file in history_files:
        # Extract model name from filename
        base_filename = os.path.basename(file).replace('_history.csv', '')
        # Load history
        df = pd.read_csv(file)
        if not df.empty:
            all_histories[base_filename] = df.to_dict('records')
    
    return all_histories

def plot_training_time_comparison(metrics, output_dir):
    """Plot training time comparison across models."""
    plt.figure(figsize=(15, 8))
    
    # Sort metrics by model type and sequence length
    sorted_metrics = sorted(metrics, key=lambda x: (x['Model Type'], x['Sequence Length'], x['Hidden Size'], x['Num Layers']))
    
    # Create model names and data lists
    model_names = [f"{m['Model Type']}_s{m['Sequence Length']}_h{m['Hidden Size']}_l{m['Num Layers']}" for m in sorted_metrics]
    train_times = [m['Training Time (s)'] for m in sorted_metrics]
    
    # Create bar chart
    plt.bar(range(len(model_names)), train_times)
    plt.xticks(range(len(model_names)), model_names, rotation=90)
    plt.title('Training Time Comparison Across Models')
    plt.xlabel('Model Configuration')
    plt.ylabel('Training Time (seconds)')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/training_time_comparison.png")
    plt.close()
    print(f"Training time comparison saved to {output_dir}/training_time_comparison.png")

def plot_model_size_comparison(metrics, output_dir):
    """Plot model size comparison across models."""
    plt.figure(figsize=(15, 8))
    
    # Sort metrics by model type and sequence length
    sorted_metrics = sorted(metrics, key=lambda x: (x['Model Type'], x['Sequence Length'], x['Hidden Size'], x['Num Layers']))
    
    # Create model names and data lists
    model_names = [f"{m['Model Type']}_s{m['Sequence Length']}_h{m['Hidden Size']}_l{m['Num Layers']}" for m in sorted_metrics]
    model_sizes = [m['Model Size (MB)'] for m in sorted_metrics]
    
    # Create bar chart
    plt.bar(range(len(model_names)), model_sizes)
    plt.xticks(range(len(model_names)), model_names, rotation=90)
    plt.title('Model Size Comparison')
    plt.xlabel('Model Configuration')
    plt.ylabel('Model Size (MB)')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/model_size_comparison.png")
    plt.close()
    print(f"Model size comparison saved to {output_dir}/model_size_comparison.png")

def plot_accuracy_comparison(metrics, output_dir):
    """Plot validation accuracy comparison across models."""
    plt.figure(figsize=(15, 8))
    
    # Sort metrics by model type and sequence length
    sorted_metrics = sorted(metrics, key=lambda x: (x['Model Type'], x['Sequence Length'], x['Hidden Size'], x['Num Layers']))
    
    # Create model names and data lists
    model_names = [f"{m['Model Type']}_s{m['Sequence Length']}_h{m['Hidden Size']}_l{m['Num Layers']}" for m in sorted_metrics]
    accuracies = [m['Final Validation Accuracy'] for m in sorted_metrics]
    
    # Create bar chart
    plt.bar(range(len(model_names)), accuracies)
    plt.xticks(range(len(model_names)), model_names, rotation=90)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Model Configuration')
    plt.ylabel('Validation Accuracy')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/validation_accuracy_comparison.png")
    plt.close()
    print(f"Validation accuracy comparison saved to {output_dir}/validation_accuracy_comparison.png")

def filter_metrics(metrics, fixed_params, variable_param):
    """Filter metrics based on fixed parameters and extract variable parameter values."""
    results = []
    labels = []
    
    for m in metrics:
        match = True
        for param, value in fixed_params.items():
            if m[param] != value:
                match = False
                break
        
        if match:
            results.append(m['Final Validation Accuracy'])
            labels.append(str(m[variable_param]))
    
    return results, labels

def plot_parameter_effects(metrics, output_dir):
    """Plot the effects of various parameters on model accuracy."""
    # Effect of hidden size on LSTM accuracy
    plt.figure(figsize=(10, 6))
    accuracies, hidden_sizes = filter_metrics(
        metrics, 
        {'Model Type': 'LSTM', 'Sequence Length': 30, 'Num Layers': 1}, 
        'Hidden Size'
    )
    if accuracies and hidden_sizes:
        plt.bar(hidden_sizes, accuracies)
        plt.title('Effect of Hidden Size on LSTM Accuracy')
        plt.xlabel('Hidden Size')
        plt.ylabel('Validation Accuracy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hidden_size_effect_lstm.png")
        plt.close()
        print(f"Hidden size effect on LSTM saved to {output_dir}/hidden_size_effect_lstm.png")
    
    # Effect of number of layers on LSTM accuracy
    plt.figure(figsize=(10, 6))
    accuracies, num_layers = filter_metrics(
        metrics, 
        {'Model Type': 'LSTM', 'Sequence Length': 30, 'Hidden Size': 128}, 
        'Num Layers'
    )
    if accuracies and num_layers:
        plt.bar(num_layers, accuracies)
        plt.title('Effect of Number of Layers on LSTM Accuracy')
        plt.xlabel('Number of Layers')
        plt.ylabel('Validation Accuracy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/num_layers_effect_lstm.png")
        plt.close()
        print(f"Number of layers effect on LSTM saved to {output_dir}/num_layers_effect_lstm.png")
    
    # Effect of hidden size on GRU accuracy
    plt.figure(figsize=(10, 6))
    accuracies, hidden_sizes = filter_metrics(
        metrics, 
        {'Model Type': 'GRU', 'Sequence Length': 30, 'Num Layers': 1}, 
        'Hidden Size'
    )
    if accuracies and hidden_sizes:
        plt.bar(hidden_sizes, accuracies)
        plt.title('Effect of Hidden Size on GRU Accuracy')
        plt.xlabel('Hidden Size')
        plt.ylabel('Validation Accuracy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hidden_size_effect_gru.png")
        plt.close()
        print(f"Hidden size effect on GRU saved to {output_dir}/hidden_size_effect_gru.png")
    
    # Effect of number of layers on GRU accuracy
    plt.figure(figsize=(10, 6))
    accuracies, num_layers = filter_metrics(
        metrics, 
        {'Model Type': 'GRU', 'Sequence Length': 30, 'Hidden Size': 128}, 
        'Num Layers'
    )
    if accuracies and num_layers:
        plt.bar(num_layers, accuracies)
        plt.title('Effect of Number of Layers on GRU Accuracy')
        plt.xlabel('Number of Layers')
        plt.ylabel('Validation Accuracy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/num_layers_effect_gru.png")
        plt.close()
        print(f"Number of layers effect on GRU saved to {output_dir}/num_layers_effect_gru.png")

def plot_training_histories(histories, output_dir):
    """Plot training histories for all models."""
    # Group histories by model type
    lstm_histories = {k: v for k, v in histories.items() if 'lstm' in k.lower()}
    gru_histories = {k: v for k, v in histories.items() if 'gru' in k.lower()}
    
    # Plot training loss curves for LSTM models
    if lstm_histories:
        plt.figure(figsize=(12, 8))
        for name, history in lstm_histories.items():
            data = [entry['loss'] for entry in history]
            epochs = [entry['epoch'] for entry in history]
            plt.plot(epochs, data, label=name)
        
        plt.title('LSTM Models: Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lstm_training_loss_curves.png")
        plt.close()
        print(f"LSTM training loss curves saved to {output_dir}/lstm_training_loss_curves.png")
    
    # Plot validation loss curves for LSTM models
    if lstm_histories:
        plt.figure(figsize=(12, 8))
        for name, history in lstm_histories.items():
            data = [entry['val_loss'] for entry in history]
            epochs = [entry['epoch'] for entry in history]
            plt.plot(epochs, data, label=name)
        
        plt.title('LSTM Models: Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lstm_validation_loss_curves.png")
        plt.close()
        print(f"LSTM validation loss curves saved to {output_dir}/lstm_validation_loss_curves.png")
    
    # Plot validation accuracy curves for LSTM models
    if lstm_histories:
        plt.figure(figsize=(12, 8))
        for name, history in lstm_histories.items():
            data = [entry['val_accuracy'] for entry in history]
            epochs = [entry['epoch'] for entry in history]
            plt.plot(epochs, data, label=name)
        
        plt.title('LSTM Models: Validation Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lstm_validation_accuracy_curves.png")
        plt.close()
        print(f"LSTM validation accuracy curves saved to {output_dir}/lstm_validation_accuracy_curves.png")
    
    # Plot training loss curves for GRU models
    if gru_histories:
        plt.figure(figsize=(12, 8))
        for name, history in gru_histories.items():
            data = [entry['loss'] for entry in history]
            epochs = [entry['epoch'] for entry in history]
            plt.plot(epochs, data, label=name)
        
        plt.title('GRU Models: Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gru_training_loss_curves.png")
        plt.close()
        print(f"GRU training loss curves saved to {output_dir}/gru_training_loss_curves.png")
    
    # Plot validation loss curves for GRU models
    if gru_histories:
        plt.figure(figsize=(12, 8))
        for name, history in gru_histories.items():
            data = [entry['val_loss'] for entry in history]
            epochs = [entry['epoch'] for entry in history]
            plt.plot(epochs, data, label=name)
        
        plt.title('GRU Models: Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gru_validation_loss_curves.png")
        plt.close()
        print(f"GRU validation loss curves saved to {output_dir}/gru_validation_loss_curves.png")
    
    # Plot validation accuracy curves for GRU models
    if gru_histories:
        plt.figure(figsize=(12, 8))
        for name, history in gru_histories.items():
            data = [entry['val_accuracy'] for entry in history]
            epochs = [entry['epoch'] for entry in history]
            plt.plot(epochs, data, label=name)
        
        plt.title('GRU Models: Validation Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gru_validation_accuracy_curves.png")
        plt.close()
        print(f"GRU validation accuracy curves saved to {output_dir}/gru_validation_accuracy_curves.png")

def main():
    # Set directories
    data_dir = 'model_data_shakespear'  # Directory containing model data CSV files
    output_dir = 'visualizations'       # Directory for saving visualization plots
    
    print(f"Loading model data from {data_dir}...")
    
    # Load all metrics from individual CSV files
    all_metrics = load_all_metrics(data_dir)
    print(f"Loaded metrics for {len(all_metrics)} models")
    
    # Load all training histories
    all_histories = load_all_histories(data_dir)
    print(f"Loaded training histories for {len(all_histories)} models")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_training_time_comparison(all_metrics, output_dir)
    plot_model_size_comparison(all_metrics, output_dir)
    plot_accuracy_comparison(all_metrics, output_dir)
    plot_parameter_effects(all_metrics, output_dir)
    plot_training_histories(all_histories, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()