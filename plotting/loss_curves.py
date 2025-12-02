import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Define model paths based on models.txt
MODELS = {
    'original': {
        'baseline': '2025-12-01_12-02-50',
        'settransformer': '2025-12-01_11-42-56'
    },
    'augmented': {
        'baseline': '2025-12-01_18-48-34',
        'settransformer': '2025-12-01_18-43-04'
    }
}

# Get the directory where CSV files are located (loss_curves subdirectory)
SCRIPT_DIR = Path(__file__).parent / 'loss_curves'

def load_loss_data(timestamp, loss_type='train'):
    """Load loss data from CSV file."""
    filename = f'run-{timestamp}_lightning_logs_version_0-tag-{loss_type}_loss.csv'
    filepath = SCRIPT_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find file: {filepath}")
    
    df = pd.read_csv(filepath)
    return df['Step'].values, df['Value'].values

def plot_comparison(data_type, loss_type='train'):
    """Plot comparison between baseline and settransformer for given data type and loss type."""
    baseline_timestamp = MODELS[data_type]['baseline']
    settransformer_timestamp = MODELS[data_type]['settransformer']
    
    # Load data
    baseline_steps, baseline_values = load_loss_data(baseline_timestamp, loss_type)
    settransformer_steps, settransformer_values = load_loss_data(settransformer_timestamp, loss_type)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_steps, baseline_values, label='Baseline', linewidth=2, alpha=0.8)
    plt.plot(settransformer_steps, settransformer_values, label='SetTransformer', linewidth=2, alpha=0.8)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(f'{loss_type.capitalize()} Loss', fontsize=12)
    plt.title(f'{data_type.capitalize()} Data: Baseline vs SetTransformer ({loss_type.capitalize()} Loss)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_filename = f'{data_type}_{loss_type}_loss_comparison.png'
    output_path = SCRIPT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    """Generate all 4 loss curve plots."""
    print("Generating loss curve plots...")
    
    # Original data plots
    print("\n1. Original data - Train loss")
    plot_comparison('original', 'train')
    
    print("\n2. Original data - Validation loss")
    plot_comparison('original', 'val')
    
    # Augmented data plots
    print("\n3. Augmented data - Train loss")
    plot_comparison('augmented', 'train')
    
    print("\n4. Augmented data - Validation loss")
    plot_comparison('augmented', 'val')
    
    print("\n✓ All plots generated successfully!")

if __name__ == '__main__':
    main()

