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

# Get the directory where CSV files are located (cer subdirectory)
SCRIPT_DIR = Path(__file__).parent / 'cer'

def load_cer_data(timestamp):
    """Load CER data from CSV file."""
    filename = f'run-{timestamp}_lightning_logs_version_0-tag-val_CER.csv'
    filepath = SCRIPT_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find file: {filepath}")
    
    df = pd.read_csv(filepath)
    return df['Step'].values, df['Value'].values

def plot_comparison(data_type):
    """Plot comparison between baseline and settransformer for given data type."""
    baseline_timestamp = MODELS[data_type]['baseline']
    settransformer_timestamp = MODELS[data_type]['settransformer']
    
    # Load data
    baseline_steps, baseline_values = load_cer_data(baseline_timestamp)
    settransformer_steps, settransformer_values = load_cer_data(settransformer_timestamp)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_steps, baseline_values, label='Baseline', linewidth=2, alpha=0.8)
    plt.plot(settransformer_steps, settransformer_values, label='SetTransformer', linewidth=2, alpha=0.8)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Validation CER (%)', fontsize=12)
    plt.title(f'{data_type.capitalize()} Data: Baseline vs SetTransformer (Validation CER)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_filename = f'{data_type}_val_cer_comparison.png'
    output_path = SCRIPT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    """Generate CER comparison plots."""
    print("Generating CER curve plots...")
    
    # Original data plot
    print("\n1. Original data - Validation CER")
    plot_comparison('original')
    
    # Augmented data plot
    print("\n2. Augmented data - Validation CER")
    plot_comparison('augmented')
    
    print("\n✓ All plots generated successfully!")

if __name__ == '__main__':
    main()


