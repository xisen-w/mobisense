import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def load_results(file_path='model_output/model_comparison_results_NEW.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_outliers(data, threshold=2):
    """Filter out outliers using z-score method with lower threshold."""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return np.where(z_scores > threshold, np.nan, data)

def plot_training_curves(results):
    """Plot training curves for each model with outlier filtering."""
    # Create figure with more space at bottom for legend
    fig = plt.figure(figsize=(15, 14))
    
    # Create color mapping for all models except PhysicsConstrained
    models = [m for m in results.keys() if m != 'PhysicsConstrained']
    color_map = dict(zip(models, plt.cm.Set2(np.linspace(0, 1, len(models)))))
    
    # Plot training loss (log scale)
    plt.subplot(2, 1, 1)
    for model_name, model_data in results.items():
        if model_name != 'PhysicsConstrained':  # Skip PhysicsConstrained in both plots
            history = model_data['history']
            
            # Filter outliers
            train_loss = filter_outliers(np.array(history['loss']))
            val_loss = filter_outliers(np.array(history['val_loss']))
            
            color = color_map[model_name]
            plt.plot(train_loss, label=f'{model_name} (Train)', color=color)
            plt.plot(val_loss, label=f'{model_name} (Val)', color=color, linestyle='--', alpha=0.7)
    
    plt.title('Training and Validation Loss Curves (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.ylim(1e-3, 1e-1)  # Adjusted y-axis limits
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot MAE
    plt.subplot(2, 1, 2)
    for model_name, model_data in results.items():
        if model_name != 'PhysicsConstrained':
            history = model_data['history']
            
            # Filter outliers
            train_mae = filter_outliers(np.array(history['mae']), threshold=1.5)
            val_mae = filter_outliers(np.array(history['val_mae']), threshold=1.5)
            
            color = color_map[model_name]
            plt.plot(train_mae, label=f'{model_name} (Train)', color=color)
            plt.plot(val_mae, label=f'{model_name} (Val)', color=color, linestyle='--', alpha=0.7)
    
    plt.title('Training and Validation MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.ylim(0, 0.5)  # Adjusted y-axis limits
    plt.grid(True)
    
    # Create a single legend for both subplots at the bottom
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    # Adjust subplot spacing to make room for legend
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('model_output/training_curves_filtered.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_final_metrics_comparison(results):
    """Plot comparison of final metrics across models."""
    metrics_data = []
    for model_name, model_data in results.items():
        if model_name != 'PhysicsConstrained':
            history = model_data['history']
            # Get the best validation metrics instead of final ones
            best_epoch = np.argmin(history['val_loss'])
            metrics_data.append({
                'Model': model_name,
                'Best Train Loss': history['loss'][best_epoch],
                'Best Val Loss': history['val_loss'][best_epoch],
                'Best Train MAE': history['mae'][best_epoch],
                'Best Val MAE': history['val_mae'][best_epoch]
            })
    
    df = pd.DataFrame(metrics_data)
    
    # Sort by validation MAE
    df = df.sort_values('Best Val MAE')
    
    # Plot best loss comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['Best Train Loss'], width, label='Train')
    plt.bar(x + width/2, df['Best Val Loss'], width, label='Validation')
    
    plt.title('Best Loss Comparison Across Models')
    plt.xticks(x, df['Model'], rotation=45, ha='right')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.ylim(1e-3, 1e-1)  # Adjusted y-axis limits
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('model_output/best_loss_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot best MAE comparison
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, df['Best Train MAE'], width, label='Train')
    plt.bar(x + width/2, df['Best Val MAE'], width, label='Validation')
    
    plt.title('Best MAE Comparison Across Models')
    plt.xticks(x, df['Model'], rotation=45, ha='right')
    plt.ylabel('MAE')
    plt.ylim(0, 0.5)  # Adjusted y-axis limits
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('model_output/best_mae_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

# Commented out component-level comparisons
"""
def plot_component_comparison(results):
    #Plot comparison of model performance across different force components.
    components = ['Left Foot X', 'Left Foot Y', 'Left Foot Z', 
                 'Right Foot X', 'Right Foot Y', 'Right Foot Z']
    
    # Prepare data for plotting
    data = []
    for model_name, model_data in results.items():
        metrics = model_data['metrics']['Components']
        for component in components:
            data.append({
                'Model': model_name,
                'Component': component,
                'MAE': metrics[component]['MAE (N)'],
                'RMSE': metrics[component]['RMSE (N)'],
                'MRE': metrics[component]['MRE (%)'] if isinstance(metrics[component]['MRE (%)'], (int, float)) else np.nan
            })
    
    df = pd.DataFrame(data)
    
    # Plot MAE comparison
    plt.figure(figsize=(15, 6))
    sns.barplot(data=df, x='Component', y='MAE', hue='Model')
    plt.title('MAE Comparison Across Components')
    plt.xticks(rotation=45)
    plt.ylabel('MAE (N)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_output/component_mae_comparison_NEW.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot RMSE comparison
    plt.figure(figsize=(15, 6))
    sns.barplot(data=df, x='Component', y='RMSE', hue='Model')
    plt.title('RMSE Comparison Across Components')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE (N)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_output/component_rmse_comparison_NEW.png', bbox_inches='tight', dpi=300)
    plt.close()
"""

def plot_physics_metrics(results):
    """Plot physics-based metrics comparison."""
    physics_data = []
    for model_name, model_data in results.items():
        metrics = model_data['metrics']['Physics']
        physics_data.append({
            'Model': model_name,
            'Vertical Force Error (N)': metrics['Vertical Force Error (N)'],
            'Vertical Force Error (%)': metrics['Vertical Force Error (%)'],
            'Predicted Total Vertical Force (N)': metrics['Predicted Total Vertical Force (N)'],
            'Body Weight Force (N)': metrics['Body Weight Force (N)']
        })
    
    df = pd.DataFrame(physics_data)
    
    # Plot vertical force error
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='Vertical Force Error (N)')
    plt.title('Vertical Force Error Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Error (N)')
    plt.tight_layout()
    plt.savefig('model_output/physics_force_error_NEW.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot predicted vs body weight force
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['Predicted Total Vertical Force (N)'], width, label='Predicted')
    plt.bar(x + width/2, df['Body Weight Force (N)'], width, label='Body Weight')
    
    plt.title('Predicted vs Body Weight Force Comparison')
    plt.xticks(x, df['Model'], rotation=45)
    plt.ylabel('Force (N)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_output/physics_force_comparison_NEW.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Create output directory if it doesn't exist
    Path('model_output').mkdir(exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Generate plots
    plot_training_curves(results)
    plot_final_metrics_comparison(results)
    
    print("Visualization completed! Check the model_output directory for the generated plots.")

if __name__ == "__main__":
    main() 