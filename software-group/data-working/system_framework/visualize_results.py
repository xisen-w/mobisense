import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

# Models to exclude from visualizations
EXCLUDE_MODELS = ["PhysicsConstrainedModel", "HybridTransformerPhysics", "PhysicsConstrained"]

def load_results(json_path):
    """Loads the results dictionary from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
        print(f"Successfully loaded results from: {json_path}")
        
        # Check if the results are nested under 'model_results'
        if 'model_results' in results and isinstance(results['model_results'], dict):
            data_to_process = results['model_results']
            print("Found results nested under 'model_results'. Processing these.")
        else:
            # Assume old format or top-level results if 'model_results' is not found/valid
            data_to_process = results
            print("Processing results from the top level (assuming no 'model_results' key or old format).")
        
        # Filter out excluded models from the actual data dictionary
        filtered_results = {k: v for k, v in data_to_process.items() if k not in EXCLUDE_MODELS and k != 'config'} # Also exclude 'config' key explicitly

        print(f"Excluded models: {EXCLUDE_MODELS}")
        print(f"Models included in visualization: {list(filtered_results.keys())}")
        return filtered_results
    except FileNotFoundError:
        print(f"Error: Results file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}. File might be corrupted or empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {json_path}: {e}")
        return None

def plot_performance_comparison(results, output_path):
    """Plots a comparison of final performance metrics (MAE, RMSE) for each model."""
    model_names = list(results.keys()) # Already filtered by load_results
    metrics_data = []

    print("\nExtracting performance metrics...")
    for model_name in model_names:
        # Check if final eval metrics exist and are not None/empty
        final_eval_metrics = results[model_name].get('final_evaluation') # Use 'final_evaluation' key
        history_stage2 = results[model_name].get('history_stage2', {}) # Use 'history_stage2' key
        # Fallback evaluation structures (less likely now but for safety)
        eval_loss = results[model_name].get('metrics', {}).get('loss') 
        eval_mae = results[model_name].get('metrics', {}).get('mae')

        mae = None
        rmse = None # Initialize rmse as None

        # Use final_eval_metrics first
        if final_eval_metrics: 
            overall_metrics = final_eval_metrics.get('Overall', {})
            mae = overall_metrics.get('MAE (N)')
            rmse = overall_metrics.get('RMSE (N)')
            # Also check for final val_mae/val_loss stored directly in final_evaluation
            if mae is None:
                 mae = final_eval_metrics.get('val_mae') # Check if final val_mae was added
            if mae is not None:
                 print(f"  {model_name}: Found final evaluation metrics - MAE={mae:.4f}, RMSE={rmse if rmse is not None else 'N/A'}")
            else:
                 # Try getting val_mae from history if not in final_eval
                 final_val_mae_hist = history_stage2.get('val_mae', [None])[-1]
                 if final_val_mae_hist is not None:
                      mae = final_val_mae_hist
                      print(f"  {model_name}: Using final val_mae from history_stage2 ({mae:.4f}) as MAE. RMSE unavailable.")
                 else:
                      print(f"  {model_name}: Final evaluation metrics found but MAE/RMSE missing.")
        # Fallback if detailed metrics are missing/incomplete
        if mae is None and eval_mae is not None:
            mae = eval_mae
            print(f"  {model_name}: Using basic evaluation MAE ({mae:.4f}). RMSE unavailable.")

        # Add to list if we found at least MAE
        if mae is not None:
             metrics_data.append({
                 'Model': model_name,
                 'MAE': mae,
                 'RMSE': rmse if rmse is not None else np.nan # Store NaN if RMSE is missing
             })
        else:
            print(f"  {model_name}: No usable performance metric (MAE) found.")


    if not metrics_data:
        print("No valid performance metrics found to plot.")
        return

    # Prepare data for plotting
    models = [item['Model'] for item in metrics_data]
    mae_values = [item['MAE'] for item in metrics_data]
    rmse_values = [item['RMSE'] for item in metrics_data]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.2), 7)) # Dynamic width
    rects1 = ax.bar(x - width/2, mae_values, width, label='MAE (N)')

    # Check if there are any valid (non-NaN) RMSE values before plotting
    valid_rmse = [r for r in rmse_values if not np.isnan(r)]
    if valid_rmse:
        rects2 = ax.bar(x + width/2, rmse_values, width, label='RMSE (N)')
        # Create custom labels, showing only non-NaN values
        rmse_labels = [f'{r:.3f}' if not np.isnan(r) else '' for r in rmse_values]
        # Add labels to the RMSE container, using the custom labels list
        ax.bar_label(rects2, labels=rmse_labels, padding=3)
    else:
        print("Skipping RMSE bars as no valid RMSE values were found.")

    ax.set_ylabel('Error Value (N)')
    ax.set_title('Model Performance Comparison (Based on Final Epoch or Evaluation)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels to MAE bars
    ax.bar_label(rects1, padding=3, fmt='%.3f')

    fig.tight_layout()

    plt.savefig(output_path)
    print(f"\nPerformance comparison plot saved to: {output_path}")
    plt.close(fig)

def plot_training_history(results, output_path, use_log_scale=False):
    """Plots the training and validation loss history for each model in separate subplots."""
    model_names = list(results.keys()) # Already filtered
    if not model_names:
         print("No models left to plot after filtering.")
         return

    log_scale_suffix = " (Log Scale)" if use_log_scale else ""
    print(f"\nPlotting training/validation loss history{log_scale_suffix}...")
    plotted_anything = False

    num_models = len(model_names)
    cols = min(3, num_models) # Max 3 columns
    rows = (num_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    plot_idx = 0
    for i, model_name in enumerate(model_names):
        history = results[model_name].get('history_stage2', {}) # Use 'history_stage2'
        train_loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])

        ax = axes_flat[plot_idx]

        if train_loss or val_loss:
            epochs = range(1, max(len(train_loss), len(val_loss)) + 1)
            if train_loss:
                ax.plot(epochs[:len(train_loss)], train_loss, label=f'Train Loss', alpha=0.8)
                # print(f"  {model_name}: Plotted {len(train_loss)} train epochs.")
                plotted_anything = True
            if val_loss:
                ax.plot(epochs[:len(val_loss)], val_loss, label=f'Val Loss', linewidth=1.5)
                # print(f"  {model_name}: Plotted {len(val_loss)} val epochs.")
                plotted_anything = True

            ax.set_title(f'{model_name}{log_scale_suffix}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            if use_log_scale:
                ax.set_yscale('log')
            ax.legend(loc='best')
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            plot_idx += 1
        else:
             print(f"  {model_name}: No 'loss' or 'val_loss' history found.")
             ax.set_title(f'{model_name} - No Data')
             ax.text(0.5, 0.5, 'No history data', ha='center', va='center')
             ax.set_xticks([])
             ax.set_yticks([])
             plot_idx += 1

    # Hide any unused subplots
    for j in range(plot_idx, len(axes_flat)):
         fig.delaxes(axes_flat[j])

    if not plotted_anything:
        print(f"No training or validation loss history found to plot{log_scale_suffix}.")
        plt.close(fig)
        return

    fig.suptitle(f'Model Training History (Individual Plots){log_scale_suffix}', fontsize=16, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout

    plt.savefig(output_path)
    print(f"\nIndividual training history plot{log_scale_suffix} saved to: {output_path}")
    plt.close(fig)

def plot_combined_log_history(results, output_path):
    """Plots the validation loss history for all included models on a single log-scale plot."""
    model_names = list(results.keys()) # Already filtered
    if not model_names:
         print("No models left for combined plot after filtering.")
         return

    plt.figure(figsize=(12, 8))
    print(f"\nPlotting combined validation loss history (Log Scale)...")
    plotted_anything = False

    for model_name in model_names:
        history = results[model_name].get('history_stage2', {}) # Use 'history_stage2'
        val_loss = history.get('val_loss', [])

        if val_loss:
            epochs = range(1, len(val_loss) + 1)
            plt.plot(epochs, val_loss, label=f'{model_name}', alpha=0.9)
            # print(f"  {model_name}: Plotted {len(val_loss)} val epochs.")
            plotted_anything = True
        else:
             print(f"  {model_name}: No 'val_loss' history found for combined plot.")

    if not plotted_anything:
        print("No validation loss history found for any included model to plot.")
        plt.close()
        return

    plt.title('Combined Validation Loss During Training (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (Log Scale)')
    plt.yscale('log')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"\nCombined validation loss plot (Log Scale) saved to: {output_path}")
    plt.close()

def plot_final_loss_histogram(results, output_path):
    """Plots a ranked bar chart of the final validation loss for each model."""
    model_names = list(results.keys()) # Already filtered
    final_losses = []

    print("\nExtracting final validation losses...")
    for model_name in model_names:
        history = results[model_name].get('history_stage2', {}) # Use 'history_stage2'
        val_loss = history.get('val_loss', [])
        if val_loss:
            final_loss = val_loss[-1]
            final_losses.append({'Model': model_name, 'Final Loss': final_loss})
            print(f"  {model_name}: Final Validation Loss = {final_loss:.4f}")
        else:
            print(f"  {model_name}: No validation loss history found.")
            # Optionally append with NaN or skip
            # final_losses.append({'Model': model_name, 'Final Loss': np.nan})

    if not final_losses:
        print("No final validation losses found to plot.")
        return

    # Sort models by final loss (ascending)
    final_losses.sort(key=lambda x: x['Final Loss'])

    # Prepare data for plotting
    models_sorted = [item['Model'] for item in final_losses]
    loss_values_sorted = [item['Final Loss'] for item in final_losses]

    x = np.arange(len(models_sorted))

    fig, ax = plt.subplots(figsize=(max(10, len(models_sorted)*1.2), 7)) # Dynamic width
    rects = ax.bar(x, loss_values_sorted)

    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Ranked Final Validation Loss (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels to bars
    ax.bar_label(rects, padding=3, fmt='%.4f')

    fig.tight_layout()

    plt.savefig(output_path)
    print(f"\nRanked final validation loss plot saved to: {output_path}")
    plt.close(fig)

# NEW FUNCTION for ranked final MAE histogram
def plot_final_mae_histogram(results, output_path):
    """Plots a ranked bar chart of the final validation MAE for each model."""
    model_names = list(results.keys()) # Already filtered
    final_maes = []

    print("\nExtracting final validation MAEs...")
    for model_name in model_names:
        # Prioritize MAE from basic evaluation if available
        eval_mae = results[model_name].get('metrics', {}).get('mae')
        final_mae = None
        source = ""

        if eval_mae is not None:
            final_mae = eval_mae
            source = "basic eval"
        else:
            # Fallback to history
            history = results[model_name].get('history_stage2', {})
            val_mae_hist = history.get('val_mae', [])
            if val_mae_hist:
                final_mae = val_mae_hist[-1]
                source = "history"

        if final_mae is not None:
            final_maes.append({'Model': model_name, 'Final MAE': final_mae})
            print(f"  {model_name}: Final Validation MAE = {final_mae:.4f} (from {source})")
        else:
            print(f"  {model_name}: No final validation MAE found.")

    if not final_maes:
        print("No final validation MAEs found to plot.")
        return

    # Sort models by final MAE (ascending)
    final_maes.sort(key=lambda x: x['Final MAE'])

    # Prepare data for plotting
    models_sorted = [item['Model'] for item in final_maes]
    mae_values_sorted = [item['Final MAE'] for item in final_maes]

    x = np.arange(len(models_sorted))

    fig, ax = plt.subplots(figsize=(max(10, len(models_sorted)*1.2), 7)) # Dynamic width
    rects = ax.bar(x, mae_values_sorted)

    ax.set_ylabel('Final Validation MAE (N)')
    ax.set_title('Ranked Final Validation MAE (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.bar_label(rects, padding=3, fmt='%.4f')
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"\nRanked final validation MAE plot saved to: {output_path}")
    plt.close(fig)

# NEW FUNCTION for ranked minimum validation loss histogram
def plot_min_loss_histogram(results, output_path):
    """Plots a ranked bar chart of the minimum validation loss achieved by each model."""
    model_names = list(results.keys()) # Already filtered
    min_losses = []

    print("\nExtracting minimum validation losses from history...")
    for model_name in model_names:
        history = results[model_name].get('history_stage2', {}) # Use 'history_stage2'
        val_loss = history.get('val_loss', [])
        if val_loss:
            min_loss = min(val_loss)
            min_losses.append({'Model': model_name, 'Min Loss': min_loss})
            print(f"  {model_name}: Minimum Validation Loss = {min_loss:.4f} (at epoch {np.argmin(val_loss) + 1})")
        else:
            print(f"  {model_name}: No validation loss history found.")

    if not min_losses:
        print("No minimum validation losses found to plot.")
        return

    # Sort models by minimum loss (ascending)
    min_losses.sort(key=lambda x: x['Min Loss'])

    models_sorted = [item['Model'] for item in min_losses]
    loss_values_sorted = [item['Min Loss'] for item in min_losses]
    x = np.arange(len(models_sorted))

    fig, ax = plt.subplots(figsize=(max(10, len(models_sorted)*1.2), 7))
    rects = ax.bar(x, loss_values_sorted)

    ax.set_ylabel('Minimum Validation Loss')
    ax.set_title('Ranked Minimum Validation Loss During Training (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.bar_label(rects, padding=3, fmt='%.4f')
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"\nRanked minimum validation loss plot saved to: {output_path}")
    plt.close(fig)

# NEW FUNCTION for ranked minimum validation MAE histogram
def plot_min_mae_histogram(results, output_path):
    """Plots a ranked bar chart of the minimum validation MAE achieved by each model."""
    model_names = list(results.keys()) # Already filtered
    min_maes = []

    print("\nExtracting minimum validation MAEs from history...")
    for model_name in model_names:
        history = results[model_name].get('history_stage2', {}) # Use 'history_stage2'
        val_mae = history.get('val_mae', [])
        if val_mae:
            min_mae_val = min(val_mae)
            min_maes.append({'Model': model_name, 'Min MAE': min_mae_val})
            print(f"  {model_name}: Minimum Validation MAE = {min_mae_val:.4f} (at epoch {np.argmin(val_mae) + 1})")
        else:
            print(f"  {model_name}: No validation MAE history found.")

    if not min_maes:
        print("No minimum validation MAEs found to plot.")
        return

    # Sort models by minimum MAE (ascending)
    min_maes.sort(key=lambda x: x['Min MAE'])

    models_sorted = [item['Model'] for item in min_maes]
    mae_values_sorted = [item['Min MAE'] for item in min_maes]
    x = np.arange(len(models_sorted))

    fig, ax = plt.subplots(figsize=(max(10, len(models_sorted)*1.2), 7))
    rects = ax.bar(x, mae_values_sorted)

    ax.set_ylabel('Minimum Validation MAE (N)')
    ax.set_title('Ranked Minimum Validation MAE During Training (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.bar_label(rects, padding=3, fmt='%.4f')
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"\nRanked minimum validation MAE plot saved to: {output_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize model comparison results from a JSON file.")
    parser.add_argument("results_json", help="Path to the model comparison results JSON file.")
    args = parser.parse_args()

    results_data = load_results(args.results_json)

    if results_data:
        # Determine base output directory and create prefix
        base_output_dir = os.path.dirname(args.results_json)
        if not base_output_dir:
            base_output_dir = '.'
        base_name = os.path.basename(args.results_json)
        file_prefix = os.path.splitext(base_name)[0].replace('model_comparison_results_', '')

        # Create a dedicated visualization folder for this run
        viz_output_dir = os.path.join(base_output_dir, f"visualizations_{file_prefix}")
        os.makedirs(viz_output_dir, exist_ok=True)
        print(f"\nSaving visualization plots to: {viz_output_dir}")

        # Generate plots and save them to the dedicated folder
        plot_performance_comparison(results_data, os.path.join(viz_output_dir, f'{file_prefix}_performance_comparison_filtered.png'))
        plot_training_history(results_data, os.path.join(viz_output_dir, f'{file_prefix}_train_history_linear_filtered.png'), use_log_scale=False)
        plot_training_history(results_data, os.path.join(viz_output_dir, f'{file_prefix}_train_history_log_filtered.png'), use_log_scale=True)
        plot_combined_log_history(results_data, os.path.join(viz_output_dir, f'{file_prefix}_train_history_log_combined.png'))
        plot_final_loss_histogram(results_data, os.path.join(viz_output_dir, f'{file_prefix}_final_loss_ranked.png'))
        # Add the new ranked plots
        plot_final_mae_histogram(results_data, os.path.join(viz_output_dir, f'{file_prefix}_final_mae_ranked.png'))
        plot_min_loss_histogram(results_data, os.path.join(viz_output_dir, f'{file_prefix}_min_loss_ranked.png'))
        plot_min_mae_histogram(results_data, os.path.join(viz_output_dir, f'{file_prefix}_min_mae_ranked.png'))

        print("\nVisualization complete.")

if __name__ == "__main__":
    main() 