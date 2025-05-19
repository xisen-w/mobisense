import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import json
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler

# Disable GPU acceleration to be consistent with training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

def load_and_prepare_sequences(file_path: str, sequence_length: int = 10, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare data specifically for GRF Estimation.
    Assumes input CSV contains IMU data and ground force targets.
    
    This is simplified version of the pipeline's load_and_prepare_sequences for inference.
    
    Args:
        file_path: Path to the CSV file
        sequence_length: Window size for sequences
        stride: Step size for sequence generation (use higher values to reduce overlap)
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Raw data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return None, None

    # Define required features and targets for GRF Estimation
    # Input: 12 IMU channels 
    imu_features = [
        f'imu{i}_acc_{axis}' for i in range(2) for axis in ['x', 'y', 'z']
    ] + [
        f'imu{i}_gyro_{axis}' for i in range(2) for axis in ['x', 'y', 'z']
    ]

    # Target: 6 GRF components (vx, vy, vz for left/right foot)
    force_targets = [
        f'ground_force_{side}_v{component}' for side in ['left', 'right'] for component in ['x', 'y', 'z']
    ]

    # Verify required columns exist
    missing_features = [col for col in imu_features if col not in df.columns]
    missing_targets = [col for col in force_targets if col not in df.columns]
    if missing_features or missing_targets:
        print(f"Error: Missing required columns.")
        if missing_features: print(f"  Missing Features: {missing_features}")
        if missing_targets: print(f"  Missing Targets: {missing_targets}")
        return None, None

    print("Processing data for GRF Estimation.")

    # Handle potential non-numeric data
    for col in imu_features + force_targets:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for NaNs introduced by coercion
    nan_check = df[imu_features + force_targets].isna().sum()
    if nan_check.sum() > 0:
        print(f"Warning: NaNs detected after converting to numeric: {nan_check[nan_check > 0]}")
        # Interpolate NaNs
        df[imu_features] = df[imu_features].interpolate(method='linear', limit_direction='both')
        df[force_targets] = df[force_targets].interpolate(method='linear', limit_direction='both')
        
        # Re-check NaNs after handling
        if df[imu_features + force_targets].isna().sum().sum() > 0:
            print("Error: NaNs remain after handling. Cannot proceed.")
            return None, None

    # Create sequences with specified stride (higher stride = less overlap)
    sequences = []
    targets = []
    
    # Use provided stride instead of hardcoded value=1
    for i in range(0, len(df) - sequence_length + 1, stride):
        seq = df[imu_features].iloc[i:i+sequence_length].values
        target = df[force_targets].iloc[i+sequence_length-1].values

        # Check for NaN in current sequence/target
        if np.isnan(seq).any() or np.isnan(target).any():
            continue  # Skip sequences with NaN
            
        sequences.append(seq)
        targets.append(target)
        
    if not sequences:
        print("Error: No valid sequences generated. Check data quality.")
        return None, None
        
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Processed dataset shape - X: {X.shape}, y: {y.shape} (with stride={stride})")
    return X, y

def standardize_data(X: np.ndarray, y: np.ndarray, 
                    fit: bool = True, 
                    scaler_X: Optional[StandardScaler] = None, 
                    scaler_y: Optional[StandardScaler] = None):
    """
    Feature-wise standardization for IMU signals (X) and force outputs (y).
    Can use existing scalers or fit new ones.
    """
    if X is None or y is None or X.size == 0 or y.size == 0:
        print("Error: Input X or y is None or empty.")
        return None, None, None, None

    samples, timesteps, features = X.shape
    X_reshaped = X.reshape(-1, features)

    if fit:
        print("CAUTION: Fitting new scalers on test data. For proper evaluation, provide scalers from training.")
        scaler_X = StandardScaler()
        X_scaled_reshaped = scaler_X.fit_transform(X_reshaped)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)
    else:
        if scaler_X is None or scaler_y is None:
            raise ValueError("Scalers must be provided when fit=False")
        try:
            X_scaled_reshaped = scaler_X.transform(X_reshaped)
            y_scaled = scaler_y.transform(y)
        except Exception as e:
            print(f"Error applying existing scaler: {e}")
            return None, None, scaler_X, scaler_y

    X_scaled = X_scaled_reshaped.reshape(samples, timesteps, features)
    return X_scaled, y_scaled, scaler_X, scaler_y

def evaluate_predictions_revised(y_true: np.ndarray, y_pred: np.ndarray, participant_weight_kg: float = 70) -> Dict:
    """
    Evaluate GRF predictions with physics-aware metrics.
    REVISED to calculate physics violations on a per-sample basis.
    """
    if y_true is None or y_pred is None or y_true.size == 0 or y_pred.size == 0:
        print("Error: Cannot evaluate predictions with empty true or predicted values.")
        return {"Error": "Input arrays are empty or None."}
    
    if y_true.shape != y_pred.shape:
        print(f"Error: Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}.")
        return {"Error": "Shape mismatch between true and predicted values."}

    # Ensure input is for force prediction (6 components)
    if y_true.ndim < 2 or y_true.shape[1] != 6:
        print(f"Error: Expected 6 target components (GRF), but got y_true shape: {y_true.shape}")
        return {"Error": f"Invalid shape for y_true: {y_true.shape}. Expected (samples, 6)."}

    # Calculate body weight force
    body_weight_force = participant_weight_kg * 9.81
    
    # Component names for better reporting
    components = [
        'Left Foot X', 'Left Foot Y', 'Left Foot Z',
        'Right Foot X', 'Right Foot Y', 'Right Foot Z'
    ]
    
    # Calculate metrics for each component
    component_metrics = {}
    for i, name in enumerate(components):
        # Handle NaNs
        if np.isnan(y_true[:, i]).any() or np.isnan(y_pred[:, i]).any():
            print(f"Warning: NaN values found in component '{name}'.")
            mae = np.nan
            rmse = np.nan
            mre = np.nan
        else:
            mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
            rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2))
            
            # Calculate relative error (as percentage)
            mask = np.abs(y_true[:, i]) > 1.0  # Threshold to avoid division by near-zero
            if np.sum(mask) > 0:
                true_masked = y_true[mask, i]
                pred_masked = y_pred[mask, i]
                if not np.isnan(true_masked).any() and not np.isnan(pred_masked).any():
                    mre = np.mean(np.abs(true_masked - pred_masked) / np.abs(true_masked)) * 100
                else:
                    mre = np.nan
            else:
                mre = np.nan
                
        component_metrics[name] = {
            'MAE (N)': float(mae) if not np.isnan(mae) else None,
            'RMSE (N)': float(rmse) if not np.isnan(rmse) else None,
            'MRE (%)': float(mre) if not np.isnan(mre) else None
        }
    
    # Physics-specific metrics - REVISED CALCULATIONS FOR MORE ACCURATE PHYSICS VIOLATION ASSESSMENT
    # 1. Vertical force constraint metrics (Y component is index 1 and 4)
    if np.isnan(y_true[:, 1]).any() or np.isnan(y_true[:, 4]).any() or \
       np.isnan(y_pred[:, 1]).any() or np.isnan(y_pred[:, 4]).any():
        print("Warning: NaN values found in vertical force components.")
        true_vert_imbalance_n = np.nan
        pred_vert_imbalance_n = np.nan
        vertical_force_error_pct = np.nan
    else:
        # For each sample, compute vertical sum
        true_vert_sum = y_true[:, 1] + y_true[:, 4]  # Left Y + Right Y
        pred_vert_sum = y_pred[:, 1] + y_pred[:, 4]
        
        # Compute difference from body weight force for each sample
        true_vert_errors = np.abs(true_vert_sum - body_weight_force)
        pred_vert_errors = np.abs(pred_vert_sum - body_weight_force)
        
        # Mean absolute error across samples
        true_vert_imbalance_n = np.mean(true_vert_errors)
        pred_vert_imbalance_n = np.mean(pred_vert_errors)
        
        # Percentage error
        vertical_force_error_pct = np.mean(pred_vert_errors / body_weight_force) * 100
        
        # Also compute maximum imbalance
        max_vert_imbalance_n = np.max(pred_vert_errors)
    
    # 2. Horizontal force balance constraint - REVISED TO PER-SAMPLE
    if np.isnan(y_pred[:, 0]).any() or np.isnan(y_pred[:, 3]).any():
        print("Warning: NaN values found in horizontal force components.")
        horizontal_imbalance = np.nan
        max_horiz_imbalance = np.nan
    else:
        # Per-sample horizontal force sum (should be close to zero)
        horiz_sum = y_pred[:, 0] + y_pred[:, 3]  # Left X + Right X
        
        # Mean absolute horizontal imbalance
        horizontal_imbalance = np.mean(np.abs(horiz_sum))
        
        # Maximum horizontal imbalance
        max_horiz_imbalance = np.max(np.abs(horiz_sum))
    
    # 3. Temporal smoothness violation
    if np.any(np.isnan(y_pred)):
        smoothness_violation = np.nan
    else:
        # Compute frame-to-frame differences (L2 norm)
        diffs = np.diff(y_pred, axis=0)
        smoothness_violation = np.mean(np.linalg.norm(diffs, axis=1))
    
    # Overall metrics
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Warning: NaN values found in arrays. Overall metrics might be inaccurate.")
        overall_mae = np.nanmean(np.abs(y_true - y_pred))
        overall_rmse = np.sqrt(np.nanmean((y_true - y_pred)**2))
    else:
        overall_mae = np.mean(np.abs(y_true - y_pred))
        overall_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Physics violation metrics dictionary - REVISED
    physics_metrics = {
        # Store both ground truth and predicted values for proper comparison
        'Ground Truth Vertical Imbalance (N)': float(true_vert_imbalance_n) if not np.isnan(true_vert_imbalance_n) else None,
        'Vertical Force Imbalance (N)': float(pred_vert_imbalance_n) if not np.isnan(pred_vert_imbalance_n) else None,
        'Max Vertical Imbalance (N)': float(max_vert_imbalance_n) if not np.isnan(max_vert_imbalance_n) else None,
        'Vertical Force Imbalance (%)': float(vertical_force_error_pct) if not np.isnan(vertical_force_error_pct) else None,
        'Horizontal Force Imbalance (N)': float(horizontal_imbalance) if not np.isnan(horizontal_imbalance) else None,
        'Max Horizontal Imbalance (N)': float(max_horiz_imbalance) if not np.isnan(max_horiz_imbalance) else None,
        'Smoothness Violation': float(smoothness_violation) if not np.isnan(smoothness_violation) else None,
        'Body Weight Force (N)': float(body_weight_force)
    }
    
    metrics = {
        'Overall': {
            'MAE (N)': float(overall_mae) if not np.isnan(overall_mae) else None,
            'RMSE (N)': float(overall_rmse) if not np.isnan(overall_rmse) else None
        },
        'Components': component_metrics,
        'Physics': physics_metrics
    }
    
    return metrics

def visualize_time_series_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                                      model_name: str, output_dir: str,
                                      body_weight_force: float,
                                      max_samples: int = 500):
    """
    Create time series visualizations comparing predicted forces with ground truth.
    REVISED to better highlight physics violations.
    
    Args:
        y_true: Ground truth force values
        y_pred: Predicted force values
        model_name: Name of the model for labeling
        output_dir: Directory to save visualizations
        body_weight_force: Expected vertical force (m*g)
        max_samples: Maximum number of samples to plot (to avoid overcrowded plots)
    """
    if y_true is None or y_pred is None:
        print("Cannot create time series visualization with None data")
        return
    
    # Limit number of samples for better visualization
    samples_to_plot = min(max_samples, y_true.shape[0])
    
    # Component names
    components = [
        'Left Foot X', 'Left Foot Y', 'Left Foot Z',
        'Right Foot X', 'Right Foot Y', 'Right Foot Z'
    ]
    
    # Create a 3x2 grid of subplots (3 axes for left/right foot)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Force Predictions Over Time: {model_name}', fontsize=16)
    
    # Time points (x-axis)
    time_points = np.arange(samples_to_plot)
    
    # Plot each component
    for i, name in enumerate(components):
        row = i % 3  # 0, 1, 2, 0, 1, 2
        col = i // 3  # 0, 0, 0, 1, 1, 1
        
        # Plot ground truth
        axes[row, col].plot(time_points, y_true[:samples_to_plot, i], 'b-', alpha=0.7, label='Ground Truth')
        # Plot prediction
        axes[row, col].plot(time_points, y_pred[:samples_to_plot, i], 'r-', alpha=0.7, label='Prediction')
        
        axes[row, col].set_title(name)
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel('Force (N)')
        axes[row, col].grid(True, alpha=0.3)
        
        if i == 0:  # Only add legend to the first subplot
            axes[row, col].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the super title
    
    # Save the figure
    save_path = os.path.join(output_dir, f'time_series_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a separate plot for total vertical force (sum of left and right Y components)
    plt.figure(figsize=(12, 6))
    total_true = y_true[:samples_to_plot, 1] + y_true[:samples_to_plot, 4]  # Left Y + Right Y
    total_pred = y_pred[:samples_to_plot, 1] + y_pred[:samples_to_plot, 4]
    
    plt.plot(time_points, total_true, 'b-', alpha=0.7, label='Ground Truth')
    plt.plot(time_points, total_pred, 'r-', alpha=0.7, label='Prediction')
    
    # Add horizontal line for expected body weight force
    plt.axhline(y=body_weight_force, color='g', linestyle='--', alpha=0.5, 
                label=f'Expected Force (m·g): {body_weight_force:.1f} N')
    
    # Calculate imbalance statistics for the plot
    true_imbalance = np.mean(np.abs(total_true - body_weight_force))
    pred_imbalance = np.mean(np.abs(total_pred - body_weight_force))
    
    plt.title(f'Total Vertical Force Over Time: {model_name}\n' + 
              f'Ground Truth Imbalance: {true_imbalance:.2f} N, ' +
              f'Prediction Imbalance: {pred_imbalance:.2f} N')
    plt.xlabel('Time Step')
    plt.ylabel('Force (N)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the total vertical force figure
    save_path = os.path.join(output_dir, f'total_vertical_force_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Additional visualization: Vertical force imbalance over time
    plt.figure(figsize=(12, 6))
    true_imbalance_ts = np.abs(total_true - body_weight_force)
    pred_imbalance_ts = np.abs(total_pred - body_weight_force)
    
    plt.plot(time_points, true_imbalance_ts, 'b-', alpha=0.7, label='Ground Truth Imbalance')
    plt.plot(time_points, pred_imbalance_ts, 'r-', alpha=0.7, label='Prediction Imbalance')
    
    plt.title(f'Vertical Force Imbalance |F - mg| Over Time: {model_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Force Imbalance (N)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the imbalance figure
    save_path = os.path.join(output_dir, f'vertical_imbalance_timeseries_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional visualization: Horizontal force imbalance over time
    plt.figure(figsize=(12, 6))
    horiz_balance_true = np.abs(y_true[:samples_to_plot, 0] + y_true[:samples_to_plot, 3])  # |Left X + Right X|
    horiz_balance_pred = np.abs(y_pred[:samples_to_plot, 0] + y_pred[:samples_to_plot, 3])
    
    plt.plot(time_points, horiz_balance_true, 'b-', alpha=0.7, label='Ground Truth')
    plt.plot(time_points, horiz_balance_pred, 'r-', alpha=0.7, label='Prediction')
    
    plt.title(f'Horizontal Force Imbalance |FL_x + FR_x| Over Time: {model_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Horizontal Imbalance (N)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the horizontal imbalance figure
    save_path = os.path.join(output_dir, f'horizontal_imbalance_timeseries_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Time series visualizations saved for {model_name}")

def load_model(model_name: str, run_id: Optional[str] = None, base_dir: str = "model_output") -> Optional[tf.keras.Model]:
    """
    Load a pre-trained Keras model (SavedModel format) based on its name and optional run_id.
    If run_id is provided, looks in run-specific directory first.
    """
    model = None
    potential_paths = []

    # Priority 1: Run-specific path if run_id is provided
    if run_id:
        run_specific_path = os.path.join(base_dir, f"{model_name}_final_{run_id}")
        potential_paths.append(run_specific_path)

    # Priority 2: Generic path (for backwards compatibility or if run_id is not available)
    generic_path = os.path.join(base_dir, f"{model_name}_final")
    if generic_path not in potential_paths:
         potential_paths.append(generic_path)

    loaded_path = None
    for model_load_path in potential_paths:
        print(f"Attempting to load SavedModel from: {model_load_path}")
        # Check if it's a directory and likely contains a SavedModel
        if os.path.isdir(model_load_path) and (os.path.exists(os.path.join(model_load_path, 'saved_model.pb')) or os.path.exists(os.path.join(model_load_path, 'saved_model.pbtxt'))):
            try:
                # Load the full model potentially containing architecture and weights
                model = tf.keras.models.load_model(model_load_path, compile=False) # compile=False as optimizer state might not match
                print(f"Successfully loaded SavedModel for {model_name} from {model_load_path}")
                loaded_path = model_load_path
                break # Stop after successful load
            except Exception as e:
                print(f"Error loading SavedModel for {model_name} from {model_load_path}: {e}")
                model = None # Ensure model is None if loading failed
        else:
             print(f"Path {model_load_path} is not a valid SavedModel directory.")

    if model is None:
        print(f"Error: Could not find or load a valid SavedModel for '{model_name}' in checked paths: {potential_paths}. Skipping.")

    return model

def create_visualizations_revised(results_df: pd.DataFrame, output_dir: str, participant_weight_kg: float = 70, all_predictions: Dict = None, y_true: np.ndarray = None):
    """
    Create visualization for physics plausibility metrics.
    REVISED to focus on key physics violations across models.
    
    Args:
        results_df: DataFrame with results for each model
        output_dir: Directory to save visualizations
        participant_weight_kg: Participant weight for body weight force calculation
        all_predictions: Dictionary of predictions for each model (for time series plots)
        y_true: Ground truth values (for time series plots)
    """
    body_weight_force = participant_weight_kg * 9.81
    
    # Filter for our target models if they exist
    target_models = [
        "SimpleDense", "LSTM", "CNNLSTM", "Transformer", "BidirectionalLSTM",
        "MultiScaleTransformer", "CrossModalAttentionTransformer", 
        "BiLSTMAttention", "CNNBiLSTMSqueezeExcitation"
    ]
    # Only include models that are actually in the results
    available_models = [model for model in target_models if model in results_df.index]
    
    if available_models:
        print(f"Focusing on target models: {available_models}")
        filtered_df = results_df.loc[available_models]
    else:
        print("Target models not found. Using all available models.")
        filtered_df = results_df
    
    # 1. CHART 1: Combined normalized physics violations with relative proportions
    plt.figure(figsize=(14, 8))
    
    # Physics violation metrics to include
    metrics = [
        'Physics.Vertical Force Imbalance (N)', 
        'Physics.Horizontal Force Imbalance (N)', 
        'Physics.Smoothness Violation'
    ]
    
    # Friendly names for the legend
    metric_names = [
        'Vertical Force Imbalance', 
        'Horizontal Force Imbalance', 
        'Temporal Smoothness Violation'
    ]
    
    # Normalize each metric to [0,1] for fair comparison
    normalized_df = filtered_df.copy()
    
    for metric in metrics:
        if metric in filtered_df.columns:
            if filtered_df[metric].max() > filtered_df[metric].min():
                normalized_df[metric] = (filtered_df[metric] - filtered_df[metric].min()) / (filtered_df[metric].max() - filtered_df[metric].min())
            else:
                normalized_df[metric] = 0 if filtered_df[metric].min() == 0 else 1
    
    # Use only metrics that exist in the dataframe
    existing_metrics = [m for m in metrics if m in normalized_df.columns]
    
    if len(existing_metrics) > 0:
        # Plot stacked bar for normalized values
        ax = normalized_df[existing_metrics].plot(kind='bar', stacked=True, figsize=(14, 8), 
                                        colormap='viridis')
        
        # Adjust legend with friendly names
        handles, labels = ax.get_legend_handles_labels()
        friendly_labels = [metric_names[metrics.index(m)] for m in existing_metrics]
        ax.legend(handles, friendly_labels, title='Violation Type', fontsize=12)
        
        plt.title('Physics Violation Comparison Across Models\n(Lower is Better)', fontsize=16)
        plt.ylabel('Normalized Violation Score', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'physics_violations_comparison.png'), dpi=150)
        plt.close()
    
    # 2. CHART 2: Vertical Force Deviations
    # Option A: If all_predictions and y_true are provided, create time series visualization
    if all_predictions and y_true is not None:
        # Create vertical force deviation time series plot
        plt.figure(figsize=(16, 10))
        
        # Limit to a reasonable number of samples for visualization
        max_samples = min(500, y_true.shape[0])
        time_points = np.arange(max_samples)
        
        # For each model, plot the absolute deviation from body weight force
        for i, model_name in enumerate(all_predictions.keys()):
            if model_name not in filtered_df.index:
                continue
                
            # Calculate total vertical force (left + right)
            y_pred = all_predictions[model_name][:max_samples]
            total_vert_pred = y_pred[:, 1] + y_pred[:, 4]  # Left Y + Right Y
            
            # Calculate deviation from body weight force
            deviation = np.abs(total_vert_pred - body_weight_force)
            
            # Plot with a unique color and slight transparency
            plt.plot(time_points, deviation, alpha=0.8, linewidth=2, 
                     label=f"{model_name} (Avg: {np.mean(deviation):.2f}N)")
        
        # Plot ground truth deviation for reference
        total_vert_true = y_true[:max_samples, 1] + y_true[:max_samples, 4]
        true_deviation = np.abs(total_vert_true - body_weight_force)
        plt.plot(time_points, true_deviation, 'k--', alpha=0.7, linewidth=2,
                 label=f"Ground Truth (Avg: {np.mean(true_deviation):.2f}N)")
        
        plt.title('Vertical Force Deviation from Body Weight Over Time\n|F - mg| where mg = ' + 
                  f'{body_weight_force:.1f}N', fontsize=16)
        plt.xlabel('Time Steps', fontsize=14)
        plt.ylabel('Force Deviation (N)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vertical_force_deviation_timeseries.png'), dpi=150)
        plt.close()
    
    # Option B: Bar chart of average vertical force deviation by model
    if 'Physics.Vertical Force Imbalance (N)' in filtered_df.columns:
        plt.figure(figsize=(14, 8))
        ax = filtered_df['Physics.Vertical Force Imbalance (N)'].plot(kind='bar', figsize=(14, 8),
                                                          color='skyblue')
                                                          
        # Add value labels above bars
        for i, v in enumerate(filtered_df['Physics.Vertical Force Imbalance (N)']):
            ax.text(i, v + 0.5, f"{v:.2f}N", ha='center', fontsize=12)
            
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Average Vertical Force Deviation by Model\n(Deviation from body weight force)', 
                  fontsize=16)
        plt.ylabel('Mean Deviation (N)', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vertical_force_deviation_by_model.png'), dpi=150)
        plt.close()
    
    print(f"Physics violation visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run inference using trained models.")
    parser.add_argument('--models', required=True, nargs='+', help='Names of the models to run inference with.')
    parser.add_argument('--data', required=True, help='Path to the input data CSV file.')
    # Made seq_len optional, default will be overridden by results_path if available
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length for input data.')
    parser.add_argument('--stride', type=int, default=10, help='Stride for creating sequences (larger stride reduces test data size).')
    # Weight path is now inferred, removing explicit --weight argument
    parser.add_argument('--scaler_path', default='model_output/scalers.joblib', help='Path to the saved scaler file (.joblib). Can be overridden by results_path.')
    parser.add_argument('--output_dir', default='model_output/inference_results', help='Directory to save inference results.')
    parser.add_argument('--visualize_time_series', action='store_true', help='Generate time series prediction plots for each model.')
    parser.add_argument('--max_series_samples', type=int, default=500, help='Maximum number of samples for time series plots.')
    # Added results_path argument
    parser.add_argument('--results_path', help='Path to the training results JSON file to load config and locate weights using standard naming convention.')
    parser.add_argument('--participant_weight_kg', type=float, default=70.0, help='Participant weight in kg for physics evaluation.')

    
    args = parser.parse_args()
    
    # --- Configuration Loading ---
    current_seq_len = args.seq_len
    current_scaler_path = args.scaler_path
    run_id_from_config = None # Initialize run_id
    print(f"Initial Sequence Length: {current_seq_len}")
    print(f"Initial Scaler Path: {current_scaler_path}")


    if args.results_path:
        print(f"\n--- Loading Configuration from Results File: {args.results_path} ---")
        try:
            with open(args.results_path, 'r') as f:
                results_data = json.load(f)

            config = None
            if 'config' in results_data:
                config = results_data['config']
                print("Found 'config' at the top level of the JSON.")
            else:
                 print("Warning: Could not find 'config' key at the top level of the results JSON. Using defaults or command-line args.")

            if config and isinstance(config, dict):
                # Extract run_id first
                run_id_from_config = config.get('run_id')
                if run_id_from_config:
                     print(f"  Found run_id in config: {run_id_from_config}")
                else:
                     print("  Warning: run_id not found in config.")

                # Override seq_len
                if args.seq_len == parser.get_default('seq_len') and 'sequence_length' in config:
                    current_seq_len = config['sequence_length']
                    print(f"  Overriding sequence_length from JSON: {current_seq_len}")
                else:
                    print(f"  Keeping sequence_length from command line/default: {current_seq_len}")
                
                # Override scaler_path
                if args.scaler_path == parser.get_default('scaler_path') and 'scaler_path' in config:
                    scaler_path_from_config = config['scaler_path']
                    # Construct absolute path relative to workspace root if path is relative
                    if not os.path.isabs(scaler_path_from_config):
                         abs_path = os.path.abspath(scaler_path_from_config)
                         if os.path.exists(abs_path):
                              current_scaler_path = abs_path
                              print(f"  Overriding scaler_path from JSON (resolved): {current_scaler_path}")
                         else:
                              print(f"  Warning: Resolved relative scaler path from JSON does not exist: {abs_path}. Keeping default/cmd arg.")
                    elif os.path.exists(scaler_path_from_config):
                         current_scaler_path = scaler_path_from_config
                         print(f"  Overriding scaler_path from JSON (absolute): {current_scaler_path}")
                    else:
                         print(f"  Warning: Absolute scaler path from JSON does not exist: {scaler_path_from_config}. Keeping default/cmd arg.")
                else:
                     print(f"  Keeping scaler_path from command line/default: {current_scaler_path}")
            else:
                 print("No valid 'config' dictionary found in JSON. Using defaults or command-line args.")

        except FileNotFoundError:
            print(f"Error: Results file not found at {args.results_path}. Using defaults or command-line args.")
            # Decide whether to exit or continue with defaults. Let's continue.
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.results_path}. Using defaults or command-line args.")
        except Exception as e:
            print(f"Error loading or parsing results JSON {args.results_path}: {e}. Using defaults or command-line args.")

    # Final config values being used
    print(f"\nFinal Sequence Length Used: {current_seq_len}")
    print(f"Final Scaler Path Used: {current_scaler_path}")
    if run_id_from_config:
        print(f"Run ID for model loading: {run_id_from_config}")
    else:
        print("Warning: No Run ID found, will attempt to load models from generic paths.")


    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving results to: {args.output_dir}")

    # --- Data Loading ---
    print("\n--- Loading and Preparing Data ---")
    X_test_raw, y_test_raw = load_and_prepare_sequences(args.data, sequence_length=current_seq_len, stride=args.stride)

    if X_test_raw is None or y_test_raw is None:
        print("Failed to load or prepare data. Exiting.")
        return
    
    # --- Scaling ---
    print(f"\n--- Loading Scaler from {current_scaler_path} ---")
    scaler_X = None
    scaler_y = None
    try:
        if not os.path.exists(current_scaler_path):
             raise FileNotFoundError(f"Scaler file not found at the specified path: {current_scaler_path}")
        scalers = joblib.load(current_scaler_path)
        scaler_X = scalers.get('scaler_X')
        scaler_y = scalers.get('scaler_y')
        if scaler_X is None or scaler_y is None:
             raise ValueError("Scaler file must contain 'scaler_X' and 'scaler_y'.")
        print("Scalers loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Cannot proceed without scalers. Please provide the correct path using --scaler_path or ensure it's valid in the results JSON.")
        return
        except Exception as e:
        print(f"Error loading scalers from {current_scaler_path}: {e}")
        return

    print("Standardizing test data...")
    X_test_scaled, _, _, _ = standardize_data(X_test_raw, y_test_raw, fit=False, scaler_X=scaler_X, scaler_y=scaler_y)

    if X_test_scaled is None:
        print("Failed to standardize test data. Exiting.")
        return
    
    # --- Inference Loop ---
    all_results = {}
    all_predictions = {} # Store predictions for combined visualization
    min_len_global = float('inf') # Track minimum prediction length across models

    print("\n--- Running Inference ---")
    for model_name in args.models:
        print(f"\nProcessing Model: {model_name}")

        # Load model using the standard path convention, passing run_id if available
        model = load_model(model_name, run_id=run_id_from_config) # Pass run_id here
        
        if model is None:
            print(f"Skipping {model_name} due to loading error.")
            continue
        
        # Perform prediction
        try:
            print(f"Predicting with {model_name}...")
            y_pred_scaled = model.predict(X_test_scaled)
            print(f"Raw prediction shape: {y_pred_scaled.shape}")

            # Inverse transform predictions
            y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled)
            print(f"Unscaled prediction shape: {y_pred_unscaled.shape}")
            print(f"True data shape: {y_test_raw.shape}")

             # Ensure shapes match after potential scaler issues / different model outputs
            current_y_test = y_test_raw # Assume full length initially
            if y_pred_unscaled.shape[0] != current_y_test.shape[0]:
                 current_min_len = min(y_pred_unscaled.shape[0], current_y_test.shape[0])
                 print(f"Warning: Mismatch in number of samples for {model_name}. True: {current_y_test.shape[0]}, Pred: {y_pred_unscaled.shape[0]}. Truncating to {current_min_len} samples for evaluation.")
                 y_pred_unscaled = y_pred_unscaled[:current_min_len, :]
                 current_y_test = current_y_test[:current_min_len, :] # Truncate true values for this model's eval
                 min_len_global = min(min_len_global, current_min_len) # Update global minimum length
            else:
                 min_len_global = min(min_len_global, y_pred_unscaled.shape[0])


            # Evaluate predictions
            print(f"Evaluating {model_name}...")
            evaluation_metrics = evaluate_predictions_revised(current_y_test, y_pred_unscaled, participant_weight_kg=args.participant_weight_kg)

            all_results[model_name] = evaluation_metrics
            all_predictions[model_name] = y_pred_unscaled # Store potentially truncated predictions

            print(f"Evaluation Metrics for {model_name}:")
            # Use a helper function for cleaner printing if needed
            for key, value in evaluation_metrics.items():
                if isinstance(value, dict): # Handle component metrics
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        val_str = f"{sub_value:.4f}" if isinstance(sub_value, (float, np.floating)) else f"{sub_value}"
                        print(f"    {sub_key}: {val_str}")
                else:
                    val_str = f"{value:.4f}" if isinstance(value, (float, np.floating)) else f"{value}"
                    print(f"  {key}: {val_str}")


            # Optional: Visualize time series for this model
            if args.visualize_time_series:
                print(f"Visualizing time series for {model_name}...")
                visualize_time_series_predictions(
                    current_y_test, # Use potentially truncated y_true for this plot
                    y_pred_unscaled,
                    model_name,
                    args.output_dir,
                    body_weight_force=args.participant_weight_kg * 9.81,
                    max_samples=args.max_series_samples
                )

        except Exception as e:
            print(f"Error during inference or evaluation for {model_name}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            all_results[model_name] = {"Error": str(e)}


    # --- Save Aggregate Results ---
    print("\n--- Saving Aggregate Results ---")
    # Convert numpy arrays in results to lists for JSON serialization if necessary (evaluate_predictions_revised should return basic types)
    results_df = pd.DataFrame.from_dict(all_results, orient='index')

    # Reorganize DataFrame for better readability (optional)
    # E.g., flatten nested component metrics
    try:
        # A more robust way to flatten
        flattened_data = {}
        for model_name, metrics in all_results.items():
            row = {}
            if isinstance(metrics, dict): # Check if metrics is a dict (might be error string)
                for key, value in metrics.items():
                     if isinstance(value, dict):
                         for sub_key, sub_value in value.items():
                             row[f"{key}_{sub_key}"] = sub_value
                     else:
                         row[key] = value
        else:
                 # Handle case where metric is just an error string
                 row['Error'] = metrics
            flattened_data[model_name] = row

        results_df_flat = pd.DataFrame.from_dict(flattened_data, orient='index')
        results_output_path = os.path.join(args.output_dir, 'inference_evaluation_metrics_flat.csv')
        results_df_flat.to_csv(results_output_path)
        print(f"Flattened evaluation metrics saved to: {results_output_path}")

        # Save original nested structure as JSON
        json_output_path = os.path.join(args.output_dir, 'inference_evaluation_metrics.json')
        # Convert potential numpy types to native python types for JSON
        def convert_numpy_to_native(obj):
             if isinstance(obj, np.integer):
                 return int(obj)
             elif isinstance(obj, np.floating):
                 # Handle NaN and Inf specifically for JSON
                 if np.isnan(obj): return None # Or 'NaN' as string
                 if np.isinf(obj): return None # Or 'Infinity'/' -Infinity' as string
                 return float(obj)
             elif isinstance(obj, np.ndarray):
                 return convert_numpy_to_native(obj.tolist()) # Recursively convert list elements
             elif isinstance(obj, dict):
                 return {k: convert_numpy_to_native(v) for k, v in obj.items()}
             elif isinstance(obj, list):
                 return [convert_numpy_to_native(elem) for elem in obj]
             # Handle cases where evaluation might return None directly
             if obj is None: return None
             return obj # Assume other types are JSON serializable

        serializable_results = convert_numpy_to_native(all_results)

        with open(json_output_path, 'w') as f:
            # Use default handler for non-serializable, though conversion func should handle most
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"Detailed evaluation metrics saved to: {json_output_path}")

    except Exception as e:
        print(f"Error processing or saving results DataFrame/JSON: {e}")
        # Fallback to just saving the raw dict as JSON
        try:
            json_output_path_raw = os.path.join(args.output_dir, 'inference_evaluation_metrics_raw.json')
            # Attempt conversion again just in case
            serializable_results_raw = convert_numpy_to_native(all_results)
            with open(json_output_path_raw, 'w') as f:
                 json.dump(serializable_results_raw, f, indent=2, default=str)
            print(f"Raw evaluation metrics dictionary saved to: {json_output_path_raw}")
        except Exception as e_raw:
             print(f"Failed to save raw JSON as well: {e_raw}")


    # --- Combined Visualizations ---
    print("\n--- Generating Combined Visualizations ---")
    if all_predictions and y_test_raw is not None and min_len_global != float('inf'):
         try:
             # Use the globally minimum length for consistent comparison
             print(f"Using globally truncated y_true (length {min_len_global}) for combined visualizations.")
             final_y_true = y_test_raw[:min_len_global, :]

             # Also truncate all predictions to this global minimum length
             final_predictions = {
                 model_name: pred[:min_len_global, :]
                 for model_name, pred in all_predictions.items()
                 if pred is not None and pred.shape[0] >= min_len_global # Ensure prediction exists and is long enough
             }

             if not final_predictions:
                 print("No valid predictions remaining after truncation for combined visualizations.")
             else:
                  # Ensure results_df_flat exists and is passed correctly
                  df_to_visualize = results_df_flat if 'results_df_flat' in locals() else pd.DataFrame.from_dict(flattened_data, orient='index')
                  create_visualizations_revised(df_to_visualize, args.output_dir, args.participant_weight_kg, final_predictions, final_y_true)
                  print("Combined visualizations generated.")
         except Exception as e:
             print(f"Error during combined visualization generation: {e}")
             import traceback
             traceback.print_exc()
    elif not all_predictions:
         print("Skipping combined visualizations: No successful predictions were made.")
    elif min_len_global == float('inf'):
         print("Skipping combined visualizations: Could not determine a consistent prediction length.")
    else:
        print("Skipping combined visualizations due to missing predictions or true values.")

    print("\nInference complete.")


if __name__ == "__main__":
    main() 