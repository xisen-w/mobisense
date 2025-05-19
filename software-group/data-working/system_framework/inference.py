import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler

# Disable GPU acceleration to be consistent with training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

def load_and_prepare_sequences(file_path: str, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare data specifically for GRF Estimation.
    Assumes input CSV contains IMU data and ground force targets.
    
    This is simplified version of the pipeline's load_and_prepare_sequences for inference.
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

    # Create sequences with overlap - inference uses all frames
    sequences = []
    targets = []
    stride = 1
    
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
    
    print(f"Processed dataset shape - X: {X.shape}, y: {y.shape}")
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

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, participant_weight_kg: float = 70) -> Dict:
    """
    Evaluate GRF predictions with physics-aware metrics.
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
    
    # Physics-specific metrics
    # 1. Vertical force constraint metrics (Y component is index 1 and 4)
    if np.isnan(y_true[:, 1]).any() or np.isnan(y_true[:, 4]).any() or \
       np.isnan(y_pred[:, 1]).any() or np.isnan(y_pred[:, 4]).any():
        print("Warning: NaN values found in vertical force components.")
        total_vertical_true = np.nan
        total_vertical_pred = np.nan
        vertical_force_error_n = np.nan
        vertical_force_error_pct = np.nan
    else:
        total_vertical_true = np.mean(y_true[:, 1] + y_true[:, 4])
        total_vertical_pred = np.mean(y_pred[:, 1] + y_pred[:, 4])
        vertical_force_error_n = np.abs(total_vertical_pred - body_weight_force)
        vertical_force_error_pct = (vertical_force_error_n / body_weight_force * 100) if body_weight_force > 1e-6 else np.nan
    
    # 2. Horizontal force balance constraint
    if np.isnan(y_pred[:, 0]).any() or np.isnan(y_pred[:, 3]).any():
        print("Warning: NaN values found in horizontal force components.")
        horizontal_imbalance = np.nan
    else:
        # Left foot X + Right foot X should sum to near zero
        horizontal_imbalance = np.mean(np.abs(y_pred[:, 0] + y_pred[:, 3]))
    
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
    
    # Physics violation metrics dictionary
    physics_metrics = {
        'True Total Vertical Force (N)': float(total_vertical_true) if not np.isnan(total_vertical_true) else None,
        'Predicted Total Vertical Force (N)': float(total_vertical_pred) if not np.isnan(total_vertical_pred) else None,
        'Body Weight Force (N)': float(body_weight_force),
        'Vertical Force Imbalance (N)': float(vertical_force_error_n) if not np.isnan(vertical_force_error_n) else None,
        'Vertical Force Imbalance (%)': float(vertical_force_error_pct) if not np.isnan(vertical_force_error_pct) else None,
        'Horizontal Force Imbalance (N)': float(horizontal_imbalance) if not np.isnan(horizontal_imbalance) else None,
        'Smoothness Violation': float(smoothness_violation) if not np.isnan(smoothness_violation) else None
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
                                      max_samples: int = 500):
    """
    Create time series visualizations comparing predicted forces with ground truth.
    
    Args:
        y_true: Ground truth force values
        y_pred: Predicted force values
        model_name: Name of the model for labeling
        output_dir: Directory to save visualizations
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
    avg_true = np.mean(total_true)
    plt.axhline(y=avg_true, color='g', linestyle='--', alpha=0.5, 
                label=f'Mean Total Force: {avg_true:.1f} N')
    
    plt.title(f'Total Vertical Force Over Time: {model_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Force (N)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the total vertical force figure
    save_path = os.path.join(output_dir, f'total_vertical_force_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Time series visualizations saved for {model_name}")

def load_model(model_name: str, base_dir: str = "model_output") -> tf.keras.Model:
    """
    Load a pre-trained model from the specified path.
    Handles loading both standard and custom models.
    """
    model_dir = f"{base_dir}/{model_name}/best_model_{model_name}"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        return None
    
    try:
        # Load model without compiling
        print(f"Loading model '{model_name}' from {model_dir}")
        model = tf.keras.models.load_model(model_dir, compile=False)
        
        # Special handling for models with custom losses if needed
        if model_name in ["HybridTransformerPhysics"]:
            print(f"Model {model_name} might have custom loss. Compiling with default loss for inference.")
        
        # Compile with MSE for inference (doesn't affect predictions)
        model.compile(optimizer="adam", loss="mse")
        return model
    
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def create_visualizations(results_df: pd.DataFrame, output_dir: str, participant_weight_kg: float = 70):
    """
    Create visualization for physics plausibility metrics.
    """
    # 1. Vertical Force Imbalance
    plt.figure(figsize=(10, 6))
    results_df['Physics.Vertical Force Imbalance (N)'].plot(kind='bar')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Vertical Force Imbalance by Model\n(Difference from expected body weight force)')
    plt.ylabel('Force Difference (N)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vertical_force_imbalance.png'))
    
    # 2. Horizontal Force Imbalance
    plt.figure(figsize=(10, 6))
    results_df['Physics.Horizontal Force Imbalance (N)'].plot(kind='bar')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Horizontal Force Imbalance by Model\n(Sum of left and right horizontal forces)')
    plt.ylabel('Force Imbalance (N)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'horizontal_force_imbalance.png'))
    
    # 3. Smoothness Violation
    plt.figure(figsize=(10, 6))
    results_df['Physics.Smoothness Violation'].plot(kind='bar')
    plt.title('Temporal Smoothness Violation by Model\n(Higher means less physically plausible)')
    plt.ylabel('Mean Frame-to-Frame L2 Norm')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_violation.png'))
    
    # 4. Validation Loss vs Physics Plausibility
    fig, ax1 = plt.figure(figsize=(12, 7)), plt.gca()
    
    # RMSE on primary y-axis
    results_df['Overall.RMSE (N)'].plot(kind='bar', ax=ax1, color='blue', alpha=0.7)
    ax1.set_ylabel('RMSE (N)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Vertical force imbalance on secondary y-axis
    ax2 = ax1.twinx()
    results_df['Physics.Vertical Force Imbalance (N)'].plot(kind='line', ax=ax2, 
                                                          color='red', marker='o')
    ax2.set_ylabel('Vertical Force Imbalance (N)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Model Performance vs Physics Plausibility')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_plausibility.png'))
    
    # 5. Combined physics violations
    plt.figure(figsize=(12, 6))
    
    # Create a combined score of physics violations (normalized)
    if ('Physics.Vertical Force Imbalance (N)' in results_df.columns and 
        'Physics.Horizontal Force Imbalance (N)' in results_df.columns and 
        'Physics.Smoothness Violation' in results_df.columns):
        
        # Normalize each metric to [0,1] for fair comparison
        metrics = ['Physics.Vertical Force Imbalance (N)', 
                  'Physics.Horizontal Force Imbalance (N)', 
                  'Physics.Smoothness Violation']
        
        normalized_df = results_df.copy()
        for metric in metrics:
            if results_df[metric].max() > results_df[metric].min():
                normalized_df[metric] = (results_df[metric] - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min())
        
        # Plot stacked bar for normalized values
        normalized_df[metrics].plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Combined Physics Violation Score by Model\n(Lower is Better)')
        plt.ylabel('Normalized Violation Score')
        plt.legend(title='Violation Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_physics_violations.png'))
    
    print(f"Visualization images saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on saved models to evaluate physics plausibility')
    parser.add_argument('--models', nargs='+', required=True, 
                        help='List of model names to evaluate')
    parser.add_argument('--data', required=True, 
                        help='Path to CSV file with IMU and force data')
    parser.add_argument('--seq_len', type=int, default=10, 
                        help='Sequence length for windowing')
    parser.add_argument('--weight', type=float, default=70.0, 
                        help='Participant weight in kg for physics calculations')
    parser.add_argument('--scaler_path', default=None,
                        help='Path to saved scalers (optional, will use fit=True if not provided)')
    parser.add_argument('--output_dir', default='inference_results',
                        help='Directory to save results and visualizations')
    parser.add_argument('--visualize_time_series', action='store_true',
                        help='Generate time series visualizations of predicted vs actual forces')
    parser.add_argument('--max_series_samples', type=int, default=500,
                        help='Maximum number of samples to include in time series plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load and prepare data
    print(f"Loading data from {args.data}")
    X, y = load_and_prepare_sequences(args.data, args.seq_len)
    
    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return
    
    # 2. Standardize data
    if args.scaler_path:
        try:
            print(f"Loading scalers from {args.scaler_path}")
            scaler_X = joblib.load(os.path.join(args.scaler_path, 'scaler_X.joblib'))
            scaler_y = joblib.load(os.path.join(args.scaler_path, 'scaler_y.joblib'))
            X_scaled, y_scaled, _, _ = standardize_data(X, y, fit=False, 
                                                      scaler_X=scaler_X, scaler_y=scaler_y)
        except Exception as e:
            print(f"Error loading scalers: {e}. Fitting new scalers.")
            X_scaled, y_scaled, scaler_X, scaler_y = standardize_data(X, y, fit=True)
    else:
        print("No scaler path provided. Fitting scalers on test data for inference.")
        X_scaled, y_scaled, scaler_X, scaler_y = standardize_data(X, y, fit=True)
    
    if X_scaled is None:
        print("Failed to standardize data. Exiting.")
        return
    
    # 3. Inference with each model
    results = {}
    all_predictions = {}  # Store predictions for time series visualization
    
    for model_name in args.models:
        print(f"\n--- Running inference for {model_name} ---")
        
        # Load model
        model = load_model(model_name)
        if model is None:
            print(f"Skipping {model_name} due to loading error.")
            continue
        
        # Run prediction
        y_pred_scaled = model.predict(X_scaled, verbose=1)
        
        # If scalers were used for y, inverse transform predictions
        if args.scaler_path or 'scaler_y' in locals():
            try:
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
            except Exception as e:
                print(f"Error inverse transforming: {e}. Using scaled predictions.")
                y_pred = y_pred_scaled
        else:
            y_pred = y_pred_scaled
        
        # Store predictions for visualization
        all_predictions[model_name] = y_pred
        
        # Evaluate physics plausibility and performance
        metrics = evaluate_predictions(y, y_pred, args.weight)
        
        # Store results
        results[model_name] = metrics
        
        # Print key metrics
        print(f"Overall MAE: {metrics['Overall']['MAE (N)']:.4f} N")
        print(f"Vertical Force Imbalance: {metrics['Physics']['Vertical Force Imbalance (N)']:.4f} N")
        print(f"Horizontal Force Imbalance: {metrics['Physics']['Horizontal Force Imbalance (N)']:.4f} N")
        print(f"Smoothness Violation: {metrics['Physics']['Smoothness Violation']:.4f}")
        
        # Create time series visualizations if requested
        if args.visualize_time_series:
            visualize_time_series_predictions(y, y_pred, model_name, args.output_dir, args.max_series_samples)
    
    # 4. Process results into a flattened DataFrame
    results_flat = {}
    for model, metrics in results.items():
        results_flat[model] = {}
        
        # Flatten metrics dictionary
        for category in ['Overall', 'Physics']:
            for metric, value in metrics[category].items():
                # Skip None values
                if value is not None:
                    results_flat[model][f"{category}.{metric}"] = value
    
    # Create DataFrame
    results_df = pd.DataFrame.from_dict(results_flat, orient='index')
    
    # 5. Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df, args.output_dir, args.weight)
    
    # 6. Save results to CSV
    csv_path = os.path.join(args.output_dir, 'inference_results.csv')
    results_df.to_csv(csv_path)
    print(f"Results saved to {csv_path}")
    
    # 7. Print summary
    print("\n=== Summary of Model Physics Plausibility ===")
    summary_cols = [
        'Overall.MAE (N)', 'Overall.RMSE (N)', 
        'Physics.Vertical Force Imbalance (N)', 
        'Physics.Horizontal Force Imbalance (N)',
        'Physics.Smoothness Violation'
    ]
    print(results_df[summary_cols].sort_values('Overall.RMSE (N)'))

if __name__ == "__main__":
    main() 