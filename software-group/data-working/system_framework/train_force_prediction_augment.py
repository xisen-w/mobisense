import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os

from scipy.interpolate import interp1d

def load_and_prepare_sequences(file_path, sequence_length=10):
    """Load and prepare data with minimal preprocessing.
    
    A sliding window approach with stride=1 is used to extract sequences.
    Only sequences with significant vertical force (above 5% of mean vertical force)
    are included.
    """
    # Load the data
    df = pd.read_csv(file_path)
    print(f"Raw data shape: {df.shape}")
    
    # Select features and targets
    imu_features = [f'imu{i}_acc_{axis}' for i in range(2) for axis in ['x', 'y', 'z']]
    imu_features += [f'imu{i}_gyro_{axis}' for i in range(2) for axis in ['x', 'y', 'z']]
    
    force_targets = [f'ground_force_{side}_{component}' for side in ['left', 'right'] for component in ['vx', 'vy', 'vz']]
    
    # Print force target statistics
    print("\nForce Targets Statistics:")
    print(df[force_targets].describe())
    
    # Compute force threshold
    mean_left_vy = df['ground_force_left_vy'].mean()
    mean_right_vy = df['ground_force_right_vy'].mean()
    mean_vertical = (mean_left_vy + mean_right_vy) / 2.0
    force_threshold = 0.05 * mean_vertical
    print(f"Using force threshold: {force_threshold:.2f} N")
    
    # Create sequences with overlap
    sequences = []
    targets = []
    stride = 1
    
    # Track sequence count
    total_sequences = 0
    filtered_sequences = 0
    
    for i in range(0, len(df) - sequence_length, stride):
        seq = df[imu_features].values[i:i+sequence_length]
        target = df[force_targets].values[i+sequence_length-1]
        
        total_sequences += 1
        
        # Use sum of vertical forces from both feet
        total_vy = np.abs(target[1]) + np.abs(target[4])
        if total_vy > force_threshold:
            sequences.append(seq)
            targets.append(target)
            filtered_sequences += 1
    
    print(f"Total sequences created: {total_sequences}")
    print(f"Sequences after filtering: {filtered_sequences}")
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Processed dataset shape - X: {X.shape}, y: {y.shape}")
    return X, y

def augment_data(X, y, noise_reps=5, warp_reps=4, noise_sigma=0.05):
    """
    Perform data augmentation on time-series sequences.
    
    Augmentation includes:
      - Adding Gaussian noise to IMU signals (noise_reps times per sequence)
      - Non-linear time warping using cubic interpolation (warp_reps times per sequence)
    
    The force targets (y) are kept unchanged.
    Returns augmented_X and augmented_y.
    """
    augmented_X = []
    augmented_y = []
    
    # Loop through each sequence
    for i in range(len(X)):
        original_seq = X[i]
        original_target = y[i]
        
        # Always keep the original sequence
        augmented_X.append(original_seq)
        augmented_y.append(original_target)
        
        # Gaussian noise augmentation: add noise_reps variations
        for _ in range(noise_reps):
            noisy_seq = original_seq + np.random.normal(loc=0.0, scale=noise_sigma, size=original_seq.shape)
            augmented_X.append(noisy_seq)
            augmented_y.append(original_target)
        
        # Time warping augmentation: create warp_reps variations using cubic interpolation
        L = original_seq.shape[0]
        orig_steps = np.arange(L)
        for _ in range(warp_reps):
            # Create a new time axis with small random perturbations
            new_steps = np.linspace(0, L-1, L) + np.random.uniform(-0.2, 0.2, size=L)
            new_steps[0] = 0
            new_steps[-1] = L-1
            new_steps = np.sort(new_steps)  # ensure monotonicity
            
            warped_seq = np.zeros_like(original_seq)
            # Interpolate each channel (feature) using cubic interpolation
            for j in range(original_seq.shape[1]):
                f = interp1d(orig_steps, original_seq[:, j], kind='cubic', fill_value="extrapolate")
                warped_seq[:, j] = f(new_steps)
            augmented_X.append(warped_seq)
            augmented_y.append(original_target)
    
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    return augmented_X, augmented_y

def standardize_data(X, y):
    """
    Feature-wise standardization for IMU signals and force outputs.
    
    For X (3D array), reshapes the data to apply StandardScaler per feature
    and then reshapes back to the original shape.
    """
    samples, timesteps, features = X.shape
    X_reshaped = X.reshape(-1, features)
    scaler_X = StandardScaler()
    X_scaled_reshaped = scaler_X.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(samples, timesteps, features)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def create_simple_model(sequence_length, n_features, n_outputs):
    """Create a simple LSTM model"""
    inputs = tf.keras.layers.Input(shape=(sequence_length, n_features))
    
    # LSTM layers
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_outputs)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_predictions(y_true, y_pred, participant_weight_kg=70):
    """Evaluate predictions with component-wise metrics"""
    # Calculate body weight force
    body_weight_force = participant_weight_kg * 9.81
    
    # Component names for better reporting
    components = ['Left Foot X', 'Left Foot Y', 'Left Foot Z', 
                  'Right Foot X', 'Right Foot Y', 'Right Foot Z']
    
    # Calculate metrics for each component
    component_metrics = {}
    for i, name in enumerate(components):
        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2))
        
        # Calculate relative error (as percentage)
        mask = np.abs(y_true[:, i]) > 1.0
        if np.sum(mask) > 0:
            mre = np.mean(np.abs(y_true[mask, i] - y_pred[mask, i]) / np.abs(y_true[mask, i])) * 100
        else:
            mre = np.nan
            
        component_metrics[name] = {
            'MAE (N)': float(mae),
            'RMSE (N)': float(rmse),
            'MRE (%)': float(mre) if not np.isnan(mre) else "N/A"
        }
    
    # Calculate vertical force constraint metrics
    total_vertical_true = np.mean(y_true[:, 1] + y_true[:, 4])
    total_vertical_pred = np.mean(y_pred[:, 1] + y_pred[:, 4])
    
    vertical_metrics = {
        'True Total Vertical Force (N)': float(total_vertical_true),
        'Predicted Total Vertical Force (N)': float(total_vertical_pred),
        'Body Weight Force (N)': float(body_weight_force),
        'Vertical Force Error (N)': float(np.abs(total_vertical_pred - body_weight_force)),
        'Vertical Force Error (%)': float(np.abs(total_vertical_pred - body_weight_force) / body_weight_force * 100)
    }
    
    # Overall metrics
    overall_mae = np.mean(np.abs(y_true - y_pred))
    overall_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    metrics = {
        'Overall': {
            'MAE (N)': float(overall_mae),
            'RMSE (N)': float(overall_rmse)
        },
        'Components': component_metrics,
        'Physics': vertical_metrics
    }
    
    return metrics

def plot_results(history, y_true, y_pred, metrics, output_dir):
    """Plot training history and prediction results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png')
    
    # Plot component-wise predictions for a sample
    sample_idx = 0  # Use the first sample for visualization
    components = ['Left Foot X', 'Left Foot Y', 'Left Foot Z', 
                  'Right Foot X', 'Right Foot Y', 'Right Foot Z']
    
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(components):
        plt.subplot(2, 3, i+1)
        plt.bar([0, 1], [y_true[sample_idx, i], y_pred[sample_idx, i]], color=['blue', 'orange'])
        plt.title(name)
        plt.xticks([0, 1], ['True', 'Predicted'])
        plt.ylabel('Force (N)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_predictions.png')
    
    # Plot vertical force comparison
    plt.figure(figsize=(10, 6))
    true_vertical = metrics['Physics']['True Total Vertical Force (N)']
    pred_vertical = metrics['Physics']['Predicted Total Vertical Force (N)']
    body_weight = metrics['Physics']['Body Weight Force (N)']
    
    plt.bar([0, 1, 2], [true_vertical, pred_vertical, body_weight], color=['blue', 'orange', 'green'])
    plt.title('Vertical Force Comparison')
    plt.xticks([0, 1, 2], ['True Vertical', 'Predicted Vertical', 'Body Weight'])
    plt.ylabel('Force (N)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vertical_force_comparison.png')
    
    # Create a bar chart for component-wise errors
    plt.figure(figsize=(12, 6))
    component_names = list(metrics['Components'].keys())
    mae_values = [metrics['Components'][comp]['MAE (N)'] for comp in component_names]
    
    plt.bar(component_names, mae_values)
    plt.title('Component-wise MAE')
    plt.xlabel('Force Component')
    plt.ylabel('MAE (N)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_errors.png')

def main():
    # Parameters
    SEQUENCE_LENGTH = 10
    EPOCHS = 150
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    PARTICIPANT_WEIGHT_KG = 70
    
    # Create output directory
    output_dir = 'model_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y = load_and_prepare_sequences(
        'software-group/data-working/assets/feb10exp/synced_IMU_forces_grf_fixed.csv',
        SEQUENCE_LENGTH
    )
    
    # Augment data
    print("\nAugmenting data...")
    X_aug, y_aug = augment_data(X, y)
    print(f"Original data shape: {X.shape}, Augmented data shape: {X_aug.shape}")
    
    # Split data (using augmented dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Standardize data (feature-wise scaling)
    X_train_scaled, y_train_scaled, scaler_X, scaler_y = standardize_data(X_train, y_train)
    X_val_scaled, y_val_scaled, _, _ = standardize_data(X_val, y_val)
    
    # The following model training code is commented out; you can uncomment and adjust as needed.
    
    # Create model
    print("Creating model...")
    model = create_simple_model(SEQUENCE_LENGTH, X_train.shape[2], y_train.shape[1])
    model.summary()
    
    # Compile model
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'{output_dir}/best_model', monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=f'{output_dir}/logs', histogram_freq=1)
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    print("\nSaving model...")
    model.save(f'{output_dir}/final_model', save_format='tf')
    
    # Save scaler parameters
    scaler_X_params = {
        'mean': scaler_X.mean_.tolist(),
        'scale': scaler_X.scale_.tolist()
    }
    
    scaler_y_params = {
        'mean': scaler_y.mean_.tolist(),
        'scale': scaler_y.scale_.tolist()
    }
    
    with open(f'{output_dir}/scaler_X.json', 'w') as f:
        json.dump(scaler_X_params, f, indent=2)
    
    with open(f'{output_dir}/scaler_y.json', 'w') as f:
        json.dump(scaler_y_params, f, indent=2)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred_scaled = model.predict(X_val_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Save model and scalers, evaluate and plot results...
      
    # Calculate metrics
    metrics = evaluate_predictions(y_val, y_pred, PARTICIPANT_WEIGHT_KG)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Overall MAE: {metrics['Overall']['MAE (N)']:.2f} N")
    print(f"Overall RMSE: {metrics['Overall']['RMSE (N)']:.2f} N")
    print("\nComponent-wise MAE:")
    for component, values in metrics['Components'].items():
        print(f"  {component}: {values['MAE (N)']:.2f} N")
    
    print("\nVertical Force Metrics:")
    print(f"  True Total Vertical Force: {metrics['Physics']['True Total Vertical Force (N)']:.2f} N")
    print(f"  Predicted Total Vertical Force: {metrics['Physics']['Predicted Total Vertical Force (N)']:.2f} N")
    print(f"  Body Weight Force: {metrics['Physics']['Body Weight Force (N)']:.2f} N")
    print(f"  Vertical Force Error: {metrics['Physics']['Vertical Force Error (N)']:.2f} N ({metrics['Physics']['Vertical Force Error (%)']:.2f}%)")
    
    # Save metrics
    with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot results
    plot_results(history, y_val, y_pred, metrics, output_dir)
    
    print("\nTraining completed successfully!")
    print(f"All outputs saved to '{output_dir}' directory")
    
    
    print("\nData augmentation and preprocessing complete!")
    print(f"All outputs will be saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()