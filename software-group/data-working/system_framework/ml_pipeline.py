import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d

# Disable GPU acceleration
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

@dataclass
class ModelConfig:
    """Configuration for model training"""
    sequence_length: int = 10
    epochs: int = 150
    batch_size: int = 16
    learning_rate: float = 0.001
    participant_weight_kg: float = 70
    validation_split: float = 0.2
    random_state: int = 42

class BaseModel(ABC):
    """Abstract base class for all models"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        self.metrics = None
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        """Build the model architecture"""
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, List[float]]:
        """Train the model"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=20, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'model_output/{self.__class__.__name__}/best_model',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        predictions = self.model.predict(X_test)
        self.metrics = evaluate_predictions(y_test, predictions, self.config.participant_weight_kg)
        return self.metrics
    
    def save_model(self, path: str):
        """Save the model"""
        os.makedirs(path, exist_ok=True)
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load the model"""
        self.model = tf.keras.models.load_model(path)

class SimpleDenseModel(BaseModel):
    """Simple feed-forward neural network"""
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_shape)(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class LSTMModel(BaseModel):
    """LSTM-based model"""
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.LSTM(32)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_shape)(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class BidirectionalLSTMModel(BaseModel):
    """Bidirectional LSTM model"""
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_shape)(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class CNNLSTMModel(BaseModel):
    """Combined CNN and LSTM model"""
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # CNN branch
        x1 = tf.keras.layers.Conv1D(64, 3, activation='relu')(inputs)
        x1 = tf.keras.layers.MaxPooling1D(2)(x1)
        x1 = tf.keras.layers.Conv1D(32, 3, activation='relu')(x1)
        x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
        
        # LSTM branch
        x2 = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
        x2 = tf.keras.layers.Dropout(0.2)(x2)
        x2 = tf.keras.layers.LSTM(32)(x2)
        
        # Combine branches
        combined = tf.keras.layers.Concatenate()([x1, x2])
        x = tf.keras.layers.Dense(64, activation='relu')(combined)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_shape)(x)
        
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class TransformerModel(BaseModel):
    """Transformer-based model"""
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Positional encoding
        x = tf.keras.layers.Dense(64)(inputs)
        
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + x)
        
        # Feed-forward network
        ffn = tf.keras.layers.Dense(64, activation='relu')(x)
        ffn = tf.keras.layers.Dense(64)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn + x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_shape)(x)
        
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class PhysicsConstrainedModel(BaseModel):
    """Model with physics-based constraints in loss function"""
    def custom_loss(self, y_true, y_pred):
        # Standard MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # Force constraint loss
        left_vy = y_pred[:, 1]  # Left foot vertical force
        right_vy = y_pred[:, 4]  # Right foot vertical force
        total_vy = left_vy + right_vy
        
        # Calculate body weight force
        body_weight_force = self.config.participant_weight_kg * 9.81
        
        # Force constraint loss (penalize deviation from body weight)
        force_constraint_loss = tf.reduce_mean(tf.square(total_vy - body_weight_force))
        
        # Combine losses with weighting
        total_loss = mse_loss + 0.1 * force_constraint_loss
        
        return total_loss
    
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Flatten the input
        x = tf.keras.layers.Flatten()(inputs)
        
        # Initial dense layer
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # First residual block
        skip1 = x
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Add()([x, skip1])
        
        # Second residual block
        skip2 = x
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Add()([x, skip2])
        
        # Output layer
        outputs = tf.keras.layers.Dense(output_shape)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss=self.custom_loss, metrics=['mae'])
        
        return model

class MLPipeline:
    """Main pipeline for model training and evaluation"""
    def __init__(self, config: ModelConfig):
        self.config = config
        # Create output directory and subdirectories for each model
        os.makedirs('model_output', exist_ok=True)
        for model_name in ['SimpleDense', 'LSTM', 'BidirectionalLSTM', 'CNNLSTM', 'Transformer', 'PhysicsConstrained']:
            os.makedirs(f'model_output/{model_name}', exist_ok=True)
        self.models = {
            'SimpleDense': SimpleDenseModel(config),
            'LSTM': LSTMModel(config),
            'BidirectionalLSTM': BidirectionalLSTMModel(config),
            'CNNLSTM': CNNLSTMModel(config),
            'Transformer': TransformerModel(config),
            'PhysicsConstrained': PhysicsConstrainedModel(config)
        }
        self.results = {}
    
    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split data"""
        # Load and prepare sequences
        X, y = load_and_prepare_sequences(data_path, self.config.sequence_length)
        
        # Augment data
        X_aug, y_aug = augment_data(X, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_aug, y_aug,
            test_size=self.config.validation_split,
            random_state=self.config.random_state
        )
        
        # Scale data
        X_train_scaled, y_train_scaled, scaler_X, scaler_y = standardize_data(X_train, y_train)
        X_val_scaled, y_val_scaled, _, _ = standardize_data(X_val, y_val)
        
        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled
    
    def train_and_evaluate(self, data_path: str):
        """Train and evaluate all models"""
        # Prepare data
        X_train, y_train, X_val, y_val = self.prepare_data(data_path)
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            
            # Build and compile model
            model.model = model.build_model(X_train.shape[1:], y_train.shape[1])
            if not isinstance(model, PhysicsConstrainedModel):  # Skip for physics-constrained model as it's already compiled
                model.model.compile(
                    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.config.learning_rate),
                    loss='mse',
                    metrics=['mae']
                )
            
            # Train model
            history = model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate model
            metrics = model.evaluate(X_val, y_val)
            
            # Store results
            self.results[name] = {
                'history': history,
                'metrics': metrics
            }
            
            # Save model
            model.save_model(f'model_output/{name}')
    
    def _convert_numpy_to_native(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_native(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_to_native(obj.tolist())
        else:
            return obj

    def plot_comparison(self):
        """Plot and save comparison of model performances."""
        # Convert results to JSON-serializable format
        json_results = self._convert_numpy_to_native(self.results)
        
        # Save results to JSON file
        results_file = os.path.join('model_output', 'model_comparison_results_NEW.json')
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Create comparison plots
        plt.figure(figsize=(12, 6))
        
        # Plot training history comparison
        plt.subplot(1, 2, 1)
        for model_name in self.results:
            history = self.results[model_name]['history']
            plt.plot(history['val_loss'], label=f'{model_name}')
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot final performance comparison
        plt.subplot(1, 2, 2)
        model_names = list(self.results.keys())
        final_mae = [self.results[model]['history']['val_mae'][-1] for model in model_names]
        plt.bar(model_names, final_mae)
        plt.title('Final Validation MAE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('MAE')
        
        plt.tight_layout()
        plt.savefig(os.path.join('model_output', 'model_comparison.png'))
        plt.close()

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

def main():
    # Configuration
    config = ModelConfig()
    
    # Create pipeline
    pipeline = MLPipeline(config)
    
    # Train and evaluate models
    pipeline.train_and_evaluate(
        'software-group/data-working/assets/mar12exp/synced_IMU_forces_grf_fixed.csv'
    )
    
    # Plot comparison
    pipeline.plot_comparison()

if __name__ == "__main__":
    main() 