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
    # New hyperparameters
    attention_heads: int = 8
    attention_dropout: float = 0.1
    physics_weight: float = 0.2
    warmup_epochs: int = 5
    # Target type configuration
    target_type: str = 'force'  # 'force' or 'angle'

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
        
        if self.config.target_type == 'force':
            # Force constraint loss for ground force predictions
            left_vy = y_pred[:, 1]  # Left foot vertical force
            right_vy = y_pred[:, 4]  # Right foot vertical force
            total_vy = left_vy + right_vy
            
            # Calculate body weight force
            body_weight_force = self.config.participant_weight_kg * 9.81
            
            # Force constraint loss (penalize deviation from body weight)
            force_constraint_loss = tf.reduce_mean(tf.square(total_vy - body_weight_force))
            
            # Combine losses with weighting
            total_loss = mse_loss + 0.1 * force_constraint_loss
        
        elif self.config.target_type == 'angle':
            # Physics constraint for angle predictions
            # Simple constraint for dorsiflexion angle: 
            # penalize predictions outside typical range (-15 to +20 degrees)
            min_angle = -15.0  # typical minimum dorsiflexion angle
            max_angle = 20.0   # typical maximum dorsiflexion angle
            
            # Penalties for exceeding range
            below_min_penalty = tf.reduce_mean(tf.maximum(0.0, min_angle - y_pred)**2)
            above_max_penalty = tf.reduce_mean(tf.maximum(0.0, y_pred - max_angle)**2)
            
            # Penalty for non-smooth predictions (continuity in movement)
            # Note: Only effective within batches
            if len(y_pred.shape) > 1:
                # For multi-dimensional output
                smoothness_penalty = tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]))
            else:
                # For single-dimensional output
                smoothness_penalty = 0.0
                
            # Combine losses
            total_loss = mse_loss + 0.1 * (below_min_penalty + above_max_penalty) + 0.05 * smoothness_penalty
        
        else:
            # Default to MSE loss if target type is unknown
            total_loss = mse_loss
            
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

class HybridTransformerPhysics(BaseModel):
    """Enhanced Hybrid model combining Transformer with physics-aware components"""
    def custom_physics_loss(self, y_true, y_pred):
        # Standard MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # Enhanced physics constraints
        left_vy = y_pred[:, 1]   # Left foot vertical force
        right_vy = y_pred[:, 4]  # Right foot vertical force
        left_vx = y_pred[:, 0]   # Left foot horizontal force
        right_vx = y_pred[:, 3]  # Right foot horizontal force
        
        # Vertical force constraint (body weight)
        total_vy = left_vy + right_vy
        body_weight_force = self.config.participant_weight_kg * 9.81
        vertical_constraint = tf.reduce_mean(tf.square(total_vy - body_weight_force))
        
        # Horizontal force balance constraint
        horizontal_constraint = tf.reduce_mean(tf.square(left_vx + right_vx))
        
        # Temporal smoothness constraint
        smoothness = tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]))
        
        # Combine all physics constraints
        physics_loss = (vertical_constraint + 0.1 * horizontal_constraint + 0.01 * smoothness)
        
        # Total loss with adaptive weighting
        total_loss = mse_loss + self.config.physics_weight * physics_loss
        
        return total_loss

    def physics_attention(self, query, key, value, physics_weights):
        # Enhanced scaled dot-product attention with physics awareness
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        
        # Apply physics-based weighting with temperature scaling
        temperature = 0.1
        physics_enhanced_logits = logits * physics_weights / temperature
        
        # Attention dropout
        attention_weights = tf.nn.softmax(physics_enhanced_logits, axis=-1)
        attention_weights = tf.keras.layers.Dropout(self.config.attention_dropout)(attention_weights)
        
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Enhanced physics-aware feature extraction
        physics_branch = tf.keras.layers.Dense(64, activation='relu')(inputs)
        physics_branch = tf.keras.layers.LayerNormalization(epsilon=1e-6)(physics_branch)
        physics_branch = tf.keras.layers.Dense(32, activation='relu')(physics_branch)
        physics_branch = tf.keras.layers.Dropout(0.1)(physics_branch)
        physics_weights = tf.keras.layers.Dense(1, activation='sigmoid')(physics_branch)
        
        # Enhanced transformer branch
        x = tf.keras.layers.Dense(128)(inputs)  # Increased dimension
        
        # Multi-layer physics-aware attention
        for _ in range(3):  # Multiple layers
            attention_output, _ = self.physics_attention(x, x, x, physics_weights)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network with increased capacity
            ffn = tf.keras.layers.Dense(256, activation='relu')(x)
            ffn = tf.keras.layers.Dropout(0.1)(ffn)
            ffn = tf.keras.layers.Dense(128)(ffn)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global feature extraction
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        physics_features = tf.keras.layers.GlobalAveragePooling1D()(physics_branch)
        
        # Combine features with residual connection
        combined = tf.keras.layers.Concatenate()([x, physics_features])
        
        # Output layers with residual connections
        x = tf.keras.layers.Dense(64, activation='relu')(combined)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_shape)(x)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # Custom learning rate schedule with warmup - fixed to not use X_train
        initial_learning_rate = self.config.learning_rate
        warmup_steps = self.config.warmup_epochs * 100  # Fixed estimate of steps per epoch
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            warmup_steps,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0  # Add gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss=self.custom_physics_loss,
            metrics=['mae']
        )
        
        self.model = model
        return model

class MultiScaleTransformer(BaseModel):
    """Multi-Scale Temporal Transformer with parallel attention mechanisms"""
    def temporal_attention_block(self, inputs, num_heads, key_dim, window_size):
        # Get static input shape information
        batch_size, seq_length, feature_dim = inputs.shape
        
        # Ensure sequence length is valid for the window size
        if seq_length < window_size:
            # If sequence is too short, use the full sequence
            window_size = seq_length
            
        # Compute number of windows (this is a static Python integer, not a tensor)
        num_windows = seq_length - window_size + 1
        
        # Create windows using tf.slice operations instead of Python list comprehension
        windows_list = []
        for i in range(num_windows):
            # Extract window using slice
            window = inputs[:, i:i+window_size, :]
            windows_list.append(window)
            
        # Stack windows along a new axis
        windows = tf.stack(windows_list, axis=1)
        
        # Reshape for attention: [batch, num_windows, window_size, features]
        # -> [batch * num_windows, window_size, features]
        reshaped_windows = tf.reshape(windows, [-1, window_size, feature_dim])
        
        # Apply attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )(reshaped_windows, reshaped_windows)
        
        # Reshape back: [batch * num_windows, window_size, features]
        # -> [batch, num_windows, window_size, features]
        attention = tf.reshape(attention, [-1, num_windows, window_size, feature_dim])
        
        # Reduce window dimension via global average pooling
        attention = tf.reduce_mean(attention, axis=2)  # [batch, num_windows, features]
        
        return attention

    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        # Extract static dimensions
        seq_length, feature_dim = input_shape
        
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Multi-scale processing with fixed window sizes
        # Fast scale (local patterns)
        fast_window_size = min(5, seq_length)
        fast_scale = self.temporal_attention_block(inputs, num_heads=4, key_dim=16, window_size=fast_window_size)
        
        # Medium scale (gait cycles)
        medium_window_size = min(10, seq_length)
        medium_scale = self.temporal_attention_block(inputs, num_heads=4, key_dim=16, window_size=medium_window_size)
        
        # Slow scale (overall patterns)
        slow_window_size = min(20, seq_length)
        slow_scale = self.temporal_attention_block(inputs, num_heads=4, key_dim=16, window_size=slow_window_size)
        
        # Concatenate all scales
        combined = tf.keras.layers.Concatenate(axis=1)([fast_scale, medium_scale, slow_scale])
        
        # Process combined features
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(combined)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_shape)(x)
        
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class CrossModalAttentionTransformer(BaseModel):
    """Transformer with cross-modal attention and uncertainty estimation"""
    def uncertainty_attention(self, query, key, value, dropout_rate=0.1, num_samples=5):
        attention_outputs = []
        
        # Monte Carlo sampling with dropout
        for _ in range(num_samples):
            # Apply dropout to create different attention patterns
            query_drop = tf.keras.layers.Dropout(dropout_rate)(query)
            key_drop = tf.keras.layers.Dropout(dropout_rate)(key)
            
            # Compute attention scores
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=16
            )(query_drop, key_drop, value)
            
            attention_outputs.append(attention)
        
        # Compute mean and variance of attention outputs
        stacked_outputs = tf.stack(attention_outputs, axis=0)
        mean_attention = tf.reduce_mean(stacked_outputs, axis=0)
        variance_attention = tf.math.reduce_variance(stacked_outputs, axis=0)
        
        # Weight attention by uncertainty (element-wise)
        confidence = 1.0 / (variance_attention + 1e-6)
        weighted_attention = mean_attention * confidence
        
        return weighted_attention, confidence

    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Split input into modalities 
        # Instead of assuming first/second half, split based on sensor type
        # First 6 channels: accelerometer (3 axes × 2 IMUs)
        # Last 6 channels: gyroscope (3 axes × 2 IMUs)
        accel_input = tf.keras.layers.Lambda(lambda x: x[:, :, :6])(inputs)
        gyro_input = tf.keras.layers.Lambda(lambda x: x[:, :, 6:])(inputs)
        
        # Process each modality
        accel_features = tf.keras.layers.Dense(32)(accel_input)
        gyro_features = tf.keras.layers.Dense(32)(gyro_input)
        
        # Cross-modal attention with uncertainty
        cross_attention, confidence = self.uncertainty_attention(
            accel_features, gyro_features, gyro_features
        )
        
        # Combine features - both should have same shape [batch, seq_len, 32]
        # Element-wise multiplication with confidence (same shape)
        weighted_accel = accel_features * confidence
        
        # Combine features
        combined = tf.keras.layers.Concatenate()([cross_attention, weighted_accel])
        
        # Process combined features
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(combined)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers with dropout for uncertainty
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_shape)(x)
        
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class MLPipeline:
    """Main pipeline for model training and evaluation"""
    def __init__(self, config: ModelConfig):
        self.config = config
        # Create output directory and subdirectories for each model
        os.makedirs('model_output', exist_ok=True)
        for model_name in ['SimpleDense', 'LSTM', 'BidirectionalLSTM', 'CNNLSTM', 'Transformer', 'PhysicsConstrained', 'HybridTransformerPhysics', 'MultiScaleTransformer', 'CrossModalAttentionTransformer']:
            os.makedirs(f'model_output/{model_name}', exist_ok=True)
        self.models = {
            'SimpleDense': SimpleDenseModel(config),
            'LSTM': LSTMModel(config),
            'BidirectionalLSTM': BidirectionalLSTMModel(config),
            'CNNLSTM': CNNLSTMModel(config),
            'Transformer': TransformerModel(config),
            'PhysicsConstrained': PhysicsConstrainedModel(config),
            'HybridTransformerPhysics': HybridTransformerPhysics(config),
            'MultiScaleTransformer': MultiScaleTransformer(config),
            'CrossModalAttentionTransformer': CrossModalAttentionTransformer(config)
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
        results_file = os.path.join('model_output', 'model_comparison_results_NEW_SECOND.json')
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
    """
    Load and prepare data with minimal preprocessing for the enhanced experiments.
    
    This function is adapted to work with the mar12exp_updated dataset format,
    which includes IMU data and dorsiflexion angle as the target.
    
    A sliding window approach with stride=1 is used to extract sequences.
    """
    # Load the data
    df = pd.read_csv(file_path)
    print(f"Raw data shape: {df.shape}")
    
    # Determine if this is the new format (with dorsiflexion_angle) or old format (with ground forces)
    if 'dorsiflexion_angle' in df.columns:
        print("Using dorsiflexion angle dataset format")
        # Select features and targets for the dorsiflexion angle format
        imu_features = [col for col in df.columns if ('acc_' in col or 'gyro_' in col) and not 'timestamp' in col]
        target_column = ['dorsiflexion_angle']
        
        # Print dataset statistics
        print("\nDorsiflexion Angle Statistics:")
        print(df[target_column].describe())
        
        # Check for NaN values in input features
        nan_count = df[imu_features].isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in features, filling with interpolation")
            df[imu_features] = df[imu_features].interpolate(method='linear', limit_direction='both')
            
        # Create sequences with overlap
        sequences = []
        targets = []
        stride = 1
        
        # Track sequence count
        total_sequences = 0
        
        for i in range(0, len(df) - sequence_length, stride):
            seq = df[imu_features].values[i:i+sequence_length]
            target = df[target_column].values[i+sequence_length-1]
            
            # Only include sequences with valid data
            if not np.isnan(target).any() and not np.isnan(seq).any():
                sequences.append(seq)
                targets.append(target)
                total_sequences += 1
        
        print(f"Total sequences created: {total_sequences}")
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"Processed dataset shape - X: {X.shape}, y: {y.shape}")
        return X, y
        
    else:
        print("Using ground force dataset format")
        # Select features and targets for the ground force format (original code)
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

def augment_data(X, y, noise_reps=3, warp_reps=3, mask_reps=2, scale_reps=2, 
              permute_reps=2, mix_reps=2, noise_sigma=0.05):
    """
    Perform enhanced data augmentation on time-series sequences.
    
    Augmentation includes:
      - Adding Gaussian noise with adaptive levels
      - Non-linear time warping using cubic interpolation
      - Channel masking/dropout to simulate sensor failures
      - Magnitude scaling to simulate different sensor sensitivities
      - Permutation of small segments to improve robustness
      - Signal mixing with controlled weights
    
    The force targets (y) are kept unchanged for all augmentation methods.
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
        
        # Compute signal statistics for adaptive augmentation
        channel_stds = np.std(original_seq, axis=0)
        
        # 1. Gaussian noise augmentation with adaptive noise levels
        for _ in range(noise_reps):
            # Adaptive noise level: scale based on each channel's standard deviation
            adaptive_noise = np.random.normal(
                loc=0.0, 
                scale=noise_sigma * channel_stds * (0.5 + np.random.random()),
                size=original_seq.shape
            )
            noisy_seq = original_seq + adaptive_noise
            augmented_X.append(noisy_seq)
            augmented_y.append(original_target)
        
        # 2. Time warping augmentation (enhanced)
        L = original_seq.shape[0]
        orig_steps = np.arange(L)
        for _ in range(warp_reps):
            # Create a new time axis with variable amplitude perturbations
            warp_sigma = 0.3 * np.random.random() + 0.1  # Random between 0.1 and 0.4
            new_steps = np.linspace(0, L-1, L) + np.random.uniform(-warp_sigma, warp_sigma, size=L)
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
        
        # 3. NEW: Channel masking/dropout - simulate sensor failures
        for _ in range(mask_reps):
            masked_seq = original_seq.copy()
            # Randomly select 1-3 channels to mask
            num_channels_to_mask = np.random.randint(1, 4)
            channels_to_mask = np.random.choice(
                original_seq.shape[1], 
                size=num_channels_to_mask, 
                replace=False
            )
            
            # Determine mask length - mask a segment of the sequence
            mask_length = np.random.randint(L // 5, L // 2)
            mask_start = np.random.randint(0, L - mask_length)
            
            # Apply masking (set to zero or mean)
            for channel in channels_to_mask:
                if np.random.random() < 0.5:
                    # Zero masking
                    masked_seq[mask_start:mask_start+mask_length, channel] = 0
                else:
                    # Mean value masking
                    masked_seq[mask_start:mask_start+mask_length, channel] = np.mean(
                        original_seq[:, channel]
                    )
            
            augmented_X.append(masked_seq)
            augmented_y.append(original_target)
        
        # 4. NEW: Magnitude scaling - simulate different sensor sensitivities
        for _ in range(scale_reps):
            scaled_seq = original_seq.copy()
            
            # Apply random scaling to each channel
            for j in range(original_seq.shape[1]):
                # Random scale factor between 0.7 and 1.3
                scale_factor = 0.7 + 0.6 * np.random.random()
                scaled_seq[:, j] = original_seq[:, j] * scale_factor
            
            augmented_X.append(scaled_seq)
            augmented_y.append(original_target)
        
        # 5. NEW: Permutation of segments - improve robustness to local signal distortions
        for _ in range(permute_reps):
            permuted_seq = original_seq.copy()
            
            # Determine number of segments to permute
            num_segments = np.random.randint(2, 5)
            min_segment_size = L // 10
            
            # Select random segment boundaries
            segment_points = np.sort(np.random.choice(
                np.arange(min_segment_size, L - min_segment_size),
                size=num_segments-1, 
                replace=False
            ))
            
            # Create segments
            segments = np.split(permuted_seq, segment_points, axis=0)
            
            # Randomly select a few adjacent segments to swap
            if len(segments) >= 2:
                swap_idx = np.random.randint(0, len(segments) - 1)
                segments[swap_idx], segments[swap_idx + 1] = segments[swap_idx + 1], segments[swap_idx]
            
            # Recombine segments
            permuted_seq = np.concatenate(segments, axis=0)
            
            augmented_X.append(permuted_seq)
            augmented_y.append(original_target)
        
        # 6. NEW: Signal mixing with other sequences - enhance generalization
        for _ in range(mix_reps):
            # Randomly select another sequence to mix with
            other_idx = np.random.choice([j for j in range(len(X)) if j != i])
            other_seq = X[other_idx]
            
            # Create random mixing weights
            alpha = 0.7 + 0.3 * np.random.random()  # Primary sequence has 70-100% weight
            mixed_seq = alpha * original_seq + (1 - alpha) * other_seq
            
            augmented_X.append(mixed_seq)
            augmented_y.append(original_target)
    
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    
    print(f"Original dataset: {X.shape}, Augmented dataset: {augmented_X.shape}")
    print(f"Augmentation factor: {len(augmented_X) / len(X):.2f}x")
    
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
    """Evaluate predictions with appropriate metrics based on data dimensions"""
    
    # Check if we're working with single-value targets (angle) or multi-value targets (forces)
    is_angle_prediction = len(y_true.shape) == 1 or y_true.shape[1] == 1
    
    if is_angle_prediction:
        # For dorsiflexion angle predictions - reshape to 1D if needed
        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
            
        # Calculate basic metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Calculate relative error as percentage
        mask = np.abs(y_true) > 0.1  # Avoid division by very small values
        if np.sum(mask) > 0:
            mre = np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100
        else:
            mre = np.nan
            
        metrics = {
            'Overall': {
                'MAE (degrees)': float(mae),
                'RMSE (degrees)': float(rmse),
                'MRE (%)': float(mre) if not np.isnan(mre) else None,
                'Correlation': float(correlation)
            },
            'Range Analysis': {
                'True Min': float(np.min(y_true)),
                'True Max': float(np.max(y_true)),
                'Predicted Min': float(np.min(y_pred)),
                'Predicted Max': float(np.max(y_pred)),
                'Range Coverage (%)': float(
                    (np.max(y_pred) - np.min(y_pred)) / (np.max(y_true) - np.min(y_true)) * 100
                ) if (np.max(y_true) - np.min(y_true)) > 0 else 0
            }
        }
        
        return metrics
    else:
        # Original implementation for ground reaction force predictions
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
                'MRE (%)': float(mre) if not np.isnan(mre) else None
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
    """
    Main function to execute the ML pipeline with enhanced data augmentation
    """
    # Configuration with slightly modified hyperparameters
    config = ModelConfig(
        sequence_length=10, 
        epochs=150, 
        batch_size=16,
        learning_rate=0.001,
        participant_weight_kg=70,
        validation_split=0.2,
        target_type='angle'  # Using angle as target instead of forces
    )
    
    # Create timestamp for this experiment run
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"enhanced_augmentation_{timestamp}"
    
    print(f"Starting experiment: {experiment_name}")
    print("Using enhanced data augmentation techniques")
    
    # Create pipeline
    pipeline = MLPipeline(config)
    
    # Train and evaluate models using the mar12exp dataset
    pipeline.train_and_evaluate(
        'software-group/data-working/assets/mar12exp/mar12exp_updated/2025-03-12_10-25-01-r4-walking3.csv'
    )
    
    # Plot comparison with new experiment name
    # Modify the plot_comparison method to use our experiment name
    original_plot_comparison = pipeline.plot_comparison
    
    def custom_plot_comparison():
        # Convert results to JSON-serializable format
        json_results = pipeline._convert_numpy_to_native(pipeline.results)
        
        # Save results to JSON file with experiment name
        results_file = os.path.join('model_output', f'model_comparison_results_{experiment_name}.json')
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Create comparison plots
        plt.figure(figsize=(12, 6))
        
        # Plot training history comparison
        plt.subplot(1, 2, 1)
        for model_name in pipeline.results:
            history = pipeline.results[model_name]['history']
            plt.plot(history['val_loss'], label=f'{model_name}')
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot final performance comparison
        plt.subplot(1, 2, 2)
        model_names = list(pipeline.results.keys())
        final_mae = [pipeline.results[model]['history']['val_mae'][-1] for model in model_names]
        plt.bar(model_names, final_mae)
        plt.title('Final Validation MAE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('MAE')
        
        plt.tight_layout()
        plt.savefig(os.path.join('model_output', f'model_comparison_{experiment_name}.png'))
        plt.close()
        
        # Create training curves visualization
        plt.figure(figsize=(15, 10))
        
        for i, model_name in enumerate(pipeline.results.keys()):
            plt.subplot(3, 3, i+1)
            history = pipeline.results[model_name]['history']
            # Plot training and validation loss
            plt.plot(history['loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title(f'{model_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join('model_output', f'training_curves_{experiment_name}.png'))
        plt.close()
        
        # Create a more detailed metrics visualization
        plt.figure(figsize=(12, 8))
        
        metrics_data = []
        for model_name in pipeline.results.keys():
            if 'metrics' in pipeline.results[model_name]:
                metric = pipeline.results[model_name]['metrics']
                if 'Overall' in metric and 'MAE (N)' in metric['Overall']:
                    metrics_data.append({
                        'Model': model_name,
                        'MAE': metric['Overall']['MAE (N)'],
                        'RMSE': metric['Overall']['RMSE (N)'] if 'RMSE (N)' in metric['Overall'] else 0
                    })
        
        if metrics_data:
            models = [item['Model'] for item in metrics_data]
            mae_values = [item['MAE'] for item in metrics_data]
            rmse_values = [item['RMSE'] for item in metrics_data]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.bar(x - width/2, mae_values, width, label='MAE')
            plt.bar(x + width/2, rmse_values, width, label='RMSE')
            
            plt.xlabel('Models')
            plt.ylabel('Error Metrics (N)')
            plt.title('Model Performance Comparison')
            plt.xticks(x, models, rotation=45)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join('model_output', f'detailed_metrics_{experiment_name}.png'))
            plt.close()
    
    # Replace the plot_comparison method with our custom version
    pipeline.plot_comparison = custom_plot_comparison
    
    # Execute the custom plot_comparison method
    pipeline.plot_comparison()
    
    print(f"Experiment {experiment_name} completed. Results saved to model_output/ directory.")

if __name__ == "__main__":
    main() 