import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
import joblib
from datetime import datetime

# Disable GPU acceleration
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

def create_normalized_physics_loss(mse_scale=1.0,
                                  vertical_scale=450000.0, horizontal_scale=375.0, smoothness_scale=65.0, 
                                  alpha=0.57, beta=0.29, gamma=0.14, lambda_phys=0.2,
                                  participant_weight_kg=70.0):
    """
    Creates a normalized physics-informed loss function and associated metrics.
    
    Args:
        mse_scale: Scale factor for MSE term.
        vertical_scale: Scale factor for vertical constraint.
        horizontal_scale: Scale factor for horizontal constraint.
        smoothness_scale: Scale factor for smoothness constraint.
        alpha, beta, gamma: Weights for the normalized physics terms.
        lambda_phys: Weight for the overall physics loss component.
        participant_weight_kg: Participant weight.
        
    Returns:
        A tuple containing:
        - normalized_physics_loss: The main loss function (scalar output).
        - custom_metrics: A list of metric functions for individual components.
    """
    
    body_weight_force = participant_weight_kg * 9.81
    
    # Use tf constants for scale factors to avoid graph retracing issues
    # Add epsilon to prevent division by zero
    epsilon = 1e-9
    mse_scale_val = tf.constant(mse_scale if mse_scale > epsilon else 1.0, dtype=tf.float32)
    vertical_scale_val = tf.constant(vertical_scale if vertical_scale > epsilon else 1.0, dtype=tf.float32)
    horizontal_scale_val = tf.constant(horizontal_scale if horizontal_scale > epsilon else 1.0, dtype=tf.float32)
    smoothness_scale_val = tf.constant(smoothness_scale if smoothness_scale > epsilon else 1.0, dtype=tf.float32)
    
    alpha = tf.constant(alpha, dtype=tf.float32)
    beta = tf.constant(beta, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    lambda_phys = tf.constant(lambda_phys, dtype=tf.float32)
    body_weight_force = tf.constant(body_weight_force, dtype=tf.float32)

    # --- Define Helper Functions (Shared Logic) ---
    def calculate_components(y_true, y_pred):
        # Standard MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        normalized_mse_loss = mse_loss / mse_scale_val
        
        # Physics Constraints
        left_vy = y_pred[..., 1] 
        right_vy = y_pred[..., 4]
        left_vx = y_pred[..., 0]
        right_vx = y_pred[..., 3]
        
        total_vy = left_vy + right_vy
        vertical_constraint = tf.reduce_mean(tf.square(total_vy - body_weight_force))
        horizontal_constraint = tf.reduce_mean(tf.square(left_vx + right_vx))
        
        vertical_normalized = vertical_constraint / vertical_scale_val
        horizontal_normalized = horizontal_constraint / horizontal_scale_val
        
        # Temporal smoothness Calculation
        y_pred_for_smoothness = tf.cond(
            tf.rank(y_pred) == 2,
            lambda: tf.expand_dims(y_pred, axis=1),
            lambda: y_pred
        )

        def calculate_smoothness_if_possible(tensor_3d):
            time_dim = tf.shape(tensor_3d)[1]
            def do_slice_and_calc():
                diffs = tensor_3d[:, 1:, :] - tensor_3d[:, :-1, :]
                sm = tf.reduce_mean(tf.square(tf.norm(diffs, axis=-1)))
                return sm
            def return_zero(): return tf.constant(0.0, dtype=tf.float32)
            smoothness_val = tf.cond(time_dim > 1, do_slice_and_calc, return_zero)
            return smoothness_val

        smoothness = calculate_smoothness_if_possible(y_pred_for_smoothness)
        smoothness_normalized = smoothness / smoothness_scale_val

        return normalized_mse_loss, vertical_normalized, horizontal_normalized, smoothness_normalized

    # --- Main Loss Function --- 
    def normalized_physics_loss(y_true, y_pred):
        norm_mse, norm_vert, norm_horiz, norm_smooth = calculate_components(y_true, y_pred)
        
        # Weighted physics loss
        weighted_physics_loss = (
            alpha * norm_vert + 
            beta * norm_horiz + 
            gamma * norm_smooth
        )
        
        # Combine with normalized MSE using lambda_phys weight
        total_loss = norm_mse + lambda_phys * weighted_physics_loss
        
        return total_loss

    # --- Custom Metric Functions ---
    def metric_norm_mse(y_true, y_pred):
        norm_mse, _, _, _ = calculate_components(y_true, y_pred)
        return norm_mse
    metric_norm_mse.__name__ = 'norm_mse' # Set name for logging

    def metric_norm_vert(y_true, y_pred):
        _, norm_vert, _, _ = calculate_components(y_true, y_pred)
        return norm_vert
    metric_norm_vert.__name__ = 'norm_vert'

    def metric_norm_horiz(y_true, y_pred):
        _, _, norm_horiz, _ = calculate_components(y_true, y_pred)
        return norm_horiz
    metric_norm_horiz.__name__ = 'norm_horiz'

    def metric_norm_smooth(y_true, y_pred):
        _, _, _, norm_smooth = calculate_components(y_true, y_pred)
        return norm_smooth
    metric_norm_smooth.__name__ = 'norm_smooth'
    
    # --- Return loss function and metrics list ---
    custom_metrics = [metric_norm_mse, metric_norm_vert, metric_norm_horiz, metric_norm_smooth]
    
    return normalized_physics_loss, custom_metrics

@dataclass
class ModelConfig:
    """Configuration for model training for GRF Estimation"""
    sequence_length: int = 10 # Corresponds to T in problem definition
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    participant_weight_kg: float = 70 # Corresponds to m for physics loss
    validation_split: float = 0.2
    random_state: int = 42
    # Hyperparameters from advanced models
    attention_heads: int = 8
    attention_dropout: float = 0.1
    physics_weight: float = 0.1 # STAGE 2: Fine-tune with physics_weight=0.1
    warmup_epochs: int = 5
    # target_type removed - pipeline focuses only on GRF

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
    
    def apply_physics_loss(self, mse_scale, vertical_scale=None, horizontal_scale=None, smoothness_scale=None):
        """Apply normalized physics loss to the model
        
        Args:
            mse_scale: Scale factor for the MSE term (e.g., mean target variance).
            vertical_scale: Scale factor for vertical constraint (V).
            horizontal_scale: Scale factor for horizontal constraint (H).
            smoothness_scale: Scale factor for smoothness constraint (S).
        """
        if self.model is None:
            raise ValueError("Model must be built before applying physics loss")

        # Prepare arguments for create_normalized_physics_loss
        # Only pass scale factors if they are explicitly provided (not None)
        loss_args = {
            'mse_scale': mse_scale,
            'alpha': 0.57,
            'beta': 0.29,
            'gamma': 0.14,
            'lambda_phys': self.config.physics_weight,
            'participant_weight_kg': self.config.participant_weight_kg
        }
        if vertical_scale is not None:
            loss_args['vertical_scale'] = vertical_scale
        if horizontal_scale is not None:
            loss_args['horizontal_scale'] = horizontal_scale
        if smoothness_scale is not None:
            loss_args['smoothness_scale'] = smoothness_scale
        
        # Create the normalized physics loss function and associated metrics using defaults if scales are not provided
        loss_fn, custom_metrics = create_normalized_physics_loss(**loss_args)
        
        # Store the loss function and metrics for later use
        self.physics_loss_fn = loss_fn
        self.custom_metrics = custom_metrics # Store metrics list
        
        # Re-compile the model with the physics loss and custom metrics
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.config.learning_rate) # Use legacy Adam for M1/M2 compatibility
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'] + custom_metrics)
        
        print(f"Applied normalized physics loss (MSE Scale={mse_scale:.4f}, λ_phys={self.config.physics_weight})")
        # Also print the names of the metrics being tracked
        metric_names = [m.__name__ for m in custom_metrics]
        print(f"Tracking metrics: ['mae', {', '.join(metric_names)}]")
    
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
                # Save best model uniquely for this class instance run
                filepath=f'model_output/{self.__class__.__name__}/best_model_{self.__class__.__name__}',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False # Save entire model
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
        # Load the best weights saved during training for evaluation
        best_model_path = f'model_output/{self.__class__.__name__}/best_model_{self.__class__.__name__}'
        if os.path.exists(best_model_path):
             print(f"Loading best weights from {best_model_path} for evaluation.")
             # Load the entire best model
             self.model = tf.keras.models.load_model(best_model_path, compile=False)
             # Re-compile if necessary (needed if custom objects like loss are used)
             if hasattr(self, 'custom_loss') or hasattr(self, 'custom_physics_loss'):
                 print("Re-compiling model with custom loss/optimizer for evaluation.")
                 optimizer = self.model.optimizer # Reuse optimizer if possible or re-instantiate
                 loss_func = getattr(self, 'custom_loss', getattr(self, 'custom_physics_loss', 'mse'))
                 self.model.compile(optimizer=optimizer, loss=loss_func, metrics=['mae'])
             else:
                 self.model.compile(loss='mse', metrics=['mae']) # Default compile
        else:
             print(f"Warning: Best model checkpoint not found at {best_model_path}. Evaluating with final weights.")

        predictions = self.model.predict(X_test)
        # evaluate_predictions now only handles force prediction
        self.metrics = evaluate_predictions(y_test, predictions, self.config.participant_weight_kg)
        return self.metrics
    
    def save_model(self, path: str):
        """Save the final model (not necessarily the best performing one)."""
        final_model_path = f"{path}_final" # Distinguish from best model
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        self.model.save(final_model_path)
        print(f"Saved final model weights to {final_model_path}")
    
    def load_model(self, path: str):
        """Load a specific model file."""
        # Compile=False is often safer for loading models with custom components outside training loops
        self.model = tf.keras.models.load_model(path, compile=False)
        print(f"Loaded model from {path}")

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


class HybridTransformerPhysics(BaseModel):
    """Enhanced Hybrid model combining Transformer with physics-aware components"""
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
        
        # Basic compile with MSE first
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        # Apply physics loss after model is built
        self.apply_physics_loss(1.0)
        
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

class BiLSTMAttentionModel(BaseModel):
    def build_model(self, input_shape, output_dim):
        """Builds a Bidirectional LSTM model with Self-Attention."""
        x_in = tf.keras.Input(shape=input_shape, name="input")
        # BiLSTM layer captures temporal dependencies, returns sequences for attention
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True), name="bilstm_1"
        )(x_in)

        # Self-Attention layer to weigh different time steps
        # Using key_dim for projection dimension as suggested
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, name="multi_head_attention"
        )(query=x, value=x, key=x) # Self-attention: query, value, key are the same

        # Add & Norm: Residual connection and Layer Normalization
        x = tf.keras.layers.LayerNormalization(name="layer_norm_1")(attention_output + x)

        # Another BiLSTM layer to process the attention-weighted sequence
        # This one does not return sequences, only the final state
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32), name="bilstm_2"
        )(x)

        # Final Dense layer for output prediction
        out = tf.keras.layers.Dense(output_dim, name="output_dense")(x)

        model = tf.keras.Model(inputs=x_in, outputs=out, name="BiLSTMAttention")
        print(f"Built BiLSTMAttentionModel.")
        model.summary() # Print model summary
        return model

class CNNBiLSTMSqueezeExcitationModel(BaseModel):
    def build_model(self, input_shape, output_dim):
        """Builds a CNN-BiLSTM model with Squeeze-and-Excitation block."""
        x_in = tf.keras.Input(shape=input_shape, name="input")

        # CNN Feature Extraction Block
        # Using 'causal' padding for time series, kernel size 3 is common
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal', name="conv1d_1")(x_in)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal', name="conv1d_2")(x)
        # x shape after Conv1D: (batch, seq_len, 64)

        # Squeeze-and-Excitation Block
        # Squeeze: Global Average Pooling across the time dimension
        se = tf.keras.layers.GlobalAveragePooling1D(name="se_squeeze")(x) # Shape: (batch, 64)
        # Excitation: Two Dense layers to learn channel weights
        se = tf.keras.layers.Dense(16, activation='relu', name="se_excite_dense_1")(se) # Reduction ratio r=4 (64/16)
        se = tf.keras.layers.Dense(64, activation='sigmoid', name="se_excite_dense_2")(se) # Get weights between 0 and 1
        # Reshape se to (batch, 1, 64) to broadcast multiplication across time steps
        se = tf.keras.layers.Reshape((1, 64), name="se_reshape")(se)
        # Scale (Excitation): Multiply original features by learned channel weights
        x = tf.keras.layers.multiply([x, se], name="se_scale") # Shape: (batch, seq_len, 64)

        # BiLSTM layer to process the locally extracted and re-weighted features
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False), name="bilstm_1" # Only need final output
        )(x)

        # Final Dense layer for output prediction
        out = tf.keras.layers.Dense(output_dim, name="output_dense")(x)

        # RENAMED Model Name
        model = tf.keras.Model(inputs=x_in, outputs=out, name="CNNBiLSTMSqueezeExcitation")
        print(f"Built CNNBiLSTMSqueezeExcitationModel.") # RENAMED Print Statement
        model.summary() # Print model summary
        return model

class MLPipeline:
    """Main pipeline for model training and evaluation for GRF Estimation."""
    def __init__(self, config: ModelConfig):
        self.config = config
        # Create output directory and subdirectories for each model
        os.makedirs('model_output', exist_ok=True)
        model_names = ['SimpleDense', 'LSTM', 'BidirectionalLSTM', 'CNNLSTM', 'Transformer',
                       'PhysicsConstrained', 'HybridTransformerPhysics', 'MultiScaleTransformer',
                       'CrossModalAttentionTransformer', 'BiLSTMAttention', 'CNNBiLSTMSqueezeExcitation']
        for model_name in model_names:
            os.makedirs(f'model_output/{model_name}', exist_ok=True)
        self.models = {
            'SimpleDense': SimpleDenseModel(config),
            'LSTM': LSTMModel(config),
            'BidirectionalLSTM': BidirectionalLSTMModel(config),
            'CNNLSTM': CNNLSTMModel(config),
            'Transformer': TransformerModel(config),
            'HybridTransformerPhysics': HybridTransformerPhysics(config),
            'MultiScaleTransformer': MultiScaleTransformer(config),
            'CrossModalAttentionTransformer': CrossModalAttentionTransformer(config),
            'BiLSTMAttention': BiLSTMAttentionModel(config),
            'CNNBiLSTMSqueezeExcitation': CNNBiLSTMSqueezeExcitationModel(config)
        }
        self.results = {}
    
    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Prepare and split data for GRF Estimation. Returns scaled data and MSE scale factor."""
        # Load and prepare sequences for GRF prediction
        X, y = load_and_prepare_sequences(data_path, self.config.sequence_length)

        if X is None or y is None:
            print("Error: Data loading and preparation failed. Cannot proceed.")
            return None, None, None, None, None # Return None tuple including MSE scale
        
        # --- Calculate MSE Scale Factor on raw 'y' before splitting/augmentation ---
        if y.ndim == 2 and y.shape[0] > 1 and y.shape[1] > 0:
            # Calculate variance across samples for each feature, then mean variance
            mse_scale = np.mean(np.var(y, axis=0))
            if mse_scale < 1e-6: # Avoid division by zero or tiny numbers
                print(f"Warning: Calculated MSE scale factor is very small ({mse_scale:.2e}). Setting to 1.0.")
                mse_scale = 1.0
            else:
                print(f"Calculated MSE scale factor (Mean Target Variance): {mse_scale:.4f}")
        else:
            print("Warning: Could not calculate MSE scale factor from y shape {y.shape}. Defaulting to 1.0.")
            mse_scale = 1.0
        # ---------------------------------------------------------------------
            
        # Augment data
        X_aug, y_aug = augment_data(X, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_aug, y_aug,
            test_size=self.config.validation_split,
            random_state=self.config.random_state
        )
        
        # Scale data - Fit scaler only on training data
        # Pass y_train *unscaled* to standardize_data if needed, but we scale y_train here
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Fit scalers ONLY on the training partition
        # Reshape X_train for scaler
        samples_train, timesteps_train, features_train = X_train.shape
        X_train_reshaped = X_train.reshape(-1, features_train)
        scaler_X.fit(X_train_reshaped)
        
        # Fit y scaler on y_train
        scaler_y.fit(y_train) 
        
        # Transform training data
        X_train_scaled_reshaped = scaler_X.transform(X_train_reshaped)
        X_train_scaled = X_train_scaled_reshaped.reshape(samples_train, timesteps_train, features_train)
        y_train_scaled = scaler_y.transform(y_train)
        
        # Transform validation data using the *same* fitted scalers
        samples_val, timesteps_val, features_val = X_val.shape
        X_val_reshaped = X_val.reshape(-1, features_val)
        X_val_scaled_reshaped = scaler_X.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled_reshaped.reshape(samples_val, timesteps_val, features_val)
        y_val_scaled = scaler_y.transform(y_val)

        # Save scalers
        with open('scaler_X.joblib', 'wb') as f: joblib.dump(scaler_X, f)
        with open('scaler_y.joblib', 'wb') as f: joblib.dump(scaler_y, f)
        
        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, mse_scale
    
    def train_and_evaluate(self, data_path: str):
        """Train and evaluate all models for GRF Estimation."""
        # Prepare data
        X_train, y_train, X_val, y_val, mse_scale = self.prepare_data(data_path)
        
        if X_train is None: # Handle potential error from prepare_data
             print("Data preparation failed. Halting training.")
             return

        # Train and evaluate each model
        for name, model_instance in self.models.items():
            print(f"\n--- Training {name} model ---")
            
            try:
                # Build and compile model
                # Input shape: (sequence_length, num_features=12)
                # Output shape: num_force_components=6
                model_instance.model = model_instance.build_model(X_train.shape[1:], y_train.shape[1])

                # --- STAGE 2: Load best weights from Stage 1 --- 
                model_class_name = model_instance.__class__.__name__
                # Correct path assuming models were saved in subdirectories like 'model_output/SimpleDenseModel/'
                best_model_path_stage1 = f'model_output/{model_class_name}/best_model_{model_class_name}' 
                if os.path.exists(best_model_path_stage1):
                    try:
                        print(f"--- STAGE 2: Loading weights from {best_model_path_stage1} for fine-tuning {name} ---")
                        # Use load_weights which is generally safer for fine-tuning the same architecture
                        model_instance.model.load_weights(best_model_path_stage1)
                    except Exception as load_err:
                        print(f"!!! Warning: Failed to load weights from {best_model_path_stage1}: {load_err}")
                        print("Proceeding with initial weights.")
                else:
                    print(f"--- STAGE 2: Warning: Weights file not found at {best_model_path_stage1}. Proceeding with initial weights for {name}. ---")
                # ------------------------------------------------

                # Apply normalized physics loss to all models (except HybridTransformerPhysics which handles it internally)
                if not isinstance(model_instance, HybridTransformerPhysics):
                    # Make sure the model is compiled with basic MSE first if needed
                    if not hasattr(model_instance.model, 'optimizer') or model_instance.model.optimizer is None:
                        print(f"Compiling {name} with default MSE before applying physics loss.")
                        model_instance.model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                            loss='mse',
                            metrics=['mae']
                        )
                    # Apply the normalized physics loss with pre-computed scale factors
                    model_instance.apply_physics_loss(
                        mse_scale=mse_scale # Only pass MSE scale, let others use defaults
                    )
                else:
                     # HybridTransformerPhysics now also needs the mse_scale
                     # We assume it's already built and compiled once in its build_model
                     # We just need to re-apply the loss with the correct mse_scale
                     print(f"Applying physics loss with correct MSE scale to {name}.")
                     model_instance.apply_physics_loss(mse_scale=mse_scale)
                
                print(f"Input shape for training: {X_train.shape}")
                print(f"Target shape for training: {y_train.shape}")
                print(f"Model input shape expected: {model_instance.model.input_shape}")
                print(f"Model output shape expected: {model_instance.model.output_shape}")
            
                # Train model
                history = model_instance.train(X_train, y_train, X_val, y_val)
            
                # Evaluate model (loads best weights internally)
                metrics = model_instance.evaluate(X_val, y_val)
            
                # Store results
                self.results[name] = {
                    'history': history,
                    'metrics': metrics
                }
            
                # Save the final model state
                model_instance.save_model(f'model_output/{name}')

                # Add final validation metrics to results
                self._add_final_val_metrics_to_results(model_instance, name)

            except Exception as e:
                print(f"!!! Error training/evaluating model {name}: {e}")
                import traceback
                traceback.print_exc()
                # Store error information
                self.results[name] = {'error': str(e)}
                # Save scalers even if training failed, might be useful
                if 'scaler_X' in locals() and 'scaler_y' in locals():
                    save_scalers(scaler_X, scaler_y, 'model_output')
        
        # After loop, save all collected results
        self.save_results_to_json()
        
        # Return the final results dictionary
        return self.results
    
    def _add_final_val_metrics_to_results(self, model_instance, model_name):
        """Helper to add final validation metrics from history to results."""
        if hasattr(model_instance, 'history') and model_instance.history:
            history = model_instance.history.history
            if model_name not in self.results:
                self.results[model_name] = {}
            
            # Add final validation values of custom metrics from history
            # Ensure we check if metrics exist in history and get the last value
            for metric_base_name in ['norm_mse', 'norm_vert', 'norm_horiz', 'norm_smooth', 'mae', 'loss']:
                metric_key = f'val_{metric_base_name}'
                if metric_key in history and history[metric_key]:
                    self.results[model_name][f'final_{metric_key}'] = history[metric_key][-1]
                else:
                    # Set to None if metric wasn't found or history is empty
                    self.results[model_name][f'final_{metric_key}'] = None 
                    print(f"Warning: Metric '{metric_key}' not found in history for {model_name}.")
        else:
            print(f"Warning: No history found for model {model_name}, cannot add final validation metrics.")

    def _convert_numpy_to_native(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_native(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            # Handle potential NaN or Inf values before converting to float
            if np.isnan(obj):
                return None # Represent NaN as null in JSON
            elif np.isinf(obj):
                # Represent infinity as a large number or string, depending on needs
                return str(obj) # e.g., "inf" or "-inf"
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            # Convert arrays to lists
            return self._convert_numpy_to_native(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void): # Handle void types if they appear
             return None
        return obj # Return object unchanged if not a numpy type or list/dict

    def save_results_to_json(self, filename: str = None):
        """Saves the collected results to a JSON file."""
        if not self.results:
            print("No results to save.")
            return

        # Generate default filename if none provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_output/model_comparison_results_{timestamp}.json"
            print(f"Generated default filename: {filename}")

        # Create output directory if it doesn't exist
        try:
            output_dir = os.path.dirname(filename) if os.path.dirname(filename) else '.'
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory for {filename}: {e}")
            # Attempt to save in the current directory as a fallback
            filename = os.path.basename(filename)
            print(f"Attempting to save in current directory as {filename}")

        # Convert numpy types in results to native python types for JSON compatibility
        serializable_results = self._convert_numpy_to_native(self.results)

        try:
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Successfully saved results to {filename}")
        except TypeError as e:
            print(f"Error saving results to JSON due to TypeError: {e}")
            print("There might be non-serializable types remaining in the results.")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

# --- Helper Functions ---

def load_and_prepare_sequences(file_path, sequence_length=10):
    """
    Load and prepare data specifically for GRF Estimation.
    Assumes input CSV contains IMU data and ground force targets.

    Uses a sliding window approach with stride=1 to extract sequences.
    Filters sequences based on a vertical force threshold to focus on stance phases.
    Input features: 12 channels (acc_x/y/z, gyro_x/y/z for 2 IMUs)
    Output targets: 6 GRF components (vx, vy, vz for left/right foot)
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
    imu_features = [
        f'imu{i}_acc_{axis}' for i in range(2) for axis in ['x', 'y', 'z']
    ] + [
        f'imu{i}_gyro_{axis}' for i in range(2) for axis in ['x', 'y', 'z']
    ]
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

    # Handle potential non-numeric data before statistics/interpolation
    for col in imu_features + force_targets:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for NaNs introduced by coercion
    nan_check = df[imu_features + force_targets].isna().sum()
    if nan_check.sum() > 0:
        print(f"Warning: NaNs detected after converting to numeric (possibly due to non-numeric entries):\n{nan_check[nan_check > 0]}")
        print("Attempting linear interpolation...")
        df[imu_features] = df[imu_features].interpolate(method='linear', limit_direction='both')
        df[force_targets] = df[force_targets].interpolate(method='linear', limit_direction='both')
        if df[imu_features + force_targets].isna().sum().sum() > 0:
            print("Error: NaNs remain after handling. Cannot proceed.")
            return None, None

    # Print force target statistics (after potential NaN handling)
    try:
        print("\nForce Targets Statistics (after potential NaN handling):")
        print(df[force_targets].describe())
    except Exception as e:
        print(f"Could not compute statistics: {e}")

    # Compute force threshold for filtering sequences (focus on stance phase)
    try:
        mean_left_vy_abs = df[force_targets[1]].abs().mean()
        mean_right_vy_abs = df[force_targets[4]].abs().mean()
        mean_vertical_abs = (mean_left_vy_abs + mean_right_vy_abs) / 2.0
        force_threshold = 0.1 * mean_vertical_abs if mean_vertical_abs > 1e-6 else 1.0
        print(f"Using force threshold (based on 10% mean abs vertical): {force_threshold:.2f} N")
    except Exception as e:
        print(f"Error calculating force threshold: {e}. Using default threshold 1.0 N")
        force_threshold = 1.0
        
    # Create sequences with overlap
    sequences = []
    targets = []
    stride = 1
    
    # Track sequence count
    total_potential_sequences = len(df) - sequence_length + 1 if len(df) >= sequence_length else 0
    filtered_sequences = 0
    nan_skipped_sequences = 0
    
    for i in range(0, len(df) - sequence_length, stride):
        seq = df[imu_features].iloc[i:i+sequence_length].values
        target = df[force_targets].iloc[i+sequence_length-1].values

        # Check for NaN in current sequence/target before filtering
        if np.isnan(seq).any() or np.isnan(target).any():
            nan_skipped_sequences += 1
            continue

        # Filter based on total vertical force (stance phase detection)
        total_vy_abs = np.abs(target[1]) + np.abs(target[4])
        if total_vy_abs > force_threshold:
            sequences.append(seq)
            targets.append(target)
            filtered_sequences += 1
    
    if not sequences:
        print("Error: No valid sequences generated after filtering. Check data quality or force threshold.")
        return None, None

    print(f"Total potential sequences: {total_potential_sequences}")
    print(f"Sequences skipped due to NaN: {nan_skipped_sequences}")
    print(f"Sequences after filtering (stance phase): {filtered_sequences}")
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Processed dataset shape - X: {X.shape}, y: {y.shape}")
    if X.size == 0 or y.size == 0:
        print("Error: Resulting X or y array is empty after processing.")
        return None, None

    return X, y


# --- Augmentation Function ---
def augment_data(X, y, noise_reps=3, warp_reps=3, mask_reps=2, scale_reps=2, 
              permute_reps=2, mix_reps=2, noise_sigma=0.05):
    """
    Perform enhanced data augmentation on time-series sequences.
    The force targets (y) are kept unchanged for all augmentation methods.
    Returns augmented_X and augmented_y.
    """
    augmented_X = []
    augmented_y = []

    if len(X) == 0:
        print("Warning: Input X to augment_data is empty. Returning empty arrays.")
        return np.array(augmented_X), np.array(augmented_y)
    
    for i in range(len(X)):
        original_seq = X[i]
        original_target = y[i]
        
        # Always keep the original sequence
        augmented_X.append(original_seq)
        augmented_y.append(original_target)
        
        channel_stds = np.std(original_seq, axis=0)
        
        # 1. Gaussian noise
        for _ in range(noise_reps):
            adaptive_noise = np.random.normal(
                loc=0.0, 
                scale=noise_sigma * channel_stds * (0.5 + np.random.random()),
                size=original_seq.shape
            )
            noisy_seq = original_seq + adaptive_noise
            augmented_X.append(noisy_seq)
            augmented_y.append(original_target)
        
        # 2. Time warping
        L = original_seq.shape[0]
        orig_steps = np.arange(L)
        for _ in range(warp_reps):
            warp_sigma = 0.3 * np.random.random() + 0.1
            new_steps = np.linspace(0, L-1, L) + np.random.uniform(-warp_sigma, warp_sigma, size=L)
            new_steps[0] = 0
            new_steps[-1] = L-1
            new_steps = np.sort(new_steps)
            
            warped_seq = np.zeros_like(original_seq)
            for j in range(original_seq.shape[1]):
                if np.all(original_seq[:, j] == original_seq[0, j]):
                    warped_seq[:, j] = original_seq[0, j]
                else:
                    try:
                        f = interp1d(orig_steps, original_seq[:, j], kind='cubic', fill_value="extrapolate")
                        warped_seq[:, j] = f(new_steps)
                    except ValueError as e:
                        print(f"Interpolation failed for channel {j}, sequence {i}: {e}. Using linear instead.")
                        f = interp1d(orig_steps, original_seq[:, j], kind='linear', fill_value="extrapolate")
                        warped_seq[:, j] = f(new_steps)
            augmented_X.append(warped_seq)
            augmented_y.append(original_target)
        
        # 3. Channel masking
        for _ in range(mask_reps):
            masked_seq = original_seq.copy()
            num_channels_to_mask = np.random.randint(1, min(4, original_seq.shape[1] + 1))
            channels_to_mask = np.random.choice(
                original_seq.shape[1], 
                size=num_channels_to_mask, 
                replace=False
            )
            
            if L > 1:
                mask_length = np.random.randint(max(1, L // 5), max(2, L // 2))
                mask_start = np.random.randint(0, max(1, L - mask_length))
            else:
                mask_length = 0
                mask_start = 0
            
            if mask_length > 0:
                for channel in channels_to_mask:
                    if np.random.random() < 0.5:
                        masked_seq[mask_start:mask_start+mask_length, channel] = 0
                    else:
                        masked_seq[mask_start:mask_start+mask_length, channel] = np.mean(
                            original_seq[:, channel]
                        )
            augmented_X.append(masked_seq)
            augmented_y.append(original_target)
        
        # 4. Magnitude scaling
        for _ in range(scale_reps):
            scaled_seq = original_seq.copy()
            for j in range(original_seq.shape[1]):
                scale_factor = 0.7 + 0.6 * np.random.random()
                scaled_seq[:, j] = original_seq[:, j] * scale_factor
            augmented_X.append(scaled_seq)
            augmented_y.append(original_target)
        
        # 5. Permutation of segments
        for _ in range(permute_reps):
            permuted_seq = original_seq.copy()
            num_segments = np.random.randint(2, 5)
            min_segment_size = max(1, L // 10)
            
            if L >= num_segments * min_segment_size and L > 1:
                try:
                    split_indices = np.sort(np.random.choice(
                        np.arange(min_segment_size, L - min_segment_size + 1),
                        size=num_segments-1, 
                        replace=False
                    ))
                    segments = np.split(permuted_seq, split_indices, axis=0)
                    np.random.shuffle(segments)
                    permuted_seq = np.concatenate(segments, axis=0)
                except ValueError as e:
                    print(f"Warning: Segment permutation failed for sequence {i}: {e}. Using original.")
                    permuted_seq = original_seq.copy()
            augmented_X.append(permuted_seq)
            augmented_y.append(original_target)
        
        # 6. Signal mixing
        if len(X) > 1:
            for _ in range(mix_reps):
                other_idx = np.random.choice([j for j in range(len(X)) if j != i])
                other_seq = X[other_idx]
                alpha = 0.7 + 0.3 * np.random.random()
                mixed_seq = alpha * original_seq + (1 - alpha) * other_seq
                augmented_X.append(mixed_seq)
                augmented_y.append(original_target)
    
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    
    if len(X) > 0 and len(augmented_X) > 0:
        print(f"Original dataset: {X.shape}, Augmented dataset: {augmented_X.shape}")
        print(f"Augmentation factor: {len(augmented_X) / len(X):.2f}x")
    elif len(X) == 0:
        print("Augmentation skipped: Input X was empty.")
    else:
        print("Warning: Augmentation resulted in an empty dataset.")
    
    return augmented_X, augmented_y

# --- Standardization Function ---
def standardize_data(X, y, fit=True, scaler_X=None, scaler_y=None):
    """
    Feature-wise standardization for IMU signals (X) and force outputs (y).
    If fit=True, fits new scalers.
    If fit=False, uses provided scalers to transform data.
    """
    if X is None or y is None or X.size == 0 or y.size == 0:
        print("Error in standardize_data: Input X or y is None or empty.")
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
            # Potentially add more debugging here, e.g., check shapes, n_features_in_
            print(f"Scaler X expected features: {getattr(scaler_X, 'n_features_in_', 'N/A')}, Input features: {X_reshaped.shape[1]}")
            print(f"Scaler Y expected features: {getattr(scaler_y, 'n_features_in_', 'N/A')}, Input features: {y.shape[1]}")
            return None, None, scaler_X, scaler_y # Return None for data, pass scalers back

    X_scaled = X_scaled_reshaped.reshape(samples, timesteps, features)

    return X_scaled, y_scaled, scaler_X, scaler_y


# --- Evaluation Function ---
def evaluate_predictions(y_true, y_pred, participant_weight_kg=70):
    """Evaluate GRF predictions with relevant metrics."""

    if y_true is None or y_pred is None or y_true.size == 0 or y_pred.size == 0:
        print("Error: Cannot evaluate predictions with empty true or predicted values.")
        return {"Error": "Input arrays are empty or None."}
    if y_true.shape != y_pred.shape:
        print(f"Error: Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}.")
        return {"Error": "Shape mismatch between true and predicted values."}

    # Ensure input is for force prediction (6 components)
    if y_true.ndim < 2 or y_true.shape[1] != 6:
        print(f"Error: evaluate_predictions expected 6 target components (GRF), but got y_true shape: {y_true.shape}")
        # Return a dictionary indicating error, or raise ValueError
        return {"Error": f"Invalid shape for y_true: {y_true.shape}. Expected (samples, 6)."}
    if y_pred.ndim < 2 or y_pred.shape[1] != 6:
        print(f"Error: evaluate_predictions expected 6 target components (GRF), but got y_pred shape: {y_pred.shape}")
        return {"Error": f"Invalid shape for y_pred: {y_pred.shape}. Expected (samples, 6)."}

    # Calculate body weight force
    body_weight_force = participant_weight_kg * 9.81
    
    # Component names for better reporting (matching problem def: x, y, z)
    components = [
        'Left Foot X', 'Left Foot Y', 'Left Foot Z',
        'Right Foot X', 'Right Foot Y', 'Right Foot Z'
    ]
    
    # Calculate metrics for each component
    component_metrics = {}
    for i, name in enumerate(components):
        # Check for NaNs in the specific component column before calculation
        if np.isnan(y_true[:, i]).any() or np.isnan(y_pred[:, i]).any():
            print(f"Warning: NaN values found in component '{name}'. Metrics for this component might be inaccurate or NaN.")
            mae = np.nan
            rmse = np.nan
            mre = np.nan
        else:
            mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
            rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2))
            
            # Calculate relative error (as percentage)
            mask = np.abs(y_true[:, i]) > 1.0 # Threshold to avoid division by near-zero
            if np.sum(mask) > 0:
                # Ensure no NaNs in the masked arrays
                true_masked = y_true[mask, i]
                pred_masked = y_pred[mask, i]
                if not np.isnan(true_masked).any() and not np.isnan(pred_masked).any():
                    mre = np.mean(np.abs(true_masked - pred_masked) / np.abs(true_masked)) * 100
                else:
                    print(f"Warning: NaN found in masked arrays for MRE calculation of {name}.")
                    mre = np.nan
            else:
                mre = np.nan # Not enough data points above threshold
        
        component_metrics[name] = {
            'MAE (N)': float(mae) if not np.isnan(mae) else None,
            'RMSE (N)': float(rmse) if not np.isnan(rmse) else None,
            'MRE (%)': float(mre) if not np.isnan(mre) else None
        }
    
    # Calculate vertical force constraint metrics (Y component is index 1 and 4)
    # Check for NaNs before calculating totals
    if np.isnan(y_true[:, 1]).any() or np.isnan(y_true[:, 4]).any() or \
       np.isnan(y_pred[:, 1]).any() or np.isnan(y_pred[:, 4]).any():
        print("Warning: NaN values found in vertical force components. Physics metrics might be inaccurate or NaN.")
        total_vertical_true = np.nan
        total_vertical_pred = np.nan
        vertical_force_error_n = np.nan
        vertical_force_error_pct = np.nan
    else:
        total_vertical_true = np.mean(y_true[:, 1] + y_true[:, 4])
        total_vertical_pred = np.mean(y_pred[:, 1] + y_pred[:, 4])
        vertical_force_error_n = np.abs(total_vertical_pred - body_weight_force)
        vertical_force_error_pct = (vertical_force_error_n / body_weight_force * 100) if body_weight_force > 1e-6 else np.nan
    
    vertical_metrics = {
        'True Total Vertical Force (N)': float(total_vertical_true) if not np.isnan(total_vertical_true) else None,
        'Predicted Total Vertical Force (N)': float(total_vertical_pred) if not np.isnan(total_vertical_pred) else None,
        'Body Weight Force (N)': float(body_weight_force),
        'Vertical Force Error (N)': float(vertical_force_error_n) if not np.isnan(vertical_force_error_n) else None,
        'Vertical Force Error (%)': float(vertical_force_error_pct) if not np.isnan(vertical_force_error_pct) else None
    }
    
    # Overall metrics
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Warning: NaN values found in overall arrays. Overall metrics might be inaccurate or NaN.")
        overall_mae = np.nanmean(np.abs(y_true - y_pred)) # Use nanmean
        overall_rmse = np.sqrt(np.nanmean((y_true - y_pred)**2)) # Use nanmean
    else:
        overall_mae = np.mean(np.abs(y_true - y_pred))
        overall_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    metrics = {
        'Overall': {
            'MAE (N)': float(overall_mae) if not np.isnan(overall_mae) else None,
            'RMSE (N)': float(overall_rmse) if not np.isnan(overall_rmse) else None
        },
        'Components': component_metrics,
        'Physics': vertical_metrics
    }
    
    return metrics


# --- Main Execution ---
def main():
    """
    Main function to execute the ML pipeline exclusively for GRF Estimation.
    """
    # Configuration for GRF Estimation
    config = ModelConfig(
        sequence_length=10, 
        epochs=50, # Consider reducing epochs for faster testing initially
        batch_size=32, # Slightly larger batch size might be okay
        learning_rate=0.001,
        participant_weight_kg=70, # Example weight - should ideally come from data or config
        validation_split=0.2,
        physics_weight=0.1,
        warmup_epochs=5
        # target_type removed
    )
    
    # Create timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Experiment name clearly indicates it's for force prediction
    experiment_name = f"run_GRF_{timestamp}"
    
    print(f"--- Starting GRF Estimation Experiment: {experiment_name} ---")
    
    # Create pipeline
    pipeline = MLPipeline(config)
    
    # --- IMPORTANT: Update this path to your actual GRF dataset ---
    # This path should contain 'ground_force_left_x', 'ground_force_left_y', etc. columns
    # AND the 12 required IMU columns (imu0_acc_..., imu0_gyro_..., imu1_acc_..., imu1_gyro_...)
    data_path = '/Users/wangxiang/Desktop/my_workspace/mobisense/software-group/data-working/assets/mar12exp/synced_IMU_forces_grf_fixed.csv'
    # Example placeholder for testing (replace with a real file path):
    # data_path = 'software-group/data-working/assets/placeholder_grf_data.csv'
    # --------------------------------------------------------------

    print(f"Using data path: {data_path}")
    if not os.path.exists(data_path):
        print(f"FATAL ERROR: Data file not found at {data_path}.")
        print("Please update the 'data_path' variable in the main() function of ml_pipeline.py.")
        return

    # Train and evaluate models using the selected dataset
    try:
        pipeline.train_and_evaluate(data_path)
    except ValueError as e:
        print(f"\nERROR during pipeline execution: {e}")
        print("Pipeline halted. Please check data format, paths, and ensure data contains valid numeric values.")
        return
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        print("Pipeline halted.")
        return

    # Save the results to a JSON file named after the experiment
    results_filename = os.path.join('model_output', f'model_comparison_results_{experiment_name}.json')
    pipeline.save_results_to_json(filename=results_filename)

    print(f"\n--- Experiment {experiment_name} completed successfully. ---")
    print(f"Results saved to {results_filename}")
    print(f"To visualize these results, run: python visualize_results.py {results_filename}")

if __name__ == "__main__":
    main() 