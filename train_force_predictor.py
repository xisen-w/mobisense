import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_and_prepare_sequences(file_path, sequence_length=10):
    """Load data and prepare sequences with sliding window"""
    # Load the data
    df = pd.read_csv(file_path)
    
    # Separate features and targets
    imu_features = []
    for i in range(2):  # IMU0 and IMU1
        for sensor in ['acc', 'gyro']:
            for axis in ['x', 'y', 'z']:
                imu_features.append(f'imu{i}_{sensor}_{axis}')
    
    force_targets = []
    for side in ['left', 'right']:
        for component in ['vx', 'vy', 'vz']:
            force_targets.append(f'ground_force_{side}_{component}')
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(df) - sequence_length):
        seq = df[imu_features].values[i:i+sequence_length]
        target = df[force_targets].values[i+sequence_length-1]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def create_simple_model(sequence_length, n_features, n_outputs):
    """Create a simpler sequence model"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(sequence_length, n_features)),
        
        # First LSTM layer
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Second LSTM layer
        tf.keras.layers.LSTM(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Dense layers for prediction
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(n_outputs)
    ])
    return model

def main():
    # Parameters
    SEQUENCE_LENGTH = 10  # Look at past 10 timesteps
    EPOCHS = 100
    BATCH_SIZE = 32  # Increased batch size
    LEARNING_RATE = 0.001
    
    # Load and prepare sequence data
    print("Loading and preparing data...")
    X, y = load_and_prepare_sequences(
        'software-group/data-working/assets/feb10exp/synced_IMU_forces_grf_fixed.csv',
        SEQUENCE_LENGTH
    )
    
    print(f"Dataset shape - X: {X.shape}, y: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit and transform
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_test_scaled = scaler_X.transform(X_test_reshaped)
    
    # Reshape back to sequences
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Scale targets
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Create and compile model
    print("Creating model...")
    model = create_simple_model(
        sequence_length=SEQUENCE_LENGTH,
        n_features=X_train.shape[-1],
        n_outputs=y_train.shape[-1]
    )
    
    # Model summary
    model.summary()
    
    # Compile model with simpler setup
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model with early stopping
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Save model
    model.save('force_predictor_model')
    
    # Make some predictions and compare
    print("\nMaking sample predictions...")
    y_pred = model.predict(X_test_scaled[:5], verbose=1)
    
    # Inverse transform predictions and actual values
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test_scaled[:5])
    
    # Print comparison
    print("\nSample Predictions vs Actual Values:")
    for i in range(5):
        print(f"\nSample {i+1}:")
        print(f"Predicted: {y_pred_original[i]}")
        print(f"Actual: {y_test_original[i]}")
        print(f"Mean Absolute Error: {np.mean(np.abs(y_pred_original[i] - y_test_original[i])):.4f}")

if __name__ == "__main__":
    main() 