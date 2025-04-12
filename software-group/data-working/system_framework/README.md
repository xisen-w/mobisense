# MobiSense IMU Experiment System

This system provides a user interface for setting up and managing IMU experiments within the MobiSense framework.

## Overview

The system allows researchers to:

1. Upload IMU data files (CSV format)
2. Register experiments with participant information
3. Visualize IMU data and dorsiflexion angle measurements
4. Manage multiple experiments in a centralized interface

The pipeline supports two primary prediction targets:

1. **Dorsiflexion Angle**: Angular measurement indicating ankle flexion, critical for understanding gait dynamics and human movement.
2. **Ground Reaction Forces**: Multi-component force vectors (6 total: left/right feet, x/y/z directions) representing the forces between the foot and ground during movement.

- **Data Upload:** Upload CSV files containing IMU sensor data
- **Participant Registration:** Record participant details (height, weight, age, etc.)
- **Optional Parameters:** Track stride length, steps per minute, injury date, and injury type
- **Experiment Management:** Create and organize experiments
- **Data Validation:** Automatically validate IMU data structure 
- **Data Visualization:** Plot acceleration, gyroscope, and angle data with interactive controls
- **Angle Data Support:** Automatically detect and visualize dorsiflexion angle data if available

## Dataset Format

The system works with two types of datasets:

1. **Dorsiflexion Angle Format**:
   - Features: Accelerometer and gyroscope readings from two IMUs (12 channels)
   - Target: Single-value dorsiflexion angle measurements
   - Example file: `mar12exp_updated/2025-03-12_10-25-01-r4-walking3.csv`

2. **Ground Force Format**:
   - Features: Same IMU data (12 channels)
   - Target: Six-component ground reaction force measurements
   - Includes force filtering based on vertical component threshold

## Data Augmentation Methods

```bash
cd software-group/data-working/system_framework
./run_app.sh
```

### Data Upload & Experiment Setup

1. Navigate to the "Data Upload & Experiment Setup" tab
2. Fill in the experiment information (name, description)
3. Enter required participant details (ID, height, weight, age, gender)
4. Optionally enter gait parameters (stride length, steps per minute)
5. Optionally enter injury information (date and type)
6. Upload an IMU data CSV file
7. Click "Register Experiment"
8. The system will validate the data structure and confirm registration

1. **Adaptive Gaussian Noise Injection**
   - **Innovation**: Instead of fixed noise levels, we implement adaptive noise scaling based on channel-specific standard deviations
   - Implementation adaptively computes appropriate noise magnitude for each channel
   - Noise is scaled by a random factor between 0.5-1.5 of the base noise parameter
   - Significantly improves model robustness to varying sensor noise levels



1. Navigate to the "View Experiments" tab
2. All registered experiments will be displayed as expandable sections
3. Click on an experiment to view its details
4. Use the "Load Data Preview" button to see the first few rows of data

### Data Visualization

3. **Channel Masking/Dropout**
   - **Innovation**: Simulates temporary sensor failures or occlusions in real-world scenarios
   - Randomly selects 1-3 sensor channels to mask
   - Creates masks of variable length (20-50% of sequence)
   - Uses two masking strategies: zero-filling and mean-value replacement
   - Teaches models to maintain prediction quality with incomplete sensor data

4. **Magnitude Scaling**
   - **Innovation**: Simulates variable sensor sensitivity between device units
   - Applies random scaling factors (0.7-1.3) independently to each channel
   - Preserves temporal patterns while modifying signal intensity
   - Increases model robustness to calibration differences and sensor drift

5. **Segment Permutation**
   - **Innovation**: Teaches model to be robust to local temporal misalignments
   - Divides sequences into 2-5 segments of varying length
   - Swaps adjacent segments to create realistic temporal distortions
   - Maintains overall sequence context while altering local dynamics
   - Particularly effective for handling phase shifts in cyclic movements

6. **Signal Mixing**
   - **Innovation**: Creates synthetic examples by weighted combination of sequences
   - Randomly pairs sequences with controlled mixing ratios (primary sequence weighted 70-100%)
   - Generates diverse training samples while preserving physiological plausibility
   - Particularly effective for addressing data scarcity in specific movement patterns

The system expects IMU data in CSV format with the following columns:

The pipeline implements multiple architectures to compare performance:

**Optional columns:**
- `imu1_timestamp`: Timestamp for IMU 1
- `imu1_acc_x`, `imu1_acc_y`, `imu1_acc_z`: Acceleration data for IMU 1
- `imu1_gyro_x`, `imu1_gyro_y`, `imu1_gyro_z`: Gyroscope data for IMU 1 
- `dorsiflexion_angle`: Angle measurement (if available)

1. **SimpleDense**: Feed-forward neural network baseline
   - Flattens input sequence and processes through dense layers
   - Architecture: Input → Flatten → Dense(64) → Dropout(0.2) → Dense(32) → Output

2. **LSTM**: Standard recurrent architecture for sequential data
   - Architecture: Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(32) → Output
   - Effectively captures temporal dependencies and sequence patterns

3. **BidirectionalLSTM**: Enhanced recurrent architecture
   - Processes sequence in both forward and backward directions
   - Captures richer contextual information compared to unidirectional LSTM

### Advanced Architectures

4. **CNNLSTMModel**: Hybrid architecture with parallel processing
   - CNN branch extracts local patterns: Conv1D(64) → MaxPool → Conv1D(32) → GlobalAvgPool
   - LSTM branch captures temporal dependencies: LSTM(64) → LSTM(32)
   - Branches are concatenated and processed through dense layers

5. **TransformerModel**: Attention-based architecture
   - Employs multi-head self-attention mechanism (4 heads)
   - Includes layer normalization and residual connections
   - GlobalAveragePooling for sequence summarization

6. **PhysicsConstrainedModel**: Physics-informed architecture
   - **Innovation**: Incorporates domain knowledge through custom loss functions
   - For dorsiflexion angle: Penalizes predictions outside physiological range (-15° to +20°)
   - For ground forces: Enforces body weight constraint on vertical forces
   - Uses residual blocks for improved gradient flow

7. **HybridTransformerPhysics**: Advanced physics-aware architecture
   - **Innovation**: Combines transformer architecture with physics-informed processing
   - Custom physics-aware attention mechanism
   - Multi-layer attention with increased capacity (128 dimensions)
   - Incorporates custom learning rate schedule with warmup

8. **MultiScaleTransformer**: Multi-resolution temporal processing
   - **Innovation**: Parallel attention mechanisms at different temporal scales
   - Fast scale (local patterns): 5-frame windows
   - Medium scale (stride patterns): 10-frame windows
   - Slow scale (overall movement): 20-frame windows

9. **CrossModalAttentionTransformer**: Cross-modality feature integration
   - **Innovation**: Uncertainty-aware cross-attention between sensor modalities
   - Separates accelerometer and gyroscope streams for specialized processing
   - Uses Monte Carlo dropout for uncertainty estimation
   - Weights features by estimated confidence

## Physics-Informed Constraints

### For Dorsiflexion Angle Prediction

- **Physiological Range Constraint**: Penalizes predictions outside normal range (-15° to +20°)
- **Smoothness Constraint**: Enforces temporal continuity in angle predictions
- Loss weighting: 0.1 for range violations, 0.05 for smoothness violations

### For Ground Reaction Force Prediction

- **Body Weight Constraint**: Total vertical force should approximate body weight
- **Horizontal Balance**: Sum of horizontal forces should approach zero
- **Temporal Consistency**: Forces should vary smoothly over time

## Evaluation Framework

### Metrics

For dorsiflexion angle:
- Mean Absolute Error (MAE) in degrees
- Root Mean Squared Error (RMSE) in degrees
- Correlation coefficient
- Mean Relative Error (MRE) as percentage
- Range coverage analysis

For ground reaction forces:
- Component-wise MAE and RMSE (in Newtons)
- Physics-based metrics (body weight approximation)
- Overall prediction accuracy across all components

### Visualization

The pipeline generates comprehensive visualizations:
- Validation loss curves for all models
- MAE comparison bar charts
- Detailed training curves for each model
- Physics constraint satisfaction metrics

## Implementation Optimizations

### Computational Efficiency

- Vectorized operations where possible
- In-place modifications for memory efficiency
- Pre-computation of channel statistics
- Efficient sequence generation with stride=1

### Robustness Enhancements

- Handling of NaN values via interpolation
- Edge case management in evaluation metrics
- Boundary checking in augmentation operations
- Learning rate scheduling with early stopping

## Future Directions

1. **Personalization**: Adapting models to individual users via transfer learning
2. **Real-time Inference**: Optimizing models for embedded deployment
3. **Multi-task Learning**: Jointly predicting multiple biomechanical variables
4. **Explainability**: Developing attention visualization tools for model interpretation

## References

1. Cui, C., Yao, K., Huang, J., Wang, Y., Suarez-Tangil, G., & Wang, W. (2022). "Sensor-based human activity recognition with deep temporal-spatial features". IEEE Internet of Things Journal.

- `streamlit_app.py`: Main Streamlit application
- `experiments.py`: Classes for experiment and participant management
- `run_app.sh`: Shell script to run the application
- `experiments/`: Directory for storing experiment data and metadata

3. Rasul, K., & Louis, O. (2021). "Time series data augmentation for neural networks". arXiv preprint arXiv:2012.14170.

4. Zhao, L., Zhou, X., Fan, M., Jiang, Z., Chen, Y., & Ma, W. (2021). "Continuous and orientation-preserving correspondence via functional maps". ACM Transactions on Graphics. 