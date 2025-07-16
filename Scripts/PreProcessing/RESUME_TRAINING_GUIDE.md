# Resume Training Feature Guide

## Overview

The Sappo Trading Bot now supports resuming training from a previously saved model. This feature allows you to continue training in multiple sessions without losing progress, which is particularly useful for long-running experiments.

## Features

### GUI Changes
- **Select Model Button**: Allows you to browse and select a `.zip` model file from the `models` directory
- **Resume Training Button**: Starts training from the selected model (only enabled when a model is selected)
- **Start New Training Button**: Renamed from "Start Training" for clarity, creates a new model from scratch

### Backend Changes
- **Continuous Training**: The training continues from where it left off, preserving the timestep counter and logs
- **TensorBoard Integration**: Logs continue seamlessly in TensorBoard without resetting
- **Model Naming**: Resumed models are saved with "resumed_" prefix for easy identification

## How to Use

### Via GUI (main_integrated.py)

1. **Start the GUI**:
   ```bash
   python main_integrated.py
   ```

2. **Select your preprocessed data** (`.npy` file)

3. **For New Training**:
   - Configure hyperparameters as needed
   - Click "Start New Training"

4. **For Resume Training**:
   - Click "Select Model" and choose a previously saved `.zip` model file
   - Configure hyperparameters (should match original training)
   - Click "Resume Training"

### Via Code

```python
from train import train_sappo_agent

# Initial training
initial_results = train_sappo_agent(
    data_path="path/to/data.npy",
    hyperparameters={'learning_rate': 0.0001, 'gamma': 0.99},
    reward_weights={'profit': 1.0, 'sharpe': 0.5, 'cost': 1.0, 'mdd': 0.5},
    total_timesteps=100000,
    model_save_dir="models"
)

# Resume training
resume_results = train_sappo_agent(
    data_path="path/to/data.npy",
    hyperparameters={'learning_rate': 0.0001, 'gamma': 0.99},
    reward_weights={'profit': 1.0, 'sharpe': 0.5, 'cost': 1.0, 'mdd': 0.5},
    total_timesteps=50000,  # Additional timesteps
    model_save_dir="models",
    resume_from_model="models/best_model_20250716_123456.zip"
)
```

## File Structure

When you resume training, the following files are created:

```
models/
├── best_model_20250716_123456.zip          # Original best model
├── final_model_20250716_123456.zip         # Original final model
├── resumed_best_model_20250716_143210.zip  # Resumed best model
└── resumed_final_model_20250716_143210.zip # Resumed final model
```

## Training Log Structure

The training results JSON now includes resume information:

```json
{
  "success": true,
  "training_summary": {
    "total_timesteps": 50000,
    "is_resumed_training": true,
    "resumed_from_model": "models/best_model_20250716_123456.zip"
  },
  "best_model_path": "models/resumed_best_model_20250716_143210.zip",
  "final_model_path": "models/resumed_final_model_20250716_143210.zip"
}
```

## Important Notes

### Hyperparameters
- **Consistency**: Use the same hyperparameters when resuming training for best results
- **Learning Rate**: You can adjust the learning rate when resuming (e.g., use a lower rate for fine-tuning)

### Timesteps
- The `total_timesteps` parameter in resume training represents **additional** timesteps, not total cumulative timesteps
- The internal timestep counter continues from where it left off

### TensorBoard Logs
- Logs continue seamlessly in TensorBoard under the same run name
- No reset of the timestep counter means graphs show continuous progress
- To view: `tensorboard --logdir=tensorboard_logs`

### Model Compatibility
- Only use models trained with the same environment and data structure
- The feature automatically handles environment setup and model loading

## Troubleshooting

### Common Issues

1. **"Resume Training" button is disabled**:
   - Make sure you've selected both a data file and a model file
   - Check that the model file exists and is a valid `.zip` file

2. **Training fails to resume**:
   - Verify the model file is not corrupted
   - Ensure the model was trained with compatible hyperparameters
   - Check that the data file structure matches the original training

3. **TensorBoard shows discontinuous logs**:
   - This might happen if you change the `tb_log_name` parameter
   - Make sure to use the same tensorboard log directory

### Testing

Run the test script to verify the resume functionality:

```bash
python test_resume_training.py
```

This will create sample data, train a model, resume training, and verify that everything works correctly.

## Benefits

1. **Interrupted Training Recovery**: Continue training after system crashes or interruptions
2. **Incremental Training**: Train models in smaller chunks over multiple sessions
3. **Hyperparameter Tuning**: Fine-tune models with different learning rates
4. **Experimentation**: Try different training durations to find optimal stopping points
5. **Resource Management**: Split long training sessions to manage computational resources

## Technical Details

### Implementation
- Uses `stable_baselines3.PPO.load()` to load the saved model
- Sets `reset_num_timesteps=False` to preserve the timestep counter
- Automatically configures the environment for the loaded model
- Preserves all training state including optimizers and replay buffers

### Validation
- The validation callback continues to evaluate the model during resumed training
- Previous evaluation history is not carried over (starts fresh)
- Best model tracking continues from the resumed state