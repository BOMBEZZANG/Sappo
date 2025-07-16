# Sappo Trading Bot - Data Preprocessing Tool

A GUI-based preprocessing tool for preparing multi-asset, multi-timeframe cryptocurrency data for Reinforcement Learning training.

## Features

- **Simple GUI**: Clean and intuitive interface with Tkinter
- **Multi-file Support**: Load multiple CSV files with different timeframes
- **Real-time Progress**: Visual progress bar and status logging
- **Data Pipeline**: Complete preprocessing including:
  - Timeframe unification to 1-hour base
  - Feature engineering (percentage changes)
  - Sliding window creation
  - Z-score normalization
- **Output**: Saves preprocessed data as NumPy .npy files

## Requirements

```bash
pip install pandas numpy scikit-learn
```

Note: tkinter is included with most Python installations.

## File Format

Input CSV files should follow the format: `upbit_{coin_name}_{timeframe}.csv`

Supported timeframes:
- `hour1` - 1-hour data
- `hour4` - 4-hour data  
- `day` - Daily data
- `week` - Weekly data

Required columns: `timestamp` (or `date`), `open`, `high`, `low`, `close`, `volume`

## Usage

### Running the GUI Application

```bash
python main.py
```

### Steps:
1. Click "Select Data Files" to choose CSV files
2. Click "Start Preprocessing" to begin the pipeline
3. Monitor progress in the status window
4. Preprocessed data will be saved as `preprocessed_data_YYYYMMDD_HHMMSS.npy`

### Testing

```bash
python test_app.py
```

This will create sample data and test the preprocessing pipeline.

## Project Structure

```
├── main.py              # Main GUI application
├── preprocessing.py     # Core preprocessing logic
├── test_app.py         # Test script
├── requirements.txt    # Python dependencies
└── Data/
    └── upbit_data/     # Your actual data files
```

## Development Phases

✅ **Phase 1**: GUI scaffolding with layout  
✅ **Phase 2**: Data loading with file selection  
✅ **Phase 3**: Core preprocessing pipeline  
✅ **Phase 4**: GUI integration with real-time feedback  
✅ **Phase 5**: Data saving and testing  

## Data Processing Pipeline

1. **Data Validation**: Checks for required columns
2. **Timeframe Unification**: Resamples all data to 1-hour frequency
3. **Feature Engineering**: Converts prices to percentage changes
4. **Sliding Windows**: Creates overlapping windows for RL training
5. **Normalization**: Applies Z-score normalization to each window
6. **Output**: Saves as 3D NumPy array (samples, window_size, features)

## Configuration

Default window size: 24 hours (can be modified in `DataPreprocessor` initialization)

## Error Handling

- Input validation for file formats
- Progress tracking with error recovery
- Detailed logging of all operations
- GUI remains responsive during processing

## Next Steps

After preprocessing, the output `.npy` files can be loaded directly into your RL training environment:

```python
import numpy as np
data = np.load('preprocessed_data_YYYYMMDD_HHMMSS.npy')
print(f"Data shape: {data.shape}")  # (samples, window_size, features)
```