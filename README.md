# Depth Sensor Adjustment Tool

A PySide6 application for making adjustments to depth sensors used in piston coring operations.

## Features

- **Load Sensor Data**: Import depth sensor data from CSV files
- **Real-time Adjustments**: Apply depth offset and scaling adjustments in real-time
- **Visual Comparison**: Compare up to two sensors side-by-side with interactive plots
- **Data Export**: Save adjusted sensor data for further analysis
- **User-friendly Interface**: Intuitive GUI for easy operation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chrisfanshier/sensor_compare.git
cd sensor_compare
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python sensor_app.py
```

### Data Format

The application expects CSV files with at least two columns:
- `depth`: Depth measurements (in meters)
- `value`: Sensor readings

Example CSV format:
```csv
depth,value
0.0,10.5
0.5,12.3
1.0,14.8
1.5,16.2
```

### Creating Sample Data

Use the provided script to generate sample sensor data for testing:

```bash
python generate_sample_data.py
```

This will create two sample CSV files with simulated depth sensor data.

### Workflow

1. **Load Data**: Click "Load Sensor 1 Data" to import your primary sensor data
2. **Apply Adjustments**:
   - Use "Depth Offset" to shift the depth values up or down
   - Use "Scale Factor" to adjust the depth scale (e.g., calibration corrections)
3. **Compare Sensors** (Optional): Load a second sensor dataset to compare with the first
4. **Visualize**: The plot automatically updates to show both sensors with depth on the y-axis (inverted to show increasing depth downward)
5. **Export**: Click "Save Adjusted Data" to export the corrected sensor data

## Adjustments

### Adjustment Order
The application applies adjustments using the formula:
```
adjusted_depth = (original_depth × scale_factor) + offset
```

This order (scale first, then offset) is chosen because:
1. **Scale Factor** corrects systematic errors in the measurement (calibration)
2. **Offset** adjusts for mounting position or reference point differences

### Depth Offset
Adds a constant value to all depth measurements. Use this to:
- Correct for instrument mounting position
- Align multiple sensor datasets
- Account for water column offset

### Scale Factor
Multiplies all depth measurements by a constant factor. Use this to:
- Apply calibration corrections
- Convert between measurement units
- Correct for systematic scaling errors

**Example:** If a sensor reads 100m but should read 110m:
- Option 1: Apply scale factor of 1.1 (systematic 10% error)
- Option 2: Apply offset of +10m (constant 10m error)
- Option 3: Apply scale 1.05 and offset +5m (combined correction)

## Requirements

- Python 3.8+
- PySide6
- NumPy
- Pandas
- Matplotlib

## License

MIT License