#!/usr/bin/env python3
"""
Example usage script showing how to use the Depth Sensor Adjustment Tool programmatically.
This demonstrates the core functionality without the GUI.
"""

import pandas as pd
from pathlib import Path


def example_workflow():
    """Demonstrate a typical workflow for adjusting sensor data."""
    
    print("="*60)
    print("Depth Sensor Adjustment Tool - Example Workflow")
    print("="*60)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample sensor data...")
    import generate_sample_data
    generate_sample_data.main()
    
    # Step 2: Load and inspect data
    print("\n2. Loading sensor data...")
    sensor1 = pd.read_csv('sensor1_sample.csv')
    sensor2 = pd.read_csv('sensor2_sample.csv')
    
    print(f"\nSensor 1:")
    print(f"  Records: {len(sensor1)}")
    print(f"  Depth range: {sensor1['depth'].min():.2f} to {sensor1['depth'].max():.2f} m")
    print(f"  First 3 readings:")
    print(sensor1.head(3).to_string(index=False))
    
    print(f"\nSensor 2:")
    print(f"  Records: {len(sensor2)}")
    print(f"  Depth range: {sensor2['depth'].min():.2f} to {sensor2['depth'].max():.2f} m")
    print(f"  First 3 readings:")
    print(sensor2.head(3).to_string(index=False))
    
    # Step 3: Apply adjustments
    print("\n3. Applying adjustments to Sensor 2...")
    
    # Correct the offset (sensor 2 has 0.5m offset)
    offset_correction = -0.5
    sensor2['depth'] = sensor2['depth'] + offset_correction
    print(f"  Applied offset correction: {offset_correction} m")
    
    # Correct the scale (sensor 2 has 1.02x scale)
    scale_correction = 1/1.02
    sensor2['depth'] = sensor2['depth'] * scale_correction
    print(f"  Applied scale correction: {scale_correction:.4f}")
    
    # Step 4: Compare sensors
    print("\n4. Comparing adjusted sensors...")
    
    # Find depth at same index points
    comparison_indices = [0, 25, 50, 75, 99]
    print("\n  Depth comparison at selected points:")
    print("  Index | Sensor 1 | Sensor 2 | Difference")
    print("  " + "-"*45)
    for idx in comparison_indices:
        d1 = sensor1['depth'].iloc[idx]
        d2 = sensor2['depth'].iloc[idx]
        diff = abs(d1 - d2)
        print(f"  {idx:5d} | {d1:8.2f} | {d2:8.2f} | {diff:10.3f}")
    
    # Step 5: Save adjusted data
    print("\n5. Saving adjusted sensor data...")
    sensor2.to_csv('sensor2_adjusted.csv', index=False)
    print("  Saved: sensor2_adjusted.csv")
    
    # Summary
    print("\n" + "="*60)
    print("Workflow completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run 'python sensor_app.py' to use the GUI application")
    print("  2. Load the sample CSV files to visualize the data")
    print("  3. Adjust offset and scaling to align sensors")
    print("  4. Export adjusted data for your analysis")
    print("="*60)


if __name__ == '__main__':
    example_workflow()
