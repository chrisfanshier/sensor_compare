#!/usr/bin/env python3
"""
Generate sample depth sensor data for testing the Depth Sensor Adjustment Tool.
"""

import numpy as np
import pandas as pd


def generate_sensor_data(filename, depth_range=(0, 50), num_points=100, 
                        noise_level=0.5, offset=0, scale=1.0):
    """
    Generate synthetic depth sensor data.
    
    Args:
        filename: Output CSV filename
        depth_range: Tuple of (min_depth, max_depth) in meters
        num_points: Number of data points to generate
        noise_level: Amount of random noise to add
        offset: Depth offset to apply
        scale: Scaling factor for depth values
    """
    # Generate depth values
    depths = np.linspace(depth_range[0], depth_range[1], num_points)
    depths = (depths * scale) + offset
    
    # Generate sensor values with some realistic variations
    # Simulate temperature/pressure changes with depth
    base_values = 10 + 0.5 * depths + 5 * np.sin(depths / 10)
    
    # Add random noise
    noise = np.random.normal(0, noise_level, num_points)
    sensor_values = base_values + noise
    
    # Create DataFrame and save
    df = pd.DataFrame({
        'depth': depths,
        'value': sensor_values
    })
    
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {num_points} data points")
    print(f"  Depth range: {depths.min():.2f} to {depths.max():.2f} m")
    print(f"  Value range: {sensor_values.min():.2f} to {sensor_values.max():.2f}")


def main():
    """Generate sample data files."""
    print("Generating sample depth sensor data...\n")
    
    # Generate first sensor data (reference sensor)
    generate_sensor_data(
        'sensor1_sample.csv',
        depth_range=(0, 50),
        num_points=100,
        noise_level=0.5,
        offset=0,
        scale=1.0
    )
    
    print()
    
    # Generate second sensor data (with slight differences for comparison)
    generate_sensor_data(
        'sensor2_sample.csv',
        depth_range=(0, 50),
        num_points=100,
        noise_level=0.6,
        offset=0.5,  # Slight offset
        scale=1.02   # Slight scale difference
    )
    
    print("\nSample data generated successfully!")
    print("You can now load these files in the Depth Sensor Adjustment Tool.")


if __name__ == '__main__':
    main()
