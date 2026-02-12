#!/usr/bin/env python3
"""
Create a static visualization showing what the application displays.
This is for demonstration purposes since we can't run the GUI in headless mode.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_app_visualization():
    """Create a visualization showing the sensor comparison plot."""
    
    # Generate or load sample data
    try:
        sensor1 = pd.read_csv('sensor1_sample.csv')
        sensor2 = pd.read_csv('sensor2_sample.csv')
    except FileNotFoundError:
        print("Generating sample data first...")
        import generate_sample_data
        generate_sample_data.main()
        sensor1 = pd.read_csv('sensor1_sample.csv')
        sensor2 = pd.read_csv('sensor2_sample.csv')
    
    # Create figure with subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Original data (showing misalignment)
    ax1.plot(sensor1['value'], sensor1['depth'], 'b-', label='Sensor 1', linewidth=2)
    ax1.plot(sensor2['value'], sensor2['depth'], 'r-', label='Sensor 2 (Original)', linewidth=2)
    ax1.set_xlabel('Sensor Value', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title('Original Data - Misaligned', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Adjusted data (showing alignment)
    sensor2_adj = sensor2.copy()
    sensor2_adj['depth'] = (sensor2_adj['depth'] - 0.5) / 1.02  # Apply corrections
    
    ax2.plot(sensor1['value'], sensor1['depth'], 'b-', label='Sensor 1', linewidth=2)
    ax2.plot(sensor2_adj['value'], sensor2_adj['depth'], 'g-', label='Sensor 2 (Adjusted)', linewidth=2)
    ax2.set_xlabel('Sensor Value', fontsize=12)
    ax2.set_ylabel('Depth (m)', fontsize=12)
    ax2.set_title('After Adjustment - Aligned', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'app_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Also create a simple diagram showing the UI layout
    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Depth Sensor Adjustment Tool', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Left panel (controls)
    from matplotlib.patches import Rectangle, FancyBboxPatch
    
    # Control panel background
    control_panel = FancyBboxPatch((0.5, 1), 3, 7.5, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='black', facecolor='lightgray', linewidth=2)
    ax.add_patch(control_panel)
    
    # Control panel text
    ax.text(2, 8, 'Control Panel', ha='center', fontsize=14, fontweight='bold')
    ax.text(0.7, 7.3, '📁 Sensor 1 Controls', fontsize=11, fontweight='bold')
    ax.text(0.9, 6.8, '• Load Data', fontsize=9)
    ax.text(0.9, 6.4, '• Depth Offset: ±1000m', fontsize=9)
    ax.text(0.9, 6.0, '• Scale Factor: 0.1-10.0', fontsize=9)
    ax.text(0.9, 5.6, '• Save Adjusted Data', fontsize=9)
    
    ax.text(0.7, 4.8, '📁 Sensor 2 Controls', fontsize=11, fontweight='bold')
    ax.text(0.9, 4.3, '• Load Data (Optional)', fontsize=9)
    ax.text(0.9, 3.9, '• Depth Offset: ±1000m', fontsize=9)
    ax.text(0.9, 3.5, '• Scale Factor: 0.1-10.0', fontsize=9)
    ax.text(0.9, 3.1, '• Save Adjusted Data', fontsize=9)
    
    ax.text(0.7, 2.3, '🔄 Reset All', fontsize=11)
    ax.text(0.7, 1.5, 'ℹ️ Instructions', fontsize=10, style='italic')
    
    # Right panel (plot)
    plot_panel = FancyBboxPatch((4, 1), 5.5, 7.5, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='white', linewidth=2)
    ax.add_patch(plot_panel)
    
    ax.text(6.75, 8, 'Sensor Comparison Plot', ha='center', fontsize=14, fontweight='bold')
    ax.text(6.75, 7.5, '(Interactive Matplotlib)', ha='center', fontsize=10, style='italic')
    
    # Simple plot representation
    ax.plot([4.5, 9], [6.5, 6.5], 'k-', linewidth=1)
    ax.plot([4.5, 4.5], [6.5, 2], 'k-', linewidth=1)
    ax.text(4.2, 4, 'Depth (m)', rotation=90, va='center', fontsize=9)
    ax.text(6.75, 1.7, 'Sensor Value', ha='center', fontsize=9)
    
    # Sample data lines
    x_vals = np.linspace(4.8, 8.8, 50)
    y_vals1 = 6.2 - 0.08 * (x_vals - 4.8) + 0.3 * np.sin((x_vals - 4.8) * 1.5)
    y_vals2 = 6.1 - 0.08 * (x_vals - 4.8) + 0.3 * np.sin((x_vals - 4.8) * 1.5 + 0.2)
    ax.plot(x_vals, y_vals1, 'b-', linewidth=2, label='Sensor 1')
    ax.plot(x_vals, y_vals2, 'r-', linewidth=2, label='Sensor 2')
    
    ax.text(5.5, 5.5, '─ Sensor 1', color='blue', fontsize=9, fontweight='bold')
    ax.text(5.5, 5.2, '─ Sensor 2', color='red', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the UI mockup
    output_file2 = 'app_interface_mockup.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Interface mockup saved to: {output_file2}")
    
    return output_file, output_file2


if __name__ == '__main__':
    print("Creating application visualizations...")
    files = create_app_visualization()
    print("\nVisualization complete!")
    print(f"Files created: {files}")
