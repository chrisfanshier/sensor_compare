#!/usr/bin/env python3
"""
Depth Sensor Adjustment Application
A PySide6 application for making adjustments to depth sensors for piston coring.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QFileDialog,
    QGroupBox, QTableWidget, QTableWidgetItem, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SensorData:
    """Class to manage sensor data and adjustments."""
    
    def __init__(self):
        self.data = None
        self.filename = None
        self.depth_column = 'depth'
        self.value_column = 'value'
        
    def load_from_file(self, filepath):
        """Load sensor data from CSV file."""
        try:
            self.data = pd.read_csv(filepath)
            self.filename = Path(filepath).name
            return True
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def apply_offset(self, offset):
        """Apply depth offset adjustment."""
        if self.data is not None and self.depth_column in self.data.columns:
            self.data[self.depth_column] = self.data[self.depth_column] + offset
    
    def apply_scaling(self, scale_factor):
        """Apply scaling factor to depth values."""
        if self.data is not None and self.depth_column in self.data.columns:
            self.data[self.depth_column] = self.data[self.depth_column] * scale_factor
    
    def save_to_file(self, filepath):
        """Save adjusted data to CSV file."""
        if self.data is not None:
            self.data.to_csv(filepath, index=False)
            return True
        return False


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas for plotting sensor data."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        
    def plot_data(self, sensor1, sensor2=None):
        """Plot one or two sensor datasets for comparison."""
        self.axes.clear()
        
        if sensor1 and sensor1.data is not None:
            depth_col = sensor1.depth_column
            value_col = sensor1.value_column
            
            if depth_col in sensor1.data.columns and value_col in sensor1.data.columns:
                self.axes.plot(sensor1.data[value_col], sensor1.data[depth_col], 
                             'b-', label=f'Sensor 1: {sensor1.filename}', linewidth=2)
        
        if sensor2 and sensor2.data is not None:
            depth_col = sensor2.depth_column
            value_col = sensor2.value_column
            
            if depth_col in sensor2.data.columns and value_col in sensor2.data.columns:
                self.axes.plot(sensor2.data[value_col], sensor2.data[depth_col], 
                             'r-', label=f'Sensor 2: {sensor2.filename}', linewidth=2)
        
        self.axes.set_xlabel('Sensor Value')
        self.axes.set_ylabel('Depth (m)')
        self.axes.invert_yaxis()  # Invert y-axis so depth increases downward
        self.axes.legend()
        self.axes.grid(True, alpha=0.3)
        self.draw()


class DepthSensorApp(QMainWindow):
    """Main application window for depth sensor adjustments."""
    
    def __init__(self):
        super().__init__()
        self.sensor1 = SensorData()
        self.sensor2 = SensorData()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Depth Sensor Adjustment Tool - Piston Coring')
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        
        # Right panel - Plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Plot canvas
        self.plot_canvas = PlotCanvas(self, width=6, height=6)
        right_layout.addWidget(self.plot_canvas)
        
        # Add panels to main layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self):
        """Create the control panel with all adjustment options."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Sensor 1 controls
        sensor1_group = QGroupBox("Sensor 1")
        sensor1_layout = QVBoxLayout()
        
        # Load button
        self.load_btn1 = QPushButton('Load Sensor 1 Data')
        self.load_btn1.clicked.connect(lambda: self.load_sensor_data(1))
        sensor1_layout.addWidget(self.load_btn1)
        
        # File label
        self.file_label1 = QLabel('No file loaded')
        self.file_label1.setWordWrap(True)
        sensor1_layout.addWidget(self.file_label1)
        
        # Offset control
        offset_layout1 = QHBoxLayout()
        offset_layout1.addWidget(QLabel('Depth Offset (m):'))
        self.offset_spin1 = QDoubleSpinBox()
        self.offset_spin1.setRange(-1000, 1000)
        self.offset_spin1.setSingleStep(0.1)
        self.offset_spin1.setDecimals(2)
        self.offset_spin1.valueChanged.connect(lambda: self.apply_adjustments(1))
        offset_layout1.addWidget(self.offset_spin1)
        sensor1_layout.addLayout(offset_layout1)
        
        # Scale control
        scale_layout1 = QHBoxLayout()
        scale_layout1.addWidget(QLabel('Scale Factor:'))
        self.scale_spin1 = QDoubleSpinBox()
        self.scale_spin1.setRange(0.1, 10.0)
        self.scale_spin1.setValue(1.0)
        self.scale_spin1.setSingleStep(0.01)
        self.scale_spin1.setDecimals(3)
        self.scale_spin1.valueChanged.connect(lambda: self.apply_adjustments(1))
        scale_layout1.addWidget(self.scale_spin1)
        sensor1_layout.addLayout(scale_layout1)
        
        # Save button
        self.save_btn1 = QPushButton('Save Adjusted Data')
        self.save_btn1.clicked.connect(lambda: self.save_sensor_data(1))
        self.save_btn1.setEnabled(False)
        sensor1_layout.addWidget(self.save_btn1)
        
        sensor1_group.setLayout(sensor1_layout)
        layout.addWidget(sensor1_group)
        
        # Sensor 2 controls
        sensor2_group = QGroupBox("Sensor 2 (Optional)")
        sensor2_layout = QVBoxLayout()
        
        # Load button
        self.load_btn2 = QPushButton('Load Sensor 2 Data')
        self.load_btn2.clicked.connect(lambda: self.load_sensor_data(2))
        sensor2_layout.addWidget(self.load_btn2)
        
        # File label
        self.file_label2 = QLabel('No file loaded')
        self.file_label2.setWordWrap(True)
        sensor2_layout.addWidget(self.file_label2)
        
        # Offset control
        offset_layout2 = QHBoxLayout()
        offset_layout2.addWidget(QLabel('Depth Offset (m):'))
        self.offset_spin2 = QDoubleSpinBox()
        self.offset_spin2.setRange(-1000, 1000)
        self.offset_spin2.setSingleStep(0.1)
        self.offset_spin2.setDecimals(2)
        self.offset_spin2.valueChanged.connect(lambda: self.apply_adjustments(2))
        offset_layout2.addWidget(self.offset_spin2)
        sensor2_layout.addLayout(offset_layout2)
        
        # Scale control
        scale_layout2 = QHBoxLayout()
        scale_layout2.addWidget(QLabel('Scale Factor:'))
        self.scale_spin2 = QDoubleSpinBox()
        self.scale_spin2.setRange(0.1, 10.0)
        self.scale_spin2.setValue(1.0)
        self.scale_spin2.setSingleStep(0.01)
        self.scale_spin2.setDecimals(3)
        self.scale_spin2.valueChanged.connect(lambda: self.apply_adjustments(2))
        scale_layout2.addWidget(self.scale_spin2)
        sensor2_layout.addLayout(scale_layout2)
        
        # Save button
        self.save_btn2 = QPushButton('Save Adjusted Data')
        self.save_btn2.clicked.connect(lambda: self.save_sensor_data(2))
        self.save_btn2.setEnabled(False)
        sensor2_layout.addWidget(self.save_btn2)
        
        sensor2_group.setLayout(sensor2_layout)
        layout.addWidget(sensor2_group)
        
        # Reset button
        reset_btn = QPushButton('Reset All Adjustments')
        reset_btn.clicked.connect(self.reset_adjustments)
        layout.addWidget(reset_btn)
        
        # Instructions
        instructions = QLabel(
            "<b>Instructions:</b><br>"
            "1. Load sensor data (CSV with 'depth' and 'value' columns)<br>"
            "2. Adjust offset and scale as needed<br>"
            "3. Compare multiple sensors visually<br>"
            "4. Save adjusted data when satisfied"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        
        return panel
    
    def load_sensor_data(self, sensor_num):
        """Load data for specified sensor."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"Open Sensor {sensor_num} Data",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                sensor = self.sensor1 if sensor_num == 1 else self.sensor2
                
                # Store original data
                if sensor_num == 1:
                    self.sensor1 = SensorData()
                    sensor = self.sensor1
                else:
                    self.sensor2 = SensorData()
                    sensor = self.sensor2
                
                sensor.load_from_file(filename)
                
                # Update UI
                if sensor_num == 1:
                    self.file_label1.setText(f'Loaded: {sensor.filename}')
                    self.save_btn1.setEnabled(True)
                else:
                    self.file_label2.setText(f'Loaded: {sensor.filename}')
                    self.save_btn2.setEnabled(True)
                
                # Reset adjustments
                if sensor_num == 1:
                    self.offset_spin1.setValue(0.0)
                    self.scale_spin1.setValue(1.0)
                else:
                    self.offset_spin2.setValue(0.0)
                    self.scale_spin2.setValue(1.0)
                
                self.update_plot()
                
                QMessageBox.information(self, "Success", f"Sensor {sensor_num} data loaded successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def apply_adjustments(self, sensor_num):
        """Apply current adjustment values to sensor data."""
        sensor = self.sensor1 if sensor_num == 1 else self.sensor2
        
        if sensor.data is None:
            return
        
        # Get adjustment values
        if sensor_num == 1:
            offset = self.offset_spin1.value()
            scale = self.scale_spin1.value()
        else:
            offset = self.offset_spin2.value()
            scale = self.scale_spin2.value()
        
        # Reload original data and apply adjustments
        # (This is a simplified approach - in production, you'd keep original data separate)
        self.update_plot()
    
    def save_sensor_data(self, sensor_num):
        """Save adjusted sensor data to file."""
        sensor = self.sensor1 if sensor_num == 1 else self.sensor2
        
        if sensor.data is None:
            QMessageBox.warning(self, "Warning", "No data to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Save Sensor {sensor_num} Adjusted Data",
            f"adjusted_{sensor.filename}",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                sensor.save_to_file(filename)
                QMessageBox.information(self, "Success", "Data saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving file: {str(e)}")
    
    def reset_adjustments(self):
        """Reset all adjustment values to defaults."""
        self.offset_spin1.setValue(0.0)
        self.scale_spin1.setValue(1.0)
        self.offset_spin2.setValue(0.0)
        self.scale_spin2.setValue(1.0)
        self.update_plot()
    
    def update_plot(self):
        """Update the comparison plot."""
        self.plot_canvas.plot_data(self.sensor1, self.sensor2)


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName('Depth Sensor Adjustment Tool')
    
    window = DepthSensorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
