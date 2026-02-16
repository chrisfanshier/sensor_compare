import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import pyqtgraph as pg

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QLineEdit, QTextEdit, QFileDialog,
                               QMessageBox, QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
                               QSplitter, QScrollArea, QSpinBox, QDoubleSpinBox, QHeaderView,
                               QTabWidget, QComboBox, QCheckBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor
import json


class CTDAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CTD Sensor Depth Analysis Tool (2-4 Sensors)")
        self.setGeometry(100, 100, 1400, 900)
        
        self.df = None
        self.df_smooth = None
        self.selection_start_idx = None
        self.selection_end_idx = None
        self.selecting = False
        self.drag_start_x = None
        self.current_plot_type = 'depths'
        self.selection_mode = False  # Toggle for selection vs pan/zoom
        self.y_axis_inverted = False  # Track Y-axis inversion state
        
        # Dynamic sensor configuration
        self.num_sensors = 3
        self.sensor_entries = []
        self.sensor_labels = ['A', 'B', 'C', 'D']
        
        # Statistics table
        self.stats_data = []
        self.last_stats = None
        self.offset_calibration = None  # Store loaded calibration
        
        # Shared dataframe for Tab 2 and Tab 3
        self.shared_df = None  # This will be the working dataframe
        
        # Tab 3 specific
        self.time_selection_start_idx = None
        self.time_selection_end_idx = None
        self.time_selecting = False
        self.calculated_time_offsets = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main widget with tabs
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab 1: Original sensor comparison tool
        tab1 = self.create_comparison_tab()
        self.tab_widget.addTab(tab1, "Sensor Comparison")
        
        # Tab 2: Offset correction tool
        tab2 = self.create_offset_correction_tab()
        self.tab_widget.addTab(tab2, "Apply Depth Corrections")
        
        # Tab 3: Time lag correction tool
        tab3 = self.create_time_lag_tab()
        self.tab_widget.addTab(tab3, "Apply Time Corrections")
    
    def create_comparison_tab(self):
        """Create the original sensor comparison interface"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel (controls)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (plot)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([400, 1000])
        
        return tab
        
    def create_left_panel(self):
        # Scrollable left panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(5)
        
        # File selection
        layout.addWidget(self.create_file_selection_group())
        
        # Parameters
        layout.addWidget(self.create_parameters_group())
        
        # Sensor configuration
        layout.addWidget(self.create_sensor_config_group())
        
        # Pre-selection
        layout.addWidget(self.create_preselection_group())
        
        # Selection info
        layout.addWidget(self.create_selection_info_group())
        
        # Action buttons
        layout.addWidget(self.create_actions_group())
        
        # Output log
        layout.addWidget(self.create_output_log_group())
        
        # Statistics table
        layout.addWidget(self.create_stats_table_group())
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll
        
    def create_file_selection_group(self):
        group = QGroupBox("File Selection")
        layout = QVBoxLayout()
        
        self.select_file_btn = QPushButton("Select CSV File")
        self.select_file_btn.clicked.connect(self.load_file)
        layout.addWidget(self.select_file_btn)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        
        group.setLayout(layout)
        return group
        
    def create_parameters_group(self):
        group = QGroupBox("Parameters")
        layout = QGridLayout()
        
        # Skip Rows
        layout.addWidget(QLabel("Skip Rows:"), 0, 0)
        self.skip_rows_spin = QSpinBox()
        self.skip_rows_spin.setRange(0, 1000)
        self.skip_rows_spin.setValue(16)
        layout.addWidget(self.skip_rows_spin, 0, 1)
        
        # Min Depth
        layout.addWidget(QLabel("Min Depth (m):"), 1, 0)
        self.min_depth_spin = QDoubleSpinBox()
        self.min_depth_spin.setRange(0, 99999)
        self.min_depth_spin.setValue(0.0)
        self.min_depth_spin.setDecimals(2)
        layout.addWidget(self.min_depth_spin, 1, 1)
        
        # Max Depth
        layout.addWidget(QLabel("Max Depth (m):"), 2, 0)
        self.max_depth_spin = QDoubleSpinBox()
        self.max_depth_spin.setRange(0, 99999)
        self.max_depth_spin.setValue(9999.0)
        self.max_depth_spin.setDecimals(2)
        layout.addWidget(self.max_depth_spin, 2, 1)
        
        # Trim Rows
        layout.addWidget(QLabel("Trim Rows:"), 3, 0)
        self.trim_rows_spin = QSpinBox()
        self.trim_rows_spin.setRange(0, 10000)
        self.trim_rows_spin.setValue(200)
        layout.addWidget(self.trim_rows_spin, 3, 1)
        
        # Smooth Window
        layout.addWidget(QLabel("Smooth Window:"), 4, 0)
        self.smooth_window_spin = QSpinBox()
        self.smooth_window_spin.setRange(1, 1000)
        self.smooth_window_spin.setValue(100)
        layout.addWidget(self.smooth_window_spin, 4, 1)
        
        group.setLayout(layout)
        return group
        
    def create_sensor_config_group(self):
        group = QGroupBox("Sensor Configuration")
        self.sensor_layout = QVBoxLayout()
        
        # Number of sensors selector
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Number of Sensors:"))
        self.num_sensors_spin = QSpinBox()
        self.num_sensors_spin.setRange(2, 4)
        self.num_sensors_spin.setValue(3)
        self.num_sensors_spin.valueChanged.connect(self.update_sensor_fields)
        num_layout.addWidget(self.num_sensors_spin)
        num_layout.addStretch()
        self.sensor_layout.addLayout(num_layout)
        
        # Datetime column
        datetime_layout = QHBoxLayout()
        datetime_layout.addWidget(QLabel("Datetime Col:"))
        self.datetime_col_edit = QLineEdit("datetime")
        datetime_layout.addWidget(self.datetime_col_edit)
        self.sensor_layout.addLayout(datetime_layout)
        
        # Sensor fields container
        self.sensor_fields_widget = QWidget()
        self.sensor_fields_layout = QGridLayout(self.sensor_fields_widget)
        self.sensor_layout.addWidget(self.sensor_fields_widget)
        
        # Initialize sensor fields
        self.update_sensor_fields()
        
        group.setLayout(self.sensor_layout)
        return group
        
    def update_sensor_fields(self):
        # Clear existing sensor fields
        while self.sensor_fields_layout.count():
            item = self.sensor_fields_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.sensor_entries = []
        self.num_sensors = self.num_sensors_spin.value()
        
        # Create sensor input fields
        default_sensors = ["230405", "230406", "236222", "236223"]
        for i in range(self.num_sensors):
            label = QLabel(f"Sensor {self.sensor_labels[i]}:")
            self.sensor_fields_layout.addWidget(label, i, 0)
            
            entry = QLineEdit(default_sensors[i])
            self.sensor_fields_layout.addWidget(entry, i, 1)
            self.sensor_entries.append(entry)
            
    def create_preselection_group(self):
        group = QGroupBox("Pre-selection (optional)")
        layout = QGridLayout()
        
        # Index selection
        layout.addWidget(QLabel("Index Start:"), 0, 0)
        self.pre_idx_start_edit = QLineEdit()
        layout.addWidget(self.pre_idx_start_edit, 0, 1)
        
        layout.addWidget(QLabel("Index End:"), 0, 2)
        self.pre_idx_end_edit = QLineEdit()
        layout.addWidget(self.pre_idx_end_edit, 0, 3)
        
        # Time selection
        layout.addWidget(QLabel("Time Start:"), 1, 0)
        self.pre_time_start_edit = QLineEdit()
        layout.addWidget(self.pre_time_start_edit, 1, 1, 1, 3)
        
        layout.addWidget(QLabel("Time End:"), 2, 0)
        self.pre_time_end_edit = QLineEdit()
        layout.addWidget(self.pre_time_end_edit, 2, 1, 1, 3)
        
        layout.addWidget(QLabel("(leave blank to skip)"), 3, 0, 1, 4)
        
        group.setLayout(layout)
        return group
        
    def create_selection_info_group(self):
        group = QGroupBox("Selected Range")
        layout = QVBoxLayout()
        
        # Selection mode toggle button
        self.selection_mode_btn = QPushButton("Enable Selection Mode")
        self.selection_mode_btn.setCheckable(True)
        self.selection_mode_btn.clicked.connect(self.toggle_selection_mode)
        self.selection_mode_btn.setStyleSheet(
            "QPushButton:checked { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        layout.addWidget(self.selection_mode_btn)
        
        help_label = QLabel("Enable Selection Mode, then double-click\non plot to start/end selection")
        help_label.setStyleSheet("QLabel { color: blue; font-style: italic; }")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        self.selection_label = QLabel("No selection")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)
        
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self.clear_selection)
        layout.addWidget(clear_btn)
        
        group.setLayout(layout)
        return group
        
    def create_actions_group(self):
        group = QGroupBox("Actions")
        layout = QVBoxLayout()
        
        process_btn = QPushButton("Process Data")
        process_btn.clicked.connect(self.process_data)
        layout.addWidget(process_btn)
        
        stats_btn = QPushButton("Show Statistics")
        stats_btn.clicked.connect(self.show_statistics)
        layout.addWidget(stats_btn)
        
        plot_depths_btn = QPushButton("Plot Depths")
        plot_depths_btn.clicked.connect(lambda: self.plot_data('depths'))
        layout.addWidget(plot_depths_btn)
        
        plot_diff_btn = QPushButton("Plot Differences")
        plot_diff_btn.clicked.connect(lambda: self.plot_data('differences'))
        layout.addWidget(plot_diff_btn)
        
        add_stats_btn = QPushButton("Add Stats to Table")
        add_stats_btn.clicked.connect(self.add_stats_to_table)
        layout.addWidget(add_stats_btn)
        
        export_btn = QPushButton("Export Table")
        export_btn.clicked.connect(self.export_stats_table)
        layout.addWidget(export_btn)
        
        offset_btn = QPushButton("Predict Offsets")
        offset_btn.clicked.connect(self.show_offset_predictor)
        layout.addWidget(offset_btn)
        
        save_offset_btn = QPushButton("Save Offset Calibration")
        save_offset_btn.clicked.connect(self.save_offset_calibration)
        layout.addWidget(save_offset_btn)
        
        load_offset_btn = QPushButton("Load Offset Calibration")
        load_offset_btn.clicked.connect(self.load_offset_calibration)
        layout.addWidget(load_offset_btn)
        
        apply_offset_btn = QPushButton("Apply Offset to Data")
        apply_offset_btn.clicked.connect(self.apply_offset_to_data)
        layout.addWidget(apply_offset_btn)
        
        group.setLayout(layout)
        return group
        
    def create_output_log_group(self):
        group = QGroupBox("Output Log")
        layout = QVBoxLayout()
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(150)
        layout.addWidget(self.output_text)
        
        group.setLayout(layout)
        return group
        
    def create_stats_table_group(self):
        group = QGroupBox("Saved Statistics")
        layout = QVBoxLayout()
        
        self.stats_table = QTableWidget()
        self.stats_table.setMaximumHeight(180)
        self.update_stats_table_columns()
        
        layout.addWidget(self.stats_table)
        
        group.setLayout(layout)
        return group
        
    def update_stats_table_columns(self):
        """Update table columns based on number of sensors"""
        cols = ["range", "n_points"]
        
        # Add difference columns dynamically
        for i in range(self.num_sensors):
            for j in range(i + 1, self.num_sensors):
                cols.append(f"{self.sensor_labels[j]}_minus_{self.sensor_labels[i]}_mean")
        
        cols.append("mean_depth_all_sensors")
        
        # Add individual sensor means
        for i in range(self.num_sensors):
            cols.append(f"Sensor_{self.sensor_labels[i]}_mean")
        
        # Add sensor identifiers
        for i in range(self.num_sensors):
            cols.append(f"Sensor_{self.sensor_labels[i]}")
        
        cols.append("source_file")
        
        self.stats_table.setColumnCount(len(cols))
        self.stats_table.setHorizontalHeaderLabels(cols)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
    def create_right_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create pyqtgraph plot widget with OpenGL acceleration
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        self.plot_widget = pg.PlotWidget(useOpenGL=True)
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Row Index')
        self.plot_widget.setLabel('left', 'Depth (m)')
        self.plot_widget.addLegend()
        
        # Performance optimization
        self.plot_widget.setDownsampling(auto=True, mode='peak')
        self.plot_widget.setClipToView(True)
        
        # Add text item for initial message
        self.initial_text = pg.TextItem('Load and process data to see plot', anchor=(0.5, 0.5))
        self.initial_text.setPos(0.5, 0.5)
        self.plot_widget.addItem(self.initial_text)
        
        # Create cursor tracking lines
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.addItem(self.hline, ignoreBounds=True)
        self.vline.setVisible(False)
        self.hline.setVisible(False)
        
        # Cursor info label
        self.cursor_label = QLabel('')
        self.cursor_label.setStyleSheet('QLabel { background-color: white; padding: 5px; border: 1px solid gray; }')
        
        # Selection region
        self.selection_region = None
        
        # Connect mouse events
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        
        layout.addWidget(self.cursor_label)
        layout.addWidget(self.plot_widget)
        
        return widget
        
    def log(self, message):
        self.output_text.append(message)
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
        QApplication.processEvents()
        
    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select CTD CSV File",
            "",
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if filename:
            self.file_label.setText(Path(filename).name)
            self.filepath = filename
            self.log(f"Selected: {Path(filename).name}")
            
    def find_sensor_column(self, df, pattern):
        """Find column matching the sensor pattern"""
        matches = [col for col in df.columns if pattern in col and 'Depth' in col]
        if matches:
            self.log(f"  Found column for {pattern}: {matches[0]}")
            return matches[0]
        self.log(f"  ERROR: No column found for pattern '{pattern}'")
        self.log(f"  Available columns: {list(df.columns)}")
        return None
        
    def clear_selection(self):
        self.selection_start_idx = None
        self.selection_end_idx = None
        self.selecting = False
        self.selection_label.setText("No selection")
        self.log("Selection cleared")
        if hasattr(self, 'current_plot_type'):
            self.plot_data(self.current_plot_type)
            
    def get_sensor_patterns(self):
        """Get sensor patterns from UI"""
        return [entry.text() for entry in self.sensor_entries]
        
    def get_sensor_col_names(self):
        """Get standardized sensor column names"""
        return [f'Sensor_{self.sensor_labels[i]}_Depth' for i in range(self.num_sensors)]
        
    def process_data(self):
        if not hasattr(self, 'filepath'):
            QMessageBox.critical(self, "Error", "Please select a file first")
            return
            
        try:
            self.log("\n" + "="*50)
            self.log("Processing...")
            
            # Load data - try different strategies and encodings
            skip_rows = self.skip_rows_spin.value()
            
            # First, try reading with comment='#'
            df = None
            for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'latin-1']:
                try:
                    df = pd.read_csv(self.filepath, comment='#', encoding=encoding)
                    self.log(f"Loaded {len(df)} rows (using comment='#', encoding={encoding})")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e1:
                    # If that fails, try skipping rows
                    try:
                        df = pd.read_csv(self.filepath, skiprows=skip_rows, encoding=encoding)
                        self.log(f"Loaded {len(df)} rows (skipping {skip_rows} rows, encoding={encoding})")
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception as e2:
                        # Last resort: try reading with auto-detect
                        try:
                            df = pd.read_csv(self.filepath, encoding=encoding)
                            self.log(f"Loaded {len(df)} rows (default read, encoding={encoding})")
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
            
            if df is None:
                raise ValueError("Could not load CSV file with any common encoding")
            
            self.log(f"Columns found: {len(df.columns)} columns")
            
            # Find sensor columns
            datetime_col = self.datetime_col_edit.text()
            sensor_patterns = self.get_sensor_patterns()
            sensor_cols = []
            
            for pattern in sensor_patterns:
                col = self.find_sensor_column(df, pattern)
                if not col:
                    raise ValueError(f"Could not find column for sensor pattern: {pattern}")
                sensor_cols.append(col)
            
            self.log(f"Found all {self.num_sensors} sensors")
            
            # Rename columns to standard names
            rename_dict = {}
            sensor_col_names = self.get_sensor_col_names()
            for orig_col, new_col in zip(sensor_cols, sensor_col_names):
                rename_dict[orig_col] = new_col
            
            df = df.rename(columns=rename_dict)
            
            # Select depth columns and handle NaN
            df_depth = df[[datetime_col] + sensor_col_names].copy()
            
            # Drop rows where ALL depth values are NaN
            df_depth = df_depth.dropna(subset=sensor_col_names, how='all')
            self.log(f"After dropping rows with all NaN depths: {len(df_depth)} rows")
            
            # For filtering, use the first non-NaN sensor
            df_depth['filter_depth'] = df_depth[sensor_col_names[0]]
            for col in sensor_col_names[1:]:
                df_depth['filter_depth'] = df_depth['filter_depth'].fillna(df_depth[col])
            
            # Filter by depth range
            min_depth = self.min_depth_spin.value()
            max_depth = self.max_depth_spin.value()
            
            df_bottom = df_depth[
                (df_depth['filter_depth'] >= min_depth) & 
                (df_depth['filter_depth'] <= max_depth)
            ].copy()
            df_bottom = df_bottom.drop(columns=['filter_depth'])
            self.log(f"Filtered by depth range: {len(df_bottom)} rows")
            
            # Trim edges
            trim_rows = self.trim_rows_spin.value()
            if len(df_bottom) > trim_rows * 2:
                df_trimmed = df_bottom.iloc[trim_rows:-trim_rows].copy()
                self.log(f"Trimmed: {len(df_trimmed)} rows")
            else:
                df_trimmed = df_bottom.copy()
            
            # Apply optional pre-selection by index
            idx_start = self.pre_idx_start_edit.text().strip()
            idx_end = self.pre_idx_end_edit.text().strip()
            if idx_start or idx_end:
                try:
                    s = int(idx_start) if idx_start else 0
                    e = int(idx_end) if idx_end else len(df_trimmed) - 1
                except ValueError:
                    raise ValueError("Invalid index pre-selection values (must be integers)")
                df_trimmed = df_trimmed.reset_index(drop=True).iloc[max(0, s):min(len(df_trimmed), e + 1)].copy()
                self.log(f"Applied index pre-selection: rows {s}-{e} -> {len(df_trimmed)} rows")
            
            # Apply optional pre-selection by time
            time_start = self.pre_time_start_edit.text().strip()
            time_end = self.pre_time_end_edit.text().strip()
            if time_start or time_end:
                if datetime_col not in df_trimmed.columns:
                    raise ValueError(f"Datetime column '{datetime_col}' not found for time pre-selection")
                df_trimmed[datetime_col] = pd.to_datetime(df_trimmed[datetime_col], errors='coerce')
                if df_trimmed[datetime_col].isnull().any():
                    raise ValueError("Found invalid datetimes when parsing datetime column")
                ts = pd.to_datetime(time_start) if time_start else df_trimmed[datetime_col].min()
                te = pd.to_datetime(time_end) if time_end else df_trimmed[datetime_col].max()
                mask = (df_trimmed[datetime_col] >= ts) & (df_trimmed[datetime_col] <= te)
                df_trimmed = df_trimmed.loc[mask].reset_index(drop=True)
                self.log(f"Applied time pre-selection: {ts} to {te} -> {len(df_trimmed)} rows")
             
            # Keep raw copy and compute raw differences
            df_raw = df_trimmed.reset_index(drop=True)
            
            # Calculate all pairwise differences (raw)
            for i in range(self.num_sensors):
                for j in range(i + 1, self.num_sensors):
                    label_i = self.sensor_labels[i]
                    label_j = self.sensor_labels[j]
                    col_i = sensor_col_names[i]
                    col_j = sensor_col_names[j]
                    df_raw[f'{label_j}_minus_{label_i}_raw'] = df_raw[col_j] - df_raw[col_i]
             
            # Apply smoothing
            smooth_window = self.smooth_window_spin.value()
            df_smooth = df_raw[[datetime_col]].copy()
            
            for col in sensor_col_names:
                df_smooth[col] = df_raw[col].rolling(window=smooth_window, min_periods=1).mean()
            
            # Calculate smoothed differences
            for i in range(self.num_sensors):
                for j in range(i + 1, self.num_sensors):
                    label_i = self.sensor_labels[i]
                    label_j = self.sensor_labels[j]
                    col_i = sensor_col_names[i]
                    col_j = sensor_col_names[j]
                    df_smooth[f'{label_j}_minus_{label_i}'] = df_smooth[col_j] - df_smooth[col_i]
            
            df_smooth = df_smooth.reset_index(drop=True)
            
            self.df = df_raw
            self.df_smooth = df_smooth
            
            # Initialize shared dataframe for Tab 2 and Tab 3
            self.shared_df = df_smooth.copy()
            
            self.log("Complete! Ready to plot.")
            
            # Update stats table columns for current sensor count
            self.update_stats_table_columns()
            
            # Plot depths by default
            self.plot_data('depths')
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
    
    def plot_data(self, plot_type):
        if self.df_smooth is None:
            QMessageBox.critical(self, "Error", "Please process data first")
            return
        
        self.current_plot_type = plot_type
        datetime_col = self.datetime_col_edit.text()
        sensor_col_names = self.get_sensor_col_names()
        
        # Remove initial text if it exists
        if hasattr(self, 'initial_text') and self.initial_text in self.plot_widget.items():
            self.plot_widget.removeItem(self.initial_text)
        
        # Clear plot
        self.plot_widget.clear()
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.addItem(self.hline, ignoreBounds=True)
        
        # Re-add legend
        self.plot_widget.addLegend()
        
        colors = ['b', 'g', 'r', 'm', 'c', 'y']
        
        # Determine if Y-axis should be inverted for this plot type
        should_invert = (plot_type == 'depths')
        
        if plot_type == 'depths':
            self.plot_widget.setLabel('left', 'Depth (m)')
            self.plot_widget.setTitle('CTD Depth Sensors (Click and drag to select range)')
            
            for i, col in enumerate(sensor_col_names):
                data = self.df_smooth[col].values
                pen = pg.mkPen(color=colors[i], width=2)
                # Use connect='finite' to skip NaN values efficiently
                self.plot_widget.plot(self.df_smooth.index.values, data,
                                     pen=pen, name=f'Sensor {self.sensor_labels[i]}',
                                     connect='finite', skipFiniteCheck=True)
        else:  # differences
            self.plot_widget.setLabel('left', 'Depth Difference (m)')
            self.plot_widget.setTitle('Sensor Differences (Click and drag to select range)')
            
            if self.selection_start_idx is not None and self.selection_end_idx is not None:
                s = self.selection_start_idx
                e = self.selection_end_idx + 1
                df_raw = self.df.iloc[s:e]
                df_sm = self.df_smooth.iloc[s:e]
            else:
                df_raw = self.df
                df_sm = self.df_smooth
            
            # Plot all pairwise differences
            color_idx = 0
            for i in range(self.num_sensors):
                for j in range(i + 1, self.num_sensors):
                    label_i = self.sensor_labels[i]
                    label_j = self.sensor_labels[j]
                    raw_col = f'{label_j}_minus_{label_i}_raw'
                    smooth_col = f'{label_j}_minus_{label_i}'
                    
                    # Plot raw (faded) - skip for performance if dataset is large
                    if len(df_raw) < 10000:
                        pen_raw = pg.mkPen(color=colors[color_idx % len(colors)], width=1, 
                                          style=Qt.DotLine)
                        self.plot_widget.plot(df_raw.index.values, df_raw[raw_col].values,
                                             pen=pen_raw, connect='finite', skipFiniteCheck=True)
                    
                    # Plot smoothed (bold)
                    pen_smooth = pg.mkPen(color=colors[color_idx % len(colors)], width=2)
                    self.plot_widget.plot(df_sm.index.values, df_sm[smooth_col].values,
                                         pen=pen_smooth, name=f'{label_j} - {label_i}',
                                         connect='finite', skipFiniteCheck=True)
                    color_idx += 1
            
            # Add zero line
            pen_zero = pg.mkPen('k', width=1, style=Qt.DashLine)
            x_range = [df_sm.index.min(), df_sm.index.max()]
            self.plot_widget.plot(x_range, [0, 0], pen=pen_zero)
            
        # Set X axis to show full range
        self.plot_widget.setXRange(0, len(self.df_smooth) - 1, padding=0)
        
        # Handle Y-axis inversion only if state needs to change
        if should_invert != self.y_axis_inverted:
            self.plot_widget.invertY(should_invert)
            self.y_axis_inverted = should_invert
        
        # Auto-range Y axis after all plotting is complete
        self.plot_widget.enableAutoRange(axis=pg.ViewBox.YAxis)
        
        # Add selection region if exists
        if self.selection_start_idx is not None and self.selection_end_idx is not None:
            if self.selection_region is not None:
                self.plot_widget.removeItem(self.selection_region)
            
            self.selection_region = pg.LinearRegionItem(
                values=[self.selection_start_idx, self.selection_end_idx],
                brush=pg.mkBrush(255, 0, 0, 50),
                movable=False
            )
            self.plot_widget.addItem(self.selection_region)
        
    def on_mouse_moved(self, pos):
        """Update cursor tracking lines and info label"""
        if self.df_smooth is None:
            return
        
        # Get mouse position in data coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Update crosshair
        self.vline.setPos(x)
        self.hline.setPos(y)
        self.vline.setVisible(True)
        self.hline.setVisible(True)
        
        # Get nearest data point
        idx = int(round(x))
        if 0 <= idx < len(self.df_smooth):
            sensor_col_names = self.get_sensor_col_names()
            info_text = f"Index: {idx}  |  "
            
            if self.current_plot_type == 'depths':
                for i, col in enumerate(sensor_col_names):
                    val = self.df_smooth[col].iloc[idx]
                    if pd.notna(val):
                        info_text += f"Sensor {self.sensor_labels[i]}: {val:.3f}m  "
            else:  # differences
                for i in range(self.num_sensors):
                    for j in range(i + 1, self.num_sensors):
                        label_i = self.sensor_labels[i]
                        label_j = self.sensor_labels[j]
                        diff_col = f'{label_j}_minus_{label_i}'
                        val = self.df_smooth[diff_col].iloc[idx]
                        if pd.notna(val):
                            info_text += f"{label_j}-{label_i}: {val:.3f}m  "
            
            self.cursor_label.setText(info_text)
    
    def toggle_selection_mode(self):
        """Toggle between selection mode and pan/zoom mode"""
        self.selection_mode = self.selection_mode_btn.isChecked()
        
        if self.selection_mode:
            # Disable pan/zoom, enable selection
            self.plot_widget.plotItem.vb.setMouseEnabled(x=False, y=False)
            self.selection_mode_btn.setText("Disable Selection Mode")
            self.log("Selection mode ENABLED - double-click to select range")
        else:
            # Enable pan/zoom, disable selection
            self.plot_widget.plotItem.vb.setMouseEnabled(x=True, y=True)
            self.selection_mode_btn.setText("Enable Selection Mode")
            self.selecting = False  # Cancel any in-progress selection
            self.log("Selection mode DISABLED - pan/zoom enabled")
    
    def on_mouse_clicked(self, event):
        """Handle mouse click for selection - use double-click to select range"""
        if self.df_smooth is None:
            return
        
        # Only allow selection when in selection mode
        if not self.selection_mode:
            return
        
        if event.double():
            # Double-click starts/ends selection
            pos = event.scenePos()
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x = mouse_point.x()
            
            if not self.selecting:
                # Start selection
                self.selecting = True
                self.drag_start_x = x
                self.log(f"Selection started at index {int(x)}")
            else:
                # End selection
                self.selecting = False
                
                start_x = min(self.drag_start_x, x)
                end_x = max(self.drag_start_x, x)
                
                self.selection_start_idx = max(0, int(start_x))
                self.selection_end_idx = min(len(self.df_smooth) - 1, int(end_x))
                
                n_points = self.selection_end_idx - self.selection_start_idx + 1
                self.selection_label.setText(
                    f"Selected:\nRows {self.selection_start_idx} to {self.selection_end_idx}\n({n_points} points)"
                )
                self.log(f"Selection completed: rows {self.selection_start_idx} to {self.selection_end_idx}")
                
                self.plot_data(self.current_plot_type)
            
    def show_statistics(self):
        if self.df_smooth is None:
            QMessageBox.critical(self, "Error", "Please process data first")
            return
        
        if self.selection_start_idx is not None and self.selection_end_idx is not None:
            df_calc = self.df_smooth.iloc[self.selection_start_idx:self.selection_end_idx+1]
            range_info = f"Selected range: rows {self.selection_start_idx}-{self.selection_end_idx} ({len(df_calc)} points)"
        else:
            df_calc = self.df_smooth
            range_info = "Full dataset (no selection)"
            
        self.log("\n" + "="*50)
        self.log("STATISTICS")
        self.log(range_info)
        self.log("="*50)
        
        sensor_col_names = self.get_sensor_col_names()
        
        # Show all pairwise differences
        for i in range(self.num_sensors):
            for j in range(i + 1, self.num_sensors):
                label_i = self.sensor_labels[i]
                label_j = self.sensor_labels[j]
                diff_col = f'{label_j}_minus_{label_i}'
                self.log(f"\n{label_j} minus {label_i}:")
                self.log(str(df_calc[diff_col].describe()))
        
        # Compute mean depth across all sensors
        mean_depth_all = df_calc[sensor_col_names].mean(axis=1).mean()
        self.log(f"\nMean depth (all sensors): {mean_depth_all:.6f}")

        # Per-sensor means
        self.last_stats = {
            "range": range_info,
            "n_points": len(df_calc),
            "mean_depth_all_sensors": float(mean_depth_all)
        }
        
        for i in range(self.num_sensors):
            label = self.sensor_labels[i]
            col = sensor_col_names[i]
            mean_val = float(df_calc[col].mean())
            self.log(f"Mean Sensor {label}: {mean_val:.6f}")
            self.last_stats[f"Sensor_{label}_mean"] = mean_val
        
        # Add difference means
        for i in range(self.num_sensors):
            for j in range(i + 1, self.num_sensors):
                label_i = self.sensor_labels[i]
                label_j = self.sensor_labels[j]
                diff_col = f'{label_j}_minus_{label_i}'
                self.last_stats[f"{label_j}_minus_{label_i}_mean"] = float(df_calc[diff_col].mean())
        
        # Add sensor identifiers and source file
        sensor_patterns = self.get_sensor_patterns()
        for i in range(self.num_sensors):
            self.last_stats[f"Sensor_{self.sensor_labels[i]}"] = sensor_patterns[i]
        
        self.last_stats["source_file"] = Path(getattr(self, "filepath", "")).name
        
    def add_stats_to_table(self):
        if self.last_stats is None:
            QMessageBox.critical(self, "Error", "No statistics available. Run 'Show Statistics' first.")
            return
        
        self.stats_data.append(self.last_stats)
        
        # Add row to table
        row_position = self.stats_table.rowCount()
        self.stats_table.insertRow(row_position)
        
        col_idx = 0
        for col_name in [self.stats_table.horizontalHeaderItem(i).text() 
                         for i in range(self.stats_table.columnCount())]:
            value = self.last_stats.get(col_name, "")
            item = QTableWidgetItem(str(value))
            self.stats_table.setItem(row_position, col_idx, item)
            col_idx += 1
        
        self.log("Added current statistics to table")
        
    def export_stats_table(self):
        if not self.stats_data:
            QMessageBox.critical(self, "Error", "No saved statistics to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Statistics",
            "",
            "CSV files (*.csv)"
        )
        
        if filename:
            df = pd.DataFrame(self.stats_data)
            df.to_csv(filename, index=False)
            self.log(f"Exported statistics to {Path(filename).name}")
    
    def show_offset_predictor(self):
        """Show window for offset prediction based on linear regression"""
        if len(self.stats_data) < 2:
            QMessageBox.critical(self, "Error", "Need at least 2 saved statistics for regression analysis")
            return
        
        # Create DataFrame from stats
        stats_df = pd.DataFrame(self.stats_data)
        
        # Create predictor window
        from PySide6.QtWidgets import QDialog
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Offset Predictor")
        dialog.setGeometry(150, 150, 1200, 800)
        
        layout = QHBoxLayout(dialog)
        
        # Left panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Prediction controls
        pred_group = QGroupBox("Prediction")
        pred_layout = QVBoxLayout()
        
        pred_layout.addWidget(QLabel("Target Depth (m):"))
        depth_spin = QDoubleSpinBox()
        depth_spin.setRange(0, 99999)
        depth_spin.setValue(700.0)
        depth_spin.setDecimals(2)
        pred_layout.addWidget(depth_spin)
        
        result_text = QTextEdit()
        result_text.setReadOnly(True)
        result_text.setMaximumHeight(200)
        pred_layout.addWidget(result_text)
        
        pred_group.setLayout(pred_layout)
        left_layout.addWidget(pred_group)
        
        # Model info
        info_group = QGroupBox("Regression Models")
        info_layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_layout.addWidget(info_text)
        
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)
        
        # Export button
        export_btn = QPushButton("Export Plot")
        left_layout.addWidget(export_btn)
        
        left_widget.setMaximumWidth(350)
        layout.addWidget(left_widget)
        
        # Right panel (plots)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Get depth data
        depth_data = stats_df['mean_depth_all_sensors'].values
        
        # Calculate number of comparisons
        num_comparisons = (self.num_sensors * (self.num_sensors - 1)) // 2
        
        # Create scroll area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        
        # Perform regressions for all sensor pairs
        regressions = []
        colors = ['b', 'g', 'r', 'm', 'c', 'y']
        
        plot_idx = 0
        for i in range(self.num_sensors):
            for j in range(i + 1, self.num_sensors):
                label_i = self.sensor_labels[i]
                label_j = self.sensor_labels[j]
                diff_col = f"{label_j}_minus_{label_i}_mean"
                
                if diff_col not in stats_df.columns:
                    continue
                
                diff_data = stats_df[diff_col].values
                
                # Remove NaN values
                mask = ~(np.isnan(depth_data) | np.isnan(diff_data))
                depth_clean = depth_data[mask]
                diff_clean = diff_data[mask]
                
                if len(depth_clean) < 2:
                    continue
                
                # Linear regression
                res = stats.linregress(depth_clean, diff_clean)
                regressions.append((label_j, label_i, res, depth_clean, diff_clean))
                
                # Create pyqtgraph plot
                plot_widget = pg.PlotWidget()
                plot_widget.setBackground('w')
                plot_widget.showGrid(x=True, y=True, alpha=0.3)
                plot_widget.setLabel('bottom', 'Mean Depth (m)')
                plot_widget.setLabel('left', f'{label_j} - {label_i} Bias (m)')
                
                r_squared = res.rvalue ** 2
                plot_widget.setTitle(f'{label_j} - {label_i} vs Depth (R² = {r_squared:.3f})')
                
                # Scatter plot
                scatter = pg.ScatterPlotItem(
                    x=depth_clean, 
                    y=diff_clean,
                    pen=None,
                    brush=pg.mkBrush(colors[plot_idx % len(colors)]),
                    size=8,
                    name='Measured Mean Bias'
                )
                plot_widget.addItem(scatter)
                
                # Trendline
                trendline_x = np.linspace(depth_clean.min(), depth_clean.max(), 100)
                trendline_y = res.slope * trendline_x + res.intercept
                pen_trend = pg.mkPen('r', width=2)
                plot_widget.plot(trendline_x, trendline_y, pen=pen_trend,
                               name=f'y = {res.slope:.6f}x + {res.intercept:.4f}')
                
                plot_widget.addLegend()
                plot_widget.setMinimumHeight(300)
                plot_layout.addWidget(plot_widget)
                
                plot_idx += 1
        
        scroll.setWidget(plot_container)
        right_layout.addWidget(scroll)
        layout.addWidget(right_widget)
        
        # Fill info text
        info_text.clear()
        for label_j, label_i, res, _, _ in regressions:
            info_text.append(f"{label_j} - {label_i} vs Depth")
            info_text.append("=" * 35)
            info_text.append(f"Slope: {res.slope:.6f}")
            info_text.append(f"Intercept: {res.intercept:.6f}")
            info_text.append(f"R²: {res.rvalue**2:.4f}")
            info_text.append(f"P-value: {res.pvalue:.6f}\n")
        
        # Prediction function
        def predict():
            target = depth_spin.value()
            result_text.clear()
            result_text.append(f"Predictions for {target}m depth:")
            result_text.append("=" * 35 + "\n")
            
            for label_j, label_i, res, _, _ in regressions:
                offset = (res.slope * target) + res.intercept
                result_text.append(f"{label_j} - {label_i} Offset: {offset:.6f} m")
            
            result_text.append(f"\nLinear Models:")
            for label_j, label_i, res, _, _ in regressions:
                result_text.append(f"{label_j}-{label_i}: y = {res.slope:.6f}x + {res.intercept:.6f}")
        
        calc_btn = QPushButton("Calculate")
        calc_btn.clicked.connect(predict)
        pred_layout.addWidget(calc_btn)
        
        # Export function
        def export_plot():
            filename, _ = QFileDialog.getSaveFileName(
                dialog,
                "Save Plot",
                "",
                "PNG files (*.png);;All files (*.*)"
            )
            if filename:
                # Export all plots as images and combine them
                from PySide6.QtWidgets import QApplication
                from PySide6.QtGui import QPixmap, QPainter
                from PySide6.QtCore import QRect
                
                # Get all plot widgets
                plot_widgets = []
                for i in range(plot_layout.count()):
                    widget = plot_layout.itemAt(i).widget()
                    if isinstance(widget, pg.PlotWidget):
                        plot_widgets.append(widget)
                
                if plot_widgets:
                    # Export first plot as example
                    exporter = pg.exporters.ImageExporter(plot_widgets[0].plotItem)
                    exporter.export(filename)
                    QMessageBox.information(dialog, "Success", f"Plot saved to {Path(filename).name}")
                else:
                    QMessageBox.warning(dialog, "Warning", "No plots to export")
        
        export_btn.clicked.connect(export_plot)
        
        dialog.exec()
    
    def save_offset_calibration(self):
        """Save offset calibration based on regression analysis"""
        if len(self.stats_data) < 2:
            QMessageBox.critical(self, "Error", "Need at least 2 saved statistics for calibration")
            return
        
        # Create DataFrame from stats
        stats_df = pd.DataFrame(self.stats_data)
        depth_data = stats_df['mean_depth_all_sensors'].values
        
        # Calculate regressions for all sensor pairs
        calibration = {
            'num_sensors': self.num_sensors,
            'sensor_labels': self.sensor_labels[:self.num_sensors],
            'regressions': []
        }
        
        for i in range(self.num_sensors):
            for j in range(i + 1, self.num_sensors):
                label_i = self.sensor_labels[i]
                label_j = self.sensor_labels[j]
                diff_col = f"{label_j}_minus_{label_i}_mean"
                
                if diff_col not in stats_df.columns:
                    continue
                
                diff_data = stats_df[diff_col].values
                mask = ~(np.isnan(depth_data) | np.isnan(diff_data))
                depth_clean = depth_data[mask]
                diff_clean = diff_data[mask]
                
                if len(depth_clean) < 2:
                    continue
                
                res = stats.linregress(depth_clean, diff_clean)
                calibration['regressions'].append({
                    'sensor_i': label_i,
                    'sensor_j': label_j,
                    'slope': float(res.slope),
                    'intercept': float(res.intercept),
                    'r_squared': float(res.rvalue ** 2),
                    'p_value': float(res.pvalue)
                })
        
        # Save to JSON file
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Offset Calibration",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(calibration, f, indent=2)
            self.log(f"Saved offset calibration to {Path(filename).name}")
            QMessageBox.information(self, "Success", f"Calibration saved to {Path(filename).name}")
    
    def load_offset_calibration(self):
        """Load saved offset calibration"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Offset Calibration",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.offset_calibration = json.load(f)
                self.log(f"Loaded offset calibration from {Path(filename).name}")
                self.log(f"Number of sensors: {self.offset_calibration['num_sensors']}")
                self.log(f"Number of regressions: {len(self.offset_calibration['regressions'])}")
                
                # Display calibration info
                info_text = "\nCalibration Models:\n" + "="*50 + "\n"
                for reg in self.offset_calibration['regressions']:
                    info_text += f"{reg['sensor_j']} - {reg['sensor_i']}: "
                    info_text += f"y = {reg['slope']:.6f}x + {reg['intercept']:.6f} "
                    info_text += f"(R² = {reg['r_squared']:.4f})\n"
                self.log(info_text)
                
                QMessageBox.information(self, "Success", f"Calibration loaded from {Path(filename).name}")
            except Exception as e:
                self.log(f"ERROR loading calibration: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load calibration: {str(e)}")
    
    def apply_offset_to_data(self):
        """Apply loaded offset calibration to current data"""
        if self.offset_calibration is None:
            QMessageBox.critical(self, "Error", "No offset calibration loaded. Load a calibration first.")
            return
        
        if self.df_smooth is None:
            QMessageBox.critical(self, "Error", "No data processed. Process data first.")
            return
        
        try:
            sensor_col_names = self.get_sensor_col_names()
            
            # Calculate mean depth for offset calculation
            mean_depth = self.df_smooth[sensor_col_names].mean(axis=1)
            
            # Apply corrections based on calibration
            # For each regression, we have sensor_j - sensor_i = slope * depth + intercept
            # This means sensor_j is offset from sensor_i by that amount
            # We'll correct each sensor relative to sensor A (first sensor)
            
            corrected_df = self.df_smooth.copy()
            
            self.log("\nApplying offset calibration...")
            for reg in self.offset_calibration['regressions']:
                sensor_i = reg['sensor_i']
                sensor_j = reg['sensor_j']
                slope = reg['slope']
                intercept = reg['intercept']
                
                # Calculate the offset for each depth point
                offset = slope * mean_depth + intercept
                
                # Find corresponding column
                col_j_name = None
                for idx, label in enumerate(self.sensor_labels[:self.num_sensors]):
                    if label == sensor_j:
                        col_j_name = sensor_col_names[idx]
                        break
                
                if col_j_name:
                    # Apply correction: subtract the offset from sensor_j to align with sensor_i
                    corrected_df[col_j_name] = self.df_smooth[col_j_name] - offset
                    self.log(f"Applied correction to Sensor {sensor_j}: subtract offset (slope={slope:.6f}, intercept={intercept:.6f})")
            
            # Store corrected data and replot
            self.df_smooth_original = self.df_smooth.copy()
            self.df_smooth = corrected_df
            
            # Update shared dataframe
            self.shared_df = corrected_df.copy()
            
            # Recalculate differences with corrected data
            for i in range(self.num_sensors):
                for j in range(i + 1, self.num_sensors):
                    label_i = self.sensor_labels[i]
                    label_j = self.sensor_labels[j]
                    col_i = sensor_col_names[i]
                    col_j = sensor_col_names[j]
                    self.df_smooth[f'{label_j}_minus_{label_i}'] = self.df_smooth[col_j] - self.df_smooth[col_i]
            
            self.log("Offset correction applied successfully!")
            self.log("Data has been corrected. Re-plot to see corrected data.")
            
            # Ask if user wants to replot
            reply = QMessageBox.question(self, "Apply Correction", 
                                        "Offset correction applied. Replot data now?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.plot_data(self.current_plot_type)
                
        except Exception as e:
            self.log(f"ERROR applying offset: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to apply offset: {str(e)}")


    def create_offset_correction_tab(self):
        """Create the offset correction and plotting interface"""
        # Initialize tab-specific data
        self.correction_df = None
        self.correction_depth_cols = []
        self.correction_sensor_assignments = {}  # Maps column name to sensor label (A/B/C)
        self.correction_calibration = None
        self.correction_manual_offsets = {}  # Manual offset for each sensor
        self.correction_ref_sensor = None
        
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel
        left_panel = self.create_correction_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (plot)
        right_panel = self.create_correction_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([450, 950])
        
        return tab
    
    def create_correction_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        
        # File selection
        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout()
        self.correction_file_label = QLabel("No file selected")
        self.correction_file_label.setWordWrap(True)
        file_layout.addWidget(self.correction_file_label)
        select_btn = QPushButton("Select CSV File")
        select_btn.clicked.connect(self.load_correction_file)
        file_layout.addWidget(select_btn)
        
        # Use Tab 1 data button
        use_tab1_btn = QPushButton("Use Data from Tab 1")
        use_tab1_btn.clicked.connect(self.use_tab1_data_for_correction)
        file_layout.addWidget(use_tab1_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Calibration file
        calib_group = QGroupBox("Calibration File")
        calib_layout = QVBoxLayout()
        self.correction_calib_label = QLabel("No calibration loaded")
        self.correction_calib_label.setWordWrap(True)
        calib_layout.addWidget(self.correction_calib_label)
        calib_btn = QPushButton("Load JSON Calibration")
        calib_btn.clicked.connect(self.load_correction_calibration)
        calib_layout.addWidget(calib_btn)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)
        
        # Sensor assignments
        self.assignment_group = QGroupBox("Sensor Assignments")
        self.assignment_layout = QGridLayout()
        self.assignment_combos = []
        self.assignment_group.setLayout(self.assignment_layout)
        layout.addWidget(self.assignment_group)
        
        # Reference sensor
        ref_group = QGroupBox("Reference Sensor")
        ref_layout = QVBoxLayout()
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference:"))
        self.ref_sensor_combo = QComboBox()
        self.ref_sensor_combo.addItems(['A', 'B', 'C'])
        self.ref_sensor_combo.currentTextChanged.connect(self.update_correction_plan)
        ref_row.addWidget(self.ref_sensor_combo)
        ref_row.addStretch()
        ref_layout.addLayout(ref_row)
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)
        
        # Correction plan (which corrections will be applied)
        self.correction_plan_group = QGroupBox("Regression Corrections")
        self.correction_plan_layout = QVBoxLayout()
        self.correction_checkboxes = {}  # Maps sensor label to checkbox
        self.correction_plan_label = QLabel("Load calibration and select reference to see plan")
        self.correction_plan_label.setWordWrap(True)
        self.correction_plan_label.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        self.correction_plan_layout.addWidget(self.correction_plan_label)
        self.correction_plan_group.setLayout(self.correction_plan_layout)
        layout.addWidget(self.correction_plan_group)
        
        # Manual offsets
        self.manual_offset_group = QGroupBox("Manual Depth Corrections")
        self.manual_offset_layout = QGridLayout()
        self.manual_offset_spinboxes = {}
        for i, label in enumerate(['A', 'B', 'C']):
            self.manual_offset_layout.addWidget(QLabel(f"Sensor {label}:"), i, 0)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1000, 1000)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(4)
            spinbox.setValue(0.0)
            self.manual_offset_layout.addWidget(spinbox, i, 1)
            self.manual_offset_layout.addWidget(QLabel("m"), i, 2)
            self.manual_offset_spinboxes[label] = spinbox
        self.manual_offset_group.setLayout(self.manual_offset_layout)
        layout.addWidget(self.manual_offset_group)
        
        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        plot_btn = QPushButton("Plot Original Data")
        plot_btn.clicked.connect(self.plot_correction_original)
        action_layout.addWidget(plot_btn)
        
        apply_btn = QPushButton("Apply Corrections & Plot")
        apply_btn.clicked.connect(self.apply_corrections_and_plot)
        action_layout.addWidget(apply_btn)
        
        export_btn = QPushButton("Export Corrected Data")
        export_btn.clicked.connect(self.export_corrected_data)
        action_layout.addWidget(export_btn)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.correction_log = QTextEdit()
        self.correction_log.setReadOnly(True)
        self.correction_log.setMaximumHeight(200)
        log_layout.addWidget(self.correction_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll
    
    def create_correction_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create plot widget with datetime axis
        self.correction_date_axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation='bottom')
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        self.correction_plot = pg.PlotWidget(
            axisItems={'bottom': self.correction_date_axis},
            useOpenGL=True
        )
        self.correction_plot.setBackground('w')
        self.correction_plot.showGrid(x=True, y=True, alpha=0.3)
        self.correction_plot.addLegend()
        self.correction_plot.getViewBox().invertY(True)
        self.correction_plot.setLabel('left', 'Depth (m)')
        
        # Performance optimizations
        self.correction_plot.setDownsampling(auto=True, mode='peak')
        self.correction_plot.setClipToView(True)
        
        # Crosshairs
        self.correction_vline = pg.InfiniteLine(angle=90, movable=False, 
                                               pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.correction_hline = pg.InfiniteLine(angle=0, movable=False,
                                               pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.correction_plot.addItem(self.correction_vline, ignoreBounds=True)
        self.correction_plot.addItem(self.correction_hline, ignoreBounds=True)
        self.correction_vline.setVisible(False)
        self.correction_hline.setVisible(False)
        
        # Info label (matching first tab style)
        self.correction_info_label = QLabel('')
        self.correction_info_label.setStyleSheet('QLabel { background-color: white; padding: 5px; border: 1px solid gray; }')
        
        layout.addWidget(self.correction_info_label)
        layout.addWidget(self.correction_plot)
        
        # Connect mouse movement (store proxy to prevent garbage collection)
        self.correction_mouse_proxy = pg.SignalProxy(
            self.correction_plot.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.correction_mouse_moved
        )
        
        return panel
    
    def correction_log_msg(self, msg):
        """Add message to correction tab log"""
        self.correction_log.append(msg)
    
    def use_tab1_data_for_correction(self):
        """Use processed data from Tab 1 for correction tab"""
        if self.shared_df is None:
            QMessageBox.warning(self, "Warning", "No data available from Tab 1. Process data in Tab 1 first.")
            return
        
        try:
            datetime_col = self.datetime_col_edit.text()
            sensor_col_names = self.get_sensor_col_names()
            
            # Use shared dataframe
            self.correction_df = self.shared_df.copy()
            self.correction_depth_cols = sensor_col_names.copy()
            
            self.correction_file_label.setText("Using data from Tab 1")
            self.correction_log_msg("Loaded data from Tab 1")
            self.correction_log_msg(f"Found {len(self.correction_depth_cols)} sensors")
            
            # Auto-assign sensors
            self.correction_sensor_assignments = {}
            for i, col in enumerate(self.correction_depth_cols):
                if i < 3:
                    self.correction_sensor_assignments[col] = self.sensor_labels[i]
            
            self.setup_sensor_assignments()
            self.correction_log_msg("Ready to apply corrections")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Tab 1 data:\n{str(e)}")
            self.correction_log_msg(f"ERROR: {str(e)}")
    
    def load_correction_file(self):
        """Load CSV file for correction"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            # Read file with comment handling - try multiple encodings
            lines = None
            for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break  # Success, stop trying encodings
                except (UnicodeDecodeError, UnicodeError):
                    continue  # Try next encoding
            
            if lines is None:
                raise ValueError("Could not decode file with any common encoding")
            
            # Look for core title and column header line
            core_title = None
            skiprows = 0
            for i, line in enumerate(lines):
                if line.startswith('# Core:'):
                    core_title = line.split(':', 1)[1].strip()
                if line.strip().startswith('datetime'):
                    skiprows = i
                    break
            
            # Read the CSV with the same encoding detection approach
            df = None
            for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'latin-1']:
                try:
                    df = pd.read_csv(file_path, skiprows=skiprows, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                raise ValueError("Could not decode CSV with any common encoding")
                
            df.columns = df.columns.str.strip()
            
            if 'datetime' not in df.columns:
                QMessageBox.warning(self, "Warning", "No 'datetime' column found")
                return
            
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Find depth columns
            depth_cols = [col for col in df.columns if 'Depth' in col]
            if not depth_cols:
                QMessageBox.critical(self, "Error", "No depth columns found")
                return
            
            self.correction_df = df
            self.correction_depth_cols = depth_cols
            
            # Update shared dataframe
            self.shared_df = df.copy()
            
            # Update file label
            self.correction_file_label.setText(f"Loaded: {Path(file_path).name}")
            if core_title:
                self.correction_file_label.setText(
                    f"Loaded: {Path(file_path).name}\nCore: {core_title}"
                )
                self.correction_plot.setTitle(core_title)
            
            self.correction_log_msg(f"Loaded file: {Path(file_path).name}")
            self.correction_log_msg(f"Found {len(depth_cols)} depth columns:\n" + 
                                   "\n".join([f"  - {col}" for col in depth_cols]))
            
            # Setup sensor assignments
            self.setup_sensor_assignments()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            self.correction_log_msg(f"ERROR loading file: {str(e)}")
    
    def setup_sensor_assignments(self):
        """Create dropdown menus to assign columns to sensors"""
        # Clear previous assignments
        while self.assignment_layout.count():
            item = self.assignment_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.assignment_combos = []
        self.correction_sensor_assignments = {}
        
        # Header
        self.assignment_layout.addWidget(QLabel("Column"), 0, 0)
        self.assignment_layout.addWidget(QLabel("Assign to"), 0, 1)
        
        # Create assignment for each depth column
        sensor_options = ['None', 'A', 'B', 'C']
        for i, col in enumerate(self.correction_depth_cols):
            row = i + 1
            
            # Short name (like p1_offset does)
            short_name = col.split('_', 1)[0]
            label = QLabel(short_name)
            label.setToolTip(col)
            self.assignment_layout.addWidget(label, row, 0)
            
            combo = QComboBox()
            combo.addItems(sensor_options)
            # Auto-assign first 3 to A, B, C
            if i < 3:
                combo.setCurrentText(['A', 'B', 'C'][i])
                self.correction_sensor_assignments[col] = ['A', 'B', 'C'][i]
            combo.currentTextChanged.connect(
                lambda text, c=col: self.update_sensor_assignment(c, text)
            )
            self.assignment_layout.addWidget(combo, row, 1)
            self.assignment_combos.append(combo)
        
        self.correction_log_msg(f"Auto-assigned first {min(3, len(self.correction_depth_cols))} columns to sensors A, B, C")
    
    def update_sensor_assignment(self, column, sensor):
        """Update sensor assignment for a column"""
        if sensor == 'None':
            if column in self.correction_sensor_assignments:
                del self.correction_sensor_assignments[column]
        else:
            self.correction_sensor_assignments[column] = sensor
    
    def load_correction_calibration(self):
        """Load JSON calibration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Offset Calibration", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                self.correction_calibration = json.load(f)
            
            self.correction_calib_label.setText(f"Loaded: {Path(file_path).name}")
            self.correction_log_msg(f"Loaded calibration from: {Path(file_path).name}")
            
            # Display regression info
            if 'regressions' in self.correction_calibration:
                for reg in self.correction_calibration['regressions']:
                    self.correction_log_msg(
                        f"  {reg['sensor_j']} - {reg['sensor_i']}: "
                        f"slope={reg['slope']:.6f}, intercept={reg['intercept']:.6f}, "
                        f"R²={reg['r_squared']:.4f}"
                    )
            
            # Update correction plan
            self.update_correction_plan()
            
            self.correction_log_msg(f"\nCurrent reference: Sensor {self.ref_sensor_combo.currentText()}")
            self.correction_log_msg("→ Check 'Regression Corrections' section to see which sensors will be corrected")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load calibration:\n{str(e)}")
            self.correction_log_msg(f"ERROR loading calibration: {str(e)}")
    
    def update_correction_plan(self, ref_sensor=None):
        """Update the display showing which corrections will be applied"""
        # Clear previous plan
        while self.correction_plan_layout.count():
            item = self.correction_plan_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.correction_checkboxes = {}
        
        if not self.correction_calibration or 'regressions' not in self.correction_calibration:
            self.correction_plan_label = QLabel("Load calibration to see correction plan")
            self.correction_plan_label.setWordWrap(True)
            self.correction_plan_label.setStyleSheet("QLabel { color: gray; font-style: italic; }")
            self.correction_plan_layout.addWidget(self.correction_plan_label)
            return
        
        ref_sensor = self.ref_sensor_combo.currentText()
        
        # Find applicable regressions for this reference sensor
        applicable = []
        for reg in self.correction_calibration['regressions']:
            sensor_i = reg['sensor_i']
            sensor_j = reg['sensor_j']
            
            # Check if this regression involves the reference sensor
            if sensor_i == ref_sensor:
                # Regression is (REF - X), so we correct X
                applicable.append({
                    'target': sensor_j,
                    'regression': f"{sensor_j} - {sensor_i}",
                    'sign': -1,  # Subtract offset
                    'reg': reg
                })
            elif sensor_j == ref_sensor:
                # Regression is (X - REF), so we correct X
                applicable.append({
                    'target': sensor_i,
                    'regression': f"{sensor_j} - {sensor_i}",
                    'sign': 1,  # Add offset
                    'reg': reg
                })
        
        if not applicable:
            label = QLabel(f"No regressions found for reference sensor {ref_sensor}")
            label.setWordWrap(True)
            label.setStyleSheet("QLabel { color: orange; }")
            self.correction_plan_layout.addWidget(label)
            return
        
        # Show applicable corrections with checkboxes
        info_label = QLabel(f"<b>Reference: Sensor {ref_sensor}</b> (will not be modified)")
        self.correction_plan_layout.addWidget(info_label)
        
        for item in applicable:
            # Calculate example offset at typical depth for tooltip
            example_depth = 250.0
            example_offset = item['reg']['slope'] * example_depth + item['reg']['intercept']
            example_correction = item['sign'] * example_offset
            
            checkbox = QCheckBox(
                f"Apply to Sensor {item['target']}: "
                f"{item['reg']['sensor_j']}-{item['reg']['sensor_i']} "
                f"(~{example_correction:+.3f}m @ 250m)"
            )
            checkbox.setChecked(True)
            checkbox.setToolTip(
                f"Regression: {item['reg']['sensor_j']} - {item['reg']['sensor_i']}\n"
                f"Slope: {item['reg']['slope']:.6f}\n"
                f"Intercept: {item['reg']['intercept']:.6f}\n"
                f"R²: {item['reg']['r_squared']:.4f}\n\n"
                f"Example @ 250m depth:\n"
                f"Offset = {example_offset:.4f}m\n"
                f"Correction = {example_correction:+.4f}m"
            )
            self.correction_checkboxes[item['target']] = {
                'checkbox': checkbox,
                'reg': item['reg'],
                'sign': item['sign']
            }
            self.correction_plan_layout.addWidget(checkbox)
    
    def plot_correction_original(self):
        """Plot original uncorrected data"""
        if self.correction_df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
        
        self.correction_plot.clear()
        self.correction_plot.addLegend()
        self.correction_plot.addItem(self.correction_vline, ignoreBounds=True)
        self.correction_plot.addItem(self.correction_hline, ignoreBounds=True)
        
        colors = ['#6baed6', '#74c476', '#fd8d3c']
        df = self.correction_df
        x = df['datetime'].astype('int64') / 1e9  # seconds since epoch
        
        for col in self.correction_depth_cols:
            if col in self.correction_sensor_assignments:
                sensor = self.correction_sensor_assignments[col]
                short_name = col.split('_', 1)[0]
                color_idx = ord(sensor) - ord('A')
                pen = pg.mkPen(color=colors[color_idx % len(colors)], width=2)
                self.correction_plot.plot(x, df[col].values, pen=pen, 
                                         name=f"{short_name} (Sensor {sensor})",
                                         connect='finite', skipFiniteCheck=True)
        
        # Auto-range to show all data
        self.correction_plot.autoRange()
        
        self.correction_log_msg("Plotted original data")
    
    def apply_corrections_and_plot(self):
        """Apply regression and manual corrections, then plot"""
        if self.correction_df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
        
        if not self.correction_sensor_assignments:
            QMessageBox.warning(self, "Warning", "Please assign columns to sensors")
            return
        
        try:
            corrected_df = self.correction_df.copy()
            ref_sensor = self.ref_sensor_combo.currentText()
            
            # Force update correction plan to ensure it matches current reference
            if not self.correction_checkboxes:
                self.update_correction_plan()
            
            self.correction_log_msg(f"\n{'='*50}")
            self.correction_log_msg(f"APPLYING CORRECTIONS")
            self.correction_log_msg(f"Reference: Sensor {ref_sensor} (will NOT be modified)")
            self.correction_log_msg(f"{'='*50}")
            
            # Get reference column for depth calculation
            ref_col = None
            for col, assigned in self.correction_sensor_assignments.items():
                if assigned == ref_sensor:
                    ref_col = col
                    break
            
            # Apply regression corrections based on checkboxes (DEPTH-DEPENDENT)
            if self.correction_checkboxes:
                if not ref_col:
                    QMessageBox.warning(self, "Warning", 
                                      f"Reference sensor {ref_sensor} not found in data. Cannot apply corrections.")
                    return
                
                self.correction_log_msg(f"\nApplying depth-dependent corrections...")
                
                for target_sensor, info in self.correction_checkboxes.items():
                    if not info['checkbox'].isChecked():
                        continue
                    
                    # Find column for this sensor
                    target_col = None
                    for col, assigned in self.correction_sensor_assignments.items():
                        if assigned == target_sensor:
                            target_col = col
                            break
                    
                    if not target_col:
                        continue
                    
                    reg = info['reg']
                    slope = reg['slope']
                    intercept = reg['intercept']
                    sign = info['sign']
                    
                    # Calculate offset at EACH depth point using reference sensor depth
                    ref_depths = corrected_df[ref_col].values
                    offsets = slope * ref_depths + intercept
                    corrections = sign * offsets
                    
                    # Apply point-by-point correction
                    corrected_df[target_col] = corrected_df[target_col] + corrections
                    
                    # Report statistics
                    mean_correction = corrections.mean()
                    std_correction = corrections.std()
                    min_correction = corrections.min()
                    max_correction = corrections.max()
                    
                    self.correction_log_msg(
                        f"\nSensor {target_sensor}: Applied regression {reg['sensor_j']}-{reg['sensor_i']}"
                    )
                    self.correction_log_msg(
                        f"  Correction range: {min_correction:+.4f}m to {max_correction:+.4f}m"
                    )
                    self.correction_log_msg(
                        f"  Mean: {mean_correction:+.4f}m ± {std_correction:.4f}m"
                    )
            
            # Apply manual offsets
            manual_applied = False
            for col, sensor in self.correction_sensor_assignments.items():
                manual_offset = self.manual_offset_spinboxes[sensor].value()
                if manual_offset != 0:
                    if not manual_applied:
                        self.correction_log_msg(f"\nManual offsets:")
                        manual_applied = True
                    corrected_df[col] = corrected_df[col] + manual_offset
                    self.correction_log_msg(
                        f"  Sensor {sensor}: {manual_offset:+.4f}m"
                    )
            
            # Plot corrected data
            self.correction_plot.clear()
            self.correction_plot.addLegend()
            self.correction_plot.addItem(self.correction_vline, ignoreBounds=True)
            self.correction_plot.addItem(self.correction_hline, ignoreBounds=True)
            
            colors = ['#6baed6', '#74c476', '#fd8d3c']
            x = corrected_df['datetime'].astype('int64') / 1e9
            
            for col in self.correction_depth_cols:
                if col in self.correction_sensor_assignments:
                    sensor = self.correction_sensor_assignments[col]
                    short_name = col.split('_', 1)[0]
                    color_idx = ord(sensor) - ord('A')
                    pen = pg.mkPen(color=colors[color_idx % len(colors)], width=2)
                    self.correction_plot.plot(x, corrected_df[col].values, pen=pen,
                                             name=f"{short_name} (Sensor {sensor} corrected)",
                                             connect='finite', skipFiniteCheck=True)
            
            # Auto-range to show all data
            self.correction_plot.autoRange()
            
            # Store corrected data for export and update shared dataframe
            self.correction_df_corrected = corrected_df
            self.shared_df = corrected_df.copy()
            
            self.correction_log_msg("Corrections applied and plotted!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply corrections:\n{str(e)}")
            self.correction_log_msg(f"ERROR: {str(e)}")
    
    def export_corrected_data(self):
        """Export corrected data to CSV"""
        if not hasattr(self, 'correction_df_corrected'):
            QMessageBox.warning(self, "Warning", "Please apply corrections first")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Corrected Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            self.correction_df_corrected.to_csv(file_path, index=False)
            self.correction_log_msg(f"Exported corrected data to: {Path(file_path).name}")
            QMessageBox.information(self, "Success", "Corrected data exported successfully!")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
            self.correction_log_msg(f"ERROR exporting: {str(e)}")
    
    def correction_mouse_moved(self, evt):
        """Handle mouse movement for crosshairs and info display"""
        if self.correction_df is None:
            return
        
        pos = evt[0]
        if self.correction_plot.sceneBoundingRect().contains(pos):
            mousePoint = self.correction_plot.plotItem.vb.mapSceneToView(pos)
            x_pos = mousePoint.x()
            y_pos = mousePoint.y()
            
            # Update crosshair lines and make visible
            self.correction_vline.setPos(x_pos)
            self.correction_hline.setPos(y_pos)
            self.correction_vline.setVisible(True)
            self.correction_hline.setVisible(True)
            
            # Find nearest data point
            df = self.correction_df_corrected if hasattr(self, 'correction_df_corrected') else self.correction_df
            timestamps = df['datetime'].astype('int64') / 1e9
            
            # Find closest timestamp
            idx = int((timestamps - x_pos).abs().argmin())
            
            # Update info label with datetime and actual depth values (matching first tab style)
            import datetime
            actual_time = df['datetime'].iloc[idx]
            x_label = actual_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Build info text in same format as first tab
            info_text = f"Index: {idx}  |  Time: {x_label}  |  "
            
            # Show actual values for assigned sensors at this data point
            for col, sensor in sorted(self.correction_sensor_assignments.items(), key=lambda x: x[1]):
                if col in df.columns:
                    short_name = col.split('_', 1)[0]
                    depth_val = df[col].iloc[idx]
                    if pd.notna(depth_val):
                        info_text += f"Sensor {sensor}: {depth_val:.3f}m  "
            
            self.correction_info_label.setText(info_text)


    # ========================================================================
    # TAB 3: TIME LAG CORRECTION
    # ========================================================================
    
    def create_time_lag_tab(self):
        """Create the time lag correction interface"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel
        left_panel = self.create_timelag_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (plots)
        right_panel = self.create_timelag_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 1000])
        
        return tab
    
    def create_timelag_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        
        # Data source
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout()
        self.timelag_source_label = QLabel("No data loaded")
        self.timelag_source_label.setWordWrap(True)
        source_layout.addWidget(self.timelag_source_label)
        
        use_tab1_btn = QPushButton("Use Data from Tab 1")
        use_tab1_btn.clicked.connect(self.use_tab1_data_for_timelag)
        source_layout.addWidget(use_tab1_btn)
        
        use_tab2_btn = QPushButton("Use Data from Tab 2")
        use_tab2_btn.clicked.connect(self.use_tab2_data_for_timelag)
        source_layout.addWidget(use_tab2_btn)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Savgol parameters
        savgol_group = QGroupBox("Savitzky-Golay Filter Parameters")
        savgol_layout = QGridLayout()
        
        savgol_layout.addWidget(QLabel("Window Length:"), 0, 0)
        self.timelag_window_spin = QSpinBox()
        self.timelag_window_spin.setRange(3, 1001)
        self.timelag_window_spin.setSingleStep(2)  # Must be odd
        self.timelag_window_spin.setValue(51)
        savgol_layout.addWidget(self.timelag_window_spin, 0, 1)
        
        savgol_layout.addWidget(QLabel("Polynomial Order:"), 1, 0)
        self.timelag_polyorder_spin = QSpinBox()
        self.timelag_polyorder_spin.setRange(1, 10)
        self.timelag_polyorder_spin.setValue(3)
        savgol_layout.addWidget(self.timelag_polyorder_spin, 1, 1)
        
        savgol_group.setLayout(savgol_layout)
        layout.addWidget(savgol_group)
        
        # Reference sensor
        ref_group = QGroupBox("Reference Sensor")
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference:"))
        self.timelag_ref_combo = QComboBox()
        self.timelag_ref_combo.addItems(['A', 'B', 'C'])
        ref_layout.addWidget(self.timelag_ref_combo)
        ref_layout.addStretch()
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)
        
        # Selection controls
        selection_group = QGroupBox("Time Range Selection")
        selection_layout = QVBoxLayout()
        
        self.timelag_selection_btn = QPushButton("Enable Selection Mode")
        self.timelag_selection_btn.setCheckable(True)
        self.timelag_selection_btn.clicked.connect(self.toggle_timelag_selection)
        self.timelag_selection_btn.setStyleSheet(
            "QPushButton:checked { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        selection_layout.addWidget(self.timelag_selection_btn)
        
        help_label = QLabel("Enable Selection Mode, then double-click\non plot to select time range for analysis")
        help_label.setStyleSheet("QLabel { color: blue; font-style: italic; }")
        help_label.setWordWrap(True)
        selection_layout.addWidget(help_label)
        
        self.timelag_selection_label = QLabel("No selection")
        self.timelag_selection_label.setWordWrap(True)
        selection_layout.addWidget(self.timelag_selection_label)
        
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self.clear_timelag_selection)
        selection_layout.addWidget(clear_btn)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Calculated offsets display
        offset_group = QGroupBox("Calculated Time Offsets")
        offset_layout = QVBoxLayout()
        self.timelag_offset_text = QTextEdit()
        self.timelag_offset_text.setReadOnly(True)
        self.timelag_offset_text.setMaximumHeight(150)
        self.timelag_offset_text.setText("Click 'Calculate Offsets' to compute time lags")
        offset_layout.addWidget(self.timelag_offset_text)
        offset_group.setLayout(offset_layout)
        layout.addWidget(offset_group)
        
        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        plot_btn = QPushButton("Plot Original Data")
        plot_btn.clicked.connect(self.plot_timelag_original)
        action_layout.addWidget(plot_btn)
        
        calc_btn = QPushButton("Calculate Offsets")
        calc_btn.clicked.connect(self.calculate_time_offsets)
        action_layout.addWidget(calc_btn)
        
        apply_btn = QPushButton("Apply Correction")
        apply_btn.clicked.connect(self.apply_time_correction)
        action_layout.addWidget(apply_btn)
        
        export_btn = QPushButton("Export Corrected Data")
        export_btn.clicked.connect(self.export_timelag_corrected)
        action_layout.addWidget(export_btn)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.timelag_log = QTextEdit()
        self.timelag_log.setReadOnly(True)
        self.timelag_log.setMaximumHeight(150)
        log_layout.addWidget(self.timelag_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll
    
    def create_timelag_right_panel(self):
        """Create right panel with two resizable plots"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create vertical splitter for two plots
        self.timelag_splitter = QSplitter(Qt.Vertical)
        
        # Top plot: Depth signals
        self.timelag_depth_plot = pg.PlotWidget()
        self.timelag_depth_plot.setBackground('w')
        self.timelag_depth_plot.showGrid(x=True, y=True, alpha=0.3)
        self.timelag_depth_plot.addLegend()
        self.timelag_depth_plot.getViewBox().invertY(True)
        self.timelag_depth_plot.setLabel('left', 'Depth (m)')
        self.timelag_depth_plot.setLabel('bottom', 'Time')
        
        # Crosshairs for depth plot
        self.timelag_depth_vline = pg.InfiniteLine(angle=90, movable=False, 
                                                   pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.timelag_depth_hline = pg.InfiniteLine(angle=0, movable=False,
                                                   pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.timelag_depth_plot.addItem(self.timelag_depth_vline, ignoreBounds=True)
        self.timelag_depth_plot.addItem(self.timelag_depth_hline, ignoreBounds=True)
        self.timelag_depth_vline.setVisible(False)
        self.timelag_depth_hline.setVisible(False)
        
        # Bottom plot: Velocity differences
        self.timelag_vel_plot = pg.PlotWidget()
        self.timelag_vel_plot.setBackground('w')
        self.timelag_vel_plot.showGrid(x=True, y=True, alpha=0.3)
        self.timelag_vel_plot.addLegend()
        self.timelag_vel_plot.setLabel('left', 'Velocity Difference (m/s)')
        self.timelag_vel_plot.setLabel('bottom', 'Time')
        
        # Crosshairs for velocity plot
        self.timelag_vel_vline = pg.InfiniteLine(angle=90, movable=False,
                                                pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.timelag_vel_hline = pg.InfiniteLine(angle=0, movable=False,
                                                pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.timelag_vel_plot.addItem(self.timelag_vel_vline, ignoreBounds=True)
        self.timelag_vel_plot.addItem(self.timelag_vel_hline, ignoreBounds=True)
        self.timelag_vel_vline.setVisible(False)
        self.timelag_vel_hline.setVisible(False)
        
        # Info labels
        self.timelag_depth_info = QLabel('')
        self.timelag_depth_info.setStyleSheet('QLabel { background-color: white; padding: 5px; border: 1px solid gray; }')
        self.timelag_vel_info = QLabel('')
        self.timelag_vel_info.setStyleSheet('QLabel { background-color: white; padding: 5px; border: 1px solid gray; }')
        
        # Add to splitter
        depth_container = QWidget()
        depth_layout = QVBoxLayout(depth_container)
        depth_layout.addWidget(self.timelag_depth_info)
        depth_layout.addWidget(self.timelag_depth_plot)
        depth_layout.setContentsMargins(0, 0, 0, 0)
        
        vel_container = QWidget()
        vel_layout = QVBoxLayout(vel_container)
        vel_layout.addWidget(self.timelag_vel_info)
        vel_layout.addWidget(self.timelag_vel_plot)
        vel_layout.setContentsMargins(0, 0, 0, 0)
        
        self.timelag_splitter.addWidget(depth_container)
        self.timelag_splitter.addWidget(vel_container)
        
        # Set initial sizes (60% depth, 40% velocity)
        self.timelag_splitter.setSizes([600, 400])
        
        layout.addWidget(self.timelag_splitter)
        
        # Connect mouse events
        self.timelag_depth_mouse_proxy = pg.SignalProxy(
            self.timelag_depth_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.timelag_depth_mouse_moved
        )
        self.timelag_vel_mouse_proxy = pg.SignalProxy(
            self.timelag_vel_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.timelag_vel_mouse_moved
        )
        self.timelag_depth_plot.scene().sigMouseClicked.connect(self.timelag_mouse_clicked)
        
        return panel
    
    def timelag_log_msg(self, msg):
        """Add message to timelag tab log"""
        self.timelag_log.append(msg)
        self.timelag_log.verticalScrollBar().setValue(
            self.timelag_log.verticalScrollBar().maximum()
        )
        QApplication.processEvents()
    
    def use_tab1_data_for_timelag(self):
        """Use data from Tab 1 for time lag correction"""
        if self.shared_df is None:
            QMessageBox.warning(self, "Warning", "No data available from Tab 1. Process data in Tab 1 first.")
            return
        
        self.timelag_df = self.shared_df.copy()
        self.timelag_source_label.setText("Using data from Tab 1")
        self.timelag_log_msg("Loaded data from Tab 1")
        self.timelag_log_msg(f"Data shape: {self.timelag_df.shape}")
    
    def use_tab2_data_for_timelag(self):
        """Use data from Tab 2 for time lag correction"""
        if self.shared_df is None:
            QMessageBox.warning(self, "Warning", "No data available from Tab 2. Apply corrections in Tab 2 first.")
            return
        
        try:
            datetime_col = self.datetime_col_edit.text()
            
            # Make sure datetime column exists and is properly formatted
            if datetime_col not in self.shared_df.columns:
                raise ValueError(f"Datetime column '{datetime_col}' not found in shared data")
            
            self.timelag_df = self.shared_df.copy()
            
            # Ensure datetime is properly parsed
            if not pd.api.types.is_datetime64_any_dtype(self.timelag_df[datetime_col]):
                self.timelag_df[datetime_col] = pd.to_datetime(self.timelag_df[datetime_col])
            
            # Find actual depth columns in the data
            depth_cols = [col for col in self.timelag_df.columns if 'Depth' in col and col != datetime_col]
            
            # Rename to standard format for Tab 3
            if len(depth_cols) > 0:
                rename_dict = {}
                for i, col in enumerate(depth_cols[:self.num_sensors]):
                    new_name = f'Sensor_{self.sensor_labels[i]}_Depth'
                    rename_dict[col] = new_name
                
                self.timelag_df = self.timelag_df.rename(columns=rename_dict)
                self.timelag_log_msg(f"Renamed columns: {rename_dict}")
            
            self.timelag_source_label.setText("Using data from Tab 2 (depth-corrected)")
            self.timelag_log_msg("Loaded data from Tab 2")
            self.timelag_log_msg(f"Data shape: {self.timelag_df.shape}")
            self.timelag_log_msg(f"Columns: {list(self.timelag_df.columns)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")
            self.timelag_log_msg(f"ERROR: {str(e)}")
    
    def toggle_timelag_selection(self):
        """Toggle selection mode for time lag analysis"""
        if self.timelag_selection_btn.isChecked():
            self.timelag_depth_plot.plotItem.vb.setMouseEnabled(x=False, y=False)
            self.timelag_selection_btn.setText("Disable Selection Mode")
            self.timelag_log_msg("Selection mode ENABLED - double-click to select range")
        else:
            self.timelag_depth_plot.plotItem.vb.setMouseEnabled(x=True, y=True)
            self.timelag_selection_btn.setText("Enable Selection Mode")
            self.time_selecting = False
            self.timelag_log_msg("Selection mode DISABLED - pan/zoom enabled")
    
    def clear_timelag_selection(self):
        """Clear time range selection"""
        self.time_selection_start_idx = None
        self.time_selection_end_idx = None
        self.time_selecting = False
        self.timelag_selection_label.setText("No selection")
        self.timelag_log_msg("Selection cleared")
        
        # Replot without selection region
        if hasattr(self, 'timelag_df') and self.timelag_df is not None:
            self.plot_timelag_original()
    
    def timelag_mouse_clicked(self, event):
        """Handle mouse clicks for time range selection"""
        if not hasattr(self, 'timelag_df') or self.timelag_df is None:
            return
        
        if not self.timelag_selection_btn.isChecked():
            return
        
        if event.double():
            pos = event.scenePos()
            mouse_point = self.timelag_depth_plot.plotItem.vb.mapSceneToView(pos)
            x = mouse_point.x()
            
            if not self.time_selecting:
                # Start selection
                self.time_selecting = True
                self.time_selection_start_x = x
                self.timelag_log_msg(f"Selection started at x={x:.2f}")
            else:
                # End selection
                self.time_selecting = False
                
                start_x = min(self.time_selection_start_x, x)
                end_x = max(self.time_selection_start_x, x)
                
                # Convert to indices
                datetime_col = self.datetime_col_edit.text()
                timestamps = self.timelag_df[datetime_col].astype('int64') / 1e9
                
                self.time_selection_start_idx = int((timestamps - start_x).abs().argmin())
                self.time_selection_end_idx = int((timestamps - end_x).abs().argmin())
                
                n_points = self.time_selection_end_idx - self.time_selection_start_idx + 1
                start_time = self.timelag_df[datetime_col].iloc[self.time_selection_start_idx]
                end_time = self.timelag_df[datetime_col].iloc[self.time_selection_end_idx]
                
                self.timelag_selection_label.setText(
                    f"Selected:\nRows {self.time_selection_start_idx} to {self.time_selection_end_idx}\n"
                    f"({n_points} points)\n"
                    f"{start_time} to {end_time}"
                )
                self.timelag_log_msg(f"Selection: rows {self.time_selection_start_idx}-{self.time_selection_end_idx}")
                
                # Replot with selection region
                self.plot_timelag_original()
    
    def plot_timelag_original(self):
        """Plot original data in time lag tab"""
        if not hasattr(self, 'timelag_df') or self.timelag_df is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        try:
            datetime_col = self.datetime_col_edit.text()
            sensor_col_names = self.get_sensor_col_names()
            
            self.timelag_log_msg(f"Plotting data...")
            self.timelag_log_msg(f"Datetime column: {datetime_col}")
            self.timelag_log_msg(f"Sensor columns: {sensor_col_names}")
            
            # Verify columns exist
            missing_cols = [col for col in sensor_col_names if col not in self.timelag_df.columns]
            if missing_cols:
                self.timelag_log_msg(f"WARNING: Missing columns: {missing_cols}")
            
            # Clear plots
            self.timelag_depth_plot.clear()
            self.timelag_depth_plot.addLegend()
            self.timelag_depth_plot.addItem(self.timelag_depth_vline, ignoreBounds=True)
            self.timelag_depth_plot.addItem(self.timelag_depth_hline, ignoreBounds=True)
            
            self.timelag_vel_plot.clear()
            self.timelag_vel_plot.addLegend()
            self.timelag_vel_plot.addItem(self.timelag_vel_vline, ignoreBounds=True)
            self.timelag_vel_plot.addItem(self.timelag_vel_hline, ignoreBounds=True)
            
            colors = ['#6baed6', '#74c476', '#fd8d3c']
            
            # Ensure datetime is in correct format
            if not pd.api.types.is_datetime64_any_dtype(self.timelag_df[datetime_col]):
                self.timelag_df[datetime_col] = pd.to_datetime(self.timelag_df[datetime_col])
            
            x = self.timelag_df[datetime_col].astype('int64') / 1e9
            
            self.timelag_log_msg(f"X data range: {x.min()} to {x.max()}")
            
            # Plot depth signals
            plots_added = 0
            for i, col in enumerate(sensor_col_names):
                if col in self.timelag_df.columns:
                    y_data = self.timelag_df[col].values
                    
                    # Check for valid data
                    valid_count = np.sum(~np.isnan(y_data))
                    self.timelag_log_msg(f"Sensor {self.sensor_labels[i]}: {valid_count} valid points")
                    
                    if valid_count > 0:
                        pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                        self.timelag_depth_plot.plot(x, y_data, pen=pen,
                                                    name=f'Sensor {self.sensor_labels[i]}',
                                                    connect='finite', skipFiniteCheck=True)
                        plots_added += 1
            
            if plots_added == 0:
                self.timelag_log_msg("ERROR: No valid data to plot!")
                QMessageBox.warning(self, "Warning", "No valid data found to plot")
                return
            
            # Add selection region if exists
            if self.time_selection_start_idx is not None and self.time_selection_end_idx is not None:
                # Use index values directly instead of time values
                start_time = self.timelag_df[datetime_col].iloc[self.time_selection_start_idx]
                end_time = self.timelag_df[datetime_col].iloc[self.time_selection_end_idx]
                
                # Convert to x-axis coordinates (seconds since epoch)
                start_x = start_time.value / 1e9  # pandas Timestamp to seconds
                end_x = end_time.value / 1e9
                
                selection_region = pg.LinearRegionItem(
                    values=[start_x, end_x],
                    brush=pg.mkBrush(255, 0, 0, 50),
                    movable=False
                )
                self.timelag_depth_plot.addItem(selection_region)
                
                # Make sure region is visible in current view
                self.timelag_depth_plot.setXRange(
                    start_x - (end_x - start_x) * 0.2,  # 20% padding
                    end_x + (end_x - start_x) * 0.2,
                    padding=0
                )
                
                self.timelag_depth_plot.autoRange()
                self.timelag_log_msg(f"Plotted {plots_added} sensors successfully")
            
        except Exception as e:
            self.timelag_log_msg(f"ERROR plotting: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to plot:\n{str(e)}")
            import traceback
            self.timelag_log_msg(traceback.format_exc())
    
    def calculate_time_offsets(self):
        """Calculate time offsets using Savgol velocity difference method"""
        if not hasattr(self, 'timelag_df') or self.timelag_df is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        if self.time_selection_start_idx is None or self.time_selection_end_idx is None:
            QMessageBox.warning(self, "Warning", "Please select a time range first")
            return
        
        try:
            self.timelag_log_msg("\n" + "="*50)
            self.timelag_log_msg("CALCULATING TIME OFFSETS")
            self.timelag_log_msg("="*50)
            
            datetime_col = self.datetime_col_edit.text()
            sensor_col_names = self.get_sensor_col_names()
            
            # Get selected data
            df_selected = self.timelag_df.iloc[self.time_selection_start_idx:self.time_selection_end_idx+1].copy()
            df_selected = df_selected.reset_index(drop=True)
            
            # Apply Savgol filter
            window_length = self.timelag_window_spin.value()
            polyorder = self.timelag_polyorder_spin.value()
            
            # Ensure window length is odd
            if window_length % 2 == 0:
                window_length += 1
                self.timelag_window_spin.setValue(window_length)
            
            self.timelag_log_msg(f"Savgol filter: window={window_length}, polyorder={polyorder}")
            
            for col in sensor_col_names:
                if col in df_selected.columns:
                    df_selected[f'{col}_filtered'] = savgol_filter(
                        df_selected[col], window_length=window_length, polyorder=polyorder
                    )
            
            # Calculate time differences
            if datetime_col not in df_selected.columns:
                raise ValueError(f"Datetime column '{datetime_col}' not found")
            
            df_selected[datetime_col] = pd.to_datetime(df_selected[datetime_col])
            time_diff = df_selected[datetime_col].diff().dt.total_seconds()
            
            # Get reference sensor
            ref_sensor_label = self.timelag_ref_combo.currentText()
            ref_idx = self.sensor_labels.index(ref_sensor_label)
            ref_col = sensor_col_names[ref_idx]
            
            self.timelag_log_msg(f"Reference sensor: {ref_sensor_label} ({ref_col})")
            
            # Calculate offsets for each sensor
            def calculate_velocity_difference_rms(df_sub, ref_col, target_col, time_offset_seconds):
                """Calculate RMS of velocity difference with time offset"""
                shifted_index = df_sub.index + time_offset_seconds / time_diff.median()
                
                df_temp = pd.DataFrame({f'{target_col}_filtered': df_sub[f'{target_col}_filtered'].values},
                                      index=shifted_index)
                
                col_interp = df_temp[f'{target_col}_filtered'].reindex(
                    df_sub.index, method='nearest', tolerance=5
                )
                
                v_ref = df_sub[f'{ref_col}_filtered'].diff() / time_diff
                v_target = col_interp.diff() / time_diff
                
                rms = np.sqrt(np.nanmean((v_ref - v_target)**2))
                return rms
            
            # Calculate offsets
            max_offset = 5.0  # seconds
            time_step = 0.025  # seconds
            time_offsets_array = np.arange(-max_offset, max_offset + time_step, time_step)
            
            self.calculated_time_offsets = {}
            
            for i, col in enumerate(sensor_col_names):
                if i == ref_idx:
                    self.calculated_time_offsets[col] = 0.0
                    continue
                
                if col not in df_selected.columns:
                    continue
                
                self.timelag_log_msg(f"Calculating offset for {self.sensor_labels[i]}...")
                
                rms_values = [calculate_velocity_difference_rms(df_selected, ref_col, col, offset)
                             for offset in time_offsets_array]
                
                optimal_offset = time_offsets_array[np.argmin(rms_values)]
                optimal_rms = min(rms_values)
                
                self.calculated_time_offsets[col] = optimal_offset
                self.timelag_log_msg(f"  {self.sensor_labels[i]}: {optimal_offset:.3f}s (RMS: {optimal_rms:.6f})")
            
            # Display results
            self.timelag_offset_text.clear()
            self.timelag_offset_text.append("Calculated Time Offsets:")
            self.timelag_offset_text.append("="*40)
            for i, col in enumerate(sensor_col_names):
                if col in self.calculated_time_offsets:
                    offset = self.calculated_time_offsets[col]
                    if offset == 0.0:
                        self.timelag_offset_text.append(f"Sensor {self.sensor_labels[i]}: 0.000s (reference)")
                    else:
                        self.timelag_offset_text.append(f"Sensor {self.sensor_labels[i]}: {offset:+.3f}s")
            
            self.timelag_log_msg("="*50)
            self.timelag_log_msg("Offset calculation complete!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate offsets:\n{str(e)}")
            self.timelag_log_msg(f"ERROR: {str(e)}")
    
    def apply_time_correction(self):
        """Apply calculated time offsets to data"""
        if not hasattr(self, 'calculated_time_offsets') or not self.calculated_time_offsets:
            QMessageBox.warning(self, "Warning", "Please calculate offsets first")
            return
        
        try:
            self.timelag_log_msg("\n" + "="*50)
            self.timelag_log_msg("APPLYING TIME CORRECTIONS")
            self.timelag_log_msg("="*50)
            
            datetime_col = self.datetime_col_edit.text()
            sensor_col_names = self.get_sensor_col_names()
            
            corrected_df = self.timelag_df.copy()
            
            # Apply time shifts
            for col, offset in self.calculated_time_offsets.items():
                if offset != 0.0:
                    # Shift timestamps
                    df_shifted = pd.DataFrame(
                        {col: corrected_df[col].values},
                        index=corrected_df.index + offset / corrected_df[datetime_col].diff().dt.total_seconds().median()
                    )
                    
                    corrected_df[col] = df_shifted[col].reindex(
                        corrected_df.index, method='nearest', tolerance=5
                    )
                    
                    sensor_idx = sensor_col_names.index(col)
                    self.timelag_log_msg(f"Applied {offset:+.3f}s shift to Sensor {self.sensor_labels[sensor_idx]}")
            
            # Store corrected data
            self.timelag_df_corrected = corrected_df
            self.shared_df = corrected_df.copy()
            
            # Replot with corrected data
            self.plot_timelag_corrected()
            
            self.timelag_log_msg("="*50)
            self.timelag_log_msg("Time corrections applied!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply corrections:\n{str(e)}")
            self.timelag_log_msg(f"ERROR: {str(e)}")
    
    def plot_timelag_corrected(self):
        """Plot corrected data with velocity differences"""
        if not hasattr(self, 'timelag_df_corrected'):
            QMessageBox.warning(self, "Warning", "No corrected data available")
            return
        
        datetime_col = self.datetime_col_edit.text()
        sensor_col_names = self.get_sensor_col_names()
        
        # Clear plots
        self.timelag_depth_plot.clear()
        self.timelag_depth_plot.addLegend()
        self.timelag_depth_plot.addItem(self.timelag_depth_vline, ignoreBounds=True)
        self.timelag_depth_plot.addItem(self.timelag_depth_hline, ignoreBounds=True)
        
        self.timelag_vel_plot.clear()
        self.timelag_vel_plot.addLegend()
        self.timelag_vel_plot.addItem(self.timelag_vel_vline, ignoreBounds=True)
        self.timelag_vel_plot.addItem(self.timelag_vel_hline, ignoreBounds=True)
        
        colors = ['#6baed6', '#74c476', '#fd8d3c']
        x = self.timelag_df_corrected[datetime_col].astype('int64') / 1e9
        
        # Plot corrected depth signals
        for i, col in enumerate(sensor_col_names):
            if col in self.timelag_df_corrected.columns:
                offset = self.calculated_time_offsets.get(col, 0.0)
                label = f'Sensor {self.sensor_labels[i]} ({offset:+.3f}s)' if offset != 0 else f'Sensor {self.sensor_labels[i]} (ref)'
                pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                self.timelag_depth_plot.plot(x, self.timelag_df_corrected[col].values, pen=pen,
                                             name=label, connect='finite', skipFiniteCheck=True)
        
        # Calculate and plot velocity differences
        window_length = self.timelag_window_spin.value()
        polyorder = self.timelag_polyorder_spin.value()
        
        if window_length % 2 == 0:
            window_length += 1
        
        df_filtered = self.timelag_df_corrected.copy()
        for col in sensor_col_names:
            if col in df_filtered.columns:
                df_filtered[f'{col}_filtered'] = savgol_filter(
                    df_filtered[col], window_length=window_length, polyorder=polyorder
                )
        
        time_diff = df_filtered[datetime_col].diff().dt.total_seconds()
        
        ref_sensor_label = self.timelag_ref_combo.currentText()
        ref_idx = self.sensor_labels.index(ref_sensor_label)
        ref_col = sensor_col_names[ref_idx]
        
        v_ref = df_filtered[f'{ref_col}_filtered'].diff() / time_diff
        
        # Plot velocity differences for each sensor vs reference
        plot_idx = 0
        for i, col in enumerate(sensor_col_names):
            if i == ref_idx or col not in df_filtered.columns:
                continue
            
            v_sensor = df_filtered[f'{col}_filtered'].diff() / time_diff
            v_diff = v_sensor - v_ref
            
            pen = pg.mkPen(color=colors[plot_idx % len(colors)], width=2)
            self.timelag_vel_plot.plot(x, v_diff.values, pen=pen,
                                      name=f'{self.sensor_labels[i]} - {ref_sensor_label}',
                                      connect='finite', skipFiniteCheck=True)
            plot_idx += 1
        
        # Add zero line to velocity plot
        pen_zero = pg.mkPen('k', width=1, style=Qt.DashLine)
        self.timelag_vel_plot.plot([x.min(), x.max()], [0, 0], pen=pen_zero)
        
        self.timelag_depth_plot.autoRange()
        self.timelag_vel_plot.autoRange()
        
        self.timelag_log_msg("Plotted corrected data with velocity differences")
    
    def export_timelag_corrected(self):
        """Export time-corrected data"""
        if not hasattr(self, 'timelag_df_corrected'):
            QMessageBox.warning(self, "Warning", "No corrected data available")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Time-Corrected Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            self.timelag_df_corrected.to_csv(file_path, index=False)
            self.timelag_log_msg(f"Exported to: {Path(file_path).name}")
            QMessageBox.information(self, "Success", "Time-corrected data exported!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
            self.timelag_log_msg(f"ERROR: {str(e)}")

    def timelag_depth_mouse_moved(self, evt):
        """Handle mouse movement on depth plot"""
        if not hasattr(self, 'timelag_df') or self.timelag_df is None:
            return
        
        pos = evt[0]
        if self.timelag_depth_plot.sceneBoundingRect().contains(pos):
            mousePoint = self.timelag_depth_plot.plotItem.vb.mapSceneToView(pos)
            x_pos = mousePoint.x()
            y_pos = mousePoint.y()
            
            self.timelag_depth_vline.setPos(x_pos)
            self.timelag_depth_hline.setPos(y_pos)
            self.timelag_depth_vline.setVisible(True)
            self.timelag_depth_hline.setVisible(True)
            
            # Show info
            datetime_col = self.datetime_col_edit.text()
            sensor_col_names = self.get_sensor_col_names()
            
            df = self.timelag_df_corrected if hasattr(self, 'timelag_df_corrected') else self.timelag_df
            timestamps = df[datetime_col].astype('int64') / 1e9
            idx = int((timestamps - x_pos).abs().argmin())
            
            actual_time = df[datetime_col].iloc[idx]
            info_text = f"Index: {idx}  |  Time: {actual_time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
            
            for i, col in enumerate(sensor_col_names):
                if col in df.columns:
                    val = df[col].iloc[idx]
                    if pd.notna(val):
                        info_text += f"Sensor {self.sensor_labels[i]}: {val:.3f}m  "
            
            self.timelag_depth_info.setText(info_text)

    def timelag_vel_mouse_moved(self, evt):
        """Handle mouse movement on velocity plot"""
        if not hasattr(self, 'timelag_df_corrected'):
            return
        
        pos = evt[0]
        if self.timelag_vel_plot.sceneBoundingRect().contains(pos):
            mousePoint = self.timelag_vel_plot.plotItem.vb.mapSceneToView(pos)
            x_pos = mousePoint.x()
            y_pos = mousePoint.y()
            
            self.timelag_vel_vline.setPos(x_pos)
            self.timelag_vel_hline.setPos(y_pos)
            self.timelag_vel_vline.setVisible(True)
            self.timelag_vel_hline.setVisible(True)
            
            datetime_col = self.datetime_col_edit.text()
            timestamps = self.timelag_df_corrected[datetime_col].astype('int64') / 1e9
            idx = int((timestamps - x_pos).abs().argmin())
            
            actual_time = self.timelag_df_corrected[datetime_col].iloc[idx]
            info_text = f"Index: {idx}  |  Time: {actual_time.strftime('%Y-%m-%d %H:%M:%S')}  |  Velocity Diff: {y_pos:.6f} m/s"
            
            self.timelag_vel_info.setText(info_text)

def main():
    app = QApplication(sys.argv)
    window = CTDAnalyzerApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()