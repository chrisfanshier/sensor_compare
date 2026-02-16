"""
DepthOffsetPanel - Control panel for the "Depth Offset" mode.

Load data, load calibration, assign sensors, apply depth corrections.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout,
    QComboBox, QCheckBox, QDoubleSpinBox, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Signal

from .base_panel import BaseModePanel
from ..widgets.log_widget import LogWidget


class DepthOffsetPanel(BaseModePanel):
    """
    Panel for the Depth Offset mode.
    
    Signals:
        load_file_requested(str): User selected a CSV file.
        load_calibration_requested(str): User selected a JSON calibration file.
        apply_corrections_requested: User clicked "Apply Corrections".
        reset_requested: User clicked "Reset".
        export_requested: User clicked "Export Corrected Data".
        plot_original_requested: User clicked "Plot Original".
    """

    # Signals
    load_file_requested = Signal(str)
    load_calibration_requested = Signal(str)
    apply_corrections_requested = Signal()
    reset_requested = Signal()
    export_requested = Signal()
    plot_original_requested = Signal()
    sensor_assignments_changed = Signal()  # User changed a sensor assignment dropdown

    def __init__(self, parent=None):
        super().__init__(parent)

        # -- Data Information --
        data_group = QGroupBox("Data Information")
        data_layout = QVBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        data_layout.addWidget(self.file_label)

        self.data_info_label = QLabel("")
        self.data_info_label.setWordWrap(True)
        data_layout.addWidget(self.data_info_label)

        load_btn = QPushButton("Load CSV File")
        load_btn.clicked.connect(self._on_load_file)
        data_layout.addWidget(load_btn)

        data_group.setLayout(data_layout)
        self._layout.addWidget(data_group)

        # -- Calibration File --
        calib_group = QGroupBox("Calibration File")
        calib_layout = QVBoxLayout()
        self.calib_label = QLabel("No calibration loaded")
        self.calib_label.setWordWrap(True)
        calib_layout.addWidget(self.calib_label)

        calib_btn = QPushButton("Load JSON Calibration")
        calib_btn.clicked.connect(self._on_load_calibration)
        calib_layout.addWidget(calib_btn)

        calib_group.setLayout(calib_layout)
        self._layout.addWidget(calib_group)

        # -- Sensor Assignments --
        self.assignment_group = QGroupBox("Sensor Assignments")
        self.assignment_layout = QGridLayout()
        self.assignment_combos: list[QComboBox] = []
        self.assignment_group.setLayout(self.assignment_layout)
        self._layout.addWidget(self.assignment_group)

        # -- Reference Sensor --
        ref_group = QGroupBox("Reference Sensor")
        ref_layout = QVBoxLayout()
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference:"))
        self.ref_sensor_combo = QComboBox()
        self.ref_sensor_combo.addItems(['A', 'B', 'C'])
        ref_row.addWidget(self.ref_sensor_combo)
        ref_row.addStretch()
        ref_layout.addLayout(ref_row)
        note = QLabel("(Reference will not be modified)")
        note.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        ref_layout.addWidget(note)
        ref_group.setLayout(ref_layout)
        self._layout.addWidget(ref_group)

        # -- Regression Corrections --
        self.correction_plan_group = QGroupBox("Regression Corrections")
        self.correction_plan_layout = QVBoxLayout()
        self.correction_checkboxes: dict[str, dict] = {}
        plan_label = QLabel("Load calibration and select reference to see plan")
        plan_label.setWordWrap(True)
        plan_label.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        self.correction_plan_layout.addWidget(plan_label)
        self.correction_plan_group.setLayout(self.correction_plan_layout)
        self._layout.addWidget(self.correction_plan_group)

        # -- Manual Offset Corrections --
        manual_group = QGroupBox("Manual Depth Corrections")
        manual_layout = QGridLayout()
        self.manual_offset_spinboxes: dict[str, QDoubleSpinBox] = {}
        for i, label in enumerate(['A', 'B', 'C']):
            manual_layout.addWidget(QLabel(f"Sensor {label}:"), i, 0)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1000, 1000)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(4)
            spinbox.setValue(0.0)
            manual_layout.addWidget(spinbox, i, 1)
            manual_layout.addWidget(QLabel("m"), i, 2)
            self.manual_offset_spinboxes[label] = spinbox
        manual_group.setLayout(manual_layout)
        self._layout.addWidget(manual_group)

        # -- Actions --
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()

        plot_orig_btn = QPushButton("Plot Original Data")
        plot_orig_btn.clicked.connect(self.plot_original_requested.emit)
        action_layout.addWidget(plot_orig_btn)

        apply_btn = QPushButton("Apply Corrections & Plot")
        apply_btn.clicked.connect(self.apply_corrections_requested.emit)
        action_layout.addWidget(apply_btn)

        reset_btn = QPushButton("Reset to Original")
        reset_btn.clicked.connect(self.reset_requested.emit)
        action_layout.addWidget(reset_btn)

        export_btn = QPushButton("Export Corrected Data")
        export_btn.clicked.connect(self.export_requested.emit)
        action_layout.addWidget(export_btn)

        action_group.setLayout(action_layout)
        self._layout.addWidget(action_group)

        # -- Log --
        self.log_widget = LogWidget("Log", max_height=200)
        self._layout.addWidget(self.log_widget)

        self._finish_layout()

    # ------------------------------------------------------------------
    # Public helpers for controller to populate UI
    # ------------------------------------------------------------------

    def update_file_info(self, filename: str, sensors: int, rows: int, core_title: str = ''):
        text = f"File: {filename}\nSensors: {sensors}  |  Rows: {rows:,}"
        if core_title:
            text += f"\nCore: {core_title}"
        self.file_label.setText(text)

    def update_calibration_info(self, filename: str, info_lines: list[str]):
        self.calib_label.setText(f"Loaded: {filename}")
        for line in info_lines:
            self.log_widget.log(line)

    def setup_sensor_assignments(self, original_columns: list[str],
                                 current_assignments: dict[str, str]):
        """
        Populate the sensor assignment dropdowns.

        Args:
            original_columns: Original CSV depth column names
                              (e.g., 'Weight Stand_230406..._Depth (m)').
            current_assignments: Maps original_column_name -> sensor label.
        """
        # Clear previous
        while self.assignment_layout.count():
            item = self.assignment_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.assignment_combos = []

        self.assignment_layout.addWidget(QLabel("Column"), 0, 0)
        self.assignment_layout.addWidget(QLabel("Assign to"), 0, 1)

        sensor_options = ['None', 'A', 'B', 'C']
        for i, orig_col in enumerate(original_columns):
            row = i + 1
            # Extract location + serial for disambiguation
            parts = orig_col.split('_')
            location = parts[0]
            if len(parts) > 1:
                serial = parts[1].removesuffix('.rsk')
                short_name = f"{location} ({serial})"
            else:
                short_name = location
            label = QLabel(short_name)
            label.setToolTip(orig_col)
            self.assignment_layout.addWidget(label, row, 0)

            combo = QComboBox()
            combo.addItems(sensor_options)
            assigned = current_assignments.get(orig_col, 'None')
            combo.setCurrentText(assigned)
            combo.currentTextChanged.connect(
                lambda text, c=orig_col: self._on_assignment_changed(c, text)
            )
            self.assignment_layout.addWidget(combo, row, 1)
            self.assignment_combos.append((orig_col, combo))

    def _on_assignment_changed(self, column: str, sensor: str):
        """User changed a sensor assignment dropdown."""
        self.sensor_assignments_changed.emit()

    def update_correction_plan(self, applicable: list[dict]):
        """Update the correction plan checkboxes."""
        while self.correction_plan_layout.count():
            item = self.correction_plan_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.correction_checkboxes = {}

        if not applicable:
            label = QLabel("No applicable corrections found")
            label.setStyleSheet("QLabel { color: orange; }")
            self.correction_plan_layout.addWidget(label)
            return

        ref = self.ref_sensor_combo.currentText()
        info_label = QLabel(f"<b>Reference: Sensor {ref}</b> (will not be modified)")
        self.correction_plan_layout.addWidget(info_label)

        for item in applicable:
            reg = item['reg']
            example_depth = 250.0
            example_offset = reg.predict_offset(example_depth)
            example_correction = item['sign'] * example_offset

            checkbox = QCheckBox(
                f"Apply to Sensor {item['target']}: "
                f"{reg.sensor_j}-{reg.sensor_i} "
                f"(~{example_correction:+.3f}m @ 250m)"
            )
            checkbox.setChecked(True)
            checkbox.setToolTip(
                f"Regression: {reg.sensor_j} - {reg.sensor_i}\n"
                f"Slope: {reg.slope:.6f}\n"
                f"Intercept: {reg.intercept:.6f}\n"
                f"R²: {reg.r_squared:.4f}"
            )
            self.correction_checkboxes[item['target']] = {
                'checkbox': checkbox,
                'reg': reg,
                'sign': item['sign'],
            }
            self.correction_plan_layout.addWidget(checkbox)

    def get_sensor_assignments(self) -> dict[str, str]:
        """Return current sensor assignments: original_column_name -> sensor_label."""
        result = {}
        for orig_col, combo in self.assignment_combos:
            text = combo.currentText()
            if text != 'None':
                result[orig_col] = text
        return result

    def get_manual_offsets(self) -> dict[str, float]:
        """Return manual offsets: sensor_label -> offset_meters."""
        return {label: spin.value() for label, spin in self.manual_offset_spinboxes.items()}

    def get_enabled_calibration_targets(self) -> list[str]:
        """Return list of sensor labels whose calibration checkbox is checked."""
        return [
            label for label, info in self.correction_checkboxes.items()
            if info['checkbox'].isChecked()
        ]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.load_file_requested.emit(file_path)

    def _on_load_calibration(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Offset Calibration", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.load_calibration_requested.emit(file_path)
