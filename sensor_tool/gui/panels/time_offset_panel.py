"""
TimeOffsetPanel - Control panel for the "Time Offset" mode.

Savgol filter parameters, reference sensor selection, time offset calculation.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout,
    QComboBox, QSpinBox, QTextEdit, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Signal

from .base_panel import BaseModePanel
from ..widgets.log_widget import LogWidget
from ..widgets.selection_controls import SelectionControls
from ...domain.models.analysis_result import TimeOffsetResult


class TimeOffsetPanel(BaseModePanel):
    """
    Panel for the Time Offset mode.
    
    Signals:
        load_file_requested(str): User selected a CSV file.
        calculate_offsets_requested: User clicked "Calculate Offsets".
        apply_correction_requested: User clicked "Apply Correction".
        reset_requested: User clicked "Reset".
        export_requested: User clicked "Export Corrected Data".
        plot_original_requested: User clicked "Plot Original Data".
        selection_mode_changed(bool): Selection mode toggled.
        clear_selection_requested: Clear selection.
    """

    load_file_requested = Signal(str)
    calculate_offsets_requested = Signal()
    apply_correction_requested = Signal()
    reset_requested = Signal()
    export_requested = Signal()
    plot_original_requested = Signal()
    selection_mode_changed = Signal(bool)
    clear_selection_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # -- Data Information --
        data_group = QGroupBox("Data Information")
        data_layout = QVBoxLayout()
        self.file_label = QLabel("No data loaded")
        self.file_label.setWordWrap(True)
        data_layout.addWidget(self.file_label)

        load_btn = QPushButton("Load CSV File")
        load_btn.clicked.connect(self._on_load_file)
        data_layout.addWidget(load_btn)

        data_group.setLayout(data_layout)
        self._layout.addWidget(data_group)

        # -- Savgol Filter --
        savgol_group = QGroupBox("Savitzky-Golay Filter Parameters")
        savgol_layout = QGridLayout()

        savgol_layout.addWidget(QLabel("Window Length:"), 0, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 1001)
        self.window_spin.setSingleStep(2)
        self.window_spin.setValue(51)
        savgol_layout.addWidget(self.window_spin, 0, 1)

        savgol_layout.addWidget(QLabel("Polynomial Order:"), 1, 0)
        self.polyorder_spin = QSpinBox()
        self.polyorder_spin.setRange(1, 10)
        self.polyorder_spin.setValue(3)
        savgol_layout.addWidget(self.polyorder_spin, 1, 1)

        savgol_group.setLayout(savgol_layout)
        self._layout.addWidget(savgol_group)

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
        note = QLabel("(Reference time will not change)")
        note.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        ref_layout.addWidget(note)
        ref_group.setLayout(ref_layout)
        self._layout.addWidget(ref_group)

        # -- Selection --
        self.selection_controls = SelectionControls("Time Range Selection")
        self.selection_controls.selection_mode_changed.connect(
            self.selection_mode_changed.emit
        )
        self.selection_controls.clear_requested.connect(
            self.clear_selection_requested.emit
        )
        self._layout.addWidget(self.selection_controls)

        # -- Calculated Offsets --
        offset_group = QGroupBox("Calculated Time Offsets")
        offset_layout = QVBoxLayout()
        self.offset_text = QTextEdit()
        self.offset_text.setReadOnly(True)
        self.offset_text.setMaximumHeight(150)
        self.offset_text.setText("Click 'Calculate Offsets' to compute time lags")
        offset_layout.addWidget(self.offset_text)
        offset_group.setLayout(offset_layout)
        self._layout.addWidget(offset_group)

        # -- Actions --
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()

        plot_btn = QPushButton("Plot Original Data")
        plot_btn.clicked.connect(self.plot_original_requested.emit)
        action_layout.addWidget(plot_btn)

        calc_btn = QPushButton("Calculate Offsets")
        calc_btn.clicked.connect(self.calculate_offsets_requested.emit)
        action_layout.addWidget(calc_btn)

        apply_btn = QPushButton("Apply Correction")
        apply_btn.clicked.connect(self.apply_correction_requested.emit)
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
        self.log_widget = LogWidget("Log", max_height=150)
        self._layout.addWidget(self.log_widget)

        self._finish_layout()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def update_file_info(self, filename: str, sensors: int, rows: int):
        self.file_label.setText(f"File: {filename}\nSensors: {sensors}  |  Rows: {rows:,}")

    def display_offsets(self, results: list[TimeOffsetResult]):
        """Update the offset display with calculation results."""
        self.offset_text.clear()
        self.offset_text.append("Calculated Time Offsets:")
        self.offset_text.append("=" * 40)
        for r in results:
            if r.is_reference:
                self.offset_text.append(f"Sensor {r.sensor_label}: 0.000s (reference)")
            else:
                self.offset_text.append(
                    f"Sensor {r.sensor_label}: {r.offset_seconds:+.3f}s  (RMS: {r.rms_value:.6f})"
                )

    def get_savgol_params(self) -> tuple[int, int]:
        """Return (window_length, polyorder)."""
        window = self.window_spin.value()
        if window % 2 == 0:
            window += 1
            self.window_spin.setValue(window)
        return window, self.polyorder_spin.value()

    def get_ref_sensor(self) -> str:
        return self.ref_sensor_combo.currentText()

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
