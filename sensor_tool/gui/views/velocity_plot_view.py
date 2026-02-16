"""
VelocityPlotView - Secondary plot for time offset mode.

Displays Savgol-filtered velocity differences between sensors
and a reference sensor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

from ...domain.models.sensor_data import SensorData

SENSOR_COLORS = ['#6baed6', '#74c476', '#fd8d3c', '#9e9ac8']


class VelocityPlotView(QWidget):
    """
    Secondary plot widget for velocity differences.
    Shows velocity difference vs time for each non-reference sensor.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info label
        self.info_label = QLabel('')
        self.info_label.setStyleSheet(
            'QLabel { background-color: white; padding: 5px; border: 1px solid gray; }'
        )
        layout.addWidget(self.info_label)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left', 'Velocity Difference (m/s)')
        self.plot_widget.setLabel('bottom', 'Time')

        # Crosshairs
        self._vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('r', width=1, style=Qt.DashLine)
        )
        self._hline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen('r', width=1, style=Qt.DashLine)
        )
        self.plot_widget.addItem(self._vline, ignoreBounds=True)
        self.plot_widget.addItem(self._hline, ignoreBounds=True)
        self._vline.setVisible(False)
        self._hline.setVisible(False)

        layout.addWidget(self.plot_widget)

        # Mouse tracking
        self._mouse_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        self._timestamps = None
        self._datetime_series = None

    def plot_velocity_differences(
        self,
        sensor_data: SensorData,
        velocity_diffs: dict[str, pd.Series],
        ref_sensor: str,
    ):
        """
        Plot velocity differences.
        
        Args:
            sensor_data: The source data (for timestamps).
            velocity_diffs: Dict mapping sensor_label -> velocity difference Series.
            ref_sensor: Reference sensor label (for legend).
        """
        self._clear_plot()

        x = sensor_data.get_timestamps_epoch()
        self._timestamps = x
        self._datetime_series = sensor_data.get_timestamps()

        color_idx = 0
        for label, v_diff in velocity_diffs.items():
            pen = pg.mkPen(
                color=SENSOR_COLORS[color_idx % len(SENSOR_COLORS)], width=2
            )
            self.plot_widget.plot(
                x, v_diff.values, pen=pen,
                name=f'{label} - {ref_sensor}',
                connect='finite', skipFiniteCheck=True,
            )
            color_idx += 1

        # Zero line
        pen_zero = pg.mkPen('k', width=1, style=Qt.DashLine)
        self.plot_widget.plot([x.min(), x.max()], [0, 0], pen=pen_zero)

        self.plot_widget.autoRange()

    def clear(self):
        self._clear_plot()
        self._timestamps = None
        self._datetime_series = None

    def _clear_plot(self):
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self.plot_widget.addItem(self._vline, ignoreBounds=True)
        self.plot_widget.addItem(self._hline, ignoreBounds=True)

    def _on_mouse_moved(self, evt):
        if self._timestamps is None:
            return

        pos = evt[0]
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return

        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x_pos = mouse_point.x()
        y_pos = mouse_point.y()

        self._vline.setPos(x_pos)
        self._hline.setPos(y_pos)
        self._vline.setVisible(True)
        self._hline.setVisible(True)

        idx = int(np.argmin(np.abs(self._timestamps - x_pos)))
        if self._datetime_series is not None and 0 <= idx < len(self._datetime_series):
            actual_time = self._datetime_series.iloc[idx]
            time_str = actual_time.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(actual_time) else 'N/A'
            self.info_label.setText(
                f"Index: {idx}  |  Time: {time_str}  |  Velocity Diff: {y_pos:.6f} m/s"
            )
