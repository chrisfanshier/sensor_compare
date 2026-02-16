"""
HeavePlotView - Secondary plot for time offset mode.

Displays isolated heave profiles from the selected range.
Shows uncorrected heave after selection, then overlays
corrected/aligned heave after offset is applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

from ...domain.models.sensor_data import SensorData

SENSOR_COLORS = ['#6baed6', '#74c476', '#fd8d3c', '#9e9ac8']
REF_COLOR = '#333333'
CORRECTED_COLOR = '#e6550d'


class HeavePlotView(QWidget):
    """
    Secondary plot widget for heave profiles.

    Two display modes:
    - Uncorrected: ref heave + mov heave (original) for selected range.
    - Corrected: ref heave + mov heave (shifted) overlaid for visual verification.
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
        self.plot_widget.setLabel('left', 'Heave (m)')
        self.plot_widget.setLabel('bottom', 'Sample')

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

        self._x_axis: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_heave_uncorrected(
        self,
        heaves: dict[str, np.ndarray],
        ref_sensor: str,
        time_axis: np.ndarray | None = None,
    ):
        """
        Plot uncorrected heave profiles for the selected range.

        Args:
            heaves: Dict mapping sensor_label -> heave array.
            ref_sensor: Reference sensor label.
            time_axis: Optional x-axis values (e.g. relative seconds).
                       If None, uses sample index.
        """
        self._clear_plot()

        n_samples = max(len(h) for h in heaves.values()) if heaves else 0
        if time_axis is not None:
            x = time_axis
            self.plot_widget.setLabel('bottom', 'Time (s)')
        else:
            x = np.arange(n_samples, dtype=float)
            self.plot_widget.setLabel('bottom', 'Sample')
        self._x_axis = x

        self.plot_widget.setTitle('Isolated Heave — Uncorrected')

        color_idx = 0
        for label, heave in heaves.items():
            if label == ref_sensor:
                pen = pg.mkPen(color=REF_COLOR, width=2)
                name = f'Ref ({label})'
            else:
                pen = pg.mkPen(
                    color=SENSOR_COLORS[color_idx % len(SENSOR_COLORS)],
                    width=2, style=Qt.DashLine,
                )
                name = f'Sensor {label} (original)'
                color_idx += 1

            plot_x = x[:len(heave)] if len(heave) <= len(x) else x
            self.plot_widget.plot(
                plot_x, heave[:len(plot_x)], pen=pen, name=name,
                connect='finite', skipFiniteCheck=True,
            )

        # Zero line
        if len(x) > 0:
            pen_zero = pg.mkPen('k', width=1, style=Qt.DashLine)
            self.plot_widget.plot(
                [float(x[0]), float(x[-1])], [0, 0], pen=pen_zero,
            )

        self.plot_widget.autoRange()

    def plot_heave_corrected(
        self,
        heaves_original: dict[str, np.ndarray],
        heaves_corrected: dict[str, np.ndarray],
        ref_sensor: str,
        offsets: dict[str, float],
        time_axis: np.ndarray | None = None,
    ):
        """
        Plot corrected/aligned heave profiles overlaying the reference.

        Args:
            heaves_original: Uncorrected heave profiles.
            heaves_corrected: Corrected (time-shifted) heave profiles.
            ref_sensor: Reference sensor label.
            offsets: Dict mapping sensor_label -> offset_seconds.
            time_axis: Optional x-axis values.
        """
        self._clear_plot()

        n_samples = max(len(h) for h in heaves_original.values()) if heaves_original else 0
        if time_axis is not None:
            x = time_axis
            self.plot_widget.setLabel('bottom', 'Time (s)')
        else:
            x = np.arange(n_samples, dtype=float)
            self.plot_widget.setLabel('bottom', 'Sample')
        self._x_axis = x

        self.plot_widget.setTitle('Isolated Heave — Corrected / Aligned')

        # Plot reference heave
        if ref_sensor in heaves_original:
            ref_h = heaves_original[ref_sensor]
            plot_x = x[:len(ref_h)]
            pen = pg.mkPen(color=REF_COLOR, width=2)
            self.plot_widget.plot(
                plot_x, ref_h[:len(plot_x)], pen=pen,
                name=f'Ref ({ref_sensor})',
                connect='finite', skipFiniteCheck=True,
            )

        # Plot corrected heave for each non-ref sensor
        color_idx = 0
        for label, heave in heaves_corrected.items():
            if label == ref_sensor:
                continue
            offset_str = f"{offsets.get(label, 0):+.4f}s"
            pen = pg.mkPen(color=CORRECTED_COLOR, width=2)
            plot_x = x[:len(heave)]
            self.plot_widget.plot(
                plot_x, heave[:len(plot_x)], pen=pen,
                name=f'Sensor {label} (aligned {offset_str})',
                connect='finite', skipFiniteCheck=True,
            )

            # Also plot original as faded dashed for comparison
            if label in heaves_original:
                orig_h = heaves_original[label]
                pen_orig = pg.mkPen(
                    color=SENSOR_COLORS[color_idx % len(SENSOR_COLORS)],
                    width=1, style=Qt.DotLine,
                )
                plot_x2 = x[:len(orig_h)]
                self.plot_widget.plot(
                    plot_x2, orig_h[:len(plot_x2)], pen=pen_orig,
                    name=f'Sensor {label} (original)',
                    connect='finite', skipFiniteCheck=True,
                )
            color_idx += 1

        # Zero line
        if len(x) > 0:
            pen_zero = pg.mkPen('k', width=1, style=Qt.DashLine)
            self.plot_widget.plot(
                [float(x[0]), float(x[-1])], [0, 0], pen=pen_zero,
            )

        self.plot_widget.autoRange()

    def clear(self):
        self._clear_plot()
        self._x_axis = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _clear_plot(self):
        self.plot_widget.setUpdatesEnabled(False)
        try:
            self.plot_widget.clear()
            self.plot_widget.addLegend()
            self._vline = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen('r', width=1, style=Qt.DashLine),
            )
            self._hline = pg.InfiniteLine(
                angle=0, movable=False,
                pen=pg.mkPen('r', width=1, style=Qt.DashLine),
            )
            self.plot_widget.addItem(self._vline, ignoreBounds=True)
            self.plot_widget.addItem(self._hline, ignoreBounds=True)
            self._vline.setVisible(False)
            self._hline.setVisible(False)
        finally:
            self.plot_widget.setUpdatesEnabled(True)

    def _on_mouse_moved(self, evt):
        if self._x_axis is None:
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

        idx = int(np.argmin(np.abs(self._x_axis - x_pos)))
        self.info_label.setText(
            f"Sample: {idx}  |  Heave: {y_pos:.6f} m"
        )
