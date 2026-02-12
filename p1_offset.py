from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QMessageBox, QLabel, QCheckBox, QSpinBox, QHBoxLayout, QDoubleSpinBox,
    QGroupBox, QScrollArea, QGridLayout
)
from PySide6.QtCore import Slot, Qt, Signal, QObject, QRunnable, QThreadPool
import sys
import pandas as pd
import pyqtgraph as pg
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

# Attempt to enable OpenGL-accelerated plotting if PyOpenGL is installed.
# pyqtgraph will use PyOpenGL when `useOpenGL=True` is set; if PyOpenGL
# is not available we gracefully fall back to the software renderer.
try:
    import OpenGL.GL  # noqa: F401
    pg.setConfigOptions(useOpenGL=True)
    _HAVE_PYOPENGL = True
except Exception:
    _HAVE_PYOPENGL = False


class _SmoothWorkerSignals(QObject):
    finished = Signal(int, object)  # run_id, list-of-smoothed-arrays-or-None


class _SmoothWorker(QRunnable):
    def __init__(self, run_id, ys, window, poly):
        super().__init__()
        self.run_id = run_id
        self.ys = ys
        self.window = window
        self.poly = poly
        self.signals = _SmoothWorkerSignals()

    def run(self):
        results = []
        # perform smoothing for each series (pure-Python / numpy work off the GUI thread)
        try:
            for y in self.ys:
                try:
                    n = len(y)
                    if n < 3:
                        results.append(None)
                        continue
                    # choose window length not greater than n and odd
                    w = min(self.window, n if n % 2 == 1 else n - 1)
                    if w <= self.poly:
                        w = self.poly + 1
                    if w % 2 == 0:
                        w += 1
                    if w > n:
                        results.append(None)
                        continue
                    y_smooth = savgol_filter(y, window_length=w, polyorder=self.poly, mode='interp')
                    results.append(y_smooth)
                except Exception:
                    results.append(None)
        except Exception:
            results = [None] * len(self.ys)
        self.signals.finished.emit(self.run_id, results)


class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Plotter (pyqtgraph)")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)
        
        self.select_button = QPushButton("Select File")
        self.select_button.clicked.connect(self.select_file)
        left_layout.addWidget(self.select_button)

        # top text label (Qt widget)
        self.label = QLabel()
        self.label.setText("")
        self.label.setTextFormat(Qt.RichText)
        # avoid layout shifts when the label text changes by constraining its width
        self.label.setWordWrap(True)
        self.label.setFixedWidth(320)
        left_layout.addWidget(self.label)

        # Savgol controls (checkbox + sliders/spinboxes) + raw toggle
        ctrl_layout = QVBoxLayout()
        
        savgol_row = QHBoxLayout()
        self.savgol_cb = QCheckBox("Savgol smoothing")
        self.savgol_cb.stateChanged.connect(self.update_smoothing)
        savgol_row.addWidget(self.savgol_cb)
        ctrl_layout.addLayout(savgol_row)

        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Window (odd)"))
        self.savgol_window = QSpinBox()
        self.savgol_window.setRange(3, 9999)
        self.savgol_window.setSingleStep(2)
        self.savgol_window.setValue(11)
        self.savgol_window.valueChanged.connect(self.on_savgol_params_changed)
        window_row.addWidget(self.savgol_window)
        ctrl_layout.addLayout(window_row)

        poly_row = QHBoxLayout()
        poly_row.addWidget(QLabel("Polyorder"))
        self.savgol_poly = QSpinBox()
        self.savgol_poly.setRange(1, 10)
        self.savgol_poly.setValue(3)
        self.savgol_poly.valueChanged.connect(self.on_savgol_params_changed)
        poly_row.addWidget(self.savgol_poly)
        ctrl_layout.addLayout(poly_row)

        # toggle to show/hide raw data curves
        self.raw_cb = QCheckBox("Show raw data")
        self.raw_cb.setChecked(True)
        self.raw_cb.stateChanged.connect(self.toggle_raw)
        ctrl_layout.addWidget(self.raw_cb)

        left_layout.addLayout(ctrl_layout)

        if not _HAS_SAVGOL:
            self.savgol_cb.setEnabled(False)
            self.savgol_cb.setToolTip("scipy not available; install scipy to enable smoothing")

        # Vertical offset controls (scrollable)
        offset_group = QGroupBox("Vertical Offsets")
        offset_layout = QVBoxLayout()
        
        # Scroll area for offset controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.offset_grid = QGridLayout(scroll_widget)
        self.offset_grid.setColumnStretch(1, 1)
        scroll.setWidget(scroll_widget)
        
        offset_layout.addWidget(scroll)
        offset_group.setLayout(offset_layout)
        left_layout.addWidget(offset_group)
        
        # Reset all offsets button
        reset_btn = QPushButton("Reset All Offsets")
        reset_btn.clicked.connect(self.reset_all_offsets)
        left_layout.addWidget(reset_btn)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.date_axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': self.date_axis})
        right_layout.addWidget(self.plot_widget)
        
        main_layout.addWidget(right_panel, stretch=1)

        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.getViewBox().invertY(True)
        self.plot_widget.setMouseEnabled(x=True, y=True)

        # --- Crosshair ---
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        # Data and plot handles
        self.data_x = None
        self.data_ys = []
        self.data_labels = []
        self.curves = []           # original PlotDataItem objects
        self.smoothed_curves = []  # smoothed PlotDataItem objects
        self.color_list = ['#6baed6', '#74c476', '#fd8d3c', '#9e9ac8', '#bcbddc', '#fbb4ae', '#a6bddb']
        
        # Offset controls
        self.offset_spinboxes = []
        self.offset_values = []  # actual offset values applied to each series

        self.is_datetime = False

        # threading for smoothing
        self.threadpool = QThreadPool()
        self._smoothing_run_id = 0

        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        # track last hovered x-index to avoid repeated label/layout updates
        self._last_mouse_idx = None

    @Slot()
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.plot_file(file_path)

    def plot_file(self, file_path):
        # clear previous
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.getViewBox().invertY(True)
        self.plot_widget.setBackground('w')
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        self.label.setText("")
        self.data_x = None
        self.data_ys = []
        self.data_labels = []
        self.curves = []
        self.smoothed_curves = []
        self.offset_values = []
        self.is_datetime = False
        
        # Clear offset controls
        for i in reversed(range(self.offset_grid.count())):
            self.offset_grid.itemAt(i).widget().setParent(None)
        self.offset_spinboxes = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        core_title = None
        skiprows = 0
        for i, line in enumerate(lines):
            if line.startswith('# Core:'):
                core_title = line.split(':', 1)[1].strip()
            if line.strip().startswith('datetime'):
                skiprows = i
                break

        if core_title:
            self.setWindowTitle(f"File Plotter — {core_title}")

        df = pd.read_csv(file_path, skiprows=skiprows)
        df.columns = df.columns.str.strip()
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        depth_cols = [col for col in df.columns if 'Depth' in col]
        if not depth_cols:
            self.show_message(
                "No depth columns found.",
                f"Expected columns with 'Depth' in name.\n\nAll columns found:\n{df.columns.tolist()}"
            )
            return

        if 'datetime' in df.columns:
            self.is_datetime = True
            x = df['datetime'].astype('int64') / 1e9  # float seconds since epoch
            self.data_x = x.values
            for i, col in enumerate(depth_cols):
                short_name = col.split('_', 1)[0]
                y = df[col].values
                self.data_ys.append(y)
                self.data_labels.append(short_name)
                self.offset_values.append(0.0)
                
                # Create offset control
                self.create_offset_control(i, short_name)
                
                # Plot with initial zero offset
                pen = pg.mkPen(color=self.color_list[i % len(self.color_list)], width=2)
                curve = self.plot_widget.plot(x, y, pen=pen, name=short_name, useOpenGL=_HAVE_PYOPENGL)
                self.curves.append(curve)
                
            self.plot_widget.setLabel('bottom', core_title or 'Datetime')
            self.plot_widget.setTitle(core_title or 'Depth Plot')
        else:
            self.is_datetime = False
            x = np.arange(len(df))
            self.data_x = x
            for i, col in enumerate(depth_cols):
                short_name = col.split('_', 1)[0]
                y = df[col].values
                self.data_ys.append(y)
                self.data_labels.append(short_name)
                self.offset_values.append(0.0)
                
                # Create offset control
                self.create_offset_control(i, short_name)
                
                # Plot with initial zero offset
                pen = pg.mkPen(color=self.color_list[i % len(self.color_list)], width=2)
                curve = self.plot_widget.plot(x, y, pen=pen, name=short_name, useOpenGL=_HAVE_PYOPENGL)
                self.curves.append(curve)
                
            self.plot_widget.setLabel('bottom', depth_cols[0])
            self.plot_widget.setTitle(core_title or 'Depth Plot')
        self.plot_widget.setLabel('left', depth_cols[0])

        # Apply smoothing if enabled
        if self.savgol_cb.isChecked():
            self.update_smoothing()

        # enforce raw visibility state after plotting
        self._apply_raw_visibility()

    def create_offset_control(self, idx, label):
        """Create offset spinbox for a series"""
        row = len(self.offset_spinboxes)
        
        # Color indicator
        color_label = QLabel("■")
        color_label.setStyleSheet(f"color: {self.color_list[idx % len(self.color_list)]}; font-size: 16pt;")
        self.offset_grid.addWidget(color_label, row, 0)
        
        # Series label
        name_label = QLabel(label)
        self.offset_grid.addWidget(name_label, row, 1)
        
        # Offset spinbox
        spinbox = QDoubleSpinBox()
        spinbox.setRange(-10000, 10000)
        spinbox.setSingleStep(0.1)
        spinbox.setDecimals(3)
        spinbox.setValue(0.0)
        spinbox.valueChanged.connect(lambda val, i=idx: self.on_offset_changed(i, val))
        self.offset_grid.addWidget(spinbox, row, 2)
        
        self.offset_spinboxes.append(spinbox)

    def on_offset_changed(self, idx, value):
        """Update offset for a specific series"""
        if idx >= len(self.offset_values):
            return
            
        self.offset_values[idx] = value
        
        # Update raw curve
        if idx < len(self.curves) and idx < len(self.data_ys):
            y_offset = self.data_ys[idx] + value
            self.curves[idx].setData(self.data_x, y_offset)
        
        # Update smoothed curve if smoothing is enabled
        if self.savgol_cb.isChecked():
            self.update_smoothing()

    def reset_all_offsets(self):
        """Reset all offset values to zero"""
        for spinbox in self.offset_spinboxes:
            spinbox.setValue(0.0)

    def on_savgol_params_changed(self, val):
        if self.savgol_cb.isChecked():
            self.update_smoothing()

    def update_smoothing(self):
        # remove previous smoothed curves on GUI thread
        for item in self.smoothed_curves:
            try:
                self.plot_widget.removeItem(item)
            except Exception:
                pass
        self.smoothed_curves = []

        if not _HAS_SAVGOL or not self.savgol_cb.isChecked() or not self.data_ys:
            return

        window = int(self.savgol_window.value())
        poly = int(self.savgol_poly.value())

        # enforce window > poly and odd (worker will also enforce)
        if window <= poly:
            window = poly + 1
        if window % 2 == 0:
            window += 1

        # start background worker to compute all smoothed series
        self._smoothing_run_id += 1
        run_id = self._smoothing_run_id
        worker = _SmoothWorker(run_id, list(self.data_ys), window, poly)
        worker.signals.finished.connect(self._on_smoothing_finished)
        self.threadpool.start(worker)

    def _on_smoothing_finished(self, run_id, results):
        # ignore results from stale runs
        if run_id != self._smoothing_run_id:
            return
        # plot results on GUI thread with offsets applied
        for i, y_smooth in enumerate(results):
            if y_smooth is None:
                continue
            # Apply offset to smoothed data
            offset = self.offset_values[i] if i < len(self.offset_values) else 0.0
            y_smooth_offset = y_smooth + offset
            pen = pg.mkPen(color=self.color_list[i % len(self.color_list)], width=2, style=Qt.DashLine)
            curve = self.plot_widget.plot(self.data_x, y_smooth_offset, pen=pen, name=None, useOpenGL=_HAVE_PYOPENGL)
            self.smoothed_curves.append(curve)

    def _apply_raw_visibility(self):
        visible = bool(self.raw_cb.isChecked())
        for c in self.curves:
            try:
                c.setVisible(visible)
            except Exception:
                pass

    def toggle_raw(self, state):
        # state may be int or Qt.CheckState; simply apply current checkbox state
        self._apply_raw_visibility()

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos) and self.data_x is not None:
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x_pos = mousePoint.x()
            if len(self.data_x) > 0 and self.data_x[0] <= x_pos <= self.data_x[-1]:
                idx = np.argmin(np.abs(self.data_x - x_pos))
                # only update label and crosshair when index changed to avoid layout thrashing
                if idx == self._last_mouse_idx:
                    # still move crosshair lines, but skip expensive UI updates
                    self.vLine.setPos(self.data_x[idx])
                    self.hLine.setPos(mousePoint.y())
                    return
                self._last_mouse_idx = int(idx)
                self.vLine.setPos(self.data_x[idx])
                self.hLine.setPos(mousePoint.y())
                if self.is_datetime:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(self.data_x[idx], datetime.timezone.utc)
                    x_label = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                else:
                    x_label = f"{self.data_x[idx]:.3f}"
                # use same gentle colors for label
                color_map = self.color_list
                y_labels = []
                for i, (label, y) in enumerate(zip(self.data_labels, self.data_ys)):
                    if idx < len(y):
                        # Show original value + offset applied
                        offset = self.offset_values[i] if i < len(self.offset_values) else 0.0
                        original_val = y[idx]
                        offset_val = original_val + offset
                        if offset != 0.0:
                            y_labels.append(
                                f"<span style='color: {color_map[i % len(color_map)]}'>{label}: {offset_val:.3f} ({original_val:.3f}+{offset:.3f})</span>"
                            )
                        else:
                            y_labels.append(
                                f"<span style='color: {color_map[i % len(color_map)]}'>{label}: {original_val:.3f}</span>"
                            )
                html = (
                    f"<span style='font-size: 12pt'>"
                    f"x={x_label}<br>"
                    + "<br>".join(y_labels) +
                    f"</span>"
                )
                self.label.setText(html)
            else:
                self.label.setText("")
        else:
            self.label.setText("")

    def show_message(self, title, text):
        QMessageBox.warning(self, title, text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    sys.exit(app.exec())