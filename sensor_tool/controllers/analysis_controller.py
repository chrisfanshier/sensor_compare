"""
AnalysisController - Central coordinator for the application.

Manages mode switching, data flow between panels/views, and calls
into domain processing logic. This keeps the GUI panels thin and
the business logic testable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..domain.models.sensor_data import SensorData, CorrectionRecord, SENSOR_LABELS
from ..domain.models.calibration import DepthCalibration
from ..domain.models.analysis_result import StatisticsResult, TimeOffsetResult
from ..domain.processing.depth_correction import DepthCorrectionProcessor
from ..domain.processing.time_correction import TimeCorrectionProcessor
from ..domain.processing.calibration_builder import CalibrationBuilder
from ..domain.processing.statistics import compute_statistics
from ..persistence.csv_loader import CSVLoader
from ..persistence.calibration_io import CalibrationIO


class AnalysisController:
    """
    Coordinates all application logic.
    
    Holds references to GUI panels and views, connects signals,
    and delegates to domain processing.
    """

    def __init__(self):
        # State
        self.sensor_data: Optional[SensorData] = None      # Primary data
        self.original_data: Optional[SensorData] = None     # Backup for reset
        self._raw_loaded_data: Optional[SensorData] = None  # Pristine copy for reassignment
        self.calibration: Optional[DepthCalibration] = None
        self.generated_calibration: Optional[DepthCalibration] = None
        self.time_offset_results: Optional[list[TimeOffsetResult]] = None
        self.collected_statistics: list[StatisticsResult] = []

        # GUI references (set by main_window after construction)
        self.main_window = None
        self.main_plot = None
        self.secondary_view = None  # velocity plot or stats table
        self.velocity_plot = None
        self.statistics_table = None

        # Panels
        self.view_panel = None
        self.depth_panel = None
        self.time_panel = None
        self.calibration_panel = None

    # ==================================================================
    # Initialization - called by MainWindow
    # ==================================================================

    def connect_signals(self):
        """Wire up all panel signals to controller methods."""
        # View Data panel
        self.view_panel.load_file_requested.connect(self.load_export_csv)
        self.view_panel.plot_depths_requested.connect(self.plot_depths)
        self.view_panel.plot_differences_requested.connect(self.plot_differences)

        # Depth Offset panel
        self.depth_panel.load_file_requested.connect(self.load_export_csv)
        self.depth_panel.load_calibration_requested.connect(self.load_calibration)
        self.depth_panel.ref_sensor_combo.currentTextChanged.connect(
            self._update_depth_correction_plan
        )
        self.depth_panel.sensor_assignments_changed.connect(
            self._on_sensor_assignments_changed
        )
        self.depth_panel.apply_corrections_requested.connect(self.apply_depth_corrections)
        self.depth_panel.reset_requested.connect(self.reset_to_original)
        self.depth_panel.export_requested.connect(self.export_corrected_csv)
        self.depth_panel.plot_original_requested.connect(self.plot_depths)

        # Time Offset panel
        self.time_panel.load_file_requested.connect(self.load_export_csv)
        self.time_panel.selection_mode_changed.connect(self._set_selection_mode)
        self.time_panel.clear_selection_requested.connect(self._clear_selection)
        self.time_panel.calculate_offsets_requested.connect(self.calculate_time_offsets)
        self.time_panel.apply_correction_requested.connect(self.apply_time_corrections)
        self.time_panel.reset_requested.connect(self.reset_to_original)
        self.time_panel.export_requested.connect(self.export_corrected_csv)
        self.time_panel.plot_original_requested.connect(self.plot_depths)

        # Create Calibration panel
        self.calibration_panel.load_cast_requested.connect(self._load_cast)
        self.calibration_panel.process_cast_requested.connect(self._process_cast)
        self.calibration_panel.selection_mode_changed.connect(self._set_selection_mode)
        self.calibration_panel.clear_selection_requested.connect(self._clear_selection)
        self.calibration_panel.add_statistics_requested.connect(self._add_statistics)
        self.calibration_panel.generate_calibration_requested.connect(
            self._generate_calibration
        )
        self.calibration_panel.save_calibration_requested.connect(
            self._save_calibration
        )

        # Main plot selection signal
        self.main_plot.selection_changed.connect(self._on_selection_changed)
        self.main_plot.selection_cleared.connect(self._on_selection_cleared)

    # ==================================================================
    # Data Loading
    # ==================================================================

    def load_export_csv(self, file_path: str):
        """Load a Sediment App batch-export CSV file."""
        try:
            # Clear previous selection to avoid stale indices
            self.main_plot.clear_selection()

            self.sensor_data = CSVLoader.load_export_csv(file_path)
            self.original_data = self.sensor_data.copy()
            # Also keep a pristine copy for reassignment (before any remap)
            self._raw_loaded_data = self.sensor_data.copy()
            self._update_all_panels_file_info()
            self.main_plot.set_sensor_data(self.sensor_data)
            self.plot_depths()
            self._log_active(
                f"Loaded: {self.sensor_data.source_file} "
                f"({self.sensor_data.row_count:,} rows, "
                f"{self.sensor_data.num_sensors} sensors)"
            )
        except Exception as e:
            self._show_error("Load Error", str(e))
            self._log_active(f"ERROR: {e}")

    def load_calibration(self, file_path: str):
        """Load a JSON calibration file."""
        try:
            self.calibration = CalibrationIO.load(file_path)
            filename = Path(file_path).name

            info_lines = [f"Loaded calibration: {filename}"]
            for reg in self.calibration.regressions:
                info_lines.append(
                    f"  {reg.sensor_j} - {reg.sensor_i}: "
                    f"slope={reg.slope:.6f}, intercept={reg.intercept:.6f}, "
                    f"R²={reg.r_squared:.4f}"
                )

            self.depth_panel.update_calibration_info(filename, info_lines)
            self._update_depth_correction_plan()
            self._log_active(f"Calibration loaded: {filename}")
        except Exception as e:
            self._show_error("Calibration Error", str(e))
            self._log_active(f"ERROR loading calibration: {e}")

    # ==================================================================
    # Plotting
    # ==================================================================

    def plot_depths(self):
        """Plot depth traces on the main plot."""
        if self.sensor_data is None:
            self._show_warning("No data loaded")
            return
        title = self.sensor_data.core_title or ''
        self.main_plot.plot_depths(self.sensor_data, title=title)

    def plot_differences(self):
        """Plot pairwise differences on the main plot."""
        if self.sensor_data is None:
            self._show_warning("No data loaded")
            return
        self.main_plot.plot_differences(self.sensor_data)

    # ==================================================================
    # Depth Offset Mode
    # ==================================================================

    def _on_sensor_assignments_changed(self):
        """
        User changed the sensor assignment dropdowns.
        Remap the sensor data columns to match the new assignment.
        """
        if self._raw_loaded_data is None:
            return

        assignments = self.depth_panel.get_sensor_assignments()
        if not assignments:
            return

        try:
            remapped = self._raw_loaded_data.remap_sensor_labels(assignments)
        except ValueError:
            # Normal transitional state: user is mid-edit with duplicate labels.
            # Wait for them to finish reassigning.
            return

        try:
            self.sensor_data = remapped
            self.original_data = remapped.copy()
            self.main_plot.set_sensor_data(self.sensor_data)
            self.plot_depths()

            panel = self.depth_panel
            panel.log_widget.log("Sensor assignments updated:")
            for orig_col, label in assignments.items():
                parts = orig_col.split('_')
                short = parts[0]
                if len(parts) > 1:
                    short = f"{short} ({parts[1].removesuffix('.rsk')})"
                panel.log_widget.log(f"  {short} → Sensor {label}")

            # Refresh correction plan if calibration loaded
            self._update_depth_correction_plan()

        except Exception as e:
            self.depth_panel.log_widget.log(f"ERROR remapping: {e}")

    def apply_depth_corrections(self):
        """
        Apply depth calibration and manual offsets.

        Mirrors the working sensor_compare_time_pyside6.py approach:
        - Uses sensor assignments to find reference and target columns
        - Applies depth-dependent regression corrections
        - Applies manual constant offsets
        - Operates on the ORIGINAL column data via assignment mapping
        """
        if self.sensor_data is None:
            self._show_warning("No data loaded")
            return

        try:
            # Work on a copy from original
            working = self.original_data.copy()
            ref_sensor = self.depth_panel.ref_sensor_combo.currentText()
            panel = self.depth_panel
            assignments = panel.get_sensor_assignments()

            panel.log_widget.log(f"\n{'='*50}")
            panel.log_widget.log(f"APPLYING CORRECTIONS")
            panel.log_widget.log(f"Reference: Sensor {ref_sensor} (will NOT be modified)")
            panel.log_widget.log(f"{'='*50}")

            # Find the reference column (the renamed Sensor_{ref}_Depth)
            ref_col = f'Sensor_{ref_sensor}_Depth'
            if ref_col not in working.df.columns:
                self._show_warning(
                    f"Reference sensor {ref_sensor} not found in data. "
                    f"Check sensor assignments."
                )
                return

            # Apply regression corrections based on checkboxes
            if panel.correction_checkboxes:
                panel.log_widget.log(f"\nApplying depth-dependent corrections...")

                for target_sensor, info in panel.correction_checkboxes.items():
                    if not info['checkbox'].isChecked():
                        continue

                    target_col = f'Sensor_{target_sensor}_Depth'
                    if target_col not in working.df.columns:
                        continue

                    reg = info['reg']
                    sign = info['sign']

                    # Calculate offset at EACH depth point using reference depth
                    ref_depths = working.df[ref_col].values
                    offsets = reg.slope * ref_depths + reg.intercept
                    corrections = sign * offsets

                    # Apply point-by-point correction
                    original_vals = working.df[target_col].values
                    working.update_depth_column(target_sensor, original_vals + corrections)

                    # Report statistics
                    mean_c = float(np.nanmean(corrections))
                    std_c = float(np.nanstd(corrections))
                    min_c = float(np.nanmin(corrections))
                    max_c = float(np.nanmax(corrections))

                    working.add_correction(CorrectionRecord(
                        correction_type='depth_calibration',
                        sensor_label=target_sensor,
                        description=(
                            f"Regression {reg.sensor_j}-{reg.sensor_i}: "
                            f"slope={reg.slope:.6f}, intercept={reg.intercept:.6f}"
                        ),
                        parameters={
                            'slope': reg.slope,
                            'intercept': reg.intercept,
                            'sign': sign,
                            'ref_sensor': ref_sensor,
                        },
                    ))

                    panel.log_widget.log(
                        f"\nSensor {target_sensor}: Applied regression "
                        f"{reg.sensor_j}-{reg.sensor_i}"
                    )
                    panel.log_widget.log(
                        f"  Correction range: {min_c:+.4f}m to {max_c:+.4f}m"
                    )
                    panel.log_widget.log(
                        f"  Mean: {mean_c:+.4f}m \u00b1 {std_c:.4f}m"
                    )

            # Apply manual offsets
            manual = panel.get_manual_offsets()
            manual_applied = False
            for label, offset in manual.items():
                if offset == 0.0:
                    continue
                col = f'Sensor_{label}_Depth'
                if col not in working.df.columns:
                    continue
                if not manual_applied:
                    panel.log_widget.log(f"\nManual offsets:")
                    manual_applied = True
                working.df[col] = working.df[col] + offset
                working.add_correction(CorrectionRecord(
                    correction_type='depth_manual',
                    sensor_label=label,
                    description=f"Manual offset: {offset:+.4f}m",
                    parameters={'offset_m': offset},
                ))
                panel.log_widget.log(f"  Sensor {label}: {offset:+.4f}m")

            self.sensor_data = working
            self.main_plot.set_sensor_data(self.sensor_data)

            # Build label suffixes showing applied corrections
            suffixes = {}
            for label in working.sensor_labels:
                short = working.get_original_short_name(label)
                suffixes[label] = f" ({short} corrected)"

            self.main_plot.plot_depths_with_labels(
                self.sensor_data, suffixes,
                title=self.sensor_data.core_title or 'Corrected Data',
            )

            panel.log_widget.log("Corrections applied and plotted!")

        except Exception as e:
            self._show_error("Correction Error", str(e))
            self.depth_panel.log_widget.log(f"ERROR: {e}")

    def _update_depth_correction_plan(self, _=None):
        """Update the correction plan display based on current calibration + ref."""
        if self.calibration is None:
            self.depth_panel.update_correction_plan([])
            return

        ref = self.depth_panel.ref_sensor_combo.currentText()
        applicable = self.calibration.get_applicable_corrections(ref)
        self.depth_panel.update_correction_plan(applicable)

    # ==================================================================
    # Time Offset Mode
    # ==================================================================

    def calculate_time_offsets(self):
        """Calculate time offsets using Savgol velocity difference method."""
        if self.sensor_data is None:
            self._show_warning("No data loaded")
            return

        sel = self.main_plot.selection
        if sel is None:
            self._show_warning("Please select a time range first")
            return

        try:
            start_idx, end_idx = sel
            panel = self.time_panel
            window, polyorder = panel.get_savgol_params()
            ref_sensor = panel.get_ref_sensor()

            panel.log_widget.log(f"\n{'='*50}")
            panel.log_widget.log("CALCULATING TIME OFFSETS")
            panel.log_widget.log(f"Range: rows {start_idx}-{end_idx}")
            panel.log_widget.log(f"Savgol: window={window}, polyorder={polyorder}")
            panel.log_widget.log(f"Reference: Sensor {ref_sensor}")
            panel.log_widget.log(f"{'='*50}")

            self.time_offset_results = TimeCorrectionProcessor.calculate_offsets(
                self.sensor_data, start_idx, end_idx,
                ref_sensor, window, polyorder,
            )

            for r in self.time_offset_results:
                if r.is_reference:
                    panel.log_widget.log(f"Sensor {r.sensor_label}: 0.000s (reference)")
                else:
                    panel.log_widget.log(
                        f"Sensor {r.sensor_label}: {r.offset_seconds:+.3f}s "
                        f"(RMS: {r.rms_value:.6f})"
                    )

            panel.display_offsets(self.time_offset_results)
            panel.log_widget.log("Offset calculation complete!")

        except Exception as e:
            self._show_error("Calculation Error", str(e))
            self.time_panel.log_widget.log(f"ERROR: {e}")

    def apply_time_corrections(self):
        """Apply calculated time offsets."""
        if self.time_offset_results is None:
            self._show_warning("Calculate offsets first")
            return

        try:
            panel = self.time_panel
            panel.log_widget.log(f"\n{'='*50}")
            panel.log_widget.log("APPLYING TIME CORRECTIONS")
            panel.log_widget.log(f"{'='*50}")

            self.sensor_data = TimeCorrectionProcessor.apply_offsets(
                self.sensor_data, self.time_offset_results
            )
            self.main_plot.set_sensor_data(self.sensor_data)

            # Build label suffixes
            suffixes = {}
            for r in self.time_offset_results:
                if not r.is_reference:
                    suffixes[r.sensor_label] = f" ({r.offset_seconds:+.3f}s)"
                else:
                    suffixes[r.sensor_label] = " (ref)"

            self.main_plot.plot_depths_with_labels(
                self.sensor_data, suffixes,
                title=self.sensor_data.core_title or 'Time-Corrected',
            )

            # Update velocity plot
            if self.velocity_plot is not None:
                ref = panel.get_ref_sensor()
                window, polyorder = panel.get_savgol_params()
                v_diffs = TimeCorrectionProcessor.compute_velocity_differences(
                    self.sensor_data, ref, window, polyorder
                )
                self.velocity_plot.plot_velocity_differences(
                    self.sensor_data, v_diffs, ref
                )

            for r in self.time_offset_results:
                if not r.is_reference:
                    panel.log_widget.log(
                        f"Applied {r.offset_seconds:+.3f}s shift to Sensor {r.sensor_label}"
                    )
            panel.log_widget.log("Time corrections applied!")

        except Exception as e:
            self._show_error("Correction Error", str(e))
            self.time_panel.log_widget.log(f"ERROR: {e}")

    # ==================================================================
    # Create Calibration Mode
    # ==================================================================

    def _load_cast(self, file_path: str):
        """Load a cast file for calibration creation."""
        try:
            panel = self.calibration_panel
            panel.cast_label.setText(f"File: {Path(file_path).name}\nReady to process.")
            # Store the path for processing
            self._cast_file_path = file_path
            panel.log_widget.log(f"Cast selected: {Path(file_path).name}")
        except Exception as e:
            self._show_error("Load Error", str(e))

    def _process_cast(self):
        """Process the loaded cast with given parameters."""
        if not hasattr(self, '_cast_file_path'):
            self._show_warning("Load a cast file first")
            return

        try:
            panel = self.calibration_panel
            params = panel.get_processing_params()
            patterns = panel.get_sensor_patterns()

            panel.log_widget.log(f"\nProcessing with patterns: {patterns}")

            data = CSVLoader.load_sensor_csv(
                self._cast_file_path,
                sensor_patterns=patterns,
                datetime_col=params['datetime_col'],
                skip_rows=params['skip_rows'],
            )

            panel.log_widget.log(f"Loaded {data.row_count} rows, {data.num_sensors} sensors")

            # Filter by depth
            df = data.df
            depth_cols = data.depth_columns
            existing = [c for c in depth_cols if c in df.columns]

            filter_depth = df[existing[0]].copy()
            for col in existing[1:]:
                filter_depth = filter_depth.fillna(df[col])

            mask = (filter_depth >= params['min_depth']) & (filter_depth <= params['max_depth'])
            df = df[mask].copy()
            panel.log_widget.log(f"After depth filter: {len(df)} rows")

            # Trim
            trim = params['trim_rows']
            if len(df) > trim * 2:
                df = df.iloc[trim:-trim].copy()
                panel.log_widget.log(f"After trim: {len(df)} rows")

            # Smooth
            smooth = params['smooth_window']
            for col in existing:
                df[col] = df[col].rolling(window=smooth, min_periods=1).mean()

            df = df.reset_index(drop=True)
            data.update_dataframe(df)

            self.sensor_data = data
            self.original_data = data.copy()
            self.main_plot.set_sensor_data(self.sensor_data)
            self.main_plot.plot_depths(self.sensor_data)

            panel.update_cast_info(data.source_file, data.num_sensors, data.row_count)
            self._update_all_panels_file_info()
            panel.log_widget.log("Cast processed and plotted!")

        except Exception as e:
            self._show_error("Processing Error", str(e))
            self.calibration_panel.log_widget.log(f"ERROR: {e}")

    def _add_statistics(self):
        """Add statistics from current selection to the collection."""
        if self.sensor_data is None:
            self._show_warning("No data loaded")
            return

        sel = self.main_plot.selection
        if sel is None:
            self._show_warning("Select a range first")
            return

        try:
            start_idx, end_idx = sel
            panel = self.calibration_panel

            # Get sensor patterns for identification
            patterns = panel.get_sensor_patterns()
            pattern_dict = {}
            for i, p in enumerate(patterns):
                if i < len(self.sensor_data.sensor_labels):
                    pattern_dict[self.sensor_data.sensor_labels[i]] = p

            stats = compute_statistics(
                self.sensor_data, start_idx, end_idx, pattern_dict
            )
            self.collected_statistics.append(stats)

            # Update stats table
            if self.statistics_table is not None:
                if self.statistics_table.count == 0:
                    self.statistics_table.set_columns(self.sensor_data.sensor_labels)
                self.statistics_table.add_statistics(stats)

            panel.update_stats_count(len(self.collected_statistics))
            panel.log_widget.log(
                f"Added stats: depth={stats.mean_depth_all_sensors:.1f}m, "
                f"n={stats.n_points}"
            )

        except Exception as e:
            self._show_error("Statistics Error", str(e))
            self.calibration_panel.log_widget.log(f"ERROR: {e}")

    def _generate_calibration(self):
        """Generate a calibration from collected statistics."""
        if len(self.collected_statistics) < 2:
            self._show_warning("Need at least 2 statistics entries")
            return

        try:
            panel = self.calibration_panel
            n = panel.num_sensors_spin.value()
            labels = SENSOR_LABELS[:n]

            self.generated_calibration = CalibrationBuilder.build(
                self.collected_statistics, n, labels
            )

            # Build summary
            summary_lines = ["Calibration generated:"]
            for reg in self.generated_calibration.regressions:
                summary_lines.append(
                    f"  {reg.sensor_j}-{reg.sensor_i}: "
                    f"y={reg.slope:.6f}x + {reg.intercept:.6f} (R²={reg.r_squared:.4f})"
                )
            summary = "\n".join(summary_lines)

            panel.update_calibration_summary(summary)
            panel.log_widget.log(summary)

        except Exception as e:
            self._show_error("Generation Error", str(e))
            self.calibration_panel.log_widget.log(f"ERROR: {e}")

    def _save_calibration(self, file_path: str):
        """Save generated calibration to file."""
        if self.generated_calibration is None:
            self._show_warning("Generate a calibration first")
            return

        try:
            CalibrationIO.save(self.generated_calibration, file_path)
            self.calibration_panel.log_widget.log(
                f"Saved calibration to {Path(file_path).name}"
            )
        except Exception as e:
            self._show_error("Save Error", str(e))

    # ==================================================================
    # Common Operations
    # ==================================================================

    def reset_to_original(self):
        """Reset data to the original loaded state."""
        if self.original_data is None:
            self._show_warning("No original data to reset to")
            return

        self.sensor_data = self.original_data.copy()
        self.main_plot.set_sensor_data(self.sensor_data)
        self.plot_depths()
        self.time_offset_results = None
        self._log_active("Data reset to original")

    def export_corrected_csv(self):
        """Export the current sensor data to CSV."""
        if self.sensor_data is None:
            self._show_warning("No data to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Data", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            self.sensor_data.df.to_csv(file_path, index=False)
            self._log_active(f"Exported to {Path(file_path).name}")
            QMessageBox.information(
                self.main_window, "Success", "Data exported successfully!"
            )
        except Exception as e:
            self._show_error("Export Error", str(e))

    # ==================================================================
    # Selection handling
    # ==================================================================

    def _set_selection_mode(self, enabled: bool):
        self.main_plot.selection_mode = enabled

    def _clear_selection(self):
        self.main_plot.clear_selection()
        # Update active panel's selection info
        mode = self.main_window.get_current_mode() if self.main_window else None
        if mode == 'Time Offset':
            self.time_panel.selection_controls.clear_info()
        elif mode == 'Create Calibration':
            self.calibration_panel.selection_controls.clear_info()

    def _on_selection_changed(self, start_idx: int, end_idx: int):
        """Called when user completes a selection on the main plot."""
        mode = self.main_window.get_current_mode() if self.main_window else None

        n = end_idx - start_idx + 1
        extra = ''
        if self.sensor_data is not None:
            dt_col = self.sensor_data.datetime_col
            if dt_col in self.sensor_data.df.columns:
                t_start = self.sensor_data.df[dt_col].iloc[start_idx]
                t_end = self.sensor_data.df[dt_col].iloc[end_idx]
                extra = f"{t_start} to {t_end}"

        if mode == 'Time Offset':
            self.time_panel.selection_controls.update_selection_info(
                start_idx, end_idx, extra
            )
        elif mode == 'Create Calibration':
            self.calibration_panel.selection_controls.update_selection_info(
                start_idx, end_idx, extra
            )

    def _on_selection_cleared(self):
        pass  # Already handled by _clear_selection

    # ==================================================================
    # Mode switching
    # ==================================================================

    def on_mode_changed(self, mode_name: str):
        """Called by MainWindow when mode changes."""
        # Show/hide secondary view
        if self.main_window is None:
            return

        if mode_name == 'Time Offset':
            self.main_window.show_secondary_view('velocity')
        elif mode_name == 'Create Calibration':
            self.main_window.show_secondary_view('statistics')
        else:
            self.main_window.hide_secondary_view()

        # Re-plot current data if available
        if self.sensor_data is not None:
            self.main_plot.set_sensor_data(self.sensor_data)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _update_all_panels_file_info(self):
        """Update file info on all panels."""
        if self.sensor_data is None:
            return

        sd = self.sensor_data
        tr = sd.time_range
        dr = sd.depth_range
        time_str = f"{tr[0]} to {tr[1]}" if tr else ''
        depth_str = f"{dr[0]:.1f}m to {dr[1]:.1f}m" if dr else ''

        self.view_panel.update_file_info(
            sd.source_file, sd.num_sensors, sd.row_count,
            time_str, depth_str, sd.core_title,
        )
        self.depth_panel.update_file_info(
            sd.source_file, sd.num_sensors, sd.row_count, sd.core_title,
        )
        self.time_panel.update_file_info(
            sd.source_file, sd.num_sensors, sd.row_count,
        )

        # Set up sensor assignments with ORIGINAL column names
        if sd.num_sensors > 0 and sd.original_depth_columns:
            # Get original column names in sensor-label order
            original_cols = []
            current_assignments = {}
            for label in sd.sensor_labels:
                renamed_col = f'Sensor_{label}_Depth'
                orig_col = sd.original_depth_columns.get(renamed_col, renamed_col)
                original_cols.append(orig_col)
                current_assignments[orig_col] = label

            self.depth_panel.setup_sensor_assignments(
                original_cols, current_assignments,
            )
            self.depth_panel.log_widget.log(
                f"Found {len(original_cols)} depth columns:"
            )
            for orig_col, label in current_assignments.items():
                parts = orig_col.split('_')
                short = parts[0]
                if len(parts) > 1:
                    short = f"{short} ({parts[1].removesuffix('.rsk')})"
                self.depth_panel.log_widget.log(
                    f"  {short} → Sensor {label}"
                )

    def _log_active(self, msg: str):
        """Log to the currently active panel."""
        mode = self.main_window.get_current_mode() if self.main_window else None
        if mode == 'View Data':
            self.view_panel.log_widget.log(msg)
        elif mode == 'Depth Offset':
            self.depth_panel.log_widget.log(msg)
        elif mode == 'Time Offset':
            self.time_panel.log_widget.log(msg)
        elif mode == 'Create Calibration':
            self.calibration_panel.log_widget.log(msg)

    def _show_error(self, title: str, msg: str):
        if self.main_window:
            QMessageBox.critical(self.main_window, title, msg)

    def _show_warning(self, msg: str):
        if self.main_window:
            QMessageBox.warning(self.main_window, "Warning", msg)
