"""
SensorData - Core domain model wrapping a DataFrame with sensor metadata.

This is the single source of truth for all sensor data in the application.
It tracks which corrections have been applied and provides access to the
underlying data in a controlled manner.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


SENSOR_LABELS = ['A', 'B', 'C', 'D']


@dataclass
class CorrectionRecord:
    """Record of a correction applied to the data."""
    correction_type: str          # 'depth_calibration', 'depth_manual', 'time_shift'
    sensor_label: str             # Which sensor was modified
    description: str              # Human-readable description
    parameters: dict = field(default_factory=dict)


class SensorData:
    """
    Wraps a pandas DataFrame containing sensor depth measurements.
    
    The DataFrame is expected to have:
      - A datetime column (default name: 'datetime')
      - One or more depth columns named 'Sensor_{label}_Depth'
    
    This class tracks corrections applied and provides convenience
    accessors for the data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        datetime_col: str = 'datetime',
        sensor_labels: Optional[list[str]] = None,
        source_file: str = '',
        core_title: str = '',
        original_depth_columns: Optional[dict[str, str]] = None,
    ):
        self._df = df.copy()
        self.datetime_col = datetime_col
        self.source_file = source_file
        self.core_title = core_title
        self.corrections: list[CorrectionRecord] = []
        # Maps Sensor_{label}_Depth -> original column name from CSV
        self.original_depth_columns: dict[str, str] = dict(original_depth_columns or {})

        # Auto-detect sensor labels from columns if not provided
        if sensor_labels is not None:
            self.sensor_labels = list(sensor_labels)
        else:
            self.sensor_labels = self._detect_sensor_labels()

        # Ensure datetime column is proper datetime type
        if self.datetime_col in self._df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self._df[self.datetime_col]):
                self._df[self.datetime_col] = pd.to_datetime(
                    self._df[self.datetime_col], errors='coerce'
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """Return the underlying DataFrame (read-only reference)."""
        return self._df

    @property
    def num_sensors(self) -> int:
        return len(self.sensor_labels)

    @property
    def depth_columns(self) -> list[str]:
        """Return list of depth column names for active sensors."""
        return [f'Sensor_{label}_Depth' for label in self.sensor_labels]

    @property
    def row_count(self) -> int:
        return len(self._df)

    @property
    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if self.datetime_col not in self._df.columns or self._df.empty:
            return None
        return (self._df[self.datetime_col].min(), self._df[self.datetime_col].max())

    @property
    def depth_range(self) -> tuple[float, float] | None:
        cols = [c for c in self.depth_columns if c in self._df.columns]
        if not cols or self._df.empty:
            return None
        min_val = self._df[cols].min().min()
        max_val = self._df[cols].max().max()
        return (float(min_val), float(max_val))

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def get_depth_series(self, label: str) -> pd.Series:
        """Get the depth series for a given sensor label."""
        col = f'Sensor_{label}_Depth'
        if col not in self._df.columns:
            raise KeyError(f"No depth column found for sensor '{label}': {col}")
        return self._df[col]

    def get_timestamps(self) -> pd.Series:
        """Return the datetime series."""
        return self._df[self.datetime_col]

    def get_timestamps_epoch(self) -> np.ndarray:
        """Return timestamps as seconds since epoch (for plotting)."""
        return self._df[self.datetime_col].astype('int64').values / 1e9

    def slice_by_index(self, start: int, end: int) -> 'SensorData':
        """Return a new SensorData sliced by row index (inclusive)."""
        sliced_df = self._df.iloc[start:end + 1].copy().reset_index(drop=True)
        new_data = SensorData(
            df=sliced_df,
            datetime_col=self.datetime_col,
            sensor_labels=self.sensor_labels,
            source_file=self.source_file,
            core_title=self.core_title,
            original_depth_columns=self.original_depth_columns,
        )
        new_data.corrections = list(self.corrections)
        return new_data

    def copy(self) -> 'SensorData':
        """Return a deep copy."""
        new_data = SensorData(
            df=self._df.copy(),
            datetime_col=self.datetime_col,
            sensor_labels=list(self.sensor_labels),
            source_file=self.source_file,
            core_title=self.core_title,
            original_depth_columns=dict(self.original_depth_columns),
        )
        new_data.corrections = list(self.corrections)
        return new_data

    def get_original_column_name(self, label: str) -> str:
        """Get the original CSV column name for a sensor label."""
        col = f'Sensor_{label}_Depth'
        return self.original_depth_columns.get(col, col)

    def get_original_short_name(self, label: str) -> str:
        """Get a short display name from the original column.

        Returns e.g. 'Weight Stand (230406)' or 'CTD Frame (236222)'.
        The serial/device ID is included when available to disambiguate
        columns that share the same location name.
        """
        orig = self.get_original_column_name(label)
        if orig.startswith('Sensor_'):
            return f'Sensor {label}'
        parts = orig.split('_')
        location = parts[0]
        if len(parts) > 1:
            serial = parts[1].removesuffix('.rsk')
            return f"{location} ({serial})"
        return location

    def remap_sensor_labels(self, new_assignment: dict[str, str]) -> 'SensorData':
        """
        Create a new SensorData with columns remapped to different sensor labels.

        Args:
            new_assignment: Maps original column name -> new sensor label.
                            e.g. {'Weight Stand_..._Depth (m)': 'B', ...}

        Returns:
            New SensorData with renamed columns and updated labels.

        Raises:
            ValueError: If two columns are assigned the same label.
        """
        # Validate: no duplicate labels
        assigned_labels: dict[str, str] = {}   # label -> orig_col
        for orig_col, new_label in new_assignment.items():
            if not new_label or new_label == 'None':
                continue
            if new_label in assigned_labels:
                raise ValueError(
                    f"Duplicate sensor label '{new_label}': "
                    f"already assigned to {assigned_labels[new_label].split('_', 1)[0]}"
                )
            assigned_labels[new_label] = orig_col

        df_new = self._df.copy()
        new_labels = []
        new_orig_map = {}

        current_to_orig = {renamed: orig for renamed, orig in self.original_depth_columns.items()}

        # Collect assigned originals: original_col -> new_label
        assigned_originals = {}
        for orig_col, new_label in new_assignment.items():
            if new_label and new_label != 'None':
                assigned_originals[orig_col] = new_label

        # Rename via temporary names to avoid collisions during swap
        # Phase 1: current col -> temp name
        temp_rename = {}
        temp_to_final = {}
        for current_col, orig_col in current_to_orig.items():
            if orig_col in assigned_originals:
                new_label = assigned_originals[orig_col]
                final_col = f'Sensor_{new_label}_Depth'
                temp_col = f'__temp_{new_label}_Depth'
                temp_rename[current_col] = temp_col
                temp_to_final[temp_col] = final_col
                new_labels.append(new_label)
                new_orig_map[final_col] = orig_col

        df_new = df_new.rename(columns=temp_rename)
        # Phase 2: temp name -> final name
        df_new = df_new.rename(columns=temp_to_final)
        new_labels.sort(key=lambda x: SENSOR_LABELS.index(x))

        new_data = SensorData(
            df=df_new,
            datetime_col=self.datetime_col,
            sensor_labels=new_labels,
            source_file=self.source_file,
            core_title=self.core_title,
            original_depth_columns=new_orig_map,
        )
        return new_data

    # ------------------------------------------------------------------
    # Mutation helpers (record corrections)
    # ------------------------------------------------------------------

    def update_depth_column(self, label: str, values: np.ndarray | pd.Series):
        """Replace depth values for a sensor."""
        col = f'Sensor_{label}_Depth'
        self._df[col] = values

    def update_dataframe(self, df: pd.DataFrame):
        """Replace the entire underlying DataFrame."""
        self._df = df

    def add_correction(self, record: CorrectionRecord):
        """Track a correction that was applied."""
        self.corrections.append(record)

    # ------------------------------------------------------------------
    # Difference calculations
    # ------------------------------------------------------------------

    def compute_pairwise_differences(self) -> dict[tuple[str, str], pd.Series]:
        """
        Compute depth differences for all sensor pairs.
        Returns dict mapping (label_j, label_i) -> Series of (j - i).
        """
        diffs = {}
        for i, label_i in enumerate(self.sensor_labels):
            for j in range(i + 1, len(self.sensor_labels)):
                label_j = self.sensor_labels[j]
                col_i = f'Sensor_{label_i}_Depth'
                col_j = f'Sensor_{label_j}_Depth'
                if col_i in self._df.columns and col_j in self._df.columns:
                    diffs[(label_j, label_i)] = self._df[col_j] - self._df[col_i]
        return diffs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_sensor_labels(self) -> list[str]:
        """Auto-detect sensor labels from column names like Sensor_A_Depth."""
        labels = []
        for col in self._df.columns:
            if col.startswith('Sensor_') and col.endswith('_Depth'):
                label = col.replace('Sensor_', '').replace('_Depth', '')
                if label in SENSOR_LABELS:
                    labels.append(label)
        return sorted(labels, key=lambda x: SENSOR_LABELS.index(x))

    def __repr__(self) -> str:
        tr = self.time_range
        time_str = f"{tr[0]} to {tr[1]}" if tr else "N/A"
        return (
            f"SensorData(sensors={self.sensor_labels}, rows={self.row_count}, "
            f"time={time_str}, source='{self.source_file}')"
        )
