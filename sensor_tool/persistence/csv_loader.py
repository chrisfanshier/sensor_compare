"""
CSV loader - handles reading sensor CSV files with multiple encoding
fallbacks and format detection.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from ..domain.models.sensor_data import SensorData, SENSOR_LABELS

# Encodings to try in order
ENCODINGS = ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'latin-1']


class CSVLoader:
    """Load sensor data from CSV files into SensorData instances."""

    @staticmethod
    def load_export_csv(
        file_path: str | Path,
        datetime_col: str = 'datetime',
    ) -> SensorData:
        """
        Load a Sediment App batch-export CSV.
        
        These files have comment lines starting with '#' and a header row
        starting with 'datetime'. Depth columns contain 'Depth' in the name.
        
        Args:
            file_path: Path to the CSV file.
            datetime_col: Name of the datetime column.
            
        Returns:
            SensorData instance.
        """
        file_path = Path(file_path)

        # Read lines to find metadata and header row
        lines = CSVLoader._read_lines(file_path)

        core_title = ''
        skiprows = 0
        for i, line in enumerate(lines):
            if line.startswith('# Core:'):
                core_title = line.split(':', 1)[1].strip()
            if line.strip().startswith(datetime_col):
                skiprows = i
                break

        # Read CSV data
        df = CSVLoader._read_csv_with_encoding(file_path, skiprows=skiprows)
        df.columns = df.columns.str.strip()

        if datetime_col not in df.columns:
            raise ValueError(f"No '{datetime_col}' column found in {file_path.name}")

        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

        # Find depth columns and auto-assign sensor labels
        depth_cols = [col for col in df.columns if 'Depth' in col]
        if not depth_cols:
            raise ValueError(f"No depth columns found in {file_path.name}")

        # Rename to standard Sensor_X_Depth format
        rename_dict = {}
        sensor_labels = []
        original_depth_columns = {}  # Sensor_X_Depth -> original column name
        for i, col in enumerate(depth_cols):
            if i < len(SENSOR_LABELS):
                label = SENSOR_LABELS[i]
                new_name = f'Sensor_{label}_Depth'
                rename_dict[col] = new_name
                original_depth_columns[new_name] = col
                sensor_labels.append(label)

        df = df.rename(columns=rename_dict)

        # Keep original column name mapping for reference
        return SensorData(
            df=df,
            datetime_col=datetime_col,
            sensor_labels=sensor_labels,
            source_file=file_path.name,
            core_title=core_title,
            original_depth_columns=original_depth_columns,
        )

    @staticmethod
    def load_sensor_csv(
        file_path: str | Path,
        sensor_patterns: list[str],
        datetime_col: str = 'datetime',
        skip_rows: int = 0,
    ) -> SensorData:
        """
        Load a CSV file using sensor name patterns to find depth columns.
        
        This is for the older format where columns are identified by
        matching sensor serial numbers (e.g., '230405', '230406').
        
        Args:
            file_path: Path to the CSV file.
            sensor_patterns: List of patterns to match column names
                             (e.g., ['230405', '230406', '236222']).
            datetime_col: Name of the datetime column.
            skip_rows: Number of rows to skip.
            
        Returns:
            SensorData instance.
        """
        file_path = Path(file_path)

        # Try reading with comment='#' first, then skiprows
        df = None
        for encoding in ENCODINGS:
            try:
                df = pd.read_csv(file_path, comment='#', encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                try:
                    df = pd.read_csv(file_path, skiprows=skip_rows, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue

        if df is None:
            raise ValueError(f"Could not load {file_path.name} with any common encoding")

        # Find sensor columns by matching patterns
        sensor_cols = []
        for pattern in sensor_patterns:
            matches = [col for col in df.columns if pattern in col and 'Depth' in col]
            if not matches:
                raise ValueError(
                    f"Could not find column for sensor pattern: {pattern}\n"
                    f"Available columns: {list(df.columns)}"
                )
            sensor_cols.append(matches[0])

        # Rename to standard format
        rename_dict = {}
        sensor_labels = []
        for i, (orig_col, pattern) in enumerate(zip(sensor_cols, sensor_patterns)):
            if i < len(SENSOR_LABELS):
                label = SENSOR_LABELS[i]
                rename_dict[orig_col] = f'Sensor_{label}_Depth'
                sensor_labels.append(label)

        df = df.rename(columns=rename_dict)

        # Handle depth column and NaN
        depth_col_names = [f'Sensor_{label}_Depth' for label in sensor_labels]
        df_depth = df[[datetime_col] + depth_col_names].copy()
        df_depth = df_depth.dropna(subset=depth_col_names, how='all')

        return SensorData(
            df=df_depth,
            datetime_col=datetime_col,
            sensor_labels=sensor_labels,
            source_file=file_path.name,
        )

    @staticmethod
    def load_raw_csv(
        file_path: str | Path,
        datetime_col: str = 'datetime',
    ) -> tuple[pd.DataFrame, str, list[str]]:
        """
        Load a CSV with minimal processing - returns the raw DataFrame
        plus detected metadata. Useful when full control is needed.
        
        Returns:
            (DataFrame, core_title, depth_column_names)
        """
        file_path = Path(file_path)
        lines = CSVLoader._read_lines(file_path)

        core_title = ''
        skiprows = 0
        for i, line in enumerate(lines):
            if line.startswith('# Core:'):
                core_title = line.split(':', 1)[1].strip()
            if line.strip().startswith(datetime_col):
                skiprows = i
                break

        df = CSVLoader._read_csv_with_encoding(file_path, skiprows=skiprows)
        df.columns = df.columns.str.strip()

        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

        depth_cols = [col for col in df.columns if 'Depth' in col]
        return df, core_title, depth_cols

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_lines(file_path: Path) -> list[str]:
        """Read file lines, trying multiple encodings."""
        for encoding in ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.readlines()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"Could not decode {file_path.name} with any common encoding")

    @staticmethod
    def _read_csv_with_encoding(file_path: Path, **kwargs) -> pd.DataFrame:
        """Read CSV trying multiple encodings."""
        for encoding in ENCODINGS:
            try:
                return pd.read_csv(file_path, encoding=encoding, **kwargs)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"Could not read CSV {file_path.name} with any common encoding")
