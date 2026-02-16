"""
Statistics computation - pure business logic, no Qt dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.sensor_data import SensorData
from ..models.analysis_result import StatisticsResult


def compute_statistics(
    sensor_data: SensorData,
    start_idx: int | None = None,
    end_idx: int | None = None,
    sensor_patterns: dict[str, str] | None = None,
) -> StatisticsResult:
    """
    Compute pairwise statistics for sensor depth data.
    
    Args:
        sensor_data: The SensorData instance.
        start_idx: Start index of selection (None for full range).
        end_idx: End index of selection (None for full range).
        sensor_patterns: Maps sensor label -> pattern/identifier string.
        
    Returns:
        StatisticsResult with all computed statistics.
    """
    df = sensor_data.df
    depth_cols = sensor_data.depth_columns
    labels = sensor_data.sensor_labels

    if start_idx is not None and end_idx is not None:
        df_calc = df.iloc[start_idx:end_idx + 1]
        range_desc = f"Selected range: rows {start_idx}-{end_idx} ({len(df_calc)} points)"
    else:
        df_calc = df
        range_desc = "Full dataset (no selection)"

    # Mean depth across all sensors
    existing_cols = [c for c in depth_cols if c in df_calc.columns]
    mean_depth_all = float(df_calc[existing_cols].mean(axis=1).mean())

    # Per-sensor means
    sensor_means = {}
    for label in labels:
        col = f'Sensor_{label}_Depth'
        if col in df_calc.columns:
            sensor_means[label] = float(df_calc[col].mean())

    # Pairwise difference means
    diff_means = {}
    for i, label_i in enumerate(labels):
        for j in range(i + 1, len(labels)):
            label_j = labels[j]
            col_i = f'Sensor_{label_i}_Depth'
            col_j = f'Sensor_{label_j}_Depth'
            if col_i in df_calc.columns and col_j in df_calc.columns:
                diff_means[(label_j, label_i)] = float(
                    (df_calc[col_j] - df_calc[col_i]).mean()
                )

    # Sensor identifiers
    idents = {}
    if sensor_patterns:
        for label in labels:
            if label in sensor_patterns:
                idents[label] = sensor_patterns[label]

    return StatisticsResult(
        range_description=range_desc,
        n_points=len(df_calc),
        mean_depth_all_sensors=mean_depth_all,
        sensor_means=sensor_means,
        difference_means=diff_means,
        sensor_identifiers=idents,
        source_file=sensor_data.source_file,
    )
