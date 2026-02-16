"""
Calibration builder - generates DepthCalibration from collected statistics.

Pure business logic, no Qt dependencies.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ..models.sensor_data import SENSOR_LABELS
from ..models.calibration import DepthCalibration, RegressionEntry
from ..models.analysis_result import StatisticsResult


class CalibrationBuilder:
    """Builds a DepthCalibration from a collection of StatisticsResult entries."""

    @staticmethod
    def build(
        statistics: list[StatisticsResult],
        num_sensors: int,
        sensor_labels: list[str] | None = None,
    ) -> DepthCalibration:
        """
        Generate a DepthCalibration from collected statistics.
        
        Fits a linear regression for each sensor pair difference vs mean depth.
        
        Args:
            statistics: List of StatisticsResult from different depths/casts.
            num_sensors: Number of sensors.
            sensor_labels: Sensor labels (defaults to SENSOR_LABELS[:num_sensors]).
            
        Returns:
            DepthCalibration with regression entries.
        """
        if len(statistics) < 2:
            raise ValueError("Need at least 2 statistics entries for regression")

        if sensor_labels is None:
            sensor_labels = SENSOR_LABELS[:num_sensors]

        depth_data = np.array([s.mean_depth_all_sensors for s in statistics])

        regressions = []
        for i in range(num_sensors):
            for j in range(i + 1, num_sensors):
                label_i = sensor_labels[i]
                label_j = sensor_labels[j]

                diff_data = np.array([
                    s.difference_means.get((label_j, label_i), np.nan)
                    for s in statistics
                ])

                # Remove NaN
                mask = ~(np.isnan(depth_data) | np.isnan(diff_data))
                depth_clean = depth_data[mask]
                diff_clean = diff_data[mask]

                if len(depth_clean) < 2:
                    continue

                res = sp_stats.linregress(depth_clean, diff_clean)
                regressions.append(RegressionEntry(
                    sensor_i=label_i,
                    sensor_j=label_j,
                    slope=float(res.slope),
                    intercept=float(res.intercept),
                    r_squared=float(res.rvalue ** 2),
                    p_value=float(res.pvalue),
                ))

        return DepthCalibration(
            num_sensors=num_sensors,
            sensor_labels=sensor_labels,
            regressions=regressions,
        )

    @staticmethod
    def predict_offsets(
        calibration: DepthCalibration,
        target_depth: float,
    ) -> dict[tuple[str, str], float]:
        """
        Predict offsets at a given depth from the calibration.
        
        Returns dict mapping (sensor_j, sensor_i) -> predicted offset.
        """
        predictions = {}
        for reg in calibration.regressions:
            offset = reg.predict_offset(target_depth)
            predictions[(reg.sensor_j, reg.sensor_i)] = offset
        return predictions
