"""
Depth correction processing - pure business logic, no Qt dependencies.

Applies depth-dependent (regression-based) and/or manual constant offsets
to sensor data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.sensor_data import SensorData, CorrectionRecord
from ..models.calibration import DepthCalibration


class DepthCorrectionProcessor:
    """
    Applies depth corrections to a SensorData instance.
    
    Can apply:
    - Regression-based corrections from a DepthCalibration
    - Manual constant offsets per sensor
    """

    @staticmethod
    def apply_calibration(
        sensor_data: SensorData,
        calibration: DepthCalibration,
        ref_sensor: str,
        enabled_targets: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Apply depth-dependent calibration corrections.
        
        Args:
            sensor_data: The SensorData to modify (mutated in place).
            calibration: The DepthCalibration containing regression models.
            ref_sensor: The reference sensor label (will not be modified).
            enabled_targets: List of sensor labels to correct. If None, all applicable.
            
        Returns:
            dict mapping sensor_label -> {mean, std, min, max} of applied corrections.
        """
        applicable = calibration.get_applicable_corrections(ref_sensor)
        
        # Find the reference column for depth lookup
        ref_col = f'Sensor_{ref_sensor}_Depth'
        if ref_col not in sensor_data.df.columns:
            raise ValueError(f"Reference sensor {ref_sensor} column not found: {ref_col}")

        ref_depths = sensor_data.df[ref_col].values
        results = {}

        for item in applicable:
            target = item['target']
            if enabled_targets is not None and target not in enabled_targets:
                continue

            reg = item['reg']
            sign = item['sign']
            target_col = f'Sensor_{target}_Depth'

            if target_col not in sensor_data.df.columns:
                continue

            # Depth-dependent offset
            offsets = reg.slope * ref_depths + reg.intercept
            corrections = sign * offsets

            # Apply
            original = sensor_data.df[target_col].values
            sensor_data.update_depth_column(target, original + corrections)

            # Record
            sensor_data.add_correction(CorrectionRecord(
                correction_type='depth_calibration',
                sensor_label=target,
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

            results[target] = {
                'mean': float(np.nanmean(corrections)),
                'std': float(np.nanstd(corrections)),
                'min': float(np.nanmin(corrections)),
                'max': float(np.nanmax(corrections)),
            }

        return results

    @staticmethod
    def apply_manual_offsets(
        sensor_data: SensorData,
        offsets: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply constant manual offsets to specified sensors.
        
        Args:
            sensor_data: The SensorData to modify (mutated in place).
            offsets: Maps sensor_label -> offset_meters.
            
        Returns:
            dict of actually applied offsets (non-zero only).
        """
        applied = {}
        for label, offset in offsets.items():
            if offset == 0.0:
                continue
            col = f'Sensor_{label}_Depth'
            if col not in sensor_data.df.columns:
                continue

            original = sensor_data.df[col].values
            sensor_data.update_depth_column(label, original + offset)

            sensor_data.add_correction(CorrectionRecord(
                correction_type='depth_manual',
                sensor_label=label,
                description=f"Manual offset: {offset:+.4f}m",
                parameters={'offset_m': offset},
            ))
            applied[label] = offset

        return applied
