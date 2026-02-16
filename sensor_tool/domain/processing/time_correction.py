"""
Time correction processing - pure business logic, no Qt dependencies.

Calculates time offsets between sensors using the Savitzky-Golay velocity
difference method, and applies time shifts to align sensors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..models.sensor_data import SensorData, CorrectionRecord
from ..models.analysis_result import TimeOffsetResult


class TimeCorrectionProcessor:
    """
    Calculates and applies time lag corrections.
    """

    @staticmethod
    def calculate_offsets(
        sensor_data: SensorData,
        start_idx: int,
        end_idx: int,
        ref_sensor: str,
        window_length: int = 51,
        polyorder: int = 3,
        max_offset: float = 5.0,
        time_step: float = 0.025,
    ) -> list[TimeOffsetResult]:
        """
        Calculate optimal time offsets using the Savgol velocity difference method.
        
        Args:
            sensor_data: The data to analyze.
            start_idx: Start row index of the selected range (inclusive).
            end_idx: End row index of the selected range (inclusive).
            ref_sensor: Reference sensor label.
            window_length: Savgol filter window (must be odd).
            polyorder: Savgol polynomial order.
            max_offset: Maximum offset to search (seconds).
            time_step: Time step for offset search grid (seconds).
            
        Returns:
            List of TimeOffsetResult for each sensor.
        """
        if window_length % 2 == 0:
            window_length += 1

        datetime_col = sensor_data.datetime_col
        sensor_labels = sensor_data.sensor_labels
        depth_cols = sensor_data.depth_columns
        ref_idx = sensor_labels.index(ref_sensor)
        ref_col = depth_cols[ref_idx]

        # Extract and prepare selected data
        df_sel = sensor_data.df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)
        df_sel[datetime_col] = pd.to_datetime(df_sel[datetime_col])
        time_diff = df_sel[datetime_col].diff().dt.total_seconds()

        # Apply Savgol filter to each sensor
        for col in depth_cols:
            if col in df_sel.columns:
                data = df_sel[col].values
                if np.sum(~np.isnan(data)) >= window_length:
                    df_sel[f'{col}_filtered'] = savgol_filter(
                        data, window_length=window_length, polyorder=polyorder
                    )
                else:
                    df_sel[f'{col}_filtered'] = data

        time_offsets_array = np.arange(-max_offset, max_offset + time_step, time_step)

        results = []
        for i, col in enumerate(depth_cols):
            label = sensor_labels[i]

            if i == ref_idx:
                results.append(TimeOffsetResult(
                    sensor_label=label, offset_seconds=0.0, rms_value=0.0, is_reference=True
                ))
                continue

            if col not in df_sel.columns:
                continue

            rms_values = []
            for offset in time_offsets_array:
                rms = TimeCorrectionProcessor._compute_velocity_rms(
                    df_sel, ref_col, col, time_diff, offset
                )
                rms_values.append(rms)

            best_idx = int(np.argmin(rms_values))
            optimal_offset = float(time_offsets_array[best_idx])
            optimal_rms = float(rms_values[best_idx])

            results.append(TimeOffsetResult(
                sensor_label=label,
                offset_seconds=optimal_offset,
                rms_value=optimal_rms,
            ))

        return results

    @staticmethod
    def apply_offsets(
        sensor_data: SensorData,
        offsets: list[TimeOffsetResult],
    ) -> SensorData:
        """
        Apply time offsets by shifting sensor data.
        
        Returns a new SensorData with corrected values.
        """
        datetime_col = sensor_data.datetime_col
        corrected_df = sensor_data.df.copy()
        time_step_median = corrected_df[datetime_col].diff().dt.total_seconds().median()

        for result in offsets:
            if result.is_reference or result.offset_seconds == 0.0:
                continue

            col = f'Sensor_{result.sensor_label}_Depth'
            if col not in corrected_df.columns:
                continue

            # Shift by reindexing
            shifted_index = corrected_df.index + result.offset_seconds / time_step_median
            df_shifted = pd.DataFrame(
                {col: corrected_df[col].values},
                index=shifted_index,
            )
            corrected_df[col] = df_shifted[col].reindex(
                corrected_df.index, method='nearest', tolerance=5
            )

        new_data = sensor_data.copy()
        new_data.update_dataframe(corrected_df)

        for result in offsets:
            if not result.is_reference and result.offset_seconds != 0.0:
                new_data.add_correction(CorrectionRecord(
                    correction_type='time_shift',
                    sensor_label=result.sensor_label,
                    description=f"Time shift: {result.offset_seconds:+.3f}s",
                    parameters={
                        'offset_seconds': result.offset_seconds,
                        'rms': result.rms_value,
                    },
                ))

        return new_data

    @staticmethod
    def compute_velocity_differences(
        sensor_data: SensorData,
        ref_sensor: str,
        window_length: int = 51,
        polyorder: int = 3,
    ) -> dict[str, pd.Series]:
        """
        Compute Savgol-filtered velocity differences between each sensor and reference.
        
        Returns dict mapping sensor_label -> velocity difference Series.
        """
        if window_length % 2 == 0:
            window_length += 1

        datetime_col = sensor_data.datetime_col
        depth_cols = sensor_data.depth_columns
        labels = sensor_data.sensor_labels
        ref_idx = labels.index(ref_sensor)
        ref_col = depth_cols[ref_idx]

        df = sensor_data.df.copy()
        time_diff = df[datetime_col].diff().dt.total_seconds()

        # Savgol filter
        for col in depth_cols:
            if col in df.columns:
                data = df[col].values
                if np.sum(~np.isnan(data)) >= window_length:
                    df[f'{col}_filtered'] = savgol_filter(
                        data, window_length=window_length, polyorder=polyorder
                    )
                else:
                    df[f'{col}_filtered'] = data

        v_ref = df[f'{ref_col}_filtered'].diff() / time_diff

        diffs = {}
        for i, col in enumerate(depth_cols):
            if i == ref_idx or col not in df.columns:
                continue
            v_sensor = df[f'{col}_filtered'].diff() / time_diff
            diffs[labels[i]] = v_sensor - v_ref

        return diffs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_velocity_rms(
        df: pd.DataFrame,
        ref_col: str,
        target_col: str,
        time_diff: pd.Series,
        time_offset_seconds: float,
    ) -> float:
        """Compute RMS of velocity difference with a given time offset."""
        shifted_index = df.index + time_offset_seconds / time_diff.median()

        df_temp = pd.DataFrame(
            {f'{target_col}_filtered': df[f'{target_col}_filtered'].values},
            index=shifted_index,
        )

        col_interp = df_temp[f'{target_col}_filtered'].reindex(
            df.index, method='nearest', tolerance=5
        )

        v_ref = df[f'{ref_col}_filtered'].diff() / time_diff
        v_target = col_interp.diff() / time_diff

        rms = float(np.sqrt(np.nanmean((v_ref - v_target) ** 2)))
        return rms
