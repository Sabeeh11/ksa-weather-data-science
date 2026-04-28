"""Feature engineering functions for meteorological modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "AIR_TEMPERATURE",
    "AIR_TEMPERATURE_DEW_POINT",
    "TEMP_DEW_SPREAD",
    "RELATIVE_HUMIDITY",
    "ATMOSPHERIC_SEA_LEVEL_PRESSURE",
    "PRESSURE_TENDENCY",
    "WIND_SPEED_RATE",
    "WIND_U",
    "WIND_V",
    "VISIBILITY_PRESENT",
    "SKY_CEILING_PRESENT",
]


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived weather features used for clustering and classification."""
    df = df.copy()

    df["TEMP_DEW_SPREAD"] = (
        df["AIR_TEMPERATURE"] - df["AIR_TEMPERATURE_DEW_POINT"]
    )

    temperature = df["AIR_TEMPERATURE"].clip(-80, 60)
    dew_point = df["AIR_TEMPERATURE_DEW_POINT"].clip(-80, 60)

    saturation_vapor_pressure = 6.112 * np.exp((17.67 * temperature) / (temperature + 243.5))
    actual_vapor_pressure = 6.112 * np.exp((17.67 * dew_point) / (dew_point + 243.5))

    relative_humidity = 100 * (actual_vapor_pressure / saturation_vapor_pressure)
    df["RELATIVE_HUMIDITY"] = relative_humidity.clip(0, 100)

    theta = np.deg2rad(df["WIND_DIRECTION_ANGLE"].fillna(0))
    df["WIND_U"] = df["WIND_SPEED_RATE"] * np.cos(theta)
    df["WIND_V"] = df["WIND_SPEED_RATE"] * np.sin(theta)

    df = df.sort_values(["STATION_ID", "OBSERVATION_DATE"])
    df["PRESSURE_TENDENCY"] = (
        df.groupby("STATION_ID")["ATMOSPHERIC_SEA_LEVEL_PRESSURE"].diff()
    )
    df["PRESSURE_TENDENCY"] = df["PRESSURE_TENDENCY"].fillna(0)

    return df


def get_feature_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Return the modeling feature matrix."""
    feature_cols = feature_cols or FEATURE_COLS
    return df[feature_cols].copy()
