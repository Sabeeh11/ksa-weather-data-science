"""Data cleaning and preprocessing utilities for the Saudi weather project."""

from __future__ import annotations

import pandas as pd
import numpy as np


CONTINUOUS_COLS = [
    "AIR_TEMPERATURE",
    "AIR_TEMPERATURE_DEW_POINT",
    "ATMOSPHERIC_SEA_LEVEL_PRESSURE",
    "WIND_SPEED_RATE",
]

ESSENTIAL_COLS = [
    "AIR_TEMPERATURE",
    "AIR_TEMPERATURE_DEW_POINT",
    "ATMOSPHERIC_SEA_LEVEL_PRESSURE",
    "WIND_SPEED_RATE",
]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw NOAA weather dataset."""
    return pd.read_csv(path)


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physical constraints and basic cleanup to the raw weather data."""
    df = df.copy()

    if "OBSERVATION_DATE" in df.columns:
        df["OBSERVATION_DATE"] = pd.to_datetime(df["OBSERVATION_DATE"], errors="coerce")

    if "LAST_LOAD_DATE" in df.columns:
        df = df.drop(columns=["LAST_LOAD_DATE"])

    df["WIND_DIRECTION_ANGLE"] = pd.to_numeric(df["WIND_DIRECTION_ANGLE"], errors="coerce")
    df.loc[
        (df["WIND_DIRECTION_ANGLE"] > 360) | (df["WIND_DIRECTION_ANGLE"] < 0),
        "WIND_DIRECTION_ANGLE",
    ] = np.nan

    if "WIND_TYPE" in df.columns:
        df.loc[df["WIND_TYPE"] == "Missing", "WIND_TYPE"] = np.nan

    df["WIND_SPEED_RATE_RAW"] = df["WIND_SPEED_RATE"]
    df["WIND_SPEED_RATE"] = pd.to_numeric(df["WIND_SPEED_RATE"], errors="coerce")
    df.loc[df["WIND_SPEED_RATE"] >= 100, "WIND_SPEED_RATE"] = np.nan

    df["SKY_CEILING_HEIGHT_RAW"] = df["SKY_CEILING_HEIGHT"]
    df["SKY_CEILING_HEIGHT"] = pd.to_numeric(df["SKY_CEILING_HEIGHT"], errors="coerce")
    df.loc[df["SKY_CEILING_HEIGHT"] > 3700, "SKY_CEILING_HEIGHT"] = np.nan
    df["SKY_CEILING_PRESENT"] = df["SKY_CEILING_HEIGHT"].notna().map({True: "yes", False: "no"})

    df["VISIBILITY_DISTANCE"] = pd.to_numeric(df["VISIBILITY_DISTANCE"], errors="coerce")
    df.loc[df["VISIBILITY_DISTANCE"] > 10000, "VISIBILITY_DISTANCE"] = np.nan
    df["VISIBILITY_PRESENT"] = df["VISIBILITY_DISTANCE"].notna().map({True: "yes", False: "no"})

    df["AIR_TEMPERATURE"] = pd.to_numeric(df["AIR_TEMPERATURE"], errors="coerce")
    df.loc[df["AIR_TEMPERATURE"] >= 99, "AIR_TEMPERATURE"] = np.nan

    df["AIR_TEMPERATURE_DEW_POINT"] = pd.to_numeric(
        df["AIR_TEMPERATURE_DEW_POINT"], errors="coerce"
    )
    df.loc[
        (df["AIR_TEMPERATURE_DEW_POINT"] >= 35)
        | (df["AIR_TEMPERATURE_DEW_POINT"] <= -60),
        "AIR_TEMPERATURE_DEW_POINT",
    ] = np.nan

    df["ATMOSPHERIC_SEA_LEVEL_PRESSURE"] = pd.to_numeric(
        df["ATMOSPHERIC_SEA_LEVEL_PRESSURE"], errors="coerce"
    )
    df.loc[
        (df["ATMOSPHERIC_SEA_LEVEL_PRESSURE"] < 870)
        | (df["ATMOSPHERIC_SEA_LEVEL_PRESSURE"] > 1080),
        "ATMOSPHERIC_SEA_LEVEL_PRESSURE",
    ] = np.nan

    return df


def select_high_quality_stations(
    df: pd.DataFrame,
    essential_cols: list[str] | None = None,
    missing_threshold: float = 0.40,
) -> pd.DataFrame:
    """Keep stations with less than the threshold missingness in all essential columns."""
    essential_cols = essential_cols or ESSENTIAL_COLS
    missing_by_station = df.groupby("STATION_ID").agg(lambda x: x.isna().mean())
    high_quality_stations = missing_by_station[essential_cols].max(axis=1) < missing_threshold
    station_ids = high_quality_stations[high_quality_stations].index
    return df[df["STATION_ID"].isin(station_ids)].copy()


def encode_binary_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yes/no weather indicators to 1/0."""
    df = df.copy()
    df["SKY_CEILING_PRESENT"] = df["SKY_CEILING_PRESENT"].map({"yes": 1, "no": 0})
    df["VISIBILITY_PRESENT"] = df["VISIBILITY_PRESENT"].map({"yes": 1, "no": 0})
    return df


def impute_station_means(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Fill continuous missing values using station-specific means."""
    df = df.copy()
    continuous_cols = continuous_cols or CONTINUOUS_COLS
    df[continuous_cols] = (
        df.groupby("STATION_ID")[continuous_cols]
        .transform(lambda x: x.fillna(x.mean()))
    )
    return df


def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete preprocessing workflow."""
    df = clean_weather_data(df)
    df = select_high_quality_stations(df)
    df = encode_binary_indicators(df)
    df = impute_station_means(df)
    return df
