"""Plotting helpers for project insights."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_station_counts(df: pd.DataFrame, output_path: str | None = None) -> None:
    station_counts = df["STATION_NAME"].value_counts()

    plt.figure(figsize=(12, 8))
    plt.barh(station_counts.index, station_counts.values)
    plt.title("Number of High-Quality Observations per Station")
    plt.xlabel("Number of Records")
    plt.ylabel("Station Name")

    for i, value in enumerate(station_counts.values):
        plt.text(value + 500, i, str(value), va="center", fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_weather_patterns(df: pd.DataFrame, output_path: str | None = None) -> None:
    counts = df["WEATHER_PATTERN_NAME"].value_counts()
    percentages = counts / counts.sum() * 100

    plt.figure(figsize=(12, 7))
    plt.bar(counts.index, counts.values)
    plt.title("Distribution of Airport Weather Patterns in Saudi Arabia")
    plt.xlabel("Weather Pattern Category")
    plt.ylabel("Number of Occurrences")
    plt.xticks(rotation=25, ha="right")

    for i, (count, pct) in enumerate(zip(counts.values, percentages.values)):
        plt.text(i, count + 1500, f"{count} ({pct:.1f}%)", ha="center", fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_model_accuracy(real_accuracy: float, shuffled_accuracy: float, output_path: str | None = None) -> None:
    labels = ["Real Labels", "Shuffled Labels"]
    values = [real_accuracy, shuffled_accuracy]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Model Accuracy: Real vs Shuffled Labels")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    for i, value in enumerate(values):
        plt.text(i, value + 0.03, f"{value:.2f}", ha="center", fontsize=12)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_station_completeness(df: pd.DataFrame, output_path: str | None = None) -> None:
    important_features = [
        "WIND_DIRECTION_ANGLE",
        "WIND_SPEED_RATE",
        "SKY_CEILING_HEIGHT",
        "VISIBILITY_DISTANCE",
        "AIR_TEMPERATURE",
        "AIR_TEMPERATURE_DEW_POINT",
        "ATMOSPHERIC_SEA_LEVEL_PRESSURE",
    ]

    station_feature_matrix = (
        df.groupby("STATION_NAME")[important_features]
        .apply(lambda x: x.notna().mean())
        .sort_values(by="AIR_TEMPERATURE")
    )

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        station_feature_matrix,
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"label": "Data Completeness (0 = missing, 1 = present)"},
    )

    plt.title("Station-Level Completeness of Key Weather Features")
    plt.xlabel("Weather Features")
    plt.ylabel("Station Name")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()
