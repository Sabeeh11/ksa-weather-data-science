"""End-to-end runner for the Saudi weather data science project."""

from __future__ import annotations

from pathlib import Path

from preprocessing import load_data, preprocess_weather_data
from feature_engineering import add_weather_features, get_feature_matrix
from clustering import run_kmeans
from modeling import train_and_evaluate, shuffled_label_validation
from pattern_mining import build_apriori_transactions, mine_association_rules
from visualization import plot_model_accuracy, plot_station_counts, plot_station_completeness, plot_weather_patterns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "saudi-hourly-weather-data_Historical.csv"
PLOTS_DIR = ROOT / "outputs" / "plots"


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place the raw NOAA CSV in the data/ folder before running."
        )

    print("Loading data...")
    df = load_data(str(DATA_PATH))

    print("Preprocessing data...")
    df_clean = preprocess_weather_data(df)

    print("Engineering features...")
    df_features = add_weather_features(df_clean)
    X = get_feature_matrix(df_features)

    print("Running K-Means clustering...")
    df_clustered, kmeans, scaler = run_kmeans(df_features, X)

    print("Training Random Forest classifier...")
    y = df_clustered["WEATHER_PATTERN"]
    real_results = train_and_evaluate(X, y)
    shuffled_results = shuffled_label_validation(X, y)

    print("\nClassification report:")
    print(real_results["classification_report"])
    print("\nConfusion matrix:")
    print(real_results["confusion_matrix"])

    print("\nShuffled-label accuracy:", round(shuffled_results["accuracy"], 4))

    print("Mining association rules...")
    transactions = build_apriori_transactions(df_clustered)
    rules = mine_association_rules(transactions)
    print(rules.head(10))

    print("Saving plots...")
    plot_station_counts(df_clustered, str(PLOTS_DIR / "station_counts.png"))
    plot_weather_patterns(df_clustered, str(PLOTS_DIR / "weather_patterns.png"))
    plot_model_accuracy(
        real_results["accuracy"],
        shuffled_results["accuracy"],
        str(PLOTS_DIR / "model_accuracy_validation.png"),
    )
    plot_station_completeness(df, str(PLOTS_DIR / "station_completeness.png"))

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
