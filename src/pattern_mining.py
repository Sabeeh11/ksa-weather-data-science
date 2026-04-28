"""Apriori association rule mining for weather events."""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def build_apriori_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create boolean event flags used for association rule mining."""
    transactions = pd.DataFrame(index=df.index)

    transactions["LOW_VISIBILITY"] = df["VISIBILITY_DISTANCE"] < 1000
    transactions["HIGH_VISIBILITY"] = df["VISIBILITY_DISTANCE"] >= 1000

    transactions["HIGH_HUMIDITY"] = df["RELATIVE_HUMIDITY"] > 80
    transactions["DRY_AIR"] = df["RELATIVE_HUMIDITY"] < 40

    transactions["FOG_LIKELY"] = df["TEMP_DEW_SPREAD"] < 2

    transactions["LOW_CEILING"] = df["SKY_CEILING_PRESENT"] == 1
    transactions["NO_CEILING"] = df["SKY_CEILING_PRESENT"] == 0

    transactions["STRONG_WIND"] = df["WIND_SPEED_RATE"] > 6
    transactions["CALM_WIND"] = df["WIND_SPEED_RATE"] < 2

    transactions["FALLING_PRESSURE"] = df["PRESSURE_TENDENCY"] < 0
    transactions["RISING_PRESSURE"] = df["PRESSURE_TENDENCY"] > 0

    return transactions.astype(bool)


def mine_association_rules(
    transactions: pd.DataFrame,
    min_support: float = 0.50,
    min_confidence: float = 0.50,
) -> pd.DataFrame:
    """Run Apriori and return sorted association rules."""
    frequent_items = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values("lift", ascending=False)
