from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from spotify_popularity.features import (
    CATEGORICAL_FEATURES,
    HIT_THRESHOLD,
    NUMERICAL_FEATURES,
    PROBABILITY_THRESHOLD,
    FeatureEncoder,
)

REGRESSOR_PARAMETERS = {
    "n_estimators": 200,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.75,
    "max_depth": None,
    "bootstrap": True,
    "random_state": 1,
    "n_jobs": -1,
}
CLASSIFIER_PARAMETERS = {
    "n_estimators": 100,
    "max_features": 0.75,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": 1,
    "n_jobs": -1,
}


def make_preparation(genre_columns):
    return ColumnTransformer([
        ("numerical", "passthrough", NUMERICAL_FEATURES),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore", drop="first", sparse_output=False
            ),
            CATEGORICAL_FEATURES,
        ),
        ("genres", "passthrough", genre_columns),
    ])


def make_regressor(genre_columns):
    return Pipeline([
        ("preparation", make_preparation(genre_columns)),
        ("model", RandomForestRegressor(**REGRESSOR_PARAMETERS)),
    ])


def make_classifier(genre_columns):
    return Pipeline([
        ("preparation", make_preparation(genre_columns)),
        ("model", RandomForestClassifier(**CLASSIFIER_PARAMETERS)),
    ])


@dataclass
class ModelBundle:
    """Serve score, hit probability and decision from the trained pair."""

    features: FeatureEncoder
    regressor: Pipeline
    classifier: Pipeline
    metadata: dict
    hit_threshold: ClassVar[int] = HIT_THRESHOLD
    probability_threshold: ClassVar[float] = PROBABILITY_THRESHOLD

    def predict(self, records):
        encoded = self.features.transform(pd.DataFrame(records))
        popularity = self.regressor.predict(encoded)
        hit_probability = self.classifier.predict_proba(encoded)[:, 1]
        return [
            {
                "popularity": float(score),
                "hit_probability": float(probability),
                "hit": bool(probability >= PROBABILITY_THRESHOLD),
            }
            for score, probability in zip(popularity, hit_probability)
        ]
