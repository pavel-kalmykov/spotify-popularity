import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

NUMERICAL_FEATURES = [
    "duration_ms", "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
CATEGORICAL_FEATURES = ["explicit", "key", "mode", "time_signature"]
HIT_THRESHOLD = 50
PROBABILITY_THRESHOLD = 0.5


def clean_dataset(frame):
    unique_rows = frame.drop_duplicates()
    return unique_rows.assign(
        popularity=unique_rows.groupby("track_id").popularity.transform("max")
    )


def track_records(frame):
    """Return one raw feature record and target per track, preserving every genre."""
    tracks = frame.groupby("track_id").first()
    records = tracks[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    records["genres"] = (
        frame.groupby("track_id").track_genre
        .agg(lambda values: sorted(set(values)))
        .reindex(tracks.index)
    )
    return records, tracks.popularity


def prepare_base_features(records):
    features = records[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    features["duration_ms"] = np.log1p(features.duration_ms)
    features["time_signature"] = features.time_signature.where(
        features.time_signature.isin([3, 4, 5, 6, 7]), "unknown"
    )
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(str)
    return features


class FeatureEncoder(TransformerMixin, BaseEstimator):
    """Apply the notebook's feature preparation to raw tracks with genre lists."""

    def fit(self, X, y=None):
        self.genre_encoder_ = MultiLabelBinarizer().fit(X["genres"])
        self.genre_columns_ = [
            f"genre={genre}" for genre in self.genre_encoder_.classes_
        ]
        return self

    def transform(self, X):
        base = prepare_base_features(X)
        encoded = self.genre_encoder_.transform(X["genres"])
        return base.join(pd.DataFrame(
            encoded, index=X.index, columns=self.genre_columns_
        ))
