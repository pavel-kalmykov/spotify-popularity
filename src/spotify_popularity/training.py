import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from spotify_popularity.features import (
    HIT_THRESHOLD,
    PROBABILITY_THRESHOLD,
    FeatureEncoder,
    clean_dataset,
    track_records,
)
from spotify_popularity.model import (
    CLASSIFIER_PARAMETERS,
    REGRESSOR_PARAMETERS,
    ModelBundle,
    make_classifier,
    make_regressor,
)


def train(dataset, output_dir):
    frame = clean_dataset(pd.read_csv(dataset, index_col=0))
    training_ids, test_ids = train_test_split(
        frame.track_id.unique(), test_size=0.2, random_state=1
    )
    training, y_train = track_records(frame[frame.track_id.isin(training_ids)])
    test, y_test = track_records(frame[frame.track_id.isin(test_ids)])
    assert set(training.index).isdisjoint(test.index)

    features = FeatureEncoder().fit(training)
    X_train = features.transform(training)
    X_test = features.transform(test)
    regressor = make_regressor(features.genre_columns_).fit(X_train, y_train)
    y_train_hit = (y_train > HIT_THRESHOLD).astype(int)
    y_test_hit = (y_test > HIT_THRESHOLD).astype(int)
    classifier = make_classifier(features.genre_columns_).fit(X_train, y_train_hit)

    predictions = regressor.predict(X_test)
    probabilities = classifier.predict_proba(X_test)[:, 1]
    hits = probabilities >= PROBABILITY_THRESHOLD
    baseline_rmse = root_mean_squared_error(
        y_test, np.full(len(y_test), y_train.mean())
    )
    metrics = {
        "regression": {
            "baseline_rmse": float(baseline_rmse),
            "test_rmse": float(root_mean_squared_error(y_test, predictions)),
        },
        "classification": {
            "test_roc_auc": float(roc_auc_score(y_test_hit, probabilities)),
            "test_precision": float(precision_score(y_test_hit, hits, zero_division=0)),
            "test_recall": float(recall_score(y_test_hit, hits, zero_division=0)),
            "test_f1": float(f1_score(y_test_hit, hits, zero_division=0)),
        },
    }
    with dataset.open("rb") as stream:
        dataset_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "training_tracks": len(training),
        "test_tracks": len(test),
        "dataset_sha256": dataset_sha256,
        "sklearn_version": sklearn.__version__,
        "regressor_parameters": REGRESSOR_PARAMETERS,
        "classifier_parameters": CLASSIFIER_PARAMETERS,
        "metrics": metrics,
    }
    bundle = ModelBundle(features, regressor, classifier, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "model.joblib", compress=3)
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train the selected popularity models.")
    parser.add_argument("--data", type=Path, default=Path("data/spotify_tracks.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    train(args.data, args.output_dir)


if __name__ == "__main__":
    main()
