import json
import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from spotify_popularity.api import app
from spotify_popularity.features import FeatureEncoder
from spotify_popularity.model import ModelBundle, make_classifier, make_regressor


@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    records = pd.read_csv("data/spotify_tracks.csv", index_col=0).head(80)
    records = records.assign(genres=[["acoustic"], ["pop"]] * (len(records) // 2))
    encoder = FeatureEncoder().fit(records)
    X = encoder.transform(records)
    regressor = make_regressor(encoder.genre_columns_)
    classifier = make_classifier(encoder.genre_columns_)
    regressor.set_params(model__n_estimators=2, model__n_jobs=1)
    classifier.set_params(model__n_estimators=2, model__n_jobs=1)
    regressor.fit(X, records.popularity)
    classifier.fit(X, (records.popularity > 50).astype(int))
    bundle = ModelBundle(encoder, regressor, classifier, {})
    path = Path(
        os.environ.get(
            "TEST_MODEL_PATH", tmp_path_factory.mktemp("model") / "model.joblib"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path, bundle


@pytest.fixture
def payload():
    record = json.loads(Path("examples/track.json").read_text())
    record["genres"] = ["acoustic"]
    return record


@pytest.fixture
def client(artifact, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(artifact[0]))
    with TestClient(app) as client:
        yield client
