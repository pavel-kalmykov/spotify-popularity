import pytest
from fastapi.testclient import TestClient

from spotify_popularity.api import app


@pytest.mark.parametrize("genres", [["acoustic"], ["acoustic", "pop"]])
def test_prediction_matches_artifact(client, artifact, payload, genres):
    # given
    payload["genres"] = genres
    expected = artifact[1].predict([payload])[0]
    # when
    response = client.post("/predict", json=payload)
    # then
    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize(
    "field",
    [
        "duration_ms",
        "genres",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "key",
        "mode",
        "explicit",
        "time_signature",
    ],
)
def test_missing_field(client, payload, field):
    # given
    del payload[field]
    # when
    response = client.post("/predict", json=payload)
    # then
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy", None),
        ("loudness", "NaN"),
        ("tempo", -1),
        ("duration_ms", -1),
        ("key", -2),
        ("key", 12),
        ("mode", 2),
        ("time_signature", 8),
        ("genres", []),
        ("genres", ["unknown"]),
        ("genres", ["acoustic", "unknown"]),
        ("unexpected", 1),
    ],
)
def test_invalid_input(client, payload, field, value):
    # given
    payload[field] = value
    # when
    response = client.post("/predict", json=payload)
    # then
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy", 0),
        ("energy", 1),
        ("duration_ms", 0),
        ("tempo", 0),
        ("key", -1),
        ("key", 11),
        ("mode", 0),
        ("mode", 1),
        ("time_signature", 0),
        ("time_signature", 1),
        ("time_signature", 7),
    ],
)
def test_valid_boundaries(client, payload, field, value):
    # given
    payload[field] = value
    expected = client.app.state.bundle.predict([payload])[0]
    # when
    response = client.post("/predict", json=payload)
    # then
    assert response.status_code == 200
    assert response.json() == expected


def test_health_reports_documented_thresholds(client):
    # when
    response = client.get("/health")
    # then
    assert response.json() == {
        "status": "ok",
        "hit_threshold": 50,
        "probability_threshold": 0.5,
    }


def test_genres_reports_fitted_vocabulary(client):
    # when
    response = client.get("/genres")
    # then
    assert response.json() == {"genres": ["acoustic", "pop"]}


def test_missing_artifact_prevents_startup(tmp_path, monkeypatch):
    # given
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    # when / then
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass


def test_corrupt_artifact_prevents_startup(tmp_path, monkeypatch):
    # given
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"not a joblib artifact")
    monkeypatch.setenv("MODEL_PATH", str(path))
    # when / then
    with pytest.raises(KeyError), TestClient(app):
        pass



@pytest.mark.parametrize(
    "field",
    [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
    ],
)
@pytest.mark.parametrize(
    "value,status", [(0, 200), (1, 200), (-0.001, 422), (1.001, 422), (None, 422)]
)
def test_unit_interval_fields(client, payload, field, value, status):
    # given
    payload[field] = value
    # when
    response = client.post("/predict", json=payload)
    # then
    assert response.status_code == status
