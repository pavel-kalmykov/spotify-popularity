import numpy as np
import pandas as pd
import pytest

from spotify_popularity.features import PROBABILITY_THRESHOLD, prepare_base_features


@pytest.mark.parametrize("probability,expected", [(PROBABILITY_THRESHOLD - 0.0001, False), (PROBABILITY_THRESHOLD, True), (PROBABILITY_THRESHOLD + 0.0001, True)])
def test_hit_probability_boundary(bundle_stub, payload, probability, expected):
    # given
    bundle_stub.classifier.predict_proba = lambda X: np.array([[1 - probability, probability]])
    # when
    result = bundle_stub.predict([payload])
    # then
    assert result[0]["hit"] is expected


@pytest.mark.parametrize("signature", [0, 1, 2])
def test_undocumented_signature_is_unknown(payload, signature):
    # given
    payload["time_signature"] = signature
    # when
    transformed = prepare_base_features(pd.DataFrame([payload]))
    # then
    assert transformed.loc[0, "time_signature"] == "unknown"


def test_duration_is_log_transformed(payload):
    # given
    raw_duration = payload["duration_ms"]
    # when
    transformed = prepare_base_features(pd.DataFrame([payload]))
    # then
    assert transformed.loc[0, "duration_ms"] == pytest.approx(np.log1p(raw_duration))
