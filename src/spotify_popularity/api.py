import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from spotify_popularity.model import ModelBundle


class TrackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    duration_ms: Annotated[int, Field(ge=0)]
    danceability: Annotated[float, Field(ge=0, le=1)]
    energy: Annotated[float, Field(ge=0, le=1)]
    loudness: float
    speechiness: Annotated[float, Field(ge=0, le=1)]
    acousticness: Annotated[float, Field(ge=0, le=1)]
    instrumentalness: Annotated[float, Field(ge=0, le=1)]
    liveness: Annotated[float, Field(ge=0, le=1)]
    valence: Annotated[float, Field(ge=0, le=1)]
    tempo: Annotated[float, Field(ge=0)]
    explicit: bool
    key: Annotated[int, Field(ge=-1, le=11)]
    mode: Literal[0, 1]
    time_signature: Annotated[int, Field(ge=0, le=7)]
    genres: Annotated[list[str], Field(min_length=1, max_length=114)]


class Prediction(BaseModel):
    popularity: float
    hit_probability: float
    hit: bool


@asynccontextmanager
async def lifespan(app):
    model_path = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
    # Only load artifacts produced by this project's training command.
    app.state.bundle = joblib.load(model_path)
    yield
    del app.state.bundle


app = FastAPI(title="Spotify track popularity", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "hit_threshold": ModelBundle.hit_threshold,
        "probability_threshold": ModelBundle.probability_threshold,
    }


@app.get("/genres")
def genres(request: Request):
    return {"genres": request.app.state.bundle.features.genre_encoder_.classes_.tolist()}


@app.post("/predict", response_model=Prediction)
def predict(track: TrackRequest, request: Request):
    bundle = request.app.state.bundle
    known = set(bundle.features.genre_encoder_.classes_)
    unknown = sorted(set(track.genres) - known)
    if unknown:
        raise HTTPException(status_code=422, detail={"unknown_genres": unknown})
    return bundle.predict([track.model_dump()])[0]
