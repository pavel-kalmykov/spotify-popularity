# Spotify Track Popularity

Prediction of a track's popularity (0-100) from its audio features (danceability, energy, tempo, loudness, ...). The project also explores the binary framing of `popularity > 50` (hit vs no hit) as a classification task, and selects between linear models and tree-based models for both.

This is the midterm project of the [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Dataset

114k Spotify tracks with 21 columns, from the [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) dataset on Hugging Face (BSD license). The dataset is committed in this repository at `data/spotify_tracks.csv` so every reviewer works on the same snapshot. To download it again:

```bash
curl -L -o data/spotify_tracks.csv "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
```

## Project structure

1. `notebook.py`: EDA, data cleaning, feature analysis, model selection and parameter tuning (Marimo notebook)
2. `train.py`: trains the final model and saves it to a file (to be added)
3. `predict.py`: web service that loads the model and serves predictions (to be added)
4. `Dockerfile`: containerizes the service (to be added)

## How to run

```bash
uv sync
uv run marimo edit notebook.py
```

## Notes

- The dataset contains duplicated `track_id` values (one track appears in several genres). The train/validation split is grouped by `track_id`, so every copy of a track stays on the same side and the metrics do not leak between splits.
