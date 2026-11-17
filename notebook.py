import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns


    # Wrapped-style dark theme: near-black canvas, green data ink, white copy.
    SPOTIFY_GREEN = "#1ED760"
    SPOTIFY_PALETTE = [SPOTIFY_GREEN, "#61B3E3", "#B3B3B3", "#FFC862"]
    sns.set_theme(
        context="notebook",
        style="dark",
        palette=SPOTIFY_PALETTE,
        rc={
            "figure.facecolor": "#121212",
            "axes.facecolor": "#121212",
            "savefig.facecolor": "#121212",
            "text.color": "#FFFFFF",
            "axes.labelcolor": "#B3B3B3",
            "axes.edgecolor": "#3A3A3A",
            "xtick.color": "#B3B3B3",
            "ytick.color": "#B3B3B3",
            "axes.titlecolor": "#FFFFFF",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#2A2A2A",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    return SPOTIFY_GREEN, mo, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spotify Track Popularity

    For the DataTalks Machine Learning Zoomcamp 2026's midterm project, I picked the Spotify tracks dataset: 114K
    tracks from 114 genres, each one with a popularity score (0-100) and
    a set of audio features Spotify computed for every song (danceability,
    energy, loudness, tempo, valence...). The dataset lives on Hugging
    Face ([here](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)), but it is cloned in this repo for reproducibility (it's fine as it has BSD license).
    """)
    return


@app.cell
def _():
    import pandas as pd


    df = pd.read_csv("data/spotify_tracks.csv", index_col=0)
    df
    return df, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data cleanse

    To begin with, we can see something strange above: There are 114K records, but track_id's unique count is 89,741. This means there must be "duplicates":
    """)
    return


@app.cell
def _(df, pd):
    dedup_df = df.drop_duplicates()
    dup_track_ids = dedup_df.track_id.duplicated(keep=False)
    pd.Series(
        {
            "total rows": len(df),
            "full-row duplicates": df.duplicated(keep=False).sum(),
            "rows after dropping exact duplicates": len(dedup_df),
            "rows sharing a track_id": dup_track_ids.sum(),
            "distinct tracks duplicated": dedup_df.loc[dup_track_ids, "track_id"].nunique(),
            "most copies of one track": dedup_df.track_id.value_counts().max(),
        }
    )
    return dedup_df, dup_track_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the repeated tracks, what are the columns that vary?
    """)
    return


@app.cell
def _(dedup_df, dup_track_ids):
    dup_tracks = dedup_df[dup_track_ids]
    dup_tracks_by_id = dup_tracks.groupby("track_id")
    varying_columns = (dup_tracks_by_id.nunique() > 1).sum()
    varying_columns[varying_columns > 0].sort_values(ascending=False)
    return dup_tracks, dup_tracks_by_id


@app.cell
def _(dup_tracks, dup_tracks_by_id):
    # track_genre is already unique: it's the same number as "distinct tracks duplicated", so no need to group by both
    varying_pop_ids = dup_tracks_by_id.popularity.nunique()
    dup_tracks[dup_tracks.track_id == varying_pop_ids[varying_pop_ids > 1].index[0]][
        ["track_id", "track_name", "track_genre", "popularity"]
    ]
    return (varying_pop_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the sample, the popularity varies only by 1, but is it the case for the rest of the songs?
    """)
    return


@app.cell
def _(dup_tracks, varying_pop_ids):
    both_varying = dup_tracks[dup_tracks.track_id.isin(varying_pop_ids[varying_pop_ids > 1].index)]
    deviations = both_varying.groupby("track_id").popularity.agg(
        gap=lambda s: s.max() - s.min(),
        values=lambda s: sorted(s),
    )
    deviations.sort_values("gap", ascending=False)
    return (both_varying,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we take a sneak peek at the values, we can infer two properties:

    1. Many tracks whose gap is significant have a minimum value of 0 and another, higher value.
    2. Most of them have a gap of less than 3.

    Let's confirm this hypothesis by comparing the minimum and maximum popularities for each track:
    """)
    return


@app.cell
def _(both_varying, sns):
    minmax = both_varying.groupby("track_id").popularity.agg(["min", "max"])
    ax = sns.scatterplot(data=minmax, x="max", y="min", alpha=0.3, s=15)
    ax.axline((0, 0), slope=1, color="#B3B3B3", linestyle="--", alpha=0.6)
    ax.set_title("Copies of a track agree... until zero appears")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From this representation we can confirm that:
    1. Most tracks have a small gap.
    2. Most tracks with a large gap have a minimum popularity of 0.
    3. There is a (very) small cluster of low-popularity tracks with a bit more mixing.

    Knowing that this data was collected from the Spotify API, a plausible explanation for these variations is that records were not collected in a single pass, and that some tracks needed time to gather their popularity. With this in mind, I will keep the maximum popularity across copies as an aggregation rule, not as a verified correction of the original score.
    """)
    return


@app.cell
def _(dedup_df):
    corrected_df = dedup_df.assign(
        popularity=dedup_df.groupby("track_id").popularity.transform("max")
    )
    return (corrected_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lastly, let's see if there are any $NaN$s in the data set:
    """)
    return


@app.cell
def _(corrected_df):
    corrected_df[corrected_df.isna().any(axis=1)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, only one record has $NaN$ values. Given that it is only one record out of 113,550 (or 89,741 tracks), and the missing fields are simply names (we won't be using them in our training), we can either discard the record or leave it as is. I'll stick to the latter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis (EDA)

    ### The Target: Popularity

    Let's start by checking our target variable to predict: The popularity we've been dealing with earlier:
    """)
    return


@app.cell
def _(corrected_df):
    track_pop = corrected_df.groupby("track_id").popularity.first()  # now all popularities for a song are the same
    track_pop.describe().round(2)
    return (track_pop,)


@app.cell
def _(sns, track_pop):
    import matplotlib.pyplot as plt


    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(
        track_pop,
        # These are integer values, so this makes the visualization better than using buckets
        discrete=True,
        ax=axes[0]
    ).set_title("All tracks")
    # Since the huge peak is only for popularity 0, it's easier to show an additional plot rather 
    # than using a log-scale. This way, you avoid its distorsion.
    sns.histplot(track_pop[track_pop > 0], discrete=True, ax=axes[1]).set_title("Without the zeros")
    fig.suptitle("One wall of zeros, two bumps of hope")
    fig.tight_layout()
    fig
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From these histograms, we can draw the following conclusions:
    1. The distribution is very right-skewed, with a huge number of tracks scoring 0.
        1. From the data, we cannot tell whether these are either not computed yet or not played (yet?).
        2. Popularity 1 also has a higher frequency than the next 2-4 scores, which might suggest 0 is a special case.
    2. For the rest of the histogram, there are two peaks:
        1. The first one, around 20-25, for maybe not-so-popular songs.
        2. The second, smaller one, around 40-50, for somewhat-popular songs.
    3. Right after the second one, the counts start to diminish, and by 60 we're already in its tail. That mark might be a great threshold of what we can consider "a hit".

    With this information, we could try to do two predictions at the same time:
    1. How well can we predict a song's popularity based on pure audio features.
    2. If we can confidently tell whether a song will be "a hit" (i.e. has a popularity greater than 50, after the three big clusters).

    Let's confirm that the 50 threshold makes sense with an [ECDF chart](https://plotly.com/blog/ecdf-plot-guide/):
    """)
    return


@app.cell
def _(sns, track_pop):
    HIT_THRESHOLD = 50
    hit_rate = (track_pop > HIT_THRESHOLD).mean()

    ecdf = sns.ecdfplot(track_pop)
    ecdf.axvline(HIT_THRESHOLD, color="#B3B3B3", linestyle="--", alpha=0.6)
    ecdf.axhline(1 - hit_rate, color="#B3B3B3", linestyle="--", alpha=0.6)
    ecdf.scatter([HIT_THRESHOLD], [1 - hit_rate], color="#FFFFFF", zorder=3, s=28)
    ecdf.annotate(
        f"~{hit_rate:.0%} of tracks above {HIT_THRESHOLD}",
        xy=(HIT_THRESHOLD, 1 - hit_rate),
        xytext=(HIT_THRESHOLD + 10, 0.55),
        arrowprops={"arrowstyle": "->"},
    )
    ecdf.set_yticks([0.25, 0.5, 1 - hit_rate, 1.0])
    ecdf.set_title("1 in 5 tracks is a hit")
    ecdf
    return (HIT_THRESHOLD,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One could argue what constitutes a hit, but the 50 threshold is practically the top quartile while staying a round number, so I'll keep that cut.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The features

    Let's now focus on analysing the feature columns. First, we define the numerical and categorical features we'd like to work with (as it was earlier mentioned, we won't be dealing with names):
    """)
    return


@app.cell
def _():
    numerical_features = ["duration_ms", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    categorical_features = ["explicit", "key", "mode", "time_signature"]  # we'll deal with the genres later
    return categorical_features, numerical_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For now, we'll ignore the genres, as they need some special consideration, and treat each track as unique (the rest of the columns are identical after the popularity fix).

    #### Numerical features

    Let's see how each numerical feature looks like independently:
    """)
    return


@app.cell
def _(numerical_features, plt, sns, tracks_one_row):
    _fig, _axes = plt.subplots(2, 5, figsize=(12, 5))
    for _col, _ax in zip(numerical_features, _axes.flat):
        sns.histplot(tracks_one_row[_col], ax=_ax, bins=40)
        _ax.set_title(_col, fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(numerical_features, tracks_one_row):
    tracks_one_row[numerical_features].skew().round(1).abs().sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Half of the features are heavily skewed, with $|skew|$ above 1.5, long tails and strong outliers. The heaviest tails are candidates for a log transform once we reach the training phase; here in the EDA I keep the raw values. The most noticeable one is `duration_ms`, so let's check individually which are the longest songs:
    """)
    return


@app.cell
def _(corrected_df):
    corrected_df.assign(minutes=lambda d: (d.duration_ms / 60_000).round(1))[
        lambda d: d.minutes > 20
    ][["artists", "album_name", "track_name", "track_genre", "minutes"]].sort_values(
        by="minutes", ascending=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we want to see how well each variable correlates with the target one, the Pearson correlation measures linear association and can be sensitive to extreme values. To explore monotonic relationships using ranks, I will use [Spearman](https://www.surveymonkey.com/learn/market-research/pearson-correlation-vs-spearman-correlation/) instead: the rank correlation, which only asks whether popularity grows when the feature grows. Standardization is not an alternative here (it leaves correlations unchanged); it belongs to the training phase, where models do care.
    """)
    return


@app.cell
def _(corrected_df, numerical_features, sns):
    tracks_one_row = corrected_df.groupby("track_id").first()
    target_corr = (
        tracks_one_row[numerical_features + ["popularity"]]
        .corr(method="spearman")["popularity"]
        .drop("popularity")
        .sort_values()
    )
    barplot = sns.barplot(x=target_corr.values, y=target_corr.index)
    barplot.axvline(0, color="#B3B3B3", linewidth=0.8, alpha=0.6)
    barplot.set_title("No single feature predicts popularity")
    return (tracks_one_row,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The audio features are individually weak: the strongest one, `instrumentalness`, sits at $-0.12$, and nothing else is above $0.07$ (!). The sound of a track barely anticipates its popularity on its own, which fits the dataset's description of the score (plays and recency, not music). Two consequences:

    1. The categorical features, absent from this chart (the genre above all), are the remaining candidates for signal.
    2. Weak individual correlations do not rule out combinations (interactions are the models' job to test).

    Let's now check how each variable correlates with each other:
    """)
    return


@app.cell
def _(SPOTIFY_GREEN, numerical_features, sns, tracks_one_row):
    import numpy as np

    feature_corr = tracks_one_row[numerical_features].corr(method="spearman")
    mask = np.triu(np.ones_like(feature_corr, dtype=bool))
    from matplotlib.colors import LinearSegmentedColormap

    # Wrapped-style diverging map: purple for negative, canvas-dark neutral at zero,
    # green for positive. Reads on the dark theme like coolwarm never could.
    spotify_diverging = LinearSegmentedColormap.from_list(
        "spotify_diverging", ["#5038A0", "#282828", SPOTIFY_GREEN]
    )

    heatmap = sns.heatmap(
        feature_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=spotify_diverging,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="#121212",
        annot_kws={"size": 8},
        cbar_kws={"label": "Spearman correlation"},
    )
    heatmap.set_title("Loud, electric, acoustic: mostly one axis")
    heatmap
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From this heatmap, we can see four dominating pairs:

    1. `energy-loudness` ($0.75$), `acousticness-energy` ($-0.71$) and `acousticness-loudness` ($-0.54$), which are three variants of the same loud-electric vs acoustic axis.
    2. `valence-danceability` ($0.48$), which moves halfway together.

    The next pair (`speechiness-energy`) has a correlation of $0.35$. Energy, loudness and acousticness contain overlapping information, which may affect the stability of linear-model coefficients.

    #### Categorical features

    Similarly to the numerical ones, let's see how categorical variable looks like independently:
    """)
    return


@app.cell
def _(categorical_features, plt, tracks_one_row):
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3))
    for _col, _ax in zip(categorical_features, _axes.flat):
        _counts = tracks_one_row[_col].value_counts()
        _ax.bar(_counts.index.astype(str), _counts.values)
        _ax.set_title(_col, fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here are some observations we can make about these variables:
    1. For all these features, only `key` looks somewhat balanced
    2. The cardinality for these features is not huge, so the data set won't explode in the one-hot-encoding phase.
    3. According to the dataset's information, "The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4."
        1. Values 6 and 7 are not present.
        2. Values 0 and 1 fall outside the documented range. Their meaning is unclear, so I will group them into an "unknown" category.

    Now, let's compare how these correlate with the target variable:
    """)
    return


@app.cell
def _(categorical_features, plt, sns, tracks_one_row):
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 4))
    for _col, _ax in zip(categorical_features, _axes.flat):
        sns.boxplot(data=tracks_one_row, x=_col, y="popularity", ax=_ax)
        _ax.set_title(_col, fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From these box and whiskers plots, we should expect to see any noticeable difference between each value, but unfortunately, most of them are practically plain, except maybe a little for `time_signature`. Put simply, none of these four variables provide any noticeable signal about popularity, at least on its own. Had the amount of variables and their cardinality been larger, it would have made sense not to take them into account, but for that very reason, it does not harm to keep them.

    However, what about the genres we decided to leave for later? One could intuitively think some genres are generally more popular than others.
    """)
    return


@app.cell
def _(corrected_df):
    genre_stats = (
        corrected_df.groupby("track_genre")
        .popularity.agg(["size", "mean"])
        .sort_values("mean", ascending=False)
    )
    genre_stats.round(1)
    return (genre_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are way too many genres, and even if we can already tell some genres are more popular than others, one could try and think about grouping some of these genres. However, there are plenty of counter-examples where two apparently-similar genres have very distant mean popularities:
    """)
    return


@app.cell
def _(genre_stats):
    genre_stats[genre_stats.index.isin(["rock", "hard-rock"])].round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we know it's better not to group genres, but the table above is still very big. What about the best vs the worst?
    """)
    return


@app.cell
def _(HIT_THRESHOLD, genre_stats, pd, sns):
    _top_bottom = pd.concat([genre_stats.head(10), genre_stats.tail(10)])
    _ax = sns.barplot(data=_top_bottom.reset_index(), y="track_genre", x="mean")
    _ax.axvline(HIT_THRESHOLD, color="#B3B3B3", linestyle="--", alpha=0.6)
    _ax.set_title("Genre spans the whole scale: 2 to 59")
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The four highest-ranked genres have mean popularity above the hit threshold, although individual tracks can fall below it.

    Since a track can carry several genres, the modeling phase will compare three ways to represent them: every row with its own genre (the main pipeline, with a split grouped by track), one row per track keeping the first genre, and one row per track with all its genres as multi-hot columns. The comparison will tell which representation predicts best.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conclusions

    All the observations made above can be summarized into the following:

    1. The `popularity` target variable, which was corrected to the maximum across unique `track_id` rows.
        1. Main task: regression.
        2. Secondary: "hit" categorization ($> 50$).
    2. The features: ten audio ones (low correlations), plus the four small categoricals (weak individually, but cheap to keep) and the genre, which carries the real signal.
    3. Known structure:
        1. `energy`, `loudness` and `acousticness` contain overlapping information.
        2. `time_signature` $0$ and $1$ are undocumented values recoded to `"unknown"`.
        3. `duration_ms` keeps its outliers and gets a log transform at training time.
    4. Split: grouped by `track_id`, so every copy of a song stays on the same side.

    ## Modeling

    Picking up the EDA's conclusions, let's start by splitting our data while keeping the same `track_id`s within a same partition. To do that, let's split the `track_id`s and then collect their records:
    """)
    return


@app.cell
def _(corrected_df, pd):
    from sklearn.model_selection import train_test_split


    train_val_ids, test_ids = train_test_split(
        corrected_df.track_id.unique(),
        test_size=0.2,
        random_state=1,
    )

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.25,
        random_state=1,
    )

    df_train = corrected_df[corrected_df.track_id.isin(train_ids)]
    df_val = corrected_df[corrected_df.track_id.isin(val_ids)]
    df_test = corrected_df[corrected_df.track_id.isin(test_ids)]

    # Let's check for overlaps just to be sure
    pd.Series(
        {
            "train / validation overlap": len(set(train_ids) & set(val_ids)),
            "train / test overlap": len(set(train_ids) & set(test_ids)),
            "validation / test overlap": len(set(val_ids) & set(test_ids)),
        }
    )
    return df_test, df_train, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before we start, let's calculate a baseline error metric that we can compare against. To do that, we'll "predict" the training tracks' mean popularity for every validation track and calculate the RMSE:
    """)
    return


@app.cell
def _(df_train, df_val):
    from sklearn.metrics import root_mean_squared_error


    train_popularity = df_train.groupby("track_id").popularity.first()
    val_popularity = df_val.groupby("track_id").popularity.first()

    baseline_prediction = train_popularity.mean()

    baseline_rmse = root_mean_squared_error(
        val_popularity,
        [baseline_prediction] * len(val_popularity),
    )
    baseline_rmse
    return baseline_rmse, root_mean_squared_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear regression

    To begin with, let's only use the numerical features to see what these provide before adding the categorical ones and compare the genres:
    """)
    return


@app.cell
def _(df_train, df_val, np, numerical_features):
    # all numerical values are the same for each track_id
    model_train = df_train.groupby("track_id").first()
    model_val = df_val.groupby("track_id").first()

    X_train_numeric = model_train[numerical_features].copy()
    X_val_numeric = model_val[numerical_features].copy()

    X_train_numeric["duration_ms"] = np.log1p(X_train_numeric["duration_ms"])
    X_val_numeric["duration_ms"] = np.log1p(X_val_numeric["duration_ms"])
    return X_train_numeric, X_val_numeric, model_train, model_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's also define some helper functions that will help us train the models using hyperparameter optimization techniques ([GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for linear regression, and [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) for the tree-based ones, as the latter take more time and we cannot be as exhaustive):
    """)
    return


@app.cell
def _(baseline_rmse, np, pd, root_mean_squared_error):
    from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge


    def negative_track_rmse(estimator, X, y):
        rows = pd.DataFrame(
            {"actual": np.asarray(y), "prediction": estimator.predict(X)},
            index=X.index,
        )
        tracks = rows.groupby(level=0).agg({"actual": "first", "prediction": "mean"})
        return -root_mean_squared_error(tracks.actual, tracks.prediction)

    def search_models(experiment, preparation, parameter_grid, X_train, X_val, y_train, y_val, *, n_iter=None):
        assert set(X_train.index).isdisjoint(X_val.index)
        X = pd.concat([X_train, X_val])
        y = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
        validation_split = PredefinedSplit(
            np.concatenate([
                np.full(len(X_train), -1, dtype=int),
                np.zeros(len(X_val), dtype=int),
            ])
        )
        pipeline = Pipeline([("preparation", preparation), ("model", Ridge())])
        search_options = dict(
            scoring=negative_track_rmse,
            cv=validation_split,
            refit=False,
            error_score="raise",
            n_jobs=1,
        )
        if n_iter is None:
            search = GridSearchCV(pipeline, parameter_grid, **search_options)
        else:
            search = RandomizedSearchCV(
                pipeline,
                parameter_grid,
                n_iter=n_iter,
                random_state=1,
                **search_options,
            )
        search.fit(X, y)
        candidates = pd.DataFrame(search.cv_results_)
        results = pd.DataFrame({
            "experiment": experiment,
            "model": [type(p["model"]).__name__ for p in candidates.params],
            "parameters": [
                {k.removeprefix("model__"): v for k, v in p.items() if k != "model"}
                for p in candidates.params
            ],
            "validation RMSE": -candidates.mean_test_score,
            "fit time (s)": candidates.mean_fit_time,
        })
        results["improvement over baseline"] = baseline_rmse - results["validation RMSE"]
        return search, results.sort_values("validation RMSE").reset_index(drop=True)

    return LinearRegression, Ridge, StandardScaler, search_models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will use linear regression with L2-regularization by using [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), trying different values of `alpha`. Note that, for no regularization, the docs recommend to use the plain `LinearRegression` class instead.
    """)
    return


@app.cell
def _(
    LinearRegression,
    Ridge,
    StandardScaler,
    X_train_numeric,
    X_val_numeric,
    model_train,
    model_val,
    search_models,
):
    linear_parameter_grid = [
        {"model": [LinearRegression()]},
        {
            "model": [Ridge()],
            "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        },
    ]

    numeric_search, numeric_results = search_models(
        "numeric", StandardScaler(), linear_parameter_grid,
        X_train_numeric, X_val_numeric,
        model_train.popularity, model_val.popularity,
    )
    numeric_results.round(4)
    return linear_parameter_grid, numeric_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, we can see that the linear regression model, regardless of its regularization, does not improve much, at least with the audio features alone. Let's continue with the categorical variables to see whether they improve the error:
    """)
    return


@app.cell
def _(
    X_train_numeric,
    X_val_numeric,
    categorical_features,
    model_train,
    model_val,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    X_train_mixed = X_train_numeric.join(model_train[categorical_features]).copy()
    X_val_mixed = X_val_numeric.join(model_val[categorical_features]).copy()

    for _frame in [X_train_mixed, X_val_mixed]:
        _frame["time_signature"] = _frame.time_signature.where(
            _frame.time_signature.isin([3, 4, 5, 6, 7]),
            "unknown",
        )
        for _column in categorical_features:
            _frame[_column] = _frame[_column].astype(str)
    return ColumnTransformer, OneHotEncoder, X_train_mixed, X_val_mixed


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    X_train_mixed,
    X_val_mixed,
    categorical_features,
    linear_parameter_grid,
    model_train,
    model_val,
    numerical_features,
    search_models,
):
    mixed_preparation = ColumnTransformer([
        ("numerical", StandardScaler(), numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            categorical_features,
        ),
    ])
    categorical_search, categorical_results = search_models(
        "numeric + categorical", mixed_preparation, linear_parameter_grid,
        X_train_mixed, X_val_mixed,
        model_train.popularity, model_val.popularity,
    )
    categorical_results.round(4)
    return (categorical_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again, the improvement is tiny: adding these categorical features only brings the best validation RMSE from 20.23 to 20.20 (!).

    Let's finish the linear regression phase by adding the genres. Since a track can belong to several, we will compare three representations:
    1. One row per track, keeping its first listed genre.
    2. One row per track, keeping every genre as a separate binary indicator (multi-hot encoding).
    3. One row per track-genre pair, keeping all the genre assignments as separate records.

    All three will use the same training and validation tracks. For the third representation, I will average the predictions across each track's rows before calculating RMSE, so every track has the same weight in the comparison.

    Let's start with the first approach:
    """)
    return


@app.cell
def _(
    X_train_mixed,
    X_val_mixed,
    categorical_features,
    model_train,
    model_val,
):
    X_train_first_genre = X_train_mixed.assign(
        track_genre=model_train.track_genre
    )
    X_val_first_genre = X_val_mixed.assign(
        track_genre=model_val.track_genre
    )

    genre_categorical_features = categorical_features + ["track_genre"]
    return X_train_first_genre, X_val_first_genre, genre_categorical_features


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    X_train_first_genre,
    X_val_first_genre,
    genre_categorical_features,
    linear_parameter_grid,
    model_train,
    model_val,
    numerical_features,
    search_models,
):
    first_genre_preparation = ColumnTransformer([
        ("numerical", StandardScaler(), numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            genre_categorical_features,
        ),
    ])
    first_genre_search, first_genre_results = search_models(
        "first genre", first_genre_preparation, linear_parameter_grid,
        X_train_first_genre, X_val_first_genre,
        model_train.popularity, model_val.popularity,
    )
    first_genre_results.round(4)
    return first_genre_preparation, first_genre_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can see that, simply adding the first listed genre brings the validation's RMSE down to 16.82. The regularization barely helps, however. As expected, a track's genre provides a much larger improvement than the other features, both categorical and numerical.

    Next, for the multi-hot encoding, we will use the [MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) class:
    """)
    return


@app.cell
def _(
    X_train_mixed,
    X_val_mixed,
    df_train,
    df_val,
    model_train,
    model_val,
    pd,
):
    from sklearn.preprocessing import MultiLabelBinarizer

    _train_genres = (
        df_train.groupby("track_id").track_genre
        .agg(lambda s: sorted(set(s)))
        .reindex(model_train.index)
    )
    _val_genres = (
        df_val.groupby("track_id").track_genre
        .agg(lambda s: sorted(set(s)))
        .reindex(model_val.index)
    )

    genre_encoder = MultiLabelBinarizer()
    _train_encoded = genre_encoder.fit_transform(_train_genres)
    _val_encoded = genre_encoder.transform(_val_genres)

    genre_columns = [
        f"genre={genre}" for genre in genre_encoder.classes_
    ]

    X_train_multigenre = X_train_mixed.join(
        pd.DataFrame(
            _train_encoded,
            index=model_train.index,
            columns=genre_columns,
        )
    )
    X_val_multigenre = X_val_mixed.join(
        pd.DataFrame(
            _val_encoded,
            index=model_val.index,
            columns=genre_columns,
        )
    )
    return X_train_multigenre, X_val_multigenre, genre_columns


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    X_train_multigenre,
    X_val_multigenre,
    categorical_features,
    genre_columns,
    linear_parameter_grid,
    model_train,
    model_val,
    numerical_features,
    search_models,
):
    multigenre_preparation = ColumnTransformer([
        ("numerical", StandardScaler(), numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            categorical_features,
        ),
        ("genres", "passthrough", genre_columns),
    ])
    multigenre_search, multigenre_results = search_models(
        "multi-hot genres", multigenre_preparation, linear_parameter_grid,
        X_train_multigenre, X_val_multigenre,
        model_train.popularity, model_val.popularity,
    )
    multigenre_results.round(4)
    return multigenre_preparation, multigenre_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This alternative slightly improves over the previous one, lowering the validation's RMSE down to 16.47. The regularization has little effect or is even counterproductive for some values.

    Lastly, let's see how the last option performs:
    """)
    return


@app.cell
def _(df_train, df_val, genre_categorical_features, np, numerical_features):
    X_train_all_genres = df_train[
        numerical_features + genre_categorical_features
    ].copy()
    X_val_all_genres = df_val[
        numerical_features + genre_categorical_features
    ].copy()

    for _frame in [X_train_all_genres, X_val_all_genres]:
        _frame["duration_ms"] = np.log1p(_frame.duration_ms)
        _frame["time_signature"] = _frame.time_signature.where(
            _frame.time_signature.isin([3, 4, 5, 6, 7]),
            "unknown",
        )
        for _column in genre_categorical_features:
            _frame[_column] = _frame[_column].astype(str)

    X_train_all_genres = X_train_all_genres.set_axis(df_train.track_id)
    X_val_all_genres = X_val_all_genres.set_axis(df_val.track_id)
    return X_train_all_genres, X_val_all_genres


@app.cell
def _(
    X_train_all_genres,
    X_val_all_genres,
    df_train,
    df_val,
    first_genre_preparation,
    linear_parameter_grid,
    search_models,
):
    all_genres_search, all_genres_results = search_models(
        "all genre rows", first_genre_preparation, linear_parameter_grid,
        X_train_all_genres, X_val_all_genres,
        df_train.popularity, df_val.popularity,
    )
    all_genres_results.round(4)
    return (all_genres_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, this alternative is basically the same as the first one, with practically the same results. Even each regularization parameter affects the errors similarly. This basically means that:

    1. Keeping all track-genre rows produces almost the same validation RMSE as keeping only the first genre.
    2. Representing all genres together as multi-hot features performs slightly better.


    Therefore, I will use the multi-hot representation for the next models.

    Let's now compare the linear ones in one chart with the best validation RMSE of each experiment. The audio features barely move the error; the genre representations are what matters.
    """)
    return


@app.cell
def _(
    SPOTIFY_GREEN,
    all_genres_results,
    baseline_rmse,
    categorical_results,
    first_genre_results,
    multigenre_results,
    numeric_results,
    pd,
    sns,
):
    _best = pd.concat([
        numeric_results.head(1),
        categorical_results.head(1),
        first_genre_results.head(1),
        multigenre_results.head(1),
        all_genres_results.head(1),
    ])
    _ax = sns.barplot(
        data=_best,
        y="experiment",
        x="validation RMSE",
        color=SPOTIFY_GREEN,
    )
    _ax.axvline(baseline_rmse, color="#B3B3B3", linestyle="--", alpha=0.6, linewidth=1)
    _ax.set_xlim(right=baseline_rmse + 3)
    _ax.text(
        baseline_rmse + 0.15,
        4.45,
        f"baseline = {baseline_rmse:.2f}",
        color="#B3B3B3",
        va="center",
    )
    _ax.set_title("Audio barely moves the error. Genre does.")
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trees

    Let's now try some non-linear, tree-based models to see if they improve the regression:
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    X_train_multigenre,
    X_val_multigenre,
    categorical_features,
    genre_columns,
    model_train,
    model_val,
    numerical_features,
    search_models,
):
    from sklearn.ensemble import RandomForestRegressor

    forest_preparation = ColumnTransformer([
        ("numerical", "passthrough", numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            categorical_features,
        ),
        ("genres", "passthrough", genre_columns),
    ])
    # Original search space (12 sampled combinations, ~5 min): n_estimators
    # [100, 200, 300], max_depth [10, 20, 40, None], min_samples_leaf [1, 2, 5, 10],
    # min_samples_split [2, 5, 10], max_features [0.5, 0.75, 1.0], bootstrap [True, False].
    # Reduced to the neighborhood of the best candidate so the notebook reruns fast;
    # the full-search evidence lives in the text above.
    forest_parameter_distributions = {
        "model": [RandomForestRegressor(random_state=1, n_jobs=-1)],
        "model__n_estimators": [200],
        "model__max_depth": [40, None],
        "model__min_samples_leaf": [1],
        "model__min_samples_split": [2],
        "model__max_features": [0.75],
        "model__bootstrap": [True],
    }
    forest_search, forest_results = search_models(
        "multi-hot genres", forest_preparation, forest_parameter_distributions,
        X_train_multigenre, X_val_multigenre,
        model_train.popularity, model_val.popularity,
    )
    forest_results.round(4)
    return forest_preparation, forest_results, forest_search


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see from the results, if the hyperparameters are properly chosen, we can beat the linear models, reducing validation RMSE to 14.33, the best result so far. However, training took considerably longer than for the linear models.

    Let's now check with XGBoost:
    """)
    return


@app.cell
def _(
    X_train_multigenre,
    X_val_multigenre,
    forest_preparation,
    model_train,
    model_val,
    search_models,
):
    from xgboost import XGBRegressor


    # Original search space (12 sampled combinations): n_estimators [100, 200, 400],
    # max_depth [3, 6, 9], learning_rate [0.03, 0.1, 0.3], min_child_weight [1, 5, 10],
    # subsample [0.8, 1.0], colsample_bytree [0.8, 1.0].
    # Reduced to the two strongest candidates from that search.
    xgb_parameter_distributions = {
        "model": [
            XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=1,
                n_jobs=-1,
            )
        ],
        "model__n_estimators": [100, 400],
        "model__max_depth": [6, 9],
        "model__learning_rate": [0.1],
        "model__min_child_weight": [1],
        "model__subsample": [1.0],
        "model__colsample_bytree": [0.8],
    }

    xgb_search, xgb_results = search_models(
        "multi-hot genres",
        forest_preparation,
        xgb_parameter_distributions,
        X_train_multigenre,
        X_val_multigenre,
        model_train.popularity,
        model_val.popularity,
    )
    xgb_results.round(4)
    return (xgb_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, among the tested configurations, XGBoost still outperforms the linear models. However, the best result still belongs to the Random Forest (optimized), so we'll keep that one as the best one.

    One could have noticed that the Random Forest training is way slower than the other alternatives. So how do they all compare in terms of performance vs time?
    """)
    return


@app.cell
def _(
    categorical_results,
    first_genre_results,
    forest_results,
    multigenre_results,
    numeric_results,
    pd,
    plt,
    sns,
    xgb_results,
):
    _linear_best = pd.concat([
        numeric_results.head(1),
        categorical_results.head(1),
        first_genre_results.head(1),
        multigenre_results.head(1),
    ]).assign(family="linear (best per experiment)")

    _families = pd.concat([
        _linear_best[["family", "fit time (s)", "validation RMSE"]],
        forest_results.assign(family="random forest")[
            ["family", "fit time (s)", "validation RMSE"]
        ],
        xgb_results.assign(family="xgboost")[
            ["family", "fit time (s)", "validation RMSE"]
        ],
    ])

    _fig, _ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=_families,
        x="fit time (s)",
        y="validation RMSE",
        hue="family",
        style="family",
        s=90,
        alpha=0.8,
        ax=_ax,
    )
    _ax.set_xscale("log")
    _ax.set_xlabel("training time per candidate (s, log scale)")
    _ax.set_ylabel("validation RMSE")
    _ax.set_title("The best score is also the slowest one")
    _ax.grid(True, which="both", axis="x", alpha=0.25)

    for _family, _marker in [
        ("linear (best per experiment)", "upper right"),
        ("random forest", "lower left"),
        ("xgboost", "upper right"),
    ]:
        _row = _families[_families.family == _family].sort_values(
            "validation RMSE"
        ).iloc[0]
        _ax.annotate(
            f"best {_family.split(' ')[0]}",
            xy=(_row["fit time (s)"], _row["validation RMSE"]),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=8,
            color="#B3B3B3",
        )

    sns.move_legend(_ax, "upper center", bbox_to_anchor=(0.5, -0.12), ncols=3, frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hit classification

    So far, we have predicted the popularity score itself. Let's now address the secondary question: can we identify tracks with popularity above 50? I will keep the same partitions and multi-hot genre representation, comparing logistic regression with a random forest classifier.

    Let's start by creating the target variable as a classification one:
    """)
    return


@app.cell
def _(HIT_THRESHOLD, model_train, model_val, pd):
    y_train_hit = (model_train.popularity > HIT_THRESHOLD).astype(int)
    y_val_hit = (model_val.popularity > HIT_THRESHOLD).astype(int)

    pd.DataFrame({
        "training share": y_train_hit.value_counts(normalize=True),
        "validation share": y_val_hit.value_counts(normalize=True),
    }).rename_axis("hit")
    return y_train_hit, y_val_hit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both partitions have around 22% hits, so the class proportions are similar. Let's compare logistic regression and random forest, selecting their parameters by validation ROC AUC with the same train/validation split as before.
    """)
    return


@app.cell
def _(
    X_train_multigenre,
    X_val_multigenre,
    multigenre_preparation,
    np,
    pd,
    y_train_hit,
    y_val_hit,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import GridSearchCV as _GridSearchCV
    from sklearn.model_selection import PredefinedSplit as _PredefinedSplit
    from sklearn.pipeline import Pipeline as _Pipeline

    _classification_X = pd.concat([
        X_train_multigenre,
        X_val_multigenre,
    ])
    _classification_y = np.concatenate([y_train_hit, y_val_hit])

    _classification_split = _PredefinedSplit(
        np.concatenate([
            np.full(len(y_train_hit), -1, dtype=int),
            np.zeros(len(y_val_hit), dtype=int),
        ])
    )

    _classification_grid = [
        {
            "model": [LogisticRegression(max_iter=2000)],
            "model__C": [0.1, 1, 10],
        },
        {
            "model": [
                RandomForestClassifier(
                    n_estimators=100,
                    max_features=0.75,
                    random_state=1,
                    n_jobs=-1,
                )
            ],
            "model__max_depth": [None],
            "model__min_samples_leaf": [1],
        },
    ]

    classification_search = _GridSearchCV(
        _Pipeline([
            ("preparation", multigenre_preparation),
            ("model", LogisticRegression()),
        ]),
        _classification_grid,
        scoring="roc_auc",
        cv=_classification_split,
        refit=False,
        error_score="raise",
        n_jobs=1,
    )
    classification_search.fit(_classification_X, _classification_y)

    _classification_cv = pd.DataFrame(classification_search.cv_results_)
    classification_results = pd.DataFrame({
        "model": [
            type(p["model"]).__name__
            for p in _classification_cv.params
        ],
        "parameters": [
            {
                k.removeprefix("model__"): v
                for k, v in p.items()
                if k != "model"
            }
            for p in _classification_cv.params
        ],
        "validation ROC AUC": _classification_cv.mean_test_score,
        "fit time (s)": _classification_cv.mean_fit_time,
    }).sort_values("validation ROC AUC", ascending=False)

    classification_results.round(4)
    return (classification_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Random Forest achieves the best validation ROC AUC (0.871), compared with 0.837 for logistic regression. Let's fit the winning configuration on training data and check its precision, recall and F1 on validation, using a probability threshold of 0.5.
    """)
    return


@app.cell
def _(
    X_train_multigenre,
    X_val_multigenre,
    classification_search,
    np,
    pd,
    y_train_hit,
    y_val_hit,
):
    from sklearn.base import clone
    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
    )

    hit_classifier = clone(
        classification_search.estimator
    ).set_params(**classification_search.best_params_)

    hit_classifier.fit(X_train_multigenre, y_train_hit)
    hit_probabilities = hit_classifier.predict_proba(X_val_multigenre)[:, 1]

    _metric_rows = []

    for _name, _probabilities in [
        ("Baseline: training hit rate",
         np.full(len(y_val_hit), y_train_hit.mean())),
        ("Selected classifier", hit_probabilities),
    ]:
        _predictions = _probabilities >= 0.5
        _metric_rows.append({
            "model": _name,
            "ROC AUC": roc_auc_score(y_val_hit, _probabilities),
            "precision": precision_score(
                y_val_hit, _predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val_hit, _predictions, zero_division=0
            ),
            "F1": f1_score(y_val_hit, _predictions, zero_division=0),
        })

    hit_validation_results = pd.DataFrame(_metric_rows)
    hit_validation_results.round(4)
    return (
        clone,
        f1_score,
        hit_probabilities,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At this threshold, roughly 7/10 predicted hits are correct, but the model detects only about half of the actual hits (precision 0.69, recall 0.52). Note that the baseline predicts no hits because its constant probability, around 0.22, never reaches 0.5, so this is expected.

    With this in mind, I will keep Random Forest for both tasks and evaluate the selected models on the test set after retraining on training and validation.

    Lastly, let's display the ROC curve which, given the 0.87 AUC, is expected to be somewhat close to the edges.
    """)
    return


@app.cell
def _(hit_probabilities, roc_auc_score, y_val_hit):
    from sklearn.metrics import RocCurveDisplay

    _ax = RocCurveDisplay.from_predictions(
        y_val_hit,
        hit_probabilities,
        name="random forest",
    )
    _ax.ax_.plot([0, 1], [0, 1], color="#B3B3B3", linestyle="--", alpha=0.6, linewidth=1)
    _ax.ax_.set_title(
        "Hits stand out from the crowd "
        f"(AUC {roc_auc_score(y_val_hit, hit_probabilities):.3f})"
    )
    _ax.ax_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Final training and test evaluation

    Now that we were able to find the best model and fix its configurations, I will next combine the training and validation data sets to fit again the models, and evaluate them on the untouched test tracks.
    """)
    return


@app.cell
def _(df_test, df_train, df_val, pd):
    from spotify_popularity.features import FeatureEncoder, track_records

    _final_training_rows = pd.concat([df_train, df_val])
    final_train_records, y_final_train = track_records(_final_training_rows)
    final_test_records, y_final_test = track_records(df_test)
    return (
        FeatureEncoder,
        final_test_records,
        final_train_records,
        y_final_test,
        y_final_train,
    )


@app.cell
def _(FeatureEncoder, final_test_records, final_train_records, pd):
    final_feature_encoder = FeatureEncoder().fit(final_train_records)
    X_final_train = final_feature_encoder.transform(final_train_records)
    X_final_test = final_feature_encoder.transform(final_test_records)
    final_genre_columns = final_feature_encoder.genre_columns_

    pd.DataFrame({
        "train + validation": [len(X_final_train), X_final_train.shape[1]],
        "test": [len(X_final_test), X_final_test.shape[1]],
    }, index=["tracks", "features"])
    return X_final_test, X_final_train, final_genre_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, the regressor:
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    X_final_test,
    X_final_train,
    categorical_features,
    clone,
    final_genre_columns,
    forest_search,
    np,
    numerical_features,
    pd,
    root_mean_squared_error,
    y_final_test,
    y_final_train,
):
    final_preparation = ColumnTransformer([
        ("numerical", "passthrough", numerical_features),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,
            ),
            categorical_features,
        ),
        ("genres", "passthrough", final_genre_columns),
    ])

    final_regressor = clone(forest_search.estimator).set_params(
        **forest_search.best_params_,
        preparation=clone(final_preparation),
    )
    final_regressor.fit(X_final_train, y_final_train)

    test_popularity_predictions = final_regressor.predict(X_final_test)

    regression_test_results = pd.DataFrame([
        {
            "model": "Baseline: training + validation mean",
            "test RMSE": root_mean_squared_error(
                y_final_test,
                np.full(len(y_final_test), y_final_train.mean()),
            ),
        },
        {
            "model": "Selected Random Forest",
            "test RMSE": root_mean_squared_error(
                y_final_test,
                test_popularity_predictions,
            ),
        },
    ])
    regression_test_results.round(4)
    return final_preparation, final_regressor, test_popularity_predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The feature importances of the final forest tell a two-sided story. Individually, the numerical audio features lead the ranking: each of the ten carries more weight than any single genre column. As a group, though, the genres sum almost as much as all the numerical features together, spread across their 114 one-hot columns. And the strongest genre columns are exactly the extreme genres from the EDA (iranian, romance, pop-film...): the forest uses them to flag the very low- and very high-popularity niches.
    """)
    return


@app.cell
def _(SPOTIFY_GREEN, final_regressor, pd, plt, sns):
    _names = final_regressor.named_steps[
        "preparation"
    ].get_feature_names_out()
    _importances = pd.Series(
        final_regressor.named_steps["model"].feature_importances_,
        index=_names,
    )

    def _group(_name):
        if _name.startswith("numerical__"):
            return "numerical"
        if _name.startswith("categorical__"):
            return "categorical"
        return "genres"

    _group_colors = {
        "numerical": "#61B3E3",
        "categorical": "#B3B3B3",
        "genres": SPOTIFY_GREEN,
    }

    _top = _importances.sort_values(ascending=False).head(15).sort_values()
    _top_groups = _top.index.map(_group)

    _by_group = _importances.groupby(_importances.index.map(_group)).sum()
    _by_group = _by_group.sort_values()

    _fig, _axes = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    sns.barplot(
        x=_top.values,
        y=_top.index,
        hue=_top_groups,
        palette=_group_colors,
        legend=False,
        ax=_axes[0],
    )
    _axes[0].set_title("The audio carries the ranking")
    _axes[0].set_xlabel("importance")
    _axes[0].set_ylabel("")

    sns.barplot(
        x=_by_group.values,
        y=_by_group.index,
        hue=_by_group.index,
        palette=_group_colors,
        legend=False,
        ax=_axes[1],
    )
    _axes[1].set_title("Genres add up to almost half")
    _axes[1].set_xlabel("total importance")
    _axes[1].set_ylabel("")

    _fig.suptitle("What the forest actually listens to")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And now the classifier:
    """)
    return


@app.cell
def _(
    HIT_THRESHOLD,
    X_final_test,
    X_final_train,
    classification_search,
    clone,
    f1_score,
    final_preparation,
    np,
    pd,
    precision_score,
    recall_score,
    roc_auc_score,
    y_final_test,
    y_final_train,
):
    y_final_train_hit = (y_final_train > HIT_THRESHOLD).astype(int)
    y_final_test_hit = (y_final_test > HIT_THRESHOLD).astype(int)

    final_classifier = clone(classification_search.estimator).set_params(
        **classification_search.best_params_,
        preparation=clone(final_preparation),
    )
    final_classifier.fit(X_final_train, y_final_train_hit)

    test_hit_probabilities = final_classifier.predict_proba(
        X_final_test
    )[:, 1]

    _test_metric_rows = []

    for _name, _probabilities in [
        (
            "Baseline: training + validation hit rate",
            np.full(len(y_final_test_hit), y_final_train_hit.mean()),
        ),
        ("Selected Random Forest", test_hit_probabilities),
    ]:
        _predictions = _probabilities >= 0.5
        _test_metric_rows.append({
            "model": _name,
            "ROC AUC": roc_auc_score(y_final_test_hit, _probabilities),
            "precision": precision_score(
                y_final_test_hit, _predictions, zero_division=0
            ),
            "recall": recall_score(
                y_final_test_hit, _predictions, zero_division=0
            ),
            "F1": f1_score(
                y_final_test_hit, _predictions, zero_division=0
            ),
        })

    classification_test_results = pd.DataFrame(_test_metric_rows)
    classification_test_results.round(4)
    return test_hit_probabilities, y_final_test_hit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Final models' visualization

    We have all the metrics, but an image is worth a thousand words: let's plot how well (or bad) do they perform against the test data!

    1. For the regression, actual vs predicted popularity.
    2. For the classification, a confusion matrix where each prediction is placed on its "quadrant".
    """)
    return


@app.cell
def _(
    HIT_THRESHOLD,
    SPOTIFY_GREEN,
    plt,
    sns,
    test_hit_probabilities,
    test_popularity_predictions,
    y_final_test,
    y_final_test_hit,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5.5))

    sns.scatterplot(
        x=y_final_test,
        y=test_popularity_predictions,
        alpha=0.15,
        s=8,
        color=SPOTIFY_GREEN,
        ax=_axes[0],
    )
    _axes[0].axline((0, 0), slope=1, color="#B3B3B3", linestyle="--", alpha=0.6, linewidth=1)
    _axes[0].set_xlabel("actual popularity")
    _axes[0].set_ylabel("predicted popularity")
    _axes[0].set_title("Predictions hug the diagonal")

    _predictions = (test_hit_probabilities >= 0.5).astype(int)
    _correct = _predictions == y_final_test_hit
    _tp = ((_predictions == 1) & (y_final_test_hit == 1)).sum()
    _fp = ((_predictions == 1) & (y_final_test_hit == 0)).sum()
    _fn = ((_predictions == 0) & (y_final_test_hit == 1)).sum()
    _tn = ((_predictions == 0) & (y_final_test_hit == 0)).sum()

    sns.scatterplot(
        x=y_final_test,
        y=test_hit_probabilities,
        hue=_correct.map({True: "correct", False: "wrong"}),
        hue_order=["correct", "wrong"],
        palette={"correct": SPOTIFY_GREEN, "wrong": "#FF6B6B"},
        alpha=0.35,
        s=12,
        ax=_axes[1],
    )
    _axes[1].axvline(HIT_THRESHOLD, color="#B3B3B3", linestyle="--", alpha=0.6, linewidth=1)
    _axes[1].axhline(0.5, color="#B3B3B3", linestyle="--", alpha=0.6, linewidth=1)
    _axes[1].set_xlabel("actual popularity")
    _axes[1].set_ylabel("hit probability")
    _axes[1].set_title("Hits live in the top right")
    sns.move_legend(
        _axes[1], "upper center", bbox_to_anchor=(0.5, -0.14), ncols=2, frameon=False
    )

    _corner = dict(fontsize=10, color="#FFFFFF", ha="center", va="center", alpha=0.85)
    _axes[1].text(5, 0.97, f"FP\n{_fp}", **_corner)
    _axes[1].text(5, 0.03, f"TN\n{_tn}", **_corner)
    _axes[1].text(97, 0.97, f"TP\n{_tp}", **_corner)
    _axes[1].text(97, 0.03, f"FN\n{_fn}", **_corner)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    On the held-out test set, the selected regressor achieves RMSE 14.21, compared with 20.45 for the constant-mean baseline, a reduction of about 30.5%.

    The classifier achieves ROC AUC of 0.8708. At the fixed probability threshold of 0.5, precision is 0.67 and recall is 0.53: roughly two thirds of predicted hits are correct, while about half of the actual hits are detected.

    The models capture some useful signal, but their predictions still have clear limits. An RMSE of 14.21 leaves substantial uncertainty around an individual track's score. For classification, however, precision is higher than recall, as the model is more selective about calling a track a hit, but it misses nearly half of the actual hits.

    The genres provided the largest improvement in our linear-model comparisons, while individual audio features had weak monotonic relationships with popularity. The results suggest that these features help estimate popularity, but do not fully explain it. The artists' identity could improve prediction by capturing differences that audio features alone do not explain. Nevertheless, it would change the question from what the music tells us about popularity to what the music and the artists' identity tell us. Therefore, I have left artist identity out of this project. Additionally, exploring it would also require distinguishing performance on new tracks by known artists from performance on entirely unseen artists.

    One limitation is that the EDA and some preparation decisions used the full dataset before splitting. The test set was excluded from model fitting and hyperparameter selection, but not from exploratory analysis, so this was not a fully blind test evaluation.

    With this, the experimentation with this dataset is done, and we can now move to the production training and deployment.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
