"""Early-warning model: at the end of week N, which still-enrolled learners will NOT finish?

Design choices that keep the model honest:
  * Features use only activity up to and including week N (no look-ahead).
  * Only learners still enrolled at week N, in courses longer than N weeks, are scored.
  * Train/test split is grouped by learner so the same person is never in both sets.
  * In-progress enrollments have no known outcome, so they are never used for training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

NUM_FEATURES = [
    "age", "duration_weeks", "price_usd", "instructor_rating", "total_videos", "total_assessments",
    "w_logins", "w_active_weeks", "w_inactive_weeks", "w_session_min", "w_videos", "w_watch_min",
    "w_assessments", "w_forum", "w_progress", "pace_ratio",
    "first_week_logins", "last_week_logins", "logins_trend",
]
CAT_FEATURES = [
    "difficulty", "category", "primary_device", "acquisition_channel",
    "education_level", "employment_status", "country",
]
FEATURES = NUM_FEATURES + CAT_FEATURES

FEATURE_LABELS = {
    "w_logins": "Logins (to date)", "w_active_weeks": "Active weeks", "w_inactive_weeks": "Inactive weeks",
    "w_session_min": "Session minutes", "w_videos": "Videos watched", "w_watch_min": "Watch time",
    "w_assessments": "Assessments attempted", "w_forum": "Forum posts", "w_progress": "Progress %",
    "pace_ratio": "Pace vs. plan", "first_week_logins": "Week-1 logins", "last_week_logins": "Latest-week logins",
    "logins_trend": "Login trend", "duration_weeks": "Course length", "price_usd": "Price",
    "instructor_rating": "Instructor rating", "total_videos": "Course videos", "total_assessments": "Course assessments",
    "difficulty": "Difficulty", "category": "Category", "primary_device": "Device",
    "acquisition_channel": "Channel", "education_level": "Education", "employment_status": "Employment",
    "country": "Country", "age": "Age",
}

MODEL_CHOICES = ["Gradient Boosting", "Random Forest", "Logistic Regression"]


def build_features(master: pd.DataFrame, weekly: pd.DataFrame, n: int) -> pd.DataFrame:
    """Features as they would look at the END of week n. One row per eligible enrollment."""
    w = weekly[weekly["week_number"] <= n].copy()
    w["active"] = (w["logins"] > 0).astype(int)
    agg = w.groupby("enrollment_id").agg(
        w_logins=("logins", "sum"), w_active_weeks=("active", "sum"),
        w_session_min=("total_session_minutes", "sum"), w_videos=("videos_watched", "sum"),
        w_watch_min=("watch_time_min", "sum"), w_assessments=("assessments_attempted", "sum"),
        w_forum=("forum_posts", "sum"),
    )
    at_n = weekly.loc[weekly["week_number"] == n, ["enrollment_id", "logins", "cumulative_progress_pct"]]
    at_n = at_n.rename(columns={"logins": "last_week_logins", "cumulative_progress_pct": "w_progress"})
    first = weekly.loc[weekly["week_number"] == 1, ["enrollment_id", "logins"]].rename(
        columns={"logins": "first_week_logins"})

    base_cols = ["enrollment_id", "learner_id", "status", "completed"] + [
        c for c in NUM_FEATURES + CAT_FEATURES if c in master.columns]
    df = master[base_cols].merge(agg, on="enrollment_id").merge(at_n, on="enrollment_id").merge(
        first, on="enrollment_id")
    df = df[df["duration_weeks"] > n].copy()  # only courses that are still running after week n

    df["w_inactive_weeks"] = n - df["w_active_weeks"]
    df["logins_trend"] = df["last_week_logins"] - df["first_week_logins"]
    expected = 100 * n / df["duration_weeks"]
    df["pace_ratio"] = df["w_progress"] / expected
    df["at_risk"] = 1 - df["completed"]
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    df["age"] = df["age"].astype(float)
    return df.reset_index(drop=True)


def _pipeline(name: str) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    clf = {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                                                        random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000),
    }[name]
    return Pipeline([("pre", pre), ("clf", clf)])


def metrics_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        accuracy=(tp + tn) / len(y_true),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def top_k_capture(y_true: np.ndarray, proba: np.ndarray, pct: float) -> dict:
    """If we contact the riskiest pct% of learners, how many real non-completers do we reach?"""
    k = max(1, int(round(len(y_true) * pct / 100)))
    order = np.argsort(-proba)[:k]
    hits = int(y_true[order].sum())
    return dict(k=k, precision=hits / k, recall=hits / max(int(y_true.sum()), 1))


@dataclass
class TrainedModel:
    name: str
    week: int
    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: np.ndarray
    proba_test: np.ndarray
    auc: float
    pr_auc: float
    base_rate: float
    n_train: int
    n_test: int
    fpr: np.ndarray
    tpr: np.ndarray
    importance: pd.DataFrame
    defaults: dict


def train(feats: pd.DataFrame, name: str, week: int) -> TrainedModel:
    labelled = feats[feats["status"].isin(["Completed", "Dropped"])].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(gss.split(labelled, groups=labelled["learner_id"]))
    X_tr, X_te = labelled.loc[tr, FEATURES], labelled.loc[te, FEATURES]
    y_tr, y_te = labelled.loc[tr, "at_risk"].to_numpy(), labelled.loc[te, "at_risk"].to_numpy()

    pipe = _pipeline(name).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)

    perm = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                  random_state=SEED, n_jobs=1)
    imp = (pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean})
           .assign(label=lambda d: d["feature"].map(FEATURE_LABELS).fillna(d["feature"]))
           .sort_values("importance", ascending=False).reset_index(drop=True))

    defaults = {c: labelled[c].median() for c in NUM_FEATURES}
    defaults.update({c: labelled[c].mode().iloc[0] for c in CAT_FEATURES})
    return TrainedModel(name, week, pipe, X_te, y_te, proba, roc_auc_score(y_te, proba),
                        average_precision_score(y_te, proba), float(y_te.mean()), len(tr), len(te),
                        fpr, tpr, imp, defaults)


def score_in_progress(model: TrainedModel, feats: pd.DataFrame) -> pd.DataFrame:
    live = feats[feats["status"] == "In Progress"].copy()
    if live.empty:
        return live.assign(risk=[])
    live["risk"] = model.pipeline.predict_proba(live[FEATURES])[:, 1]
    return live


def recommend_action(row: pd.Series) -> str:
    """Simple, explainable playbook so each flagged learner gets a concrete next step."""
    if row["last_week_logins"] == 0:
        return "Re-engagement email / push nudge (inactive last week)"
    if row["w_assessments"] == 0:
        return "Prompt first quiz attempt with a short practice set"
    if row["pace_ratio"] < 0.6:
        return "Send catch-up plan: 2-3 short videos to get back on pace"
    if row["logins_trend"] < 0:
        return "Check-in message: activity is declining"
    return "Monitor; light-touch encouragement"


def what_if_row(model: TrainedModel, overrides: dict, week: int) -> pd.DataFrame:
    row = dict(model.defaults)
    row.update(overrides)
    row["w_inactive_weeks"] = week - row["w_active_weeks"]
    row["logins_trend"] = row["last_week_logins"] - row["first_week_logins"]
    row["pace_ratio"] = row["w_progress"] / (100 * week / row["duration_weeks"])
    return pd.DataFrame([row])[FEATURES]
