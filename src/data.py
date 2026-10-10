"""Data loading, merging and feature derivation. Pure pandas - no Streamlit imports,
so everything here can be unit-tested and reused in notebooks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SNAPSHOT_DATE = pd.Timestamp("2026-09-30")

STATUS_ORDER = ["Completed", "Dropped", "In Progress"]
STATUS_COLORS = {"Completed": "#2E9E6B", "Dropped": "#E4572E", "In Progress": "#8FA3B8"}

# label shown in the UI for each behavioural metric in enrollments.csv
BEHAVIOUR_METRICS = {
    "total_logins": "Total logins",
    "active_weeks": "Active weeks",
    "avg_session_minutes": "Avg session (min)",
    "videos_watched": "Videos watched",
    "total_watch_time_min": "Watch time (min)",
    "avg_video_completion_pct": "Avg video completion (%)",
    "assessments_attempted": "Assessments attempted",
    "avg_assessment_score": "Avg assessment score",
    "forum_posts": "Forum posts",
    "progress_pct": "Course progress (%)",
}

# dimensions the user can slice completion by
SEGMENT_DIMENSIONS = {
    "category": "Course category",
    "difficulty": "Course difficulty",
    "price_band": "Price band",
    "primary_device": "Primary device",
    "acquisition_channel": "Acquisition channel",
    "employment_status": "Employment status",
    "education_level": "Education level",
    "age_band": "Age band",
    "country": "Country",
    "engagement_tier": "Early engagement (wk 1-2 logins)",
}


def load_tables(data_dir: Path = DATA_DIR):
    learners = pd.read_csv(data_dir / "learners.csv", parse_dates=["signup_date"])
    courses = pd.read_csv(data_dir / "courses.csv")
    enroll = pd.read_csv(
        data_dir / "enrollments.csv", parse_dates=["enrollment_date", "completion_date"]
    )
    weekly = pd.read_csv(data_dir / "weekly_activity.csv")
    return learners, courses, enroll, weekly


def _ordered(values, order: list[str]) -> pd.Categorical:
    return pd.Categorical(values, categories=order, ordered=True)


def build_master(learners, courses, enroll, weekly) -> pd.DataFrame:
    """One row per enrollment with learner + course attributes and derived metrics."""
    df = enroll.merge(learners, on="learner_id", how="left", validate="m:1").merge(
        courses, on="course_id", how="left", validate="m:1"
    )

    # --- cleaning -----------------------------------------------------------
    df["education_level"] = df["education_level"].fillna("Unknown")
    df["status"] = _ordered(df["status"], STATUS_ORDER)

    # --- bands --------------------------------------------------------------
    age_labels = ["Under 22", "22-25", "26-30", "31-40", "41+"]
    age_band = pd.cut(df["age"], [0, 21, 25, 30, 40, 120], labels=age_labels).astype(object)
    age_band = pd.Series(age_band, index=df.index).fillna("Unknown")
    df["age_band"] = _ordered(age_band, age_labels + ["Unknown"])

    price_labels = ["Free", "$1-30", "$31-80", "$81+"]
    price_band = pd.cut(df["price_usd"], [-1, 0, 30, 80, 10_000], labels=price_labels)
    df["price_band"] = _ordered(price_band.astype(object), price_labels)
    df["difficulty"] = _ordered(df["difficulty"], ["Beginner", "Intermediate", "Advanced"])
    df["education_level"] = _ordered(
        df["education_level"], ["High School", "Diploma", "Bachelor's", "Master's", "PhD", "Unknown"]
    )

    # --- derived behaviour metrics -----------------------------------------
    df["logins_per_active_week"] = np.where(
        df["active_weeks"] > 0, df["total_logins"] / df["active_weeks"].replace(0, np.nan), 0.0
    )
    df["watch_hours"] = df["total_watch_time_min"] / 60
    df["assessment_coverage_pct"] = (
        100 * df["assessments_attempted"] / df["total_assessments"]
    ).clip(upper=100)

    # --- early engagement (weeks 1-2) ---------------------------------------
    early = (
        weekly[weekly["week_number"] <= 2]
        .groupby("enrollment_id")
        .agg(
            early_logins=("logins", "sum"),
            early_videos=("videos_watched", "sum"),
            early_assessments=("assessments_attempted", "sum"),
        )
    )
    df = df.join(early, on="enrollment_id")
    for c in ["early_logins", "early_videos", "early_assessments"]:
        df[c] = df[c].fillna(0)

    lo, hi = df["early_logins"].quantile([0.33, 0.67]).tolist()
    tier = np.select(
        [df["early_logins"] <= lo, df["early_logins"] > hi], ["Low", "High"], default="Medium"
    )
    df["engagement_tier"] = _ordered(tier, ["Low", "Medium", "High"])

    # --- time ---------------------------------------------------------------
    df["enrollment_month"] = df["enrollment_date"].dt.to_period("M").dt.to_timestamp()
    df["is_closed"] = df["status"] != "In Progress"
    return df


def completion_by(master: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Completion rate (closed enrollments only) by a dimension."""
    closed = master[master["is_closed"]]
    out = (
        closed.groupby(dim, observed=True)
        .agg(enrollments=("completed", "size"), completion_rate=("completed", "mean"))
        .reset_index()
    )
    out["completion_rate"] *= 100
    dtype = master[dim].dtype
    if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
        out = out.sort_values(dim)
    else:
        out = out.sort_values("completion_rate", ascending=False)
    return out.reset_index(drop=True)


def load_all():
    learners, courses, enroll, weekly = load_tables()
    master = build_master(learners, courses, enroll, weekly)
    return master, weekly, courses, learners
