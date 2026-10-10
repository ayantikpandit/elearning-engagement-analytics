"""Descriptive analytics: retention curves, dropout hazard, auto-generated insights."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_GROUP = 30  # hide points built on fewer enrollments than this


def _closed_with_retention(master: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    last = weekly.groupby("enrollment_id")["week_number"].max().rename("weeks_retained")
    closed = master[master["is_closed"]].join(last, on="enrollment_id")
    closed["weeks_retained"] = closed["weeks_retained"].fillna(0)
    return closed


def retention_curve(master, weekly, split: str | None = None) -> pd.DataFrame:
    """Share of learners still enrolled / active in each week.

    Only closed enrollments (Completed or Dropped) whose course lasts at least k weeks
    are counted at week k, so short courses do not distort later weeks.
    """
    closed = _closed_with_retention(master, weekly)
    max_w = int(closed["duration_weeks"].max())
    rows = []
    for k in range(1, max_w + 1):
        sub = closed[closed["duration_weeks"] >= k]
        active_ids = set(
            weekly.loc[(weekly["week_number"] == k) & (weekly["logins"] > 0), "enrollment_id"]
        )
        groups = sub.groupby(split, observed=True) if split else [("All learners", sub)]
        for name, g in groups:
            if len(g) < MIN_GROUP:
                continue
            rows.append(
                dict(
                    week=k,
                    group=str(name),
                    eligible=len(g),
                    still_enrolled=100 * (g["weeks_retained"] >= k).mean(),
                    active_that_week=100 * g["enrollment_id"].isin(active_ids).mean(),
                )
            )
    return pd.DataFrame(rows)


def last_active_distribution(master: pd.DataFrame) -> pd.DataFrame:
    """For learners who did not complete: the last week in which they logged in (0 = never started)."""
    nc = master[master["status"] == "Dropped"]
    dist = nc.groupby("last_active_week").size().rename("learners").reset_index()
    dist["share"] = 100 * dist["learners"] / max(len(nc), 1)
    dist["cumulative_share"] = dist["share"].cumsum()
    dist["label"] = dist["last_active_week"].map(lambda w: "Never started" if w == 0 else f"Week {int(w)}")
    return dist


def weekly_by_status(master: pd.DataFrame, weekly: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average weekly metric among learners still enrolled that week, by final status."""
    w = weekly.merge(master[["enrollment_id", "status", "duration_weeks"]], on="enrollment_id")
    w = w[w["status"].isin(["Completed", "Dropped"])]
    out = (
        w.groupby(["status", "week_number"], observed=True)[metric]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "value", "size": "n"})
    )
    return out[out["n"] >= MIN_GROUP]


def key_insights(master: pd.DataFrame, weekly: pd.DataFrame) -> list[dict]:
    """Plain-English findings computed live from the filtered data."""
    closed = master[master["is_closed"]]
    if len(closed) < 100:
        return [dict(title="Not enough data", text="Widen the sidebar filters to see insights.")]
    out: list[dict] = []
    overall = 100 * closed["completed"].mean()
    out.append(dict(title="Baseline",
                    text=f"{overall:.0f}% of {len(closed):,} finished enrollments ended in completion; "
                         f"the rest dropped out or stalled."))

    t = closed.groupby("engagement_tier", observed=True)["completed"].mean() * 100
    if {"Low", "High"} <= set(t.index):
        out.append(dict(
            title="Week 1-2 behaviour is a strong early signal",
            text=f"Learners with high early activity complete at {t['High']:.0f}% versus {t['Low']:.0f}% "
                 f"for low early activity ({t['High'] / max(t['Low'], 0.1):.1f}x)."))

    dist = last_active_distribution(master)
    if not dist.empty:
        never = dist.loc[dist["last_active_week"] == 0, "share"].sum()
        by2 = dist.loc[dist["last_active_week"] <= 2, "share"].sum()
        out.append(dict(
            title="Where learners disappear",
            text=f"{by2:.0f}% of non-completers were last seen in week 2 or earlier, and {never:.0f}% "
                 f"never logged in at all. Onboarding and the first two weeks are a major lever."))

    a = closed.groupby(closed["early_assessments"] > 0)["completed"].mean() * 100
    if {True, False} <= set(a.index):
        out.append(dict(
            title="First assessment attempt matters",
            text=f"Attempting an assessment in weeks 1-2 goes with {a[True]:.0f}% completion versus "
                 f"{a[False]:.0f}% without one."))

    d = closed.groupby("difficulty", observed=True)["completed"].mean() * 100
    if len(d) >= 2:
        out.append(dict(
            title="Difficulty",
            text=f"Completion falls from {d.iloc[0]:.0f}% ({d.index[0]}) to {d.iloc[-1]:.0f}% ({d.index[-1]})."))

    for dim, label in [("primary_device", "Device"), ("price_band", "Price"),
                       ("acquisition_channel", "Acquisition channel")]:
        g = closed.groupby(dim, observed=True).agg(n=("completed", "size"), r=("completed", "mean"))
        g = g[g["n"] >= MIN_GROUP]
        if len(g) >= 2:
            best, worst = g["r"].idxmax(), g["r"].idxmin()
            out.append(dict(
                title=label,
                text=f"Best: {best} ({100 * g['r'].max():.0f}%). Weakest: {worst} ({100 * g['r'].min():.0f}%)."))
    return out
