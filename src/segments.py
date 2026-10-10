"""Behavioural segmentation of enrollments with K-Means (k=4, ordered by engagement)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = ["total_logins", "active_weeks", "avg_session_minutes", "videos_watched",
                "assessments_attempted", "forum_posts", "assessment_coverage_pct"]
SEG_NAMES = ["Ghosts", "Skimmers", "Steady Learners", "Power Learners"]
SEG_COLORS = {"Ghosts": "#B0B7C3", "Skimmers": "#E9A23B", "Steady Learners": "#5B8DEF", "Power Learners": "#2E9E6B"}
SEG_PLAYBOOK = {
    "Ghosts": "Signed up but barely engaged. Test onboarding friction; trigger a day-2 nudge and a one-click first lesson.",
    "Skimmers": "Dip in and out, rarely assess. Add micro-quizzes, streaks and deadline reminders.",
    "Steady Learners": "Consistent but could stall mid-course. Reinforce with milestones, peer forums and certificates.",
    "Power Learners": "Highly engaged. Turn into advocates: reviews, referrals, advanced-track upsell.",
}


def segment(master: pd.DataFrame, k: int = 4, seed: int = 42):
    X = master[SEG_FEATURES].fillna(0).to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    composite = pd.Series(Z.mean(axis=1)).groupby(km.labels_).mean().sort_values()
    rank = {cl: i for i, cl in enumerate(composite.index)}
    names = SEG_NAMES if k == 4 else [f"Segment {i + 1}" for i in range(k)]
    seg = pd.Series(km.labels_).map(rank).map(dict(enumerate(names))).to_numpy()

    out = master.copy()
    out["segment"] = pd.Categorical(seg, categories=names, ordered=True)
    pca = PCA(n_components=2, random_state=seed).fit_transform(Z)
    out["pc1"], out["pc2"] = pca[:, 0], pca[:, 1]
    zdf = pd.DataFrame(Z, columns=SEG_FEATURES, index=out.index)
    zprofile = zdf.groupby(out["segment"], observed=True).mean()
    return out, zprofile
