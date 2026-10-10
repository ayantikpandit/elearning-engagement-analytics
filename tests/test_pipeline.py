import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import model as M
from src.analysis import key_insights, retention_curve
from src.data import load_all
from src.segments import segment

master, weekly, courses, learners = load_all()


def test_shapes_and_keys():
    assert len(master) >= 1000
    assert master["enrollment_id"].is_unique
    assert set(master["status"].astype(str)) == {"Completed", "Dropped", "In Progress"}
    assert master["learner_id"].isin(learners["learner_id"]).all()
    assert weekly["enrollment_id"].isin(master["enrollment_id"]).all()


def test_no_future_information_in_features():
    banned = {"progress_pct", "drop_week", "completion_date", "certificate_issued", "last_active_week",
              "days_since_last_login", "total_logins", "status", "completed"}
    assert banned.isdisjoint(M.FEATURES)


def test_feature_window_respects_week():
    f1, f3 = M.build_features(master, weekly, 1), M.build_features(master, weekly, 3)
    assert f1["w_logins"].max() <= f3["w_logins"].max()
    assert (f3["duration_weeks"] > 3).all()


def test_model_beats_chance():
    f = M.build_features(master, weekly, 2)
    t = M.train(f, "Logistic Regression", 2)
    assert t.auc > 0.65


def test_retention_monotone_and_insights_run():
    c = retention_curve(master, weekly)
    assert (np.diff(c["still_enrolled"]) <= 1e-9).all()
    assert len(key_insights(master, weekly)) >= 4


def test_segments_cover_all_rows():
    seg, z = segment(master)
    assert seg["segment"].notna().all() and len(z) == 4
