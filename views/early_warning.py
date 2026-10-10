import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import model as M
from src.ui import section, show, show_df


@st.cache_data(show_spinner=False)
def _features(master: pd.DataFrame, weekly: pd.DataFrame, week: int) -> pd.DataFrame:
    return M.build_features(master, weekly, week)


@st.cache_resource(show_spinner="Training model...")
def _train(_feats: pd.DataFrame, name: str, week: int, key: int):
    return M.train(_feats, name, week)


def render(full_master: pd.DataFrame, weekly: pd.DataFrame, filtered_ids: set) -> None:
    st.markdown(
        "**Question:** at the end of week *N*, which still-enrolled learners are unlikely to finish? "
        "The model trains on the **full dataset** (sidebar filters only narrow the at-risk list below) and uses "
        "activity up to week *N* only, so it never sees the future.")
    c1, c2, c3 = st.columns(3)
    week = c1.slider("Score learners at end of week", 1, 4, 2, key="ew_week")
    name = c2.selectbox("Model", M.MODEL_CHOICES, key="ew_model")
    thr = c3.slider("Risk threshold", 0.2, 0.9, 0.5, 0.05, key="ew_thr",
                    help="Learners with predicted risk above this are flagged.")

    feats = _features(full_master, weekly, week)
    model = _train(feats, name, week, len(feats))

    m = M.metrics_at(model.y_test, model.proba_test, thr)
    k = st.columns(5)
    k[0].metric("ROC AUC", f"{model.auc:.3f}")
    k[1].metric("PR AUC", f"{model.pr_auc:.3f}", help=f"Baseline (random) = {model.base_rate:.2f}")
    k[2].metric("Precision", f"{m['precision']:.2f}", help="Of learners flagged, share who truly would not finish.")
    k[3].metric("Recall", f"{m['recall']:.2f}", help="Of learners who would not finish, share we flagged.")
    k[4].metric("F1", f"{m['f1']:.2f}")
    st.caption(f"Held-out test set: {model.n_test:,} enrollments (grouped by learner; {model.n_train:,} used to train). "
               f"{100 * model.base_rate:.0f}% of test learners did not complete.")

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=model.fpr, y=model.tpr, mode="lines", name=f"{name} (AUC {model.auc:.2f})",
                                 line=dict(color="#3B6FD4", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="#B0B7C3", dash="dash")))
        fig.update_layout(title="ROC curve")
        fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate")
        show(fig, 380)
    with b:
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fig = px.imshow(cm, text_auto=True, x=["Predicted: completes", "Predicted: at risk"],
                        y=["Actually completed", "Actually did not"], color_continuous_scale="Blues",
                        title=f"Confusion matrix at threshold {thr:.2f}")
        show(fig, 380, legend=False)

    c, d = st.columns(2)
    with c:
        imp = model.importance.head(12).iloc[::-1]
        fig = px.bar(imp, x="importance", y="label", orientation="h",
                     title="What drives the prediction? (permutation importance, drop in AUC)")
        fig.update_traces(marker_color="#3B6FD4")
        fig.update_xaxes(title="AUC drop when feature is shuffled"); fig.update_yaxes(title=None)
        show(fig, 420, legend=False)
    with d:
        dfp = pd.DataFrame({"risk": model.proba_test,
                            "outcome": np.where(model.y_test == 1, "Did not complete", "Completed")})
        fig = px.histogram(dfp, x="risk", color="outcome", nbins=30, barmode="overlay", opacity=0.65,
                           color_discrete_map={"Completed": "#2E9E6B", "Did not complete": "#E4572E"},
                           title="Predicted risk by actual outcome")
        fig.add_vline(x=thr, line_dash="dash", line_color="#1F2937")
        fig.update_xaxes(title="Predicted risk of not completing")
        show(fig, 420)

    section("Intervention budget", "If support staff can only contact a fraction of learners, who should they call first?")
    pct = st.slider("Contact the riskiest ...% of learners", 5, 60, 20, 5, key="ew_budget")
    cap = M.top_k_capture(model.y_test, model.proba_test, pct)
    e = st.columns(3)
    e[0].metric("Learners contacted (test set)", f"{cap['k']:,}")
    e[1].metric("Precision", f"{100 * cap['precision']:.0f}%", help="Share of contacted learners who truly would not finish.")
    e[2].metric("Share of all non-completers reached", f"{100 * cap['recall']:.0f}%")

    section("At-risk learners (in progress)",
            f"Scored with activity through week {week}. Respects the sidebar filters.")
    live = M.score_in_progress(model, feats)
    live = live[live["enrollment_id"].isin(filtered_ids)].copy()
    if live.empty:
        st.info("No in-progress enrollments with activity through this week match the current filters.")
    else:
        live["risk_band"] = pd.cut(live["risk"], [-0.01, 0.4, 0.7, 1.01], labels=["Low", "Medium", "High"])
        live["suggested_action"] = live.apply(M.recommend_action, axis=1)
        live = live.sort_values("risk", ascending=False)
        counts = live["risk_band"].value_counts()
        z = st.columns(3)
        z[0].metric("High risk", int(counts.get("High", 0)))
        z[1].metric("Medium risk", int(counts.get("Medium", 0)))
        z[2].metric("Low risk", int(counts.get("Low", 0)))
        cols = ["enrollment_id", "learner_id", "category", "difficulty", "w_logins", "w_videos",
                "w_assessments", "w_progress", "risk", "risk_band", "suggested_action"]
        view = live[cols].rename(columns={"w_logins": "logins", "w_videos": "videos", "w_assessments": "assessments",
                                          "w_progress": "progress_%"})
        view["risk"] = view["risk"].round(3)
        show_df(view, height=320)
        st.download_button("Download at-risk list (CSV)", view.to_csv(index=False).encode(),
                           "at_risk_learners.csv", "text/csv")

    section("What-if simulator", "Change a learner's behaviour and watch the predicted risk move.")
    d0 = model.defaults
    s1, s2, s3 = st.columns(3)
    difficulty = s1.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"], key="wi_diff")
    duration = s1.selectbox("Course length (weeks)", sorted(feats["duration_weeks"].unique().tolist()),
                            index=1, key="wi_dur")
    device = s1.selectbox("Device", sorted(feats["primary_device"].unique().tolist()), key="wi_dev")
    logins = s2.slider("Total logins so far", 0, 30, int(d0["w_logins"]), key="wi_logins")
    active = s2.slider("Active weeks so far", 0, week, min(week, int(round(d0["w_active_weeks"]))), key="wi_active")
    last = s2.slider("Logins in latest week", 0, 15, int(d0["last_week_logins"]), key="wi_last")
    videos = s3.slider("Videos watched", 0, 30, int(d0["w_videos"]), key="wi_videos")
    assess = s3.slider("Assessments attempted", 0, 6, int(d0["w_assessments"]), key="wi_assess")
    progress = s3.slider("Progress (%)", 0, 100, int(round(d0["w_progress"])), key="wi_prog")
    row = M.what_if_row(model, dict(difficulty=difficulty, duration_weeks=duration, primary_device=device,
                                    w_logins=logins, w_active_weeks=active, last_week_logins=last,
                                    w_videos=videos, w_assessments=assess, w_progress=float(progress)), week)
    risk = float(model.pipeline.predict_proba(row)[0, 1])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * risk, number={"suffix": "%"},
        title={"text": "Predicted risk of NOT completing"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1F2937"},
               "steps": [{"range": [0, 40], "color": "#CDEBDD"}, {"range": [40, 70], "color": "#FBE3B5"},
                         {"range": [70, 100], "color": "#F6C5B8"}]}))
    show(gauge, 300, legend=False)
