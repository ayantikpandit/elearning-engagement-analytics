import pandas as pd
import plotly.express as px
import streamlit as st

from src.segments import SEG_COLORS, SEG_NAMES, SEG_PLAYBOOK, segment
from src.ui import section, show, show_df

PRETTY = {"total_logins": "Logins", "active_weeks": "Active weeks", "avg_session_minutes": "Session length",
          "videos_watched": "Videos", "assessments_attempted": "Assessments", "forum_posts": "Forum posts",
          "assessment_coverage_pct": "Assessment coverage"}


@st.cache_data(show_spinner=False)
def _segment(master: pd.DataFrame):
    return segment(master)


def render(master: pd.DataFrame, weekly: pd.DataFrame) -> None:
    if len(master) < 200:
        st.info("Widen the filters: segmentation needs at least 200 enrollments.")
        return
    seg, z = _segment(master)
    closed = seg[seg["is_closed"]]
    section("Behavioural segments",
            "K-Means (k=4) on logins, active weeks, session length, videos, assessments and forum activity. "
            "Segments are ordered by overall engagement.")

    prof = seg.groupby("segment", observed=True).agg(
        enrollments=("completed", "size"),
        avg_logins=("total_logins", "mean"), avg_videos=("videos_watched", "mean"),
        avg_assessments=("assessments_attempted", "mean"), avg_progress=("progress_pct", "mean"))
    prof["completion_rate"] = closed.groupby("segment", observed=True)["completed"].mean() * 100
    prof["share"] = 100 * prof["enrollments"] / prof["enrollments"].sum()
    prof = prof.reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(prof, x="segment", y="share", color="segment", color_discrete_map=SEG_COLORS,
                     text=prof["share"].round(0).astype(int).astype(str) + "%", title="Segment size (% of enrollments)")
        fig.update_xaxes(title=None); fig.update_yaxes(title=None)
        show(fig, 340, legend=False)
    with c2:
        fig = px.bar(prof, x="segment", y="completion_rate", color="segment", color_discrete_map=SEG_COLORS,
                     text=prof["completion_rate"].round(0).fillna(0).astype(int).astype(str) + "%",
                     title="Completion rate by segment")
        fig.update_xaxes(title=None); fig.update_yaxes(title=None, range=[0, 105])
        show(fig, 340, legend=False)

    c3, c4 = st.columns(2)
    with c3:
        zz = z.rename(columns=PRETTY)
        fig = px.imshow(zz, text_auto=".1f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-2, zmax=2,
                        title="Segment profile (z-scores vs. average learner)")
        show(fig, 380, legend=False)
    with c4:
        sample = seg.sample(min(2000, len(seg)), random_state=3)
        fig = px.scatter(sample, x="pc1", y="pc2", color="segment", color_discrete_map=SEG_COLORS,
                         category_orders={"segment": SEG_NAMES}, opacity=0.6, title="Segments in 2-D (PCA)")
        fig.update_xaxes(title="Component 1"); fig.update_yaxes(title="Component 2")
        show(fig, 380)

    section("Profile table and playbook")
    show_df(prof.round(1))
    for name in SEG_NAMES:
        if name in set(prof["segment"].astype(str)):
            st.markdown(f"**{name}:** {SEG_PLAYBOOK[name]}")
    st.caption("Segments describe behaviour over the whole enrollment, so low-engagement segments naturally "
               "contain learners who left early. Use the Early-Warning tab for forward-looking prediction.")
