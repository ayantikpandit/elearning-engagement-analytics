import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import last_active_distribution, retention_curve, weekly_by_status
from src.data import STATUS_COLORS
from src.ui import section, show

SPLITS = {"None": None, "Difficulty": "difficulty", "Category": "category", "Device": "primary_device",
          "Price band": "price_band", "Early engagement": "engagement_tier",
          "Acquisition channel": "acquisition_channel"}
WEEKLY_METRICS = {"logins": "Logins per week", "videos_watched": "Videos watched per week",
                  "watch_time_min": "Watch minutes per week", "assessments_attempted": "Assessments per week",
                  "total_session_minutes": "Session minutes per week"}


def render(master: pd.DataFrame, weekly: pd.DataFrame) -> None:
    if (master["is_closed"]).sum() < 100:
        st.info("Not enough finished enrollments for the current filters.")
        return
    section("Retention curve",
            "Of learners whose course is long enough to reach week k, what share is still around in week k?")
    c1, c2 = st.columns([1, 1])
    split_label = c1.selectbox("Split by", list(SPLITS), key="rt_split")
    kind = c2.radio("Measure", ["Active that week (>=1 login)", "Still enrolled (not formally dropped)"],
                    horizontal=True, key="rt_kind")
    y = "active_that_week" if kind.startswith("Active") else "still_enrolled"
    curve = retention_curve(master, weekly, SPLITS[split_label])
    fig = px.line(curve, x="week", y=y, color="group", markers=True,
                  hover_data={"eligible": True}, title=kind)
    fig.update_yaxes(title="% of learners", range=[0, 102])
    fig.update_xaxes(title="Course week", dtick=1)
    show(fig, 400, legend=split_label != "None")
    st.caption("'Active' captures silent disengagement: many learners stay enrolled but stop logging in.")

    c3, c4 = st.columns(2)
    with c3:
        dist = last_active_distribution(master)
        fig = px.bar(dist, x="label", y="share", hover_data={"learners": True, "cumulative_share": ":.0f"},
                     title="When do non-completers last show up?")
        fig.update_traces(marker_color="#E4572E")
        fig.update_yaxes(title="% of non-completers")
        fig.update_xaxes(title=None, categoryorder="array", categoryarray=dist["label"].tolist())
        show(fig, 380, legend=False)
    with c4:
        metric = st.selectbox("Weekly behaviour", list(WEEKLY_METRICS), format_func=WEEKLY_METRICS.get, key="rt_metric")
        wb = weekly_by_status(master, weekly, metric)
        fig = px.line(wb, x="week_number", y="value", color="status", markers=True,
                      color_discrete_map=STATUS_COLORS, title=f"{WEEKLY_METRICS[metric]} by final outcome")
        fig.update_xaxes(title="Course week", dtick=1)
        fig.update_yaxes(title="Average")
        show(fig, 380)
    st.caption("Completers and droppers separate within the first two weeks, which is what makes early "
               "warning possible (see the Early-Warning Model tab).")
