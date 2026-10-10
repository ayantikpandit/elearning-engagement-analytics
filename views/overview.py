import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import SEGMENT_DIMENSIONS, STATUS_COLORS, STATUS_ORDER, completion_by
from src.ui import kpi, section, show, show_df


def render(master: pd.DataFrame, weekly: pd.DataFrame) -> None:
    closed = master[master["is_closed"]]
    c = st.columns(6)
    kpi(c[0], "Enrollments", f"{len(master):,}")
    kpi(c[1], "Learners", f"{master['learner_id'].nunique():,}")
    kpi(c[2], "Completion rate", f"{100 * closed['completed'].mean():.1f}%",
        "Completed / (Completed + Dropped). In-progress enrollments are excluded.")
    kpi(c[3], "Avg logins", f"{master['total_logins'].mean():.1f}")
    kpi(c[4], "Avg progress", f"{master['progress_pct'].mean():.0f}%")
    kpi(c[5], "Never logged in", f"{100 * (master['total_logins'] == 0).mean():.1f}%",
        "Enrollments with zero logins across the whole course.")

    left, right = st.columns([1, 2])
    with left:
        counts = master["status"].value_counts().reindex(STATUS_ORDER).reset_index()
        counts.columns = ["status", "enrollments"]
        fig = px.pie(counts, names="status", values="enrollments", hole=0.55,
                     color="status", color_discrete_map=STATUS_COLORS, title="Enrollment outcomes")
        fig.update_traces(textinfo="percent+label", sort=False)
        show(fig, 360, legend=False)
    with right:
        m = (master.groupby(["enrollment_month", "status"], observed=True).size()
             .rename("enrollments").reset_index())
        fig = px.bar(m, x="enrollment_month", y="enrollments", color="status",
                     color_discrete_map=STATUS_COLORS, category_orders={"status": STATUS_ORDER},
                     title="Enrollments by month and outcome")
        fig.update_xaxes(title=None)
        show(fig, 360)

    section("Who completes?", "Completion rate by learner or course attribute (finished enrollments only).")
    dim = st.selectbox("Slice by", list(SEGMENT_DIMENSIONS), format_func=SEGMENT_DIMENSIONS.get, key="ov_dim")
    tbl = completion_by(master, dim)
    fig = px.bar(tbl, x=dim, y="completion_rate", text=tbl["completion_rate"].round(1).astype(str) + "%",
                 hover_data={"enrollments": True, "completion_rate": ":.1f"},
                 title=f"Completion rate by {SEGMENT_DIMENSIONS[dim].lower()}")
    fig.add_hline(y=100 * master[master["is_closed"]]["completed"].mean(), line_dash="dot",
                  annotation_text="overall", line_color="#8FA3B8")
    fig.update_traces(marker_color="#3B6FD4", textposition="outside")
    fig.update_yaxes(title="Completion rate (%)", range=[0, max(tbl["completion_rate"].max() * 1.2, 10)])
    fig.update_xaxes(title=None)
    show(fig, 380, legend=False)
    with st.expander("Show table"):
        show_df(tbl.rename(columns={dim: SEGMENT_DIMENSIONS[dim]}).round(1))
