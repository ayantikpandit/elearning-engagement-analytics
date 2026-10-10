import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import BEHAVIOUR_METRICS, STATUS_COLORS
from src.ui import section, show, show_df

LOGIN_BINS = [-1, 2, 5, 10, 20, 10_000]
LOGIN_LABELS = ["0-2", "3-5", "6-10", "11-20", "21+"]
ASSESS_BINS = [-1, 0, 25, 50, 75, 100]
ASSESS_LABELS = ["None", "1-25%", "26-50%", "51-75%", "76-100%"]


def _binned_completion(df: pd.DataFrame, col: str) -> pd.DataFrame:
    x = df[col]
    try:
        b = pd.qcut(x, 5, duplicates="drop")
    except ValueError:
        b = pd.cut(x, 4)
    g = df.assign(bin=b.astype(str)).groupby("bin", observed=True).agg(
        n=("completed", "size"), rate=("completed", "mean"), lo=(col, "min")).reset_index()
    g["rate"] *= 100
    return g.sort_values("lo")


def render(master: pd.DataFrame, weekly: pd.DataFrame) -> None:
    closed = master[master["is_closed"]]
    if len(closed) < 50:
        st.info("Not enough finished enrollments for the current filters.")
        return

    section("Completers vs. droppers",
            "Average behaviour of learners who finished versus those who did not (in-progress excluded).")
    cols = list(BEHAVIOUR_METRICS)
    tbl = closed.groupby("status", observed=True)[cols].mean().T
    tbl = tbl[["Completed", "Dropped"]]
    tbl["Completed / Dropped"] = tbl["Completed"] / tbl["Dropped"].replace(0, np.nan)
    tbl.index = [BEHAVIOUR_METRICS[i] for i in tbl.index]
    show_df(tbl.round(2).reset_index(names="Metric"))

    c1, c2 = st.columns(2)
    with c1:
        metric = st.selectbox("Metric", cols, format_func=BEHAVIOUR_METRICS.get, key="dr_metric")
        fig = px.box(closed, x="status", y=metric, color="status", color_discrete_map=STATUS_COLORS,
                     category_orders={"status": ["Completed", "Dropped"]}, points=False,
                     title=f"{BEHAVIOUR_METRICS[metric]} distribution")
        fig.update_xaxes(title=None)
        show(fig, 380, legend=False)
    with c2:
        g = _binned_completion(closed.dropna(subset=[metric]), metric)
        fig = px.bar(g, x="bin", y="rate", text=g["rate"].round(0).astype(int).astype(str) + "%",
                     hover_data={"n": True}, title=f"Completion rate by {BEHAVIOUR_METRICS[metric].lower()} (quintiles)")
        fig.update_traces(marker_color="#3B6FD4", textposition="outside")
        fig.update_yaxes(title="Completion rate (%)", range=[0, 110])
        fig.update_xaxes(title=None)
        show(fig, 380, legend=False)

    section("Combined effect: logins x assessments",
            "Cells with fewer than 15 enrollments are blanked out.")
    h = closed.assign(
        logins_band=pd.cut(closed["total_logins"], LOGIN_BINS, labels=LOGIN_LABELS),
        assess_band=pd.cut(closed["assessment_coverage_pct"], ASSESS_BINS, labels=ASSESS_LABELS),
    )
    pv = h.pivot_table(index="assess_band", columns="logins_band", values="completed",
                       aggfunc=["mean", "size"], observed=False)
    rate = (pv["mean"] * 100).where(pv["size"] >= 15)
    fig = px.imshow(rate, text_auto=".0f", aspect="auto", color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                    labels=dict(x="Total logins", y="Assessment coverage", color="Completion %"),
                    title="Completion rate (%)")
    fig.update_yaxes(autorange="reversed")
    show(fig, 380, legend=False)

    c3, c4 = st.columns(2)
    with c3:
        corr_cols = cols + ["completed"]
        corr = closed[corr_cols].corr()
        labels = [BEHAVIOUR_METRICS.get(c, "Completed (0/1)") for c in corr_cols]
        corr.index, corr.columns = labels, labels
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation matrix")
        show(fig, 480, legend=False)
    with c4:
        sample = closed.sample(min(1500, len(closed)), random_state=1)
        fig = px.scatter(sample, x="total_logins", y="watch_hours", color="status", opacity=0.6,
                         color_discrete_map=STATUS_COLORS, title="Logins vs. watch hours (sample of 1,500)")
        fig.update_xaxes(title="Total logins")
        fig.update_yaxes(title="Watch hours")
        show(fig, 480)
    st.caption("Correlation is not causation: in this dataset engagement and completion move together, "
               "but the data cannot prove that nudging logins alone would raise completion.")
