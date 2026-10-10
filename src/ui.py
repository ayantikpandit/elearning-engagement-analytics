"""Small shared Streamlit helpers so styling lives in one place."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.data import STATUS_COLORS

PALETTE = ["#3B6FD4", "#2E9E6B", "#E9A23B", "#E4572E", "#8E6BBF", "#2BA5B5", "#B0B7C3"]


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13), colorway=PALETTE, showlegend=legend,
        title=dict(x=0.0, font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def show(fig: go.Figure, height: int = 380, legend: bool = True) -> None:
    st.plotly_chart(style(fig, height, legend), width="stretch")


def show_df(df, **kw) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kw)


def kpi(col, label: str, value: str, help: str | None = None) -> None:
    col.metric(label, value, help=help)


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


STATUS_MAP = STATUS_COLORS
