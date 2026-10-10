import pandas as pd
import streamlit as st

from src.analysis import key_insights

RECS = [
    ("Fix the first two weeks", "Trigger a nudge within 48 hours of enrollment if there is no login, and make the "
     "first lesson one click away. Early activity is the strongest predictor of completion."),
    ("Make the first assessment easy", "Add a short, low-stakes quiz in week 1. Learners who attempt an assessment "
     "early complete far more often."),
    ("Run the early-warning list weekly", "Every Monday, contact the highest-risk learners using the suggested "
     "actions (re-engagement, catch-up plan, quiz prompt)."),
    ("De-risk advanced and premium courses", "Add prerequisites check, pacing guides, and money-back or milestone "
     "incentives where completion is lowest."),
    ("Treat silent disengagement as dropout", "Many learners stay enrolled but stop logging in. Track 'active last "
     "week' as the health metric, not enrollment status."),
]


def render(master: pd.DataFrame, weekly: pd.DataFrame) -> None:
    st.subheader("Key findings (recomputed for the current filters)")
    for item in key_insights(master, weekly):
        st.markdown(f"- **{item['title']}.** {item['text']}")
    st.subheader("Recommended actions")
    for title, text in RECS:
        st.markdown(f"**{title}.** {text}")
    st.info("This dataset is synthetic. Findings illustrate the analytical method; they are not real-world evidence.")
