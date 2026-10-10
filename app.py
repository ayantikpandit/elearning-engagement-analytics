"""Online Learning Engagement Analytics - Streamlit app.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.data import SEGMENT_DIMENSIONS, load_all
from views import data_explorer, drivers, early_warning, insights, overview, retention, segments

st.set_page_config(page_title="Learning Engagement Analytics", page_icon="📚", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def get_data():
    return load_all()


master, weekly, courses, learners = get_data()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Filters")
st.sidebar.caption("Apply to every tab (the model is always trained on all data).")

FILTERS = {
    "category": "Course category", "difficulty": "Difficulty", "primary_device": "Device",
    "acquisition_channel": "Acquisition channel", "employment_status": "Employment", "country": "Country",
}
mask = pd.Series(True, index=master.index)
for col, label in FILTERS.items():
    options = [str(x) for x in master[col].cat.categories] if hasattr(master[col], "cat") else sorted(master[col].unique())
    chosen = st.sidebar.multiselect(label, options, default=[], placeholder="All")
    if chosen:
        mask &= master[col].astype(str).isin(chosen)

dmin, dmax = master["enrollment_date"].min().date(), master["enrollment_date"].max().date()
date_range = st.sidebar.date_input("Enrollment date", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(date_range, tuple) and len(date_range) == 2:
    mask &= master["enrollment_date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))

filtered = master[mask]
filtered_weekly = weekly[weekly["enrollment_id"].isin(filtered["enrollment_id"])]
st.sidebar.markdown(f"**{len(filtered):,}** of {len(master):,} enrollments selected")
st.sidebar.divider()
st.sidebar.caption("Data: AI-generated synthetic dataset (seed 42). Snapshot date 30 Sep 2026.")

# --------------------------------------------------------------------- main
st.title("📚 Online Learning Engagement Analytics")
st.caption("Why do some learners complete courses while others drop out? "
           "Logins, video consumption, assessments, progress and completion behaviour.")

if filtered.empty:
    st.warning("No enrollments match the current filters.")
    st.stop()

tabs = st.tabs(["Overview", "Engagement drivers", "Drop-off & retention", "Segments",
                "Early-warning model", "Insights", "Data"])
with tabs[0]:
    overview.render(filtered, filtered_weekly)
with tabs[1]:
    drivers.render(filtered, filtered_weekly)
with tabs[2]:
    retention.render(filtered, filtered_weekly)
with tabs[3]:
    segments.render(filtered, filtered_weekly)
with tabs[4]:
    early_warning.render(master, weekly, set(filtered["enrollment_id"]))
with tabs[5]:
    insights.render(filtered, filtered_weekly)
with tabs[6]:
    data_explorer.render(filtered)
