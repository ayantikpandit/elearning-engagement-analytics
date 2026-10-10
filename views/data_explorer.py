from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui import show_df

DICT_PATH = Path(__file__).resolve().parents[1] / "data" / "DATA_DICTIONARY.md"


def render(master: pd.DataFrame) -> None:
    st.markdown(f"Showing **{len(master):,}** enrollments after filters.")
    cols = st.multiselect("Columns", master.columns.tolist(),
                          default=["enrollment_id", "learner_id", "course_id", "status", "total_logins",
                                   "videos_watched", "assessments_attempted", "progress_pct", "completed"])
    show_df(master[cols].head(500) if cols else master.head(500))
    st.download_button("Download filtered data (CSV)", master.to_csv(index=False).encode(),
                       "filtered_enrollments.csv", "text/csv")
    with st.expander("Data dictionary"):
        if DICT_PATH.exists():
            st.markdown(DICT_PATH.read_text())
