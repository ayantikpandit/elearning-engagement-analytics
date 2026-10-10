# Online Learning Engagement Analytics

> **Business question:** an online-learning platform wants to understand *why some learners complete courses while others drop out*.
> This app analyses login frequency, video consumption, assessment attempts, course progress and completion behaviour, and turns the findings into an **early-warning model** that flags at-risk learners while there is still time to help.

**Live app:** `<ADD STREAMLIT CLOUD LINK>`  |  **Author:** `<YOUR NAME>`  |  **Stack:** Python, Pandas, Plotly, scikit-learn, Streamlit

![App screenshot](docs/screenshot_overview.png)  <!-- add after you run the app -->

## What the app does
| Tab | What you can do |
|---|---|
| Overview | KPIs, outcome mix, enrollments over time, completion rate sliced by 10 dimensions |
| Engagement drivers | Completers vs. droppers, quintile "dose-response" bars, logins x assessments heatmap, correlations |
| Drop-off & retention | Retention curves (active vs. enrolled), last-seen week of non-completers, weekly behaviour by outcome |
| Segments | K-Means personas (Ghosts, Skimmers, Steady, Power) with an intervention playbook per persona |
| Early-warning model | Score learners at end of week 1-4, 3 models, ROC/confusion matrix, feature importance, intervention-budget view, at-risk list with suggested actions (CSV export), what-if simulator |
| Insights | Auto-written findings and recommendations that update with the sidebar filters |
| Data | Filtered table, CSV download, data dictionary |

## Data
AI-generated **synthetic** data (reproducible: `cd data && python generate_dataset.py`, seed 42).
4 linked tables: learners (1,500), courses (28), enrollments (5,000), weekly_activity (~30.7k rows). See `data/DATA_DICTIONARY.md`.
The generator uses hidden learner traits (motivation, free time) plus course difficulty, price, device and noise, so patterns are realistic but **not real-world evidence**.

## Method highlights
- **No look-ahead:** model features use only weeks 1..N; target-leaking columns (final progress, drop week, completion date, certificate) are excluded and unit-tested.
- **Grouped split:** train/test split is by learner, so one person never appears on both sides.
- **Honest scope:** in-progress enrollments are excluded from training and completion-rate maths.
- **Explainability:** permutation importance plus a rule-based "suggested action" per flagged learner.
- **Operational view:** "if we can contact only the riskiest X%, how many real non-completers do we reach?"

## Headline results (current data; recomputed live in the app)
- ~32% of finished enrollments complete.
- High early (week 1-2) activity: ~69% completion vs ~8% for low early activity.
- Attempting an assessment in weeks 1-2: ~48% completion vs ~7% without.
- Completion falls from ~38% (Beginner) to ~19% (Advanced); ~35% (Free) to ~19% ($81+).
- Early-warning model ROC AUC: 0.82 (end of week 1), 0.90 (week 2), 0.92 (week 3), 0.93 (week 4).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q tests                              # optional
```

## Deploy (free)
1. Push this folder to a public GitHub repo.
2. Go to share.streamlit.io, connect the repo, set main file to `app.py`, deploy.

## Project structure
```
app.py              Streamlit entry point + sidebar filters
src/data.py         loading, merging, derived features
src/analysis.py     retention, last-active distribution, auto insights
src/model.py        early-warning pipeline (features, training, scoring, what-if)
src/segments.py     K-Means personas
src/ui.py           shared chart styling
views/              one module per tab
tests/              pytest checks (shapes, leakage guard, model sanity)
data/               CSVs, generator, data dictionary
```

## Limitations
Synthetic data; association not causation; model thresholds and playbooks are illustrative and need A/B testing before real use.
