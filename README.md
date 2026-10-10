# Online Learning Engagement Analytics

> **Business question:** why do some learners complete online courses while others drop out?
> This project analyses login frequency, video consumption, assessment attempts, course progress and completion behaviour, then turns the findings into an **early-warning model** that flags at-risk learners while there is still time to help them.

**Live app:** http://192.168.1.33:8501/ *(local-network link, active while the app is running on my machine)*
**Author:** Ayantik Pandit  |  **Stack:** Python, Pandas, Plotly, scikit-learn, Streamlit, GitHub

---

## My approach

1. **Frame the problem.** Defined "completion" as finishing a course (enough video progress and assessments), and separated learners into Completed, Dropped and In Progress. In-progress enrollments have no known outcome, so they are excluded from completion rates and model training.
2. **Build a clean analytical base.** Joined four tables (learners, courses, enrollments, weekly activity) into one row per enrollment, handled missing values, and engineered features such as early (week 1-2) engagement, assessment coverage, price and age bands, and pace versus plan.
3. **Explore what separates completers from droppers.** Compared behaviour by outcome, sliced completion by learner and course attributes, and tested combined effects (logins x assessments).
4. **Study when learners disengage.** Built retention curves and looked at the last week each non-completer was active, to find the critical window for intervention.
5. **Segment learners.** Used K-Means on engagement signals to create four behavioural personas, each with a suggested intervention.
6. **Predict risk early.** Trained models that use only activity up to week *N* to flag learners unlikely to finish, with a grouped train/test split by learner and checks that no outcome-related column leaks into the features.
7. **Ship it.** Packaged everything as an interactive Streamlit app with sidebar filters, exportable at-risk lists and a what-if simulator, plus unit tests.

## Dataset overview

| Table | Rows | Grain |
|---|---|---|
| learners | 1,500 | one row per learner (demographics, device, acquisition channel) |
| courses | 28 | one row per course (category, difficulty, length, price, rating) |
| enrollments | 5,000 | one row per learner-course pair (main analysis table) |
| weekly_activity | ~30,700 | one row per enrollment per week (logins, videos, watch time, assessments, forum posts, progress) |

5,000 enrollments across 1,461 learners and 28 courses. Full column descriptions are in `data/DATA_DICTIONARY.md`.

## Analysis and key findings

**Outcome mix.** 64.5% of enrollments ended in dropout, 30% in completion and 5.5% are still in progress. Among finished enrollments the **completion rate is 31.7%**, and **10.4% of enrollments never logged in at all**.

**Early engagement is the strongest signal.**
- Learners with high activity in weeks 1-2 complete at about **69%**, versus about **8%** for low early activity (roughly 9x).
- Attempting an assessment in the first two weeks goes with about **48%** completion versus **7%** for learners who did not.

**Engagement tiers, not demographics, separate outcomes.**

| Segment (K-Means) | Share of enrollments | Completion rate | Profile |
|---|---|---|---|
| Ghosts | 20% | 0% | Enrolled but barely engaged |
| Skimmers | 36% | ~4% | Dip in and out, rarely assess |
| Steady Learners | 31% | ~59% | Consistent weekly activity |
| Power Learners | 12% | ~96% | High logins, videos and assessments |

**Course and learner attributes matter, but less.**
- Difficulty: completion falls from about 38% (Beginner) to about 19% (Advanced).
- Price: about 35% for free courses versus about 19% for courses priced $81+.
- Employment: job seekers (36.5%) and students (34.3%) complete more than working professionals (27.0%).
- Category, device and acquisition channel differ only by a few percentage points (for example, 29% for Programming to 33% for Design).

**When learners disappear.** About 32% of non-completers were last seen in week 2 or earlier, and 16% never logged in. Most non-completers stay enrolled until the course ends without finishing, so "active that week" is a better health metric than enrollment status.

**Early-warning model.** Using only activity up to the end of week *N*:

| Score at end of week | ROC AUC |
|---|---|
| 1 | ~0.82 |
| 2 | ~0.90 |
| 3 | ~0.92 |
| 4 | ~0.93 |

At week 2, flagging learners at a 0.5 risk threshold gives precision of about 0.87 and recall of about 0.88. If support can only contact the riskiest 30% of learners, about 98% of those contacted are true non-completers, reaching roughly 45% of all learners who would not finish. The most influential features are session time, watch time, assessments attempted and pace versus plan.

## Recommendations

1. **Fix the first two weeks.** Trigger a nudge within 48 hours if there is no login, and make the first lesson one click away.
2. **Make the first assessment easy.** Add a short, low-stakes quiz in week 1.
3. **Run the early-warning list weekly.** Contact the highest-risk learners using the suggested actions (re-engagement message, catch-up plan, quiz prompt).
4. **De-risk advanced and premium courses.** Add prerequisite checks, pacing guides and milestone incentives where completion is lowest.
5. **Track "active last week" as the health metric** instead of enrollment status.

## What the app does

| Tab | What you can do |
|---|---|
| Overview | KPIs, outcome mix, enrollments over time, completion rate sliced by 10 dimensions |
| Engagement drivers | Completers vs. droppers, dose-response bars, logins x assessments heatmap, correlations |
| Drop-off & retention | Retention curves, last-seen week of non-completers, weekly behaviour by outcome |
| Segments | Behavioural personas with an intervention playbook |
| Early-warning model | Choose scoring week, model and threshold; ROC, confusion matrix, feature importance, intervention-budget view, at-risk list (CSV export), what-if simulator |
| Insights | Auto-written findings and recommendations that update with the filters |
| Data | Filtered table, CSV download, data dictionary |

## Method notes
- **No look-ahead:** model features use only weeks 1 to N; outcome-linked columns (final progress, drop week, completion date, certificate) are excluded and covered by a unit test.
- **Grouped split:** train/test split is by learner, so one person never appears on both sides.
- **Explainability:** permutation importance plus a rule-based suggested action for every flagged learner.
- **Operational view:** "if we can contact only the riskiest X%, how many real non-completers do we reach?"

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q tests          # optional
```

## Project structure
```
app.py              Streamlit entry point and sidebar filters
src/data.py         loading, merging, derived features
src/analysis.py     retention, last-active distribution, auto insights
src/model.py        early-warning pipeline (features, training, scoring, what-if)
src/segments.py     K-Means personas
src/ui.py           shared chart styling
views/              one module per tab
tests/              pytest checks (shapes, leakage guard, model sanity)
data/               source tables and data dictionary
```

## Limitations
Findings show association, not causation, so nudging logins alone is not proven to raise completion. Risk thresholds and intervention playbooks are illustrative and should be validated with an A/B test before operational use.
