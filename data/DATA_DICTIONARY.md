# Synthetic Dataset: Online Learning Engagement Analytics
AI-generated synthetic data (seed=42, reproducible via `generate_dataset.py`). Snapshot date: 2026-09-30.

| File | Rows | Grain |
|---|---|---|
| learners.csv | 1,500 | one row per learner |
| courses.csv | 28 | one row per course |
| enrollments.csv | 5,000 | one row per learner-course enrollment (MAIN TABLE) |
| weekly_activity.csv | ~30,700 | one row per enrollment per active-or-inactive week (until drop/completion/snapshot) |

Joins: enrollments.learner_id -> learners.learner_id; enrollments.course_id -> courses.course_id; weekly_activity.enrollment_id -> enrollments.enrollment_id

## enrollments.csv (key columns)
- status: Completed / Dropped / In Progress (In Progress should be excluded from completion modelling)
- completed: 1 if Completed else 0 (target variable)
- total_logins, active_weeks, avg_session_minutes: login behaviour
- videos_watched, total_watch_time_min, avg_video_completion_pct: video consumption
- assessments_attempted, avg_assessment_score: assessment behaviour (score is NaN if no attempts)
- progress_pct: videos_watched / course total_videos
- last_active_week, days_since_last_login (NaN for completed), drop_week (NaN unless Dropped)
- forum_posts, completion_date, certificate_issued, course_rating_by_learner (NaN where not given)

## weekly_activity.csv
week_number, logins, total_session_minutes, videos_watched, watch_time_min, assessments_attempted, forum_posts, cumulative_progress_pct

## Deliberate data-cleaning cases
~2% missing education_level and ~1% missing age in learners.csv; structural NaNs in enrollments.csv as described above.

## Caution
Leakage: progress_pct, last_active_week, drop_week, completion_date and certificate_issued are outcome-linked. Exclude them (or use only early-weeks data from weekly_activity.csv) when building a predictive model.
