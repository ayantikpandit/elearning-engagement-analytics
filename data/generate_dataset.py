"""
Synthetic dataset generator: Online Learning Engagement Analytics
Produces 4 linked CSVs:
  learners.csv            (1,500 rows)
  courses.csv             (28 rows)
  enrollments.csv         (5,000 rows)  <- main analysis table
  weekly_activity.csv     (~30k rows)   <- time-series / drop-off curves

Run:  python generate_dataset.py
Seed is fixed so results are reproducible.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
SNAPSHOT = pd.Timestamp("2026-09-30")

N_LEARNERS = 1500
N_COURSES = 30
N_ENROLL = 5000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- courses
categories = {
    "Data Science": ["Python for Data Analysis", "SQL Masterclass", "Machine Learning Basics",
                     "Tableau Dashboards", "Statistics for Analysts"],
    "Finance": ["Financial Statement Analysis", "Equity Valuation", "Excel for Finance",
                "Corporate Finance Fundamentals", "Fixed Income Essentials"],
    "Programming": ["Python Foundations", "JavaScript Essentials", "Git and GitHub",
                    "Web Development Bootcamp", "Data Structures Primer"],
    "Business": ["Business Strategy 101", "Product Management", "Project Management",
                 "Negotiation Skills", "Business Communication"],
    "Design": ["UI/UX Fundamentals", "Figma in Practice", "Graphic Design Basics"],
    "Marketing": ["Digital Marketing", "SEO Essentials", "Brand Building",
                  "Social Media Analytics", "Content Strategy"],
}
rows = []
cid = 1
for cat, titles in categories.items():
    for t in titles:
        diff = rng.choice(["Beginner", "Intermediate", "Advanced"], p=[0.45, 0.4, 0.15])
        weeks = int(rng.choice([4, 6, 8, 10, 12], p=[0.15, 0.25, 0.3, 0.2, 0.1]))
        videos = int(weeks * rng.integers(4, 7))
        rows.append(dict(
            course_id=f"C{cid:03d}", course_title=t, category=cat, difficulty=diff,
            duration_weeks=weeks, total_videos=videos,
            avg_video_length_min=int(rng.integers(8, 20)),
            total_assessments=int(weeks * rng.integers(1, 3)),
            price_usd=int(rng.choice([0, 19, 29, 49, 79, 99, 149],
                                     p=[0.12, 0.12, 0.2, 0.2, 0.16, 0.12, 0.08])),
            instructor_rating=round(float(np.clip(rng.normal(4.4, 0.3), 3.4, 5.0)), 1),
        ))
        cid += 1
courses = pd.DataFrame(rows)
N_COURSES = len(courses)

# ---------------------------------------------------------------- learners
edu = ["High School", "Diploma", "Bachelor's", "Master's", "PhD"]
learners = pd.DataFrame({
    "learner_id": [f"L{i:04d}" for i in range(1, N_LEARNERS + 1)],
    "age": np.clip(rng.normal(27, 7, N_LEARNERS), 17, 60).astype(int),
    "gender": rng.choice(["Male", "Female", "Other"], N_LEARNERS, p=[0.52, 0.45, 0.03]),
    "country": rng.choice(["India", "USA", "UK", "Germany", "Brazil", "Nigeria", "Canada", "Australia"],
                          N_LEARNERS, p=[0.38, 0.16, 0.09, 0.07, 0.08, 0.07, 0.08, 0.07]),
    "education_level": rng.choice(edu, N_LEARNERS, p=[0.1, 0.12, 0.5, 0.25, 0.03]),
    "employment_status": rng.choice(["Student", "Working Professional", "Job Seeker", "Freelancer"],
                                    N_LEARNERS, p=[0.42, 0.33, 0.15, 0.10]),
    "primary_device": rng.choice(["Mobile", "Laptop", "Tablet"], N_LEARNERS, p=[0.38, 0.56, 0.06]),
    "acquisition_channel": rng.choice(["Organic Search", "Social Media", "Referral", "Paid Ads", "Email"],
                                      N_LEARNERS, p=[0.30, 0.22, 0.18, 0.18, 0.12]),
    "signup_date": (pd.Timestamp("2025-06-01") +
                    pd.to_timedelta(rng.integers(0, 120, N_LEARNERS), unit="D")).date,
})
# latent traits (NOT exported) driving behaviour
learners["_motivation"] = rng.normal(0, 1, N_LEARNERS)
learners["_hours_free"] = np.clip(rng.normal(0, 1, N_LEARNERS), -2, 2)

# ------------------------------------------------------ enrollments + weekly
# unique learner-course pairs
pairs = set()
while len(pairs) < N_ENROLL:
    pairs.add((int(rng.integers(0, N_LEARNERS)), int(rng.integers(0, N_COURSES))))
pairs = sorted(pairs)
rng.shuffle(pairs)

diff_pen = {"Beginner": 0.0, "Intermediate": -0.25, "Advanced": -0.55}
edu_bonus = {"High School": -0.15, "Diploma": -0.05, "Bachelor's": 0.05, "Master's": 0.15, "PhD": 0.2}
emp_bonus = {"Student": 0.1, "Working Professional": -0.1, "Job Seeker": 0.2, "Freelancer": 0.0}

enroll_rows, weekly_rows = [], []
for i, (li, ci) in enumerate(pairs, start=1):
    L = learners.iloc[li]
    C = courses.iloc[ci]
    W = int(C.duration_weeks)

    enroll_date = pd.Timestamp("2025-10-01") + pd.Timedelta(days=int(rng.integers(0, 335)))
    weeks_elapsed = int((SNAPSHOT - enroll_date).days // 7)
    max_weeks = min(W, max(weeks_elapsed, 1))

    # latent engagement for this enrollment
    e = (0.85 * L._motivation + 0.30 * L._hours_free
         + diff_pen[C.difficulty] + edu_bonus[L.education_level] + emp_bonus[L.employment_status]
         - 0.004 * C.price_usd + 0.25 * (C.instructor_rating - 4.4)
         - (0.15 if L.primary_device == "Mobile" else 0)
         + rng.normal(0, 0.45))

    base_logins = np.exp(0.75 + 0.42 * e)
    videos_per_week = C.total_videos / W
    assess_per_week = C.total_assessments / W

    cum_videos = cum_assess = 0
    wk = []
    dropped, drop_week, last_active = False, None, 0
    scores = []
    for w in range(1, max_weeks + 1):
        decay = np.exp(-0.06 * (w - 1))              # natural fatigue
        # hazard of dropping this week
        hazard = sigmoid(-4.2 - 1.2 * e + 0.08 * w)
        if w > 1 and rng.random() < hazard:
            dropped, drop_week = True, w
            break
        # weekly behaviour
        p_active = sigmoid(0.9 + 1.0 * e - 0.05 * w)
        active = rng.random() < p_active
        if active:
            logins = max(1, rng.poisson(base_logins * decay))
            session = float(np.clip(rng.normal(22 + 7 * e, 7), 4, 90))
            v_watch = min(rng.poisson(videos_per_week * sigmoid(1.4 + 1.1 * e) * 1.4),
                          C.total_videos - cum_videos)
            vc_rate = float(np.clip(rng.normal(0.72 + 0.1 * e, 0.12), 0.15, 1.0))
            watch_min = v_watch * C.avg_video_length_min * vc_rate
            a_att = min(rng.poisson(assess_per_week * sigmoid(1.2 + 1.0 * e) * 1.4),
                        C.total_assessments - cum_assess)
            for _ in range(a_att):
                scores.append(float(np.clip(rng.normal(68 + 9 * e, 12), 0, 100)))
            forum = int(rng.poisson(max(0.05, 0.25 + 0.3 * e)))
            last_active = w
        else:
            logins, session, v_watch, vc_rate, watch_min, a_att, forum = 0, 0.0, 0, 0.0, 0.0, 0, 0
        cum_videos += v_watch
        cum_assess += a_att
        _r = dict(
            enrollment_id=f"E{i:05d}", week_number=w, logins=int(logins),
            total_session_minutes=round(logins * session, 1),
            videos_watched=int(v_watch), watch_time_min=round(watch_min, 1),
            assessments_attempted=int(a_att), forum_posts=forum,
            cumulative_progress_pct=round(100 * cum_videos / C.total_videos, 1),
        )
        weekly_rows.append(_r)
        wk.append(_r)

    progress = 100 * cum_videos / C.total_videos
    reached_end = (not dropped) and (max_weeks == W)
    if dropped:
        status = "Dropped"
    elif reached_end:
        status = "Completed" if (progress >= 65 and cum_assess >= 0.45 * C.total_assessments) else "Dropped"
        if status == "Dropped":
            drop_week = W          # stalled / never finished
    else:
        status = "In Progress"

    total_logins = sum(r["logins"] for r in wk)
    active_weeks = sum(1 for r in wk if r["logins"] > 0)
    total_sess = sum(r["total_session_minutes"] for r in wk)
    last_login_date = enroll_date + pd.Timedelta(days=7 * (last_active - 1) + int(rng.integers(0, 7)))

    enroll_rows.append(dict(
        enrollment_id=f"E{i:05d}", learner_id=L.learner_id, course_id=C.course_id,
        enrollment_date=enroll_date.date(),
        total_logins=total_logins,
        active_weeks=active_weeks,
        avg_session_minutes=round(total_sess / total_logins, 1) if total_logins else 0.0,
        videos_watched=cum_videos,
        total_watch_time_min=round(sum(r["watch_time_min"] for r in wk), 1),
        avg_video_completion_pct=round(min(100.0, 100 * sum(r['watch_time_min'] for r in wk) /
                                           (cum_videos * C.avg_video_length_min)), 1) if cum_videos else 0.0,
        assessments_attempted=cum_assess,
        avg_assessment_score=round(float(np.mean(scores)), 1) if scores else np.nan,
        forum_posts=sum(r["forum_posts"] for r in wk),
        progress_pct=round(progress, 1),
        last_active_week=last_active,
        days_since_last_login=int((SNAPSHOT - last_login_date).days) if status != "Completed" else np.nan,
        drop_week=drop_week if status == "Dropped" else np.nan,
        status=status,
        completed=1 if status == "Completed" else 0,
        completion_date=(enroll_date + pd.Timedelta(days=7 * W - int(rng.integers(0, 5)))).date()
        if status == "Completed" else None,
        certificate_issued=int(status == "Completed" and rng.random() < 0.85),
        course_rating_by_learner=(int(np.clip(round(rng.normal(3.6 + 0.5 * e, 0.8)), 1, 5))
                                  if status == "Completed" or (status == "Dropped" and rng.random() < 0.25)
                                  else np.nan),
    ))

enroll = pd.DataFrame(enroll_rows)
weekly = pd.DataFrame(weekly_rows)

# ---------------------------------------------------- light realistic mess
# ~2% missing education_level, ~1% missing age (for a cleaning step)
learners.loc[learners.sample(frac=0.02, random_state=1).index, "education_level"] = np.nan
learners.loc[learners.sample(frac=0.01, random_state=2).index, "age"] = np.nan

learners = learners.drop(columns=["_motivation", "_hours_free"])

learners.to_csv("learners.csv", index=False)
courses.to_csv("courses.csv", index=False)
enroll.to_csv("enrollments.csv", index=False)
weekly.to_csv("weekly_activity.csv", index=False)

print("learners", learners.shape, "| courses", courses.shape,
      "| enrollments", enroll.shape, "| weekly", weekly.shape)
print(enroll.status.value_counts(normalize=True).round(3))
