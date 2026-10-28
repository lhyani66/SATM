# SATM Backend Plan v2.0 (approved by Salem, 2026-10-28)

## 1. Backend structure (Flask, kept)
- Split app.py into modules: auth, tasks, predictions, scheduling; configs for local / staging / production.
- Server owns priority, deadline detection and schedule building (one source of truth for web + mobile).
- Deadlines stored as real dates, not "days left".
- Auth: cookie sessions for web + token auth for mobile; password reset (PR #2, needs SMTP).
- Input validation, clear errors, rate-limited login, versioned API (/api/v1).
- Automated tests (pytest) + CI on every push.
- Background job for reminders / push notifications.

## 2. Deployment
- Render always-on plan (~$7/mo) + free tiers: Neon Postgres, Expo push, Sentry.
- Budget: $10-30/mo, never above ~$100; start ~100 users, designed to scale by upgrading plans.
- Staging deploys from `develop`, production from `main`.
- Health check, error alerts, DB migrations, backups before risky changes, secrets only in Render env vars.

## 3. AI / models (hybrid)
- Launch: ready-made LLM API parses task text (category, course, due date, first time guess) with per-user caps; acts as a "teacher".
- Own text model: fine-tune a small pretrained multilingual open-source model (MiniLM-type) with SATM heads; trained on API labels + student corrections; replaces the API once it matches quality.
- Per-student models (scikit-learn / LightGBM) learn time estimates and habits from actual times and corrections.
- Dates via existing library (dateparser).
- Current scikit-learn ensemble kept as fallback.
- Evaluation baseline + versioned offline retraining; deploy a new model only if it beats the old one.
- Original data was AI-generated: use only as a starting point.

## 4. Features tied to backend
- Save student corrections; track actual time (focus timer suggested).
- English first, Arabic later.
- Reminders + push notifications.
- Basics: task editing, calendar view, search/filters, settings, progress charts, board view, onboarding.
- Suggested: recurring tasks, one-sentence quick add, offline sync, privacy opt-out + account deletion.
