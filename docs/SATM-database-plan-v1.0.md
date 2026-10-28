# SATM Database Plan v1.0 (approved by Salem, 2026-10-28)

## 1. Where it lives
- PostgreSQL on Neon (free tier is enough for ~100 users; upgrade the plan to grow).
- Use Postgres locally too (not SQLite) so local and production behave the same.
- Staging uses its own Neon database branch; tests never touch real student data.
- Every schema change goes through a migration file (Alembic / Flask-Migrate), run on staging first, after a backup.

## 2. Planning modes
- **Priority list (default):** tasks ranked by priority with predicted time, no time-of-day schedule. Optional "hours available today" highlights the top tasks that fit.
- **Scheduled day (opt-in):** tasks placed into free slots around the student's timetable (classes, busy times, optional prayer times).
- Stored as `users.planning_mode`; timetable is optional; `schedule_blocks` used only in scheduled mode.

## 3. Tables

### Core (v1)
| Table | Purpose / key fields |
|---|---|
| users | id, email, password_hash, name, timezone, planning_mode, settings (JSONB), training_opt_in, created_at, updated_at |
| courses | id, user_id, name, color, weight (credit hours), updated_at, deleted_at |
| tasks | id, user_id, course_id, title, raw_text, category, importance, due_at (UTC), estimated_minutes, status (todo/doing/done), recurrence_template_id, completed_at, created_at, updated_at, deleted_at |
| recurrence_templates | id, user_id, course_id, title, rule (e.g. weekly Monday), active |
| predictions | id, task_id, source (llm / own_model / fallback), model_version, category, importance, estimated_minutes, confidence, created_at |
| corrections | id, task_id, field, predicted_value, corrected_value, created_at (training data) |
| work_sessions | id, task_id, user_id, started_at, ended_at, minutes (actual time, from focus timer) |
| auth_tokens / password_reset_tokens | mobile sign-in tokens and reset tokens (hashed), expires_at |

### Added later via migrations (when the feature ships)
| Table | Purpose |
|---|---|
| timetable | user_id, day_of_week, start, end, kind (class / busy / prayer / free) |
| schedule_blocks | generated plan: task_id, start_at, end_at (scheduled mode only) |
| reminders | task_id, remind_at, sent_at, channel |
| devices | user_id, platform, push_token |
| model_versions | name, version, metrics (JSONB), deployed_at |

## 4. Design rules
- **UUID primary keys** so the phone can create tasks offline without ID clashes.
- **Priority is computed on request**, never stored (it changes as due dates approach).
- **Recurring tasks** = one template + a real task row per occurrence.
- **All timestamps in UTC**; each user's timezone stored separately.
- **Durations as whole minutes** (integers).
- **Soft deletes** (`deleted_at`) + `updated_at` on every row, enabling undo and offline sync.
- **Indexes:** (user_id, status, due_at) and (user_id, updated_at).
- **Privacy:** account deletion cascades to all user data; training exports exclude names/emails and respect training_opt_in.

## 5. Migrating existing data
- Old `deadline` (days left) becomes `due_at` = created_at + days.
- Old `time_est` (hours, float) becomes `estimated_minutes` (int).
- Existing integer IDs mapped to UUIDs during migration.

## 6. Alternatives considered
- Firebase / NoSQL: rejected; makes reporting and training-data queries much harder.
- Supabase: also Postgres; no real gain over Neon for the cost of moving.
