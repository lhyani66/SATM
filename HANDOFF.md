# SATM — Handoff

**From:** Salem · **To:** you · **Date:** 19 Sep 2026

You're taking over SATM to build the mobile app (iOS + Android). Everything you
need is in this repo. This file is the map. Read it top to bottom once, then use
it as a reference.

---

## 0. The short version

- SATM is a **working, deployed web app**. Don't rebuild it — build a phone app
  that talks to its API.
- The backend (Flask, Python) is the brain. It does the AI and stores the data.
  **You will barely touch it.**
- The web frontend (`SATM.html`) is your **reference design and reference
  logic**. Some of the app's logic lives *only* there, not in the API — you have
  to port it (section 6).
- Recommended app stack: **Expo (React Native)**. One codebase, both stores,
  test on your real phone in 10 minutes, no Mac needed (section 7).
- Work in small steps. Milestone 1 is just "the phone shows the task list". Get
  there first (section 8).

---

## 1. What SATM is

**Smart Academic Task Manager.** A student types a task the way they'd say it
out loud — *"finish bio hw ASAP, exam is tomorrow"* — and three machine-learning
models read it and agree on:

| | |
|---|---|
| **Category** | Homework / Exam / Quiz / Project |
| **Importance** | high / medium / low |
| **Time estimate** | hours, 0.5–8 |

The app also pulls the deadline out of the words ("tomorrow", "in 3 days",
"next week"), scores every task's priority, and builds a day schedule in 45-min
blocks with breaks. Tasks can be ticked off.

It's Salem's CS graduation project. The models were trained on 1,000 labelled
academic tasks.

---

## 2. What exists today, and where

| Thing | Where | You need access? |
|---|---|---|
| **Live site** | https://satm.onrender.com | No — it's public. Register an account and play. |
| **Code** | https://github.com/lhyani66/SATM | **Yes** — Salem adds you as a collaborator |
| **Hosting** (backend) | Render, free tier | Not for now |
| **Database** | Neon (PostgreSQL), free tier | **No** — you never need prod DB access |

> **Salem — to-do before handing over:**
> GitHub → repo → Settings → Collaborators → add your brother.
> That's the only access he needs to start.

### How it's wired

```
 Phone / browser
       │  HTTPS, JSON
       ▼
 Render  ──  app.py (Flask)  ──  9 model .pkl files
       │
       ▼
 Neon PostgreSQL   (users + tasks)
```

Locally it's the same thing with SQLite instead of Neon. Zero setup.

---

## 3. The repo, file by file

```
SATM/
├── app.py                 ← THE BACKEND. All API routes + ML pipeline + auth. ~300 lines.
├── SATM.html              ← THE WEB APP. One file: HTML + CSS + JS. Your reference.
├── models/                ← 9 trained models. Don't touch. Don't retrain unless asked.
│   ├── tfidf_vectorizer.pkl
│   ├── category_model.pkl,  cat_model_svm.pkl,  cat_model_gb.pkl
│   ├── importance_model.pkl, imp_model_svm.pkl, imp_model_gb.pkl
│   ├── time_model.pkl,      time_model_svr.pkl, time_model_gb.pkl
│   ├── deadline_scaler.pkl
│   └── category_encoder.pkl
├── SATM.ipynb             ← Notebook that trained the original 3 models
├── train_ensemble.py      ← Trains the other 6 (SVM + GradientBoosting)
├── satm_dataset_1000_rows.csv ← Training data
├── smoke_test.py          ← Runs the whole API end to end. Your first command.
├── requirements.txt       ← Python deps, pinned. Don't bump scikit-learn (see gotchas).
├── render.yaml, Procfile  ← Render deploy config. Leave alone.
├── .env.example           ← Copy to .env for local overrides. .env is gitignored.
├── README.md              ← Public readme
├── HANDOFF.md             ← This file
├── SATM_Project_Brief.pdf ← Project brief for the university
└── SATM_Study_Guide.pdf   ← Salem's defense notes — explains the ML choices
```

Ignore `instance/` (local SQLite, auto-created) and `debug-*.log`.

---

## 4. Run it on your machine

You need **Python 3.11+** and **git**. Check:

```bash
python --version
git --version
```

If either fails, install from python.org and git-scm.com, then reopen your
terminal.

### 4.1 Get the code

```bash
cd C:\dev
git clone https://github.com/lhyani66/SATM.git
cd SATM
```

`git clone` downloads the repo **and its whole history**. You now have
everything Salem has.

### 4.2 Install and run

```bash
pip install -r requirements.txt
python app.py
```

First run downloads some NLTK language data (~1 min). Then open
**http://127.0.0.1:5000**. Register any email/password — it's your local
database, nothing leaves your machine.

### 4.3 Prove the API works

With the server running, in a **second** terminal:

```bash
python smoke_test.py
```

It registers a throwaway account, asks the AI to classify a task, saves it,
ticks it off, lists it, deletes it, logs out — and asserts every step. You
should see six `ok` lines and `ALL GOOD`.

**That's the whole product in one command.** Your app is a nicer way to send
those same requests and show the answers. Read `smoke_test.py` — it's 50
lines and it's the API contract in runnable form.

Same script against the live server (first run takes up to a minute — see
"Render sleeps" in gotchas):

```bash
python smoke_test.py https://satm.onrender.com
```

### 4.4 Git — the five commands you actually need

Git is a save system with history. You'll use these and almost nothing else:

```bash
git status
```
What changed since the last save.

```bash
git add -A
```
Stage everything for saving.

```bash
git commit -m "Add task list screen"
```
Save a snapshot with a message. **Do this every time something works.**
Small commits, often. Message = what you did, in plain words.

```bash
git push
```
Upload your commits to GitHub. Do this at the end of every session.

```bash
git pull
```
Download anything Salem pushed. Do this at the start of every session.

If git ever says something scary and you don't understand it — **don't guess,
don't delete the folder.** Copy the message and ask Salem. Almost everything
in git is reversible if you don't panic.

---

## 5. The API — your contract

**Base URL:** `https://satm.onrender.com` (production) or
`http://127.0.0.1:5000` (local).

Every request and response is JSON. Set the header
`Content-Type: application/json` on anything with a body.

### 5.1 How login works (read this — it affects the app)

Auth is **cookie-based**. When you log in, the server sends back a
`session` cookie. Every later request must send that cookie back.

React Native's `fetch` stores and sends cookies automatically on iOS and
Android. Pass `credentials: 'include'` on every call to be explicit:

```js
const API = 'https://satm.onrender.com';

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (res.status === 401) throw new Error('Not logged in');
  return res.json();
}

// usage
await api('/api/login', { method: 'POST', body: JSON.stringify({ email, password }) });
const tasks = await api('/api/tasks');
```

Cookies work but they're the old way for mobile. If they give you trouble,
the upgrade is **token auth** — the backend returns a token on login, the app
stores it and sends `Authorization: Bearer <token>`. That's a ~20-line change
in `app.py`. Do it only if cookies actually bite you. Ask Salem first.

`401` on any request = you're logged out. Send the user to the login screen.

### 5.2 Endpoints

**`POST /api/register`** — create account, logs you in
```json
→ { "email": "a@b.com", "password": "Strong!1" }
← 201 { "email": "a@b.com" }
← 400 { "error": "Email and password are required" }
← 409 { "error": "An account with this email already exists" }
```
The web frontend enforces password rules (8+ chars, upper, lower, number,
symbol) **in the browser only**. The API accepts anything. Copy the rules
into the app so behaviour matches.

**`POST /api/login`**
```json
→ { "email": "a@b.com", "password": "Strong!1" }
← 200 { "email": "a@b.com" }
← 401 { "error": "Invalid email or password" }
```

**`POST /api/logout`** → `200 { "message": "Logged out" }`

**`GET /api/me`** — who am I? Call on app launch to check for a saved session.
```json
← 200 { "email": "a@b.com" }
← 401 (not logged in)
```

**`POST /api/predict`** — the AI. Doesn't save anything.
```json
→ { "text": "finish bio hw ASAP exam tomorrow", "deadline": 1 }
← 200 {
    "category":    "Homework",
    "cat_agree":   3,          // how many of the 3 models agreed (2 or 3)
    "importance":  "high",
    "imp_agree":   3,
    "time_est":    2.5,        // hours
    "time_spread": 0.3         // disagreement between models, in hours
  }
```
`deadline` is **days from now** (integer). Default 3 if omitted. The web app
gets it from `detectDeadline()` (section 6) and lets the user override.
Show `cat_agree`/`imp_agree` as a confidence badge — "3/3 agree" is the
web's wording.

**`GET /api/tasks`** — all of the logged-in user's tasks, newest first
```json
← 200 [
  { "id": 12, "task_text": "finish bio hw", "category": "Homework",
    "importance": "high", "deadline": 1, "time_est": 2.5, "status": "todo" }
]
```
`status` is `"todo"` or `"done"`. Sorting by priority is **your job** (section 6).

**`POST /api/tasks`** — save a task (after the user confirms the prediction)
```json
→ { "task_text": "...", "category": "Homework", "importance": "high",
    "deadline": 1, "time_est": 2.5 }
← 201 { "id": 13 }
```
Only `task_text` is required. Everything else is whatever you pass.

**`PATCH /api/tasks/:id`** — update any of `status`, `category`,
`importance`, `deadline`, `time_est`
```json
→ { "status": "done" }        // this is how "tick the task" works
← 200 { "message": "Updated" }
← 404 { "error": "Task not found" }
```

**`DELETE /api/tasks/:id`** → `200 { "message": "Deleted" }` / `404`

Users can only see and change their own tasks. The API enforces that — you
don't have to.

---

## 6. Logic that lives ONLY in the browser — you must port this

The API does prediction and storage. **Everything below runs in `SATM.html`
and nowhere else.** Copy it into the app. It's all plain JavaScript, so if
you use Expo it moves over almost unchanged. Search `SATM.html` for the
function name to find the original.

### 6.1 `detectDeadline(text)` → days or `null`

Reads the task text and guesses the deadline. Run it as the user types; if it
returns a number, pre-fill the deadline field and show an "auto-detected"
badge; the user can still change it.

```js
function detectDeadline(text) {
  const t = text.toLowerCase();
  if (/\b(asap|urgent|right now|due now|due today|today|tonight)\b/.test(t)) return 1;
  if (/\btomorrow\b/.test(t)) return 2;
  let m;
  m = t.match(/in\s+(\d+)\s+days?/);   if (m) return parseInt(m[1]);
  m = t.match(/(\d+)\s+days?\b/);       if (m) return parseInt(m[1]);
  m = t.match(/in\s+(\d+)\s+weeks?/);   if (m) return parseInt(m[1]) * 7;
  m = t.match(/(\d+)\s+weeks?\b/);      if (m) return parseInt(m[1]) * 7;
  m = t.match(/in\s+(\d+)\s+months?/);  if (m) return parseInt(m[1]) * 30;
  m = t.match(/(\d+)\s+months?\b/);     if (m) return parseInt(m[1]) * 30;
  const words = {one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,a:1,an:1};
  for (const [w, n] of Object.entries(words)) {
    if (new RegExp(`in\\s+${w}\\s+days?`).test(t))  return n;
    if (new RegExp(`in\\s+${w}\\s+weeks?`).test(t)) return n * 7;
  }
  if (/in\s+a\s+month/.test(t)) return 30;
  if (/next\s+week/.test(t))    return 7;
  if (/next\s+month/.test(t))   return 30;
  return null;
}
```

### 6.2 `calcPriority(deadline, importance)` → 5–100

```js
function calcPriority(dl, imp) {
  const p = 50
    + (imp === 'high' ? 25 : imp === 'low' ? -20 : 0)
    + (dl <= 1 ? 30 : dl <= 2 ? 20 : dl <= 3 ? 10 : dl <= 5 ? 5 : 0);
  return Math.min(100, Math.max(5, p));
}
```

Sort the task list by this, highest first. Colour: `>= 75` red, `>= 50`
amber, else green. The recommendation text the web shows:

| score | text |
|---|---|
| ≥ 75 | Start immediately — this is your highest priority. Block focused time today. |
| ≥ 50 | Schedule within *{deadline}*. Work in 45-min focused sessions. |
| else | Low urgency — handle after higher-priority tasks are done. |

### 6.3 What counts as "urgent"

```js
const urgent = openTasks.filter(t => t.deadline <= 2 || t.importance === 'high').length;
```

`openTasks` = tasks with `status !== 'done'`. The summary numbers (tasks,
hours, urgent) are computed from **open tasks only** — finishing a task must
make the numbers go down.

### 6.4 The day schedule

Start at 09:00. Walk the tasks in priority order. Each task takes
`time_est` hours. After 3 hours of work, insert a 1-hour break.

```js
function buildSchedule(openTasksByPriority) {
  let t = 9 * 60, worked = 0, rows = [];
  for (const task of openTasksByPriority) {
    if (worked >= 180) { rows.push({ time: t, break: true, mins: 60 }); t += 60; worked = 0; }
    const mins = Math.round((task.time_est || 1) * 60);
    rows.push({ time: t, task, mins });
    t += mins; worked += mins;
  }
  return rows;   // time is minutes since midnight
}
```

### 6.5 Deadline in words

`1` → "today", `2` → "tomorrow", else `"N days"`.

---

## 7. Recommended stack: Expo (React Native)

**Why this and not Flutter / native Swift+Kotlin:**

- One codebase → both App Store and Google Play.
- It's JavaScript. The logic in section 6 copies over almost verbatim. You
  learn one language, not three.
- **Expo Go** — install the app on your phone, scan a QR code, see your app
  running on real hardware. No Xcode, no Android Studio, no emulators to set up.
- **EAS Build** builds the iOS version in the cloud. **You don't need a Mac.**
- Biggest community, most tutorials, most answers on Stack Overflow.

**Start here:**

```bash
npx create-expo-app@latest satm-mobile
cd satm-mobile
npx expo start
```

Install **Expo Go** from the App Store / Play Store on your phone, scan the QR
code. You're running a mobile app. That's day one.

Keep the mobile app in **its own repo** (`C:\dev\satm-mobile`), not inside
this one. Different language, different deploy, different history. Link them
in each README.

**Store facts, so there are no surprises later:**

| | Google Play | Apple App Store |
|---|---|---|
| Developer account | $25 once | $99 / year |
| Review time | hours to 1 day | 1–3 days, can reject |
| Build from Windows | yes | via EAS Build (cloud) |
| Test on your phone | Expo Go | Expo Go |

Don't buy either account until the app works end to end in Expo Go.

---

## 8. Milestones — what "done" looks like at each step

Do these **in order**. Each one is small enough to finish. Commit when each
one works.

| # | Milestone | You're done when |
|---|---|---|
| 1 | **Backend runs locally** | `python smoke_test.py` prints `ALL GOOD` |
| 2 | **Expo Go shows a screen** | Your phone shows "Hello SATM" |
| 3 | **Login** | Login screen → `/api/login` → next screen. `401` → error shown |
| 4 | **Task list** | `/api/tasks` displayed, sorted by `calcPriority`, coloured badges |
| 5 | **Tick a task** | Tap → `PATCH {status:'done'}` → row greys out, counts drop |
| 6 | **Add a task** | Type → `detectDeadline` pre-fills → `/api/predict` → confirm → `POST /api/tasks` |
| 7 | **Schedule tab** | `buildSchedule` rendered as a timeline |
| 8 | **Register + logout + session restore** | `/api/me` on launch skips login if still valid |
| 9 | **Looks like SATM** | Section 9 applied |
| 10 | **Store builds** | `eas build` succeeds for both platforms |

Milestones 1–5 are the real product. If you get there, everything else is
polish. Don't start 6 before 5 works.

**Expectation on pace:** there is no deadline from Salem. A beginner doing
this seriously in evenings should reach milestone 5 in 3–4 weeks. Ask for
help the moment you're stuck for more than an hour — not after a week.

---

## 9. Design — so the app looks like SATM

The web app has a specific identity. Match it; don't invent a new one.

**Concept:** a small shop at the end of an alley on a rainy night. Warm, quiet,
premium. Not a corporate dashboard.

| Token | Value |
|---|---|
| Background | `#0A0C09` — warm near-black, never blue-black |
| Text | `#E9EFE4` / secondary `#A2AE9A` / muted `#717D6B` |
| **Accent** | `#52DE63` → `#17A63F` gradient ("Meadow" green) |
| High priority | `#FB7185` rose |
| Medium | `#FBBF24` amber |
| Low | `#34D399` emerald |
| Radius | 12–14px cards, pill badges |
| Borders | hairlines, `rgba(235,255,225,.085)` |

**Fonts** (all free on Google Fonts, available for Expo via
`@expo-google-fonts`):
- Headings: **Bricolage Grotesque** 800, tight letter-spacing
- Body: **Onest**
- Numbers, times, labels: **IBM Plex Mono**, uppercase, wide tracking

**Logo:** the "Ess" mark — an S built from three stacked bars, fading
downward. SVG is in `SATM.html`, search `<svg viewBox="0 0 32 32"`. Use
`react-native-svg`.

**Task rows:** a soft status gradient sweeps in from the right edge — rose for
high, amber for medium, emerald for low. That's the signature detail.

**Moods** (nice-to-have, do last): the web shifts palette by state —
*Rain* (cold blue) after 21:00, *Panic* (ember orange) at 5+ urgent tasks,
*All clear* (dawn gold) when everything's ticked. Open `SATM.html` and press
`R` / `P` to see them. Port these after milestone 9, not before.

---

## 10. Gotchas and tips

**Render sleeps.** Free tier. The first request after ~15 min idle takes
**30–60 seconds** while the container wakes up. Your app must show a loading
state, not a spinner-forever or a crash. Before any demo, open the site 5 min
early to warm it.

**Don't upgrade scikit-learn.** The `.pkl` models were saved by
scikit-learn 1.8.0 and will fail to load under a different version. Everything
in `requirements.txt` is pinned for a reason. If you ever need to retrain:
`python train_ensemble.py` after running the notebook.

**CORS doesn't exist on native.** A native app's `fetch` ignores CORS, so the
backend's lack of CORS headers is fine. It *will* bite you if you run
`expo start --web`. Test on the phone, not the browser.

**`curl` in PowerShell is a trap.** On Windows, `curl` is an alias for
`Invoke-WebRequest`, and even `curl.exe` mangles the `\"` quotes inside JSON.
Any tutorial that says "just curl it" will fail for you with a confusing
error. Use `smoke_test.py` as your template instead, or use PowerShell's own
`Invoke-RestMethod` with a hashtable body.

**`deadline` is days, `time_est` is hours.** Integers and floats. Don't send
strings.

**Password rules live in the browser.** The API accepts any password. Copy
the rules into the app (section 5.2).

**Local dev = SQLite, no setup.** You never need the Neon connection string.
If someone gives you one, don't paste it into a file, a chat, or a screenshot.

**The prod database is real.** Accounts on satm.onrender.com are real
people's (well, Salem's and yours). Test against **local** for anything
destructive.

**Small commits.** `git commit` every time a milestone step works. If you break
something, `git log` shows you when it last worked.

**Ask early.** An hour stuck = ask. Screenshot the error, paste the exact
message. "It doesn't work" isn't a question.

---

## 11. Rules

1. **Never commit secrets.** No passwords, tokens, or connection strings in
   any file. `.env` is gitignored — that's where they go. Use `<PLACEHOLDER>`
   in examples.
2. **Don't push to `main` on this repo without telling Salem.** Pushing to
   `main` here **redeploys the live site**. For backend changes, make a branch
   (`git checkout -b token-auth`) and ask.
3. **The mobile app is your repo.** Push to it freely.
4. **Don't retrain the models** unless Salem asks. They're the graded part.
5. **Commit at the end of every session.** Even if it's half done. Message:
   "WIP: task list screen".

---

## 12. Where to look when stuck

| Question | Answer is in |
|---|---|
| What does endpoint X return? | Section 5, or `app.py` — the code is short |
| How does the web do Y? | `SATM.html` — search for the button text, follow the `onclick` |
| Why these three models? | `SATM_Study_Guide.pdf` |
| What's the priority formula? | Section 6.2 |
| Expo question | https://docs.expo.dev — genuinely good docs |
| Git question | Section 4.4, then Salem |
| Anything else | Salem |

Good luck. The hard part — the AI, the backend, the deploy — is done. You're
building the nice front door.
