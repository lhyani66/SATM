# SATM — Smart Academic Task Manager

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-27ae60)

**Live demo: [satm.onrender.com](https://satm.onrender.com)**

A web application that uses machine learning to help students manage their academic workload. Describe a task in plain language — the AI classifies it, estimates importance and time, detects the deadline from your words, and builds a prioritized daily schedule automatically.

---

## Features

- **Natural language input** — write tasks the way you think: *"finish bio hw ASAP, exam is tomorrow"*
- **Three-stage ML pipeline** — predicts category, importance, and time estimate independently
- **Deadline detection** — extracts deadlines from text automatically (*"in 3 days"*, *"next week"*, *"ASAP"*)
- **Priority scoring** — ranks all tasks by urgency and importance with a visual bar
- **Auto-generated schedule** — builds a daily study plan with 45-min sessions and automatic breaks
- **Password-validated auth** — register with email + strong password, sessions persist across visits
- **Clean dark UI** — minimal, distraction-free interface

---

## ML Pipeline

Each task runs through three sequential models:

```
Raw text
   │
   ▼
TF-IDF vectorizer  ──►  Category model     →  "Homework" / "Exam" / "Quiz" / "Project"
   │
   ├── + deadline  ──►  Importance model   →  high / medium / low
   │
   └── + category  ──►  Time model         →  estimated hours (clamped 0.5–8)
```

All models were trained on a labelled dataset of 1,000 academic tasks. Preprocessing mirrors the training notebook exactly: lowercasing, punctuation removal, emoji stripping, abbreviation expansion (`hw → homework`, etc.), stopword removal, and lemmatization.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML / CSS / JavaScript |
| Backend | Python · Flask · Flask-SQLAlchemy |
| Database | SQLite (local) |
| ML | scikit-learn · TF-IDF · Random Forest |
| NLP | NLTK — tokenization, lemmatization, stopwords |
| Auth | Werkzeug password hashing · Flask sessions |
| Deployment | Render · gunicorn |

---

## Project Structure

```
SATM/
├── app.py                   # Flask backend — routes, ML pipeline, auth
├── SATM.html                # Single-page frontend
├── SATM.ipynb               # Model training notebook
├── requirements.txt         # Python dependencies
├── Procfile                 # gunicorn start command
├── render.yaml              # One-click Render deployment config
├── .env.example             # Environment variable template
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── category_model.pkl
│   ├── importance_model.pkl
│   ├── time_model.pkl
│   ├── deadline_scaler.pkl
│   └── category_encoder.pkl
└── instance/
    └── satm.db              # SQLite database (auto-created on first run)
```

---

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/lhyani66/SATM.git
cd SATM
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the server**
```bash
python app.py
```

**4. Open in your browser**
```
http://127.0.0.1:5000
```

The database is created automatically on first run. No setup needed.

---

## Environment Variables

Copy `.env.example` to `.env` for local overrides:

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | dev fallback | Flask session signing key — **must be set in production** |
| `DATABASE_URL` | `sqlite:///satm.db` | Database connection string |

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/register` | — | Create account |
| `POST` | `/api/login` | — | Sign in |
| `POST` | `/api/logout` | — | Sign out |
| `GET` | `/api/me` | ✓ | Get current user |
| `POST` | `/api/predict` | ✓ | Run ML prediction on a task |
| `GET` | `/api/tasks` | ✓ | List all tasks |
| `POST` | `/api/tasks` | ✓ | Save a task |
| `DELETE` | `/api/tasks/:id` | ✓ | Delete a task |
| `PATCH` | `/api/tasks/:id` | ✓ | Update task fields |

**Predict request body:**
```json
{ "text": "finish the ML homework by tonight", "deadline": 1 }
```

**Predict response:**
```json
{ "category": "Homework", "importance": "high", "time_est": 3.2 }
```

---

## Deployment

The app is configured for [Render](https://render.com) with `render.yaml` for one-click deployment:

1. Push this repository to GitHub
2. Go to Render → **New Web Service** → connect your repo
3. Render auto-detects `render.yaml` and configures everything
4. Set `SECRET_KEY` in Render's environment variables dashboard

`gunicorn` is used as the production WSGI server. The database tables are created automatically on startup.

---

## License

[MIT](LICENSE) — graduation project, Computer Science
