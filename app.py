import os
import re
import secrets
import string
from datetime import datetime, timedelta, timezone
from functools import wraps

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from scipy.sparse import csr_matrix, hstack
from werkzeug.security import check_password_hash, generate_password_hash

import mailer

import nltk
for _resource, _path in [('stopwords', 'corpora/stopwords'),
                          ('wordnet',   'corpora/wordnet'),
                          ('punkt_tab', 'tokenizers/punkt_tab')]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
# Render hands out postgres:// URLs, but SQLAlchemy 2.x only accepts postgresql://
_db_url = os.environ.get('DATABASE_URL', 'sqlite:///satm.db')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Managed Postgres drops idle connections; recycle before it bites.
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 300}

# Sessions are signed with this key. A known key means forgeable logins, so
# production must supply a real one; local SQLite dev keeps working unchanged.
_secret = os.environ.get('SECRET_KEY')
if not _secret:
    if _db_url.startswith('postgresql://'):
        raise RuntimeError('SECRET_KEY must be set when running against PostgreSQL')
    _secret = 'satm-dev-secret-change-before-deploy'
app.secret_key = _secret

db = SQLAlchemy(app)


def _utcnow():
    """UTC with the tzinfo stripped.

    SQLite hands datetimes back naive whatever went in, so storing an aware one
    and comparing it to `datetime.now(timezone.utc)` raises TypeError on local
    dev while working on Postgres. Storing naive UTC everywhere behaves the same
    on both.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ── Database tables ───────────────────────────────────────────────────────────
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    tasks    = db.relationship('Task', backref='user', lazy=True)

# Reset codes are hashed like passwords: a leaked database should not hand
# anyone a working code. One live row per user — asking for a code throws away
# the previous one.
class PasswordReset(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    code_hash  = db.Column(db.String(200), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    attempts   = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: _utcnow())

class Task(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    task_text  = db.Column(db.String(500), nullable=False)
    category   = db.Column(db.String(50))
    importance = db.Column(db.String(20))
    deadline   = db.Column(db.Integer)
    time_est   = db.Column(db.Float)
    status     = db.Column(db.String(20), default='todo')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# Runs under any WSGI server (gunicorn, flask dev, etc.)
with app.app_context():
    db.create_all()

# ── Load ML models ────────────────────────────────────────────────────────────
MODEL_DIR       = os.path.join(os.path.dirname(__file__), 'models')
tfidf           = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
deadline_scaler = joblib.load(os.path.join(MODEL_DIR, 'deadline_scaler.pkl'))
category_enc    = joblib.load(os.path.join(MODEL_DIR, 'category_encoder.pkl'))

# Category ensemble (classification — TF-IDF only)
cat_model_rf    = joblib.load(os.path.join(MODEL_DIR, 'category_model.pkl'))
cat_model_svm   = joblib.load(os.path.join(MODEL_DIR, 'cat_model_svm.pkl'))
cat_model_gb    = joblib.load(os.path.join(MODEL_DIR, 'cat_model_gb.pkl'))

# Importance ensemble (classification — TF-IDF + deadline)
imp_model_rf    = joblib.load(os.path.join(MODEL_DIR, 'importance_model.pkl'))
imp_model_svm   = joblib.load(os.path.join(MODEL_DIR, 'imp_model_svm.pkl'))
imp_model_gb    = joblib.load(os.path.join(MODEL_DIR, 'imp_model_gb.pkl'))

# Time ensemble (regression — TF-IDF + encoded category)
time_model_rf   = joblib.load(os.path.join(MODEL_DIR, 'time_model.pkl'))
time_model_svr  = joblib.load(os.path.join(MODEL_DIR, 'time_model_svr.pkl'))
time_model_gb   = joblib.load(os.path.join(MODEL_DIR, 'time_model_gb.pkl'))

# ── Preprocessing ─────────────────────────────────────────────────────────────
_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()
_ABBREVS    = {'hw': 'homework', 'ppt': 'presentation', 'bio': 'biology', 'idk': 'unknown'}

def _clean(text):
    text = str(text).lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
                  r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    for abbr, full in _ABBREVS.items():
        text = text.replace(abbr, full)
    return text

def _preprocess(text):
    tokens = word_tokenize(_clean(text))
    return ' '.join(
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words
    )

def _majority_vote(votes):
    counts  = {v: votes.count(v) for v in set(votes)}
    winner  = max(counts, key=counts.get)
    return winner, counts[winner]   # (label, agreement count out of 3)

def run_prediction(task_text, deadline):
    vec = tfidf.transform([_preprocess(task_text)])

    # Category ensemble (majority vote)
    cat_votes  = [
        cat_model_rf.predict(vec)[0],
        cat_model_svm.predict(vec)[0],
        cat_model_gb.predict(vec.toarray())[0],
    ]
    category, cat_agree = _majority_vote(cat_votes)

    # Importance ensemble (majority vote)
    dl_scaled = deadline_scaler.transform([[deadline]])
    vec_imp   = hstack([vec, csr_matrix(dl_scaled)])
    imp_votes = [
        imp_model_rf.predict(vec_imp)[0].lower(),
        imp_model_svm.predict(vec_imp)[0].lower(),
        imp_model_gb.predict(vec_imp.toarray())[0].lower(),
    ]
    importance, imp_agree = _majority_vote(imp_votes)

    # Time ensemble (average of three regressors, then clamp)
    cat_enc   = category_enc.transform([category]).reshape(-1, 1)
    vec_time  = hstack([vec, csr_matrix(cat_enc)])
    time_preds = [
        float(time_model_rf.predict(vec_time)[0]),
        float(time_model_svr.predict(vec_time)[0]),
        float(time_model_gb.predict(vec_time.toarray())[0]),
    ]
    time_est = round(max(0.5, min(8.0, sum(time_preds) / 3)), 1)
    # Spread between highest and lowest estimate (rounded to 1 dp)
    time_spread = round(max(time_preds) - min(time_preds), 1)

    return {
        'category':    category,
        'cat_agree':   cat_agree,
        'importance':  importance,
        'imp_agree':   imp_agree,
        'time_est':    time_est,
        'time_spread': time_spread,
    }

# ── Auth helpers ──────────────────────────────────────────────────────────────
def current_user_id():
    return session.get('user_id')

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user_id():
            return jsonify({'error': 'Not logged in'}), 401
        return f(*args, **kwargs)
    return wrapper

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    data     = request.get_json()
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'An account with this email already exists'}), 409
    user = User(email=email, password=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    session['user_id'] = user.id
    return jsonify({'email': user.email}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data     = request.get_json()
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    user     = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid email or password'}), 401
    session['user_id'] = user.id
    return jsonify({'email': user.email}), 200

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'}), 200

# ── Password reset ────────────────────────────────────────────────────────────
# A six-digit code emailed to the address on the account, typed back into the
# app. No reset link, deliberately: a link has to reopen this specific app,
# which needs Apple universal-link setup and fails quietly when it goes wrong.
RESET_CODE_TTL_MIN   = 15
RESET_MAX_ATTEMPTS   = 5
RESET_RESEND_SECONDS = 60

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    data  = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # The same answer whether or not the address has an account. Any difference
    # — wording, status, timing — turns this into a way to ask the server which
    # email addresses are registered.
    sent = jsonify({'message': 'If that email has an account, a reset code is on its way.'}), 200

    user = User.query.filter_by(email=email).first()
    if not user:
        return sent

    # No relay configured in production would mean promising an email that
    # cannot arrive, so say it is unavailable instead of lying.
    if not mailer.is_configured() and _db_url.startswith('postgresql://'):
        return jsonify({'error': 'Password reset is unavailable right now.'}), 503

    # One code a minute per account: enough to recover from a mistyped address,
    # not enough to use this endpoint to post mail at someone.
    recent = (PasswordReset.query
              .filter_by(user_id=user.id)
              .order_by(PasswordReset.created_at.desc())
              .first())
    if recent and (_utcnow() - recent.created_at).total_seconds() < RESET_RESEND_SECONDS:
        return sent

    PasswordReset.query.filter_by(user_id=user.id).delete()
    code = f'{secrets.randbelow(1_000_000):06d}'
    db.session.add(PasswordReset(
        user_id=user.id,
        code_hash=generate_password_hash(code),
        expires_at=_utcnow() + timedelta(minutes=RESET_CODE_TTL_MIN),
    ))
    db.session.commit()

    if mailer.is_configured():
        try:
            mailer.send_reset_code(user.email, code, RESET_CODE_TTL_MIN)
        except Exception:
            app.logger.exception('Reset code for %s could not be sent', user.email)
            return jsonify({'error': 'The email could not be sent. Try again shortly.'}), 502
    else:
        # Local development with no relay: the log is the inbox, so the flow is
        # testable end to end without signing up anywhere. Only ever reached on
        # SQLite — the guard above stops this path in production.
        app.logger.warning('SMTP not configured; reset code for %s is %s', user.email, code)

    return sent

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    data     = request.get_json(silent=True) or {}
    email    = (data.get('email') or '').strip().lower()
    code     = (data.get('code') or '').strip()
    password = data.get('password') or ''
    if not email or not code or not password:
        return jsonify({'error': 'Email, code and new password are required'}), 400

    # One message for every way of being wrong — unknown address, no code
    # outstanding, expired, mistyped — so none of them can be told apart.
    invalid = jsonify({'error': 'That code is wrong or has expired'}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return invalid

    row = (PasswordReset.query
           .filter_by(user_id=user.id)
           .order_by(PasswordReset.created_at.desc())
           .first())
    if not row:
        return invalid

    if row.expires_at < _utcnow():
        db.session.delete(row)
        db.session.commit()
        return invalid

    # Five guesses, then the code is spent: a six-digit code is only strong
    # while it cannot be tried a million times.
    if row.attempts >= RESET_MAX_ATTEMPTS:
        db.session.delete(row)
        db.session.commit()
        return jsonify({'error': 'Too many wrong codes. Ask for a new one.'}), 429

    if not check_password_hash(row.code_hash, code):
        row.attempts += 1
        db.session.commit()
        return invalid

    user.password = generate_password_hash(password)
    PasswordReset.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    # Nothing is signed in here — the user proved the code, not the old
    # password — so end this request's session and let them sign in fresh.
    session.clear()
    return jsonify({'message': 'Password updated'}), 200

@app.route('/api/me', methods=['GET'])
@require_auth
def me():
    user = db.session.get(User, current_user_id())
    return jsonify({'email': user.email}), 200

# ── Predict route ─────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
@require_auth
def predict():
    data     = request.get_json()
    text     = (data.get('text') or '').strip()
    deadline = int(data.get('deadline') or 3)
    if not text:
        return jsonify({'error': 'Task text is required'}), 400
    return jsonify(run_prediction(text, deadline)), 200

# ── Task routes ───────────────────────────────────────────────────────────────
@app.route('/api/tasks', methods=['GET'])
@require_auth
def get_tasks():
    tasks = Task.query.filter_by(user_id=current_user_id()).order_by(Task.created_at.desc()).all()
    return jsonify([{
        'id':         t.id,
        'task_text':  t.task_text,
        'category':   t.category,
        'importance': t.importance,
        'deadline':   t.deadline,
        'time_est':   t.time_est,
        'status':     t.status,
    } for t in tasks]), 200

@app.route('/api/tasks', methods=['POST'])
@require_auth
def add_task():
    data      = request.get_json()
    task_text = (data.get('task_text') or '').strip()
    if not task_text:
        return jsonify({'error': 'Task text is required'}), 400
    task = Task(
        user_id    = current_user_id(),
        task_text  = task_text,
        category   = data.get('category'),
        importance = data.get('importance'),
        deadline   = data.get('deadline'),
        time_est   = data.get('time_est'),
    )
    db.session.add(task)
    db.session.commit()
    return jsonify({'id': task.id}), 201

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@require_auth
def delete_task(task_id):
    task = Task.query.filter_by(id=task_id, user_id=current_user_id()).first()
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    db.session.delete(task)
    db.session.commit()
    return jsonify({'message': 'Deleted'}), 200

@app.route('/api/tasks/<int:task_id>', methods=['PATCH'])
@require_auth
def update_task(task_id):
    task = Task.query.filter_by(id=task_id, user_id=current_user_id()).first()
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    data = request.get_json()
    for field in ('status', 'category', 'importance', 'deadline', 'time_est'):
        if field in data:
            setattr(task, field, data[field])
    db.session.commit()
    return jsonify({'message': 'Updated'}), 200

# ── Serve the frontend ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'SATM.html')

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
