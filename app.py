import os
import re
import string
from datetime import datetime, timezone
from functools import wraps

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from scipy.sparse import csr_matrix, hstack
from werkzeug.security import check_password_hash, generate_password_hash

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
app.secret_key = os.environ.get('SECRET_KEY', 'satm-dev-secret-change-before-deploy')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///satm.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ── Database tables ───────────────────────────────────────────────────────────
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    tasks    = db.relationship('Task', backref='user', lazy=True)

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
    app.run(debug=True)
