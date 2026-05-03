"""
Trains two additional importance models (SVM + Gradient Boosting) to complement
the existing Random Forest. All three are saved and used together in an ensemble.
Run once: py train_ensemble.py
"""
import os, re, string
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import nltk
for resource, path in [('stopwords','corpora/stopwords'),
                        ('wordnet','corpora/wordnet'),
                        ('punkt_tab','tokenizers/punkt_tab')]:
    try: nltk.data.find(path)
    except LookupError: nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()
_ABBREVS    = {'hw':'homework','ppt':'presentation','bio':'biology','idk':'unknown'}

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
    return ' '.join(_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('satm_dataset_1000_rows.csv')
df['Importance'] = df['Importance'].str.title()
df['Category']   = df['Category'].str.title()
df['processed']  = df['Task Text'].apply(_preprocess)

# ── Load existing fitted transformers ─────────────────────────────────────────
MODEL_DIR       = 'models'
tfidf           = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
deadline_scaler = joblib.load(os.path.join(MODEL_DIR, 'deadline_scaler.pkl'))

# ── Build features (identical to app.py pipeline) ────────────────────────────
X_tfidf   = tfidf.transform(df['processed'])
dl_scaled = deadline_scaler.transform(df[['Deadline (days)']].values)
X = hstack([X_tfidf, csr_matrix(dl_scaled)])
y = df['Importance']

# ── Same split as notebook (random_state=42, stratify=Category) ───────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Category']
)

print(f"Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}\n")

# ── SVM ───────────────────────────────────────────────────────────────────────
print("Training SVM (LinearSVC)...")
svm = LinearSVC(random_state=42, max_iter=3000)
svm.fit(X_tr, y_tr)
svm_preds = svm.predict(X_te)
svm_acc   = accuracy_score(y_te, svm_preds)
print(f"  Accuracy: {svm_acc:.4f}")
print(classification_report(y_te, svm_preds))

# ── Gradient Boosting ─────────────────────────────────────────────────────────
print("Training Gradient Boosting (150 estimators)...")
gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
gb.fit(X_tr.toarray(), y_tr)
gb_preds = gb.predict(X_te.toarray())
gb_acc   = accuracy_score(y_te, gb_preds)
print(f"  Accuracy: {gb_acc:.4f}")
print(classification_report(y_te, gb_preds))

# ── Evaluate existing RF for comparison ──────────────────────────────────────
rf = joblib.load(os.path.join(MODEL_DIR, 'importance_model.pkl'))
rf_preds = rf.predict(X_te)
rf_acc   = accuracy_score(y_te, rf_preds)
print(f"Random Forest accuracy (existing): {rf_acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(svm, os.path.join(MODEL_DIR, 'imp_model_svm.pkl'))
joblib.dump(gb,  os.path.join(MODEL_DIR, 'imp_model_gb.pkl'))

print("\n=== ENSEMBLE SUMMARY ===")
print(f"  Random Forest    : {rf_acc:.4f}")
print(f"  SVM              : {svm_acc:.4f}")
print(f"  Gradient Boosting: {gb_acc:.4f}")
print("\nSaved: imp_model_svm.pkl, imp_model_gb.pkl")
