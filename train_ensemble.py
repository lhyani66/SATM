"""
Trains two additional models (SVM + Gradient Boosting) for EACH prediction:
  - Category   : LinearSVC + GradientBoostingClassifier
  - Importance : LinearSVC + GradientBoostingClassifier
  - Time        : LinearSVR + GradientBoostingRegressor

All six new models are saved alongside the existing RF models.
Run once: py train_ensemble.py
"""
import os, re, string
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

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
category_enc    = joblib.load(os.path.join(MODEL_DIR, 'category_encoder.pkl'))

# ── Shared split (random_state=42, stratify=Category) ─────────────────────────
idx_tr, idx_te = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=df['Category']
)
train_df = df.loc[idx_tr]
test_df  = df.loc[idx_te]
print(f"Train: {len(train_df)}  Test: {len(test_df)}\n")

# ── Pre-compute TF-IDF for all rows ───────────────────────────────────────────
X_tfidf_tr = tfidf.transform(train_df['processed'])
X_tfidf_te = tfidf.transform(test_df['processed'])

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CATEGORY  (features = TF-IDF only)
# ═══════════════════════════════════════════════════════════════════════════════
y_cat_tr = train_df['Category']
y_cat_te = test_df['Category']

print("=" * 60)
print("CATEGORY MODELS")
print("=" * 60)

print("\nTraining Category SVM (LinearSVC)...")
cat_svm = LinearSVC(random_state=42, max_iter=3000)
cat_svm.fit(X_tfidf_tr, y_cat_tr)
cat_svm_acc = accuracy_score(y_cat_te, cat_svm.predict(X_tfidf_te))
print(f"  Accuracy: {cat_svm_acc:.4f}")

print("\nTraining Category Gradient Boosting...")
cat_gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
cat_gb.fit(X_tfidf_tr.toarray(), y_cat_tr)
cat_gb_acc = accuracy_score(y_cat_te, cat_gb.predict(X_tfidf_te.toarray()))
print(f"  Accuracy: {cat_gb_acc:.4f}")

cat_rf_acc = accuracy_score(y_cat_te,
    joblib.load(os.path.join(MODEL_DIR, 'category_model.pkl')).predict(X_tfidf_te))
print(f"\n  Random Forest (existing): {cat_rf_acc:.4f}")
print(f"  SVM                     : {cat_svm_acc:.4f}")
print(f"  Gradient Boosting       : {cat_gb_acc:.4f}")

joblib.dump(cat_svm, os.path.join(MODEL_DIR, 'cat_model_svm.pkl'))
joblib.dump(cat_gb,  os.path.join(MODEL_DIR, 'cat_model_gb.pkl'))
print("  Saved: cat_model_svm.pkl, cat_model_gb.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. IMPORTANCE  (features = TF-IDF + scaled deadline)
# ═══════════════════════════════════════════════════════════════════════════════
dl_tr = deadline_scaler.transform(train_df[['Deadline (days)']].values)
dl_te = deadline_scaler.transform(test_df[['Deadline (days)']].values)
X_imp_tr = hstack([X_tfidf_tr, csr_matrix(dl_tr)])
X_imp_te = hstack([X_tfidf_te, csr_matrix(dl_te)])
y_imp_tr  = train_df['Importance']
y_imp_te  = test_df['Importance']

print("\n" + "=" * 60)
print("IMPORTANCE MODELS")
print("=" * 60)

print("\nTraining Importance SVM (LinearSVC)...")
imp_svm = LinearSVC(random_state=42, max_iter=3000)
imp_svm.fit(X_imp_tr, y_imp_tr)
imp_svm_acc = accuracy_score(y_imp_te, imp_svm.predict(X_imp_te))
print(f"  Accuracy: {imp_svm_acc:.4f}")

print("\nTraining Importance Gradient Boosting...")
imp_gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
imp_gb.fit(X_imp_tr.toarray(), y_imp_tr)
imp_gb_acc = accuracy_score(y_imp_te, imp_gb.predict(X_imp_te.toarray()))
print(f"  Accuracy: {imp_gb_acc:.4f}")

imp_rf_acc = accuracy_score(y_imp_te,
    joblib.load(os.path.join(MODEL_DIR, 'importance_model.pkl')).predict(X_imp_te))
print(f"\n  Random Forest (existing): {imp_rf_acc:.4f}")
print(f"  SVM                     : {imp_svm_acc:.4f}")
print(f"  Gradient Boosting       : {imp_gb_acc:.4f}")

joblib.dump(imp_svm, os.path.join(MODEL_DIR, 'imp_model_svm.pkl'))
joblib.dump(imp_gb,  os.path.join(MODEL_DIR, 'imp_model_gb.pkl'))
print("  Saved: imp_model_svm.pkl, imp_model_gb.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TIME  (features = TF-IDF + encoded category, regression)
# ═══════════════════════════════════════════════════════════════════════════════
cat_enc_tr = category_enc.transform(train_df['Category']).reshape(-1, 1)
cat_enc_te = category_enc.transform(test_df['Category']).reshape(-1, 1)
X_time_tr  = hstack([X_tfidf_tr, csr_matrix(cat_enc_tr)])
X_time_te  = hstack([X_tfidf_te, csr_matrix(cat_enc_te)])
y_time_tr  = train_df['Time Taken (hours)']
y_time_te  = test_df['Time Taken (hours)']

print("\n" + "=" * 60)
print("TIME ESTIMATION MODELS")
print("=" * 60)

print("\nTraining Time SVR (LinearSVR)...")
time_svr = LinearSVR(random_state=42, max_iter=5000)
time_svr.fit(X_time_tr, y_time_tr)
svr_preds   = np.clip(time_svr.predict(X_time_te), 0.5, 8.0)
svr_mae     = mean_absolute_error(y_time_te, svr_preds)
svr_r2      = r2_score(y_time_te, svr_preds)
print(f"  MAE: {svr_mae:.4f}  R²: {svr_r2:.4f}")

print("\nTraining Time Gradient Boosting (150 estimators)...")
time_gb = GradientBoostingRegressor(n_estimators=150, random_state=42)
time_gb.fit(X_time_tr.toarray(), y_time_tr)
gb_preds    = np.clip(time_gb.predict(X_time_te.toarray()), 0.5, 8.0)
gb_mae      = mean_absolute_error(y_time_te, gb_preds)
gb_r2       = r2_score(y_time_te, gb_preds)
print(f"  MAE: {gb_mae:.4f}  R²: {gb_r2:.4f}")

rf_time     = joblib.load(os.path.join(MODEL_DIR, 'time_model.pkl'))
rf_preds_t  = np.clip(rf_time.predict(X_time_te), 0.5, 8.0)
rf_mae      = mean_absolute_error(y_time_te, rf_preds_t)
rf_r2       = r2_score(y_time_te, rf_preds_t)
print(f"\n  Random Forest (existing): MAE={rf_mae:.4f}  R²={rf_r2:.4f}")
print(f"  SVR                     : MAE={svr_mae:.4f}  R²={svr_r2:.4f}")
print(f"  Gradient Boosting       : MAE={gb_mae:.4f}  R²={gb_r2:.4f}")

joblib.dump(time_svr, os.path.join(MODEL_DIR, 'time_model_svr.pkl'))
joblib.dump(time_gb,  os.path.join(MODEL_DIR, 'time_model_gb.pkl'))
print("  Saved: time_model_svr.pkl, time_model_gb.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ENSEMBLE SUMMARY")
print("=" * 60)
print(f"\nCategory   RF={cat_rf_acc:.4f}  SVM={cat_svm_acc:.4f}  GB={cat_gb_acc:.4f}")
print(f"Importance RF={imp_rf_acc:.4f}  SVM={imp_svm_acc:.4f}  GB={imp_gb_acc:.4f}")
print(f"Time (MAE) RF={rf_mae:.4f}  SVR={svr_mae:.4f}  GB={gb_mae:.4f}")
print("\nAll models saved to models/")
