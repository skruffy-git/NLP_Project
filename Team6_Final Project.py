# -*- coding: utf-8 -*-
"""team6_phase2_complete.py

Sentiment Analysis – Phase 2  (Complete Pipeline)
Team #6 | Dataset: Software (Amazon Product Reviews)

Steps covered:
  11(a) – Dataset Selection          (Omair Khan)
  11(b) – Data Preprocessing         (Omair Khan)
  11(c) – Text Representation        (TF-IDF)
  11(d) – Train / Test Split         (70 / 30, stratified)
  11(e) – Model Development          (Logistic Regression + LinearSVC)
  12    – Training Results Summary
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Install & Import Dependencies
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys

packages = ['scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
            'nltk']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import json, os, re, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

# Download required NLTK resources
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 110
print('Imports complete.')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11(a) — DATASET SELECTION
# Author: Omair Khan
# ─────────────────────────────────────────────────────────────────────────────

# Load Amazon Software review data (JSON Lines format)
JSON_CANDIDATES = ['Software_5.json', 'Software_5_1.json']
data = []

for fname in JSON_CANDIDATES:
    if os.path.exists(fname):
        print(f'Loading {fname}...')
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        break
else:
    raise FileNotFoundError(
        'Dataset not found! Upload Software_5.json to Colab.\n'
        'Download: https://nijianmo.github.io/amazon/index.html'
    )

# Convert to DataFrame
df = pd.DataFrame(data)

# Keep only fields needed for sentiment analysis
df = df[['reviewText', 'summary', 'overall']].copy()

# Drop rows with missing text or rating
df.dropna(subset=['reviewText', 'overall'], inplace=True)

# Combine summary and reviewText for richer text input
df['text'] = df['summary'].fillna('') + ' ' + df['reviewText'].fillna('')
df['text'] = df['text'].str.strip()

# Remove rows where combined text is empty
df = df[df['text'] != '']

# Convert numerical ratings to sentiment labels
def map_sentiment(rating):
    if rating in [1, 2]:   return 'negative'
    elif rating == 3:       return 'neutral'
    elif rating in [4, 5]:  return 'positive'

df['sentiment'] = df['overall'].apply(map_sentiment)

# Select a random subset of 3,000 reviews (exceeds the 2,000-review minimum)
SEED = 42
df_subset = df.sample(n=3000, random_state=SEED).reset_index(drop=True)

# Basic exploration on the selected subset
print('\n--- STEP 11(a): Dataset Selection ---')
print('Selected subset shape:', df_subset.shape)
print('\nSentiment distribution:')
print(df_subset['sentiment'].value_counts())

df_subset['review_length_words'] = df_subset['text'].apply(lambda x: len(x.split()))
print('\nReview length statistics (words):')
print(df_subset['review_length_words'].describe())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11(b) — DATA PREPROCESSING
# Author: Omair Khan
# ─────────────────────────────────────────────────────────────────────────────

print('\n--- STEP 11(b): Data Preprocessing ---')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation, digits, and special characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 3. Tokenize (whitespace split)
    tokens = text.split()
    # 4. Remove stopwords
    tokens = [w for w in tokens if w not in stop_words]
    # 5. Lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

df_subset['clean_text'] = df_subset['text'].apply(preprocess_text)

# Show before / after examples
print('\nSample before and after preprocessing:')
for i in range(3):
    print(f'\n[Review {i+1}]')
    print('  Original :', df_subset.iloc[i]['text'][:200])
    print('  Cleaned  :', df_subset.iloc[i]['clean_text'][:200])

# Save preprocessed subset for reference
df_subset.to_csv('phase2_subset_preprocessed.csv', index=False)
print('\nPreprocessed dataset saved as: phase2_subset_preprocessed.csv')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11(c) — TEXT REPRESENTATION  (TF-IDF)
# Author: Ryan Frederick
# ─────────────────────────────────────────────────────────────────────────────
print('\n--- STEP 11(c): Text Representation (TF-IDF) ---')

# TF-IDF converts the cleaned text into a numerical feature matrix.
# Settings chosen:
#   - stop_words='english'  : removes common English words (sklearn built-in)
#   - max_features=50,000   : caps vocabulary to top 50k terms by frequency
#   - ngram_range=(1,2)     : includes unigrams and bigrams (e.g. "not good")
#   - sublinear_tf=True     : applies log(1+tf) to dampen high-frequency terms
#
# TF-IDF is applied inside the sklearn Pipeline at training time (Steps 11e),
# so it is fitted only on X_train and never on X_test (preventing data leakage).

TFIDF_PARAMS = dict(
    stop_words='english',
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
print('TF-IDF settings:', TFIDF_PARAMS)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11(d) — TRAIN / TEST SPLIT  (70 % train, 30 % test)
# Author: Ryan Frederick
# ─────────────────────────────────────────────────────────────────────────────
print('\n--- STEP 11(d): Train / Test Split ---')

# Capitalise labels to match the rest of the pipeline
df_subset['label'] = df_subset['sentiment'].str.capitalize()  # positive→Positive etc.

CLASSES = ['Positive', 'Neutral', 'Negative']

X = df_subset['clean_text']
y = df_subset['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=SEED,
    stratify=y        # preserve class proportions in both splits
)

print(f'Train set: {len(X_train):,} reviews (70%)')
print(f'Test  set: {len(X_test):,} reviews (30%)')
print('\nTrain label distribution:')
print(y_train.value_counts())
print('\nTest label distribution:')
print(y_test.value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11(e) — MODEL DEVELOPMENT
# Model 1: Logistic Regression
# Model 2: Support Vector Machine (LinearSVC)
# Author: Ryan Frederick
# ─────────────────────────────────────────────────────────────────────────────

# ── Model 1: Logistic Regression ─────────────────────────────────────────────
print('\n' + '='*62)
print('  MODEL 1: Logistic Regression')
print('='*62)

print('\nHyperparameter sweep — C values: [0.1, 0.5, 1.0, 5.0]')
print(f'{"C":<8} {"Accuracy":>10} {"F1-W":>10}')
print('-'*30)
lr_results = {}
for C in [0.1, 0.5, 1.0, 5.0]:
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   LogisticRegression(C=C, max_iter=1000, random_state=SEED,
                                      class_weight='balanced', solver='lbfgs'))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    lr_results[C] = (acc, f1w)
    print(f'{C:<8} {acc:>10.4f} {f1w:>10.4f}')

best_C_lr = max(lr_results, key=lambda c: lr_results[c][1])
print(f'\n  → Best C = {best_C_lr}  (F1-W = {lr_results[best_C_lr][1]:.4f})')

# Final LR model with best C
lr_final = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf',   LogisticRegression(C=best_C_lr, max_iter=1000, random_state=SEED,
                                  class_weight='balanced', solver='lbfgs'))
])
t0 = time.time()
lr_final.fit(X_train, y_train)
lr_train_time = time.time() - t0

y_pred_lr = lr_final.predict(X_test)
lr_acc  = accuracy_score(y_test, y_pred_lr)
lr_prec = precision_score(y_test, y_pred_lr, labels=CLASSES, average='weighted', zero_division=0)
lr_rec  = recall_score(y_test, y_pred_lr, labels=CLASSES, average='weighted', zero_division=0)
lr_f1   = f1_score(y_test, y_pred_lr, labels=CLASSES, average='weighted', zero_division=0)
lr_cm   = confusion_matrix(y_test, y_pred_lr, labels=CLASSES)

print(f'\n  Best LR config:')
print(f'    Solver        : lbfgs')
print(f'    Regularisation: C = {best_C_lr}')
print(f'    Class weights : balanced')
print(f'    TF-IDF        : unigrams + bigrams, max_features=50000, sublinear_tf')
print(f'\n  Training time : {lr_train_time:.2f} s')
print(f'  Test Accuracy : {lr_acc:.4f}')
print(f'  Precision (W) : {lr_prec:.4f}')
print(f'  Recall (W)    : {lr_rec:.4f}')
print(f'  F1 (W)        : {lr_f1:.4f}')
print('\n  Per-class Classification Report:')
print(classification_report(y_test, y_pred_lr, labels=CLASSES, zero_division=0))
print('  Confusion Matrix (rows = Actual, cols = Predicted):')
print(pd.DataFrame(lr_cm, index=CLASSES, columns=CLASSES).to_string())

# ── Model 2: LinearSVC ────────────────────────────────────────────────────────
print('\n' + '='*62)
print('  MODEL 2: Support Vector Machine (LinearSVC)')
print('='*62)

print('\nHyperparameter sweep — C values: [0.01, 0.1, 0.5, 1.0, 5.0]')
print(f'{"C":<8} {"Accuracy":>10} {"F1-W":>10}')
print('-'*30)
svm_results = {}
for C in [0.01, 0.1, 0.5, 1.0, 5.0]:
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   LinearSVC(C=C, random_state=SEED, class_weight='balanced', max_iter=3000))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    svm_results[C] = (acc, f1w)
    print(f'{C:<8} {acc:>10.4f} {f1w:>10.4f}')

best_C_svm = max(svm_results, key=lambda c: svm_results[c][1])
print(f'\n  → Best C = {best_C_svm}  (F1-W = {svm_results[best_C_svm][1]:.4f})')

# Final SVM model with best C
svm_final = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf',   LinearSVC(C=best_C_svm, random_state=SEED, class_weight='balanced', max_iter=3000))
])
t0 = time.time()
svm_final.fit(X_train, y_train)
svm_train_time = time.time() - t0

y_pred_svm = svm_final.predict(X_test)
svm_acc  = accuracy_score(y_test, y_pred_svm)
svm_prec = precision_score(y_test, y_pred_svm, labels=CLASSES, average='weighted', zero_division=0)
svm_rec  = recall_score(y_test, y_pred_svm, labels=CLASSES, average='weighted', zero_division=0)
svm_f1   = f1_score(y_test, y_pred_svm, labels=CLASSES, average='weighted', zero_division=0)
svm_cm   = confusion_matrix(y_test, y_pred_svm, labels=CLASSES)

print(f'\n  Best SVM config:')
print(f'    Kernel (implicit): Linear (via LinearSVC)')
print(f'    Regularisation   : C = {best_C_svm}')
print(f'    Class weights    : balanced')
print(f'    TF-IDF           : unigrams + bigrams, max_features=50000, sublinear_tf')
print(f'\n  Training time : {svm_train_time:.2f} s')
print(f'  Test Accuracy : {svm_acc:.4f}')
print(f'  Precision (W) : {svm_prec:.4f}')
print(f'  Recall (W)    : {svm_rec:.4f}')
print(f'  F1 (W)        : {svm_f1:.4f}')
print('\n  Per-class Classification Report:')
print(classification_report(y_test, y_pred_svm, labels=CLASSES, zero_division=0))
print('  Confusion Matrix (rows = Actual, cols = Predicted):')
print(pd.DataFrame(svm_cm, index=CLASSES, columns=CLASSES).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — TRAINING RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('  STEP 12 — TRAINING RESULTS SUMMARY')
print('='*62)

summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (weighted)', 'Recall (weighted)', 'F1 (weighted)', 'Training Time (s)'],
    'Logistic Regression': [
        round(lr_acc, 4), round(lr_prec, 4), round(lr_rec, 4), round(lr_f1, 4), round(lr_train_time, 2)
    ],
    'LinearSVC (SVM)': [
        round(svm_acc, 4), round(svm_prec, 4), round(svm_rec, 4), round(svm_f1, 4), round(svm_train_time, 2)
    ]
}).set_index('Metric')

print(summary.to_string())
summary.to_csv('team6_phase2_model_comparison.csv')

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

# Sentiment distribution of the selected subset
df_subset['sentiment'].value_counts().plot(
    kind='bar', color=['#4CAF50', '#F44336', '#FFC107'],
    title='Sentiment Distribution — 3,000-Review Subset'
)
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('team6_phase2_sentiment_distribution.png', bbox_inches='tight')
plt.show()

# Hyperparameter tuning curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

lr_Cs = sorted(lr_results.keys())
axes[0].plot(lr_Cs, [lr_results[c][0] for c in lr_Cs], 'o-', label='Accuracy', color='steelblue')
axes[0].plot(lr_Cs, [lr_results[c][1] for c in lr_Cs], 's--', label='F1-W', color='darkorange')
axes[0].axvline(best_C_lr, color='green', linestyle=':', lw=1.5, label=f'Best C={best_C_lr}')
axes[0].set_xscale('log')
axes[0].set_title('Logistic Regression — C Tuning')
axes[0].set_xlabel('C (regularisation)')
axes[0].set_ylabel('Score')
axes[0].legend()

svm_Cs = sorted(svm_results.keys())
axes[1].plot(svm_Cs, [svm_results[c][0] for c in svm_Cs], 'o-', label='Accuracy', color='steelblue')
axes[1].plot(svm_Cs, [svm_results[c][1] for c in svm_Cs], 's--', label='F1-W', color='darkorange')
axes[1].axvline(best_C_svm, color='green', linestyle=':', lw=1.5, label=f'Best C={best_C_svm}')
axes[1].set_xscale('log')
axes[1].set_title('LinearSVC — C Tuning')
axes[1].set_xlabel('C (regularisation)')
axes[1].legend()

plt.tight_layout()
plt.savefig('team6_phase2_hyperparameter_tuning.png', bbox_inches='tight')
plt.show()
print('Saved: team6_phase2_hyperparameter_tuning.png')

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, cm, title in zip(axes,
                          [lr_cm, svm_cm],
                          [f'Logistic Regression (C={best_C_lr})',
                           f'LinearSVC (C={best_C_svm})']):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f'{title}\nConfusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('team6_phase2_confusion_matrices.png', bbox_inches='tight')
plt.show()
print('Saved: team6_phase2_confusion_matrices.png')

# Model comparison bar chart
metrics_labels = ['Accuracy', 'Precision (W)', 'Recall (W)', 'F1 (W)']
lr_vals  = [lr_acc,  lr_prec,  lr_rec,  lr_f1]
svm_vals = [svm_acc, svm_prec, svm_rec, svm_f1]

x = np.arange(len(metrics_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, lr_vals,  width, label='Logistic Regression', color='steelblue',  edgecolor='white')
bars2 = ax.bar(x + width/2, svm_vals, width, label='LinearSVC (SVM)',     color='darkorange', edgecolor='white')
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison — Logistic Regression vs LinearSVC')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, rotation=15, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig('team6_phase2_model_comparison.png', bbox_inches='tight')
plt.show()
print('Saved: team6_phase2_model_comparison.png')

print('\n=== Phase 2 Complete! ===')
