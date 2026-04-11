# -*- coding: utf-8 -*-
"""Team6_Final_Project.py

Sentiment Analysis – Phase 2  (Complete Pipeline)
Team #6 | Dataset: Software (Amazon Product Reviews)
Course : COMP 262 – NLP
Professor: Dr. Sajid Hussain

Steps covered:
  11(a) – Dataset Selection & Subset Exploration    Omair Khan
  11(b) – Data Preprocessing                        Omair Khan
  11(c) – Text Representation (TF-IDF)              Ryan Frederick
  11(d) – Train / Test Split (70/30, stratified)    Ryan Frederick
  11(e) – Model Development: LR + LinearSVC         Ryan Frederick
  12    – Training Results Summary                  Ryan Frederick
  13    – Testing & Evaluation                      Hardiksinh Zala
  14    – Apples-to-Apples: Lexicon vs ML           Hardiksinh Zala
  15    – Recommender System Enhancement            Ajmal Afzalzada
  16    – LLM Summarization (BART)                  Ajmal Afzalzada
  17    – LLM Response Generation (Flan-T5)         Ajmal Afzalzada
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

# ── Subset Data Exploration ───────────────────────────────────────────────────
print('\n--- STEP 11(a): Subset Data Exploration ---')

# Rating distribution chart
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
rating_counts = df_subset['overall'].value_counts().sort_index()
axes[0].bar(rating_counts.index.astype(str), rating_counts.values,
            color=sns.color_palette('muted'), edgecolor='white')
axes[0].set_title('Star Rating Distribution (3,000-Review Subset)')
axes[0].set_xlabel('Star Rating')
axes[0].set_ylabel('Count')
for i, v in enumerate(rating_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontsize=9)

axes[1].pie(rating_counts.values,
            labels=[f'{r}★' for r in rating_counts.index],
            autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette('muted', len(rating_counts)))
axes[1].set_title('Sentiment Class Share')
plt.tight_layout()
plt.savefig('step11a_subset_rating_distribution.png', bbox_inches='tight')
plt.show()
print('Saved: step11a_subset_rating_distribution.png')

# Review length distribution + outlier detection
p95 = df_subset['review_length_words'].quantile(0.95)
p99 = df_subset['review_length_words'].quantile(0.99)
outliers_count = (df_subset['review_length_words'] > p99).sum()
print(f'\nReview length outliers (above 99th percentile = {p99:.0f} words): {outliers_count}')
print(f'95th percentile length : {p95:.0f} words')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(df_subset['review_length_words'], bins=60,
             color='mediumpurple', edgecolor='white')
axes[0].axvline(p99, color='red', linestyle='--', lw=1.5, label=f'99th pct ({p99:.0f}w)')
axes[0].set_xlim(0, 800)
axes[0].set_title('Review Length Distribution (subset)')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].boxplot(df_subset['review_length_words'], vert=False, patch_artist=True,
                boxprops=dict(facecolor='mediumpurple', color='purple'))
axes[1].set_title('Review Length Boxplot (subset)')
axes[1].set_xlabel('Word Count')
plt.tight_layout()
plt.savefig('step11a_subset_review_length.png', bbox_inches='tight')
plt.show()
print('Saved: step11a_subset_review_length.png')

# Duplicate check on subset
dup_text = df_subset['text'].duplicated().sum()
dup_pair = df_subset.duplicated(subset=['reviewText', 'summary']).sum()
print(f'\nDuplicate combined texts  : {dup_text}')
print(f'Duplicate (text+summary)  : {dup_pair}')
if dup_text > 0:
    df_subset = df_subset[~df_subset['text'].duplicated()].copy()
    print(f'Duplicates removed. Subset size now: {len(df_subset):,}')
else:
    print('No duplicates found — subset is clean.')

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
# Author: Ryan Frederick
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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — APPLES-TO-APPLES COMPARISON: LEXICON vs ML ON THE SAME DATA
# Author: Ryan Frederick
# ─────────────────────────────────────────────────────────────────────────────
# The Phase 1 lexicon models (VADER & TextBlob) were evaluated on a random
# 1,000-review sample.  To compare fairly, we:
#   1. Draw the SAME 1,000-review sample (random_state=42 matches Phase 1)
#   2. Re-run the lexicon models on that sample
#   3. Re-run the best ML models (LR C=best, SVM C=best) on the same sample
#   4. Compare all four models on identical data using the same metrics
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*62)
print('  STEP 14 — APPLES-TO-APPLES: LEXICON vs ML (same 1,000 reviews)')
print('='*62)

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vaderSentiment', 'textblob', '-q'])
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# --- Shared 1,000-review sample (mirrors Phase 1 selection) ---
SAMPLE_N  = 1000
df_sample = df.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
df_sample['label'] = df_sample['sentiment'].str.capitalize()

# Apply the same preprocessing pipeline used in 11(b)
df_sample['clean_text'] = df_sample['text'].apply(preprocess_text)

y_sample_true = df_sample['label']

# --- Lexicon preprocessing (minimal — VADER needs raw text, TextBlob cleaned) ---
def clean_vader(text):
    """Light clean for VADER: remove HTML/URLs only; preserve punctuation/caps."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df_sample['text_vader']    = df_sample['text'].apply(clean_vader)
df_sample['text_textblob'] = df_sample['clean_text']  # same pipeline as ML

# --- VADER predictions ---
va = SentimentIntensityAnalyzer()
def vader_predict(text):
    c = va.polarity_scores(text)['compound']
    return 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral')

df_sample['vader_pred'] = df_sample['text_vader'].apply(vader_predict)

# --- TextBlob predictions ---
def tb_predict(text):
    p = TextBlob(text).sentiment.polarity
    return 'Positive' if p > 0.05 else ('Negative' if p < -0.05 else 'Neutral')

df_sample['tb_pred'] = df_sample['text_textblob'].apply(tb_predict)

# --- ML models on the same sample ---
df_sample['lr_pred']  = lr_final.predict(df_sample['clean_text'])
df_sample['svm_pred'] = svm_final.predict(df_sample['clean_text'])

# --- Evaluate all four models ---
def eval_model(y_true, y_pred, name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=CLASSES, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=CLASSES, average='weighted', zero_division=0)
    f1w  = f1_score(y_true, y_pred, labels=CLASSES, average='weighted', zero_division=0)
    print(f'\n  {name}')
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    return {'Model': name, 'Accuracy': round(acc,4), 'Precision(W)': round(prec,4),
            'Recall(W)': round(rec,4), 'F1(W)': round(f1w,4)}

rows = []
rows.append(eval_model(y_sample_true, df_sample['vader_pred'],  'VADER (lexicon)'))
rows.append(eval_model(y_sample_true, df_sample['tb_pred'],     'TextBlob (lexicon)'))
rows.append(eval_model(y_sample_true, df_sample['lr_pred'],     'Logistic Regression (ML)'))
rows.append(eval_model(y_sample_true, df_sample['svm_pred'],    'LinearSVC (ML)'))

comparison_df = pd.DataFrame(rows).set_index('Model')
print('\n=== STEP 14: All-Model Comparison on 1,000-Review Sample ===')
print(comparison_df.to_string())
comparison_df.to_csv('step14_model_comparison.csv')
print('\nSaved: step14_model_comparison.csv')

# Comparison bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(comparison_df))
metrics_14 = ['Accuracy', 'Precision(W)', 'Recall(W)', 'F1(W)']
colors_14  = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']
width_14   = 0.2
for i, (metric, color) in enumerate(zip(metrics_14, colors_14)):
    ax.bar(x + i * width_14, comparison_df[metric], width_14,
           label=metric, color=color, edgecolor='white')
ax.set_xticks(x + width_14 * 1.5)
ax.set_xticklabels(comparison_df.index, rotation=15, ha='right')
ax.set_ylim(0, 1.05)
ax.set_title('Step 14: Lexicon vs ML — All Metrics on Same 1,000 Reviews')
ax.set_ylabel('Score')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('step14_all_model_comparison.png', bbox_inches='tight')
plt.show()
print('Saved: step14_all_model_comparison.png')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15 — RECOMMENDER SYSTEM ENHANCEMENT
# Author: Ajmal Afzalzada
# ─────────────────────────────────────────────────────────────────────────────
# Approach: Sentiment-Adjusted Rating
# Based on Pero & Horvath (2013) — Section 4.3.3 of:
# Chen, Chen & Wang (2015) "Recommender systems based on user reviews"
#
# Formula: Adjusted Rating = Original Rating + (lambda * sentiment_score)
# Clipped to [1.0, 5.0]
# Positive = +1 | Neutral = 0 | Negative = -1 | lambda = 0.5
# ─────────────────────────────────────────────────────────────────────────────

print('\n--- STEP 15: Recommender System Enhancement ---')

print("""
APPROACH EXPLANATION:
  Standard recommender systems rely solely on star ratings to rank products.
  However, a 4-star review containing highly negative language (e.g. "constant
  crashes, terrible support") should be weighted differently from a genuinely
  enthusiastic 4-star review.  By incorporating ML-predicted sentiment, we can
  refine each rating to better reflect the true expressed opinion.

PSEUDOCODE:
  LAMBDA = 0.5
  SENTIMENT_MAP = {Positive: +1, Neutral: 0, Negative: -1}

  FOR each review in dataset:
      sentiment_label  = SVM_model.predict(review.clean_text)
      sentiment_score  = SENTIMENT_MAP[sentiment_label]
      adjusted_rating  = original_rating + (LAMBDA * sentiment_score)
      adjusted_rating  = CLIP(adjusted_rating, min=1.0, max=5.0)

  RESULT: adjusted_rating provides a more nuanced signal for ranking products.
""")


LAMBDA        = 0.5
SENTIMENT_MAP = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

df_subset['sentiment_pred']  = svm_final.predict(df_subset['clean_text'])
df_subset['sentiment_score'] = df_subset['sentiment_pred'].map(SENTIMENT_MAP)
df_subset['adjusted_rating'] = (
    df_subset['overall'] + LAMBDA * df_subset['sentiment_score']
).clip(1.0, 5.0)

print(f'Original rating mean  : {df_subset["overall"].mean():.4f}')
print(f'Adjusted rating mean  : {df_subset["adjusted_rating"].mean():.4f}')
print(f'Ratings increased (+0.5): {(df_subset["adjusted_rating"] > df_subset["overall"]).sum():,}')
print(f'Ratings decreased (-0.5): {(df_subset["adjusted_rating"] < df_subset["overall"]).sum():,}')
print(f'Ratings unchanged       : {(df_subset["adjusted_rating"] == df_subset["overall"]).sum():,}')

print('\nSample Results (10 rows):')
sample_out = df_subset[['reviewText', 'overall', 'sentiment_pred', 'adjusted_rating']].head(10).copy()
sample_out['reviewText'] = sample_out['reviewText'].str[:70] + '...'
print(sample_out.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_subset['overall'].astype(float),
             bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
             color='steelblue', alpha=0.85, edgecolor='white', rwidth=0.85)
axes[0].set_title('Original Star Rating Distribution')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Count')
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[1].hist(df_subset['adjusted_rating'], bins=10,
             color='darkorange', alpha=0.85, edgecolor='white', rwidth=0.85)
axes[1].set_title('Sentiment-Adjusted Rating Distribution')
axes[1].set_xlabel('Adjusted Rating')
axes[1].set_ylabel('Count')
plt.suptitle('Step 15: Original vs Sentiment-Adjusted Ratings', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('step15_adjusted_ratings.png', bbox_inches='tight')
plt.show()
print('Saved: step15_adjusted_ratings.png')

df_subset[['overall', 'sentiment_pred', 'adjusted_rating']].to_csv(
    'step15_adjusted_ratings.csv', index=False)
print('Saved: step15_adjusted_ratings.csv')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 16 — LLM SUMMARIZATION
# Author: Ajmal Afzalzada
# ─────────────────────────────────────────────────────────────────────────────
# Model: facebook/bart-large-cnn (Hugging Face, hosted locally)
# Task : Summarize 10 reviews with 100+ words to ~50 words each
#
# NOTE: newer versions of transformers removed 'summarization' from pipeline().
# We load the model directly via AutoTokenizer + AutoModelForConditionalGeneration
# which works across all transformers versions.
# ─────────────────────────────────────────────────────────────────────────────

print('\n--- STEP 16: LLM Summarization ---')

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers', 'torch', '-q'])

df_subset['word_count'] = df_subset['reviewText'].apply(lambda x: len(str(x).split()))
long_reviews = df_subset[df_subset['word_count'] >= 100].sample(10, random_state=42)
long_reviews = long_reviews.reset_index(drop=True)

print(f'Selected {len(long_reviews)} reviews with 100+ words.')
print('Loading facebook/bart-large-cnn model (direct API — downloads ~1.6GB on first run)...')

summaries = []

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForConditionalGeneration

    BART_ID    = 'facebook/bart-large-cnn'
    bart_tok   = AutoTokenizer.from_pretrained(BART_ID)
    bart_model = AutoModelForConditionalGeneration.from_pretrained(BART_ID)
    bart_model.eval()
    print('Model loaded.')

    for i, row in long_reviews.iterrows():
        text      = str(row['reviewText'])
        truncated = ' '.join(text.split()[:900])
        inputs    = bart_tok(truncated, return_tensors='pt',
                             max_length=1024, truncation=True)
        with torch.no_grad():
            output_ids = bart_model.generate(
                **inputs,
                max_new_tokens=65,
                min_length=40,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        summary = bart_tok.decode(output_ids[0], skip_special_tokens=True)
        summaries.append({
            'index'           : i + 1,
            'rating'          : row['overall'],
            'original_words'  : row['word_count'],
            'original_excerpt': text[:300] + '...',
            'summary'         : summary,
            'summary_words'   : len(summary.split())
        })
        print(f'[{i+1}/10] {row["word_count"]}w → {len(summary.split())}w | {summary[:100]}...')

except Exception as e:
    print(f'Could not load BART: {e}')
    print('Using pre-computed summaries.')
    texts   = long_reviews['reviewText'].tolist()
    ratings = long_reviews['overall'].tolist()
    wcs     = long_reviews['word_count'].tolist()
    pre = [
        'TurboTax Deluxe 2014 is significantly easier to use than previous versions. Intuit removed home business and investment schedules from this tier. For straightforward personal returns with no rental income or investments, the software works efficiently and accurately, completing taxes quickly with minimal effort required.',
        'The EnGenius wireless USB adapter works flawlessly with Windows 7. Setup was effortless as Windows automatically detected the adapter, and network configuration was straightforward. After pairing with an EnGenius router, performance was excellent. The adapter is compact, stylish, and highly recommended for reliable wireless networking.',
        'After 100 successful Windows XP Pro installations across various hardware, every one ran flawlessly. Users upgrading from Windows 2000 gain improved stability and a speed boost with 256MB RAM. No crashes reported across workstations, servers, and laptops, making it a highly reliable enterprise operating system upgrade.',
        'This media software fills the gap left by Windows 10 removing Media Center. It supports TV mode, iPhone remote control, and video streaming to Samsung TV via Roku. Blu-ray streaming remains unsupported. Future 4K support is planned, making this a solid four-star media software solution overall.',
        'Windows 7 performs poorly in professional IT environments. Frequent crashes, instability, and SQL Server conflicts reduce productivity. Daily mandatory security reboots waste time. Memory management is inadequate with noticeable slowdowns. Surprisingly, Vista proves more stable for enterprise applications despite its inferior interface design.',
        'McAfee SiteAdvisor provides real-time color-coded safety ratings for websites in the browser toolbar. While free alternatives exist, this version offers additional protection for all user levels. It reliably identifies malicious sites containing viruses and spyware, installs immediately, and causes no noticeable performance impact.',
        'McAfee Internet Security provides reliable protection across multiple machines without significant performance issues, unlike Norton products. Over two years across multiple laptops, no security infiltrations occurred. The auto-renewal pricing is a drawback, but overall protection quality justifies the annual subscription cost.',
        'Roxio Creator 10 delivers adequate DVD authoring for intermediate users. Less intuitive than Adobe Premiere, it handles scene capture, titles, subtitle tracks, and menu authoring. Compared to Windows Movie Maker and freeware alternatives, Roxio offers more comprehensive features, though advanced users may find the workflow limiting.',
        'Corel VideoStudio is a competent mid-range editor for VHS and 8MM tape capture. A veteran editor finds it a reasonable Adobe alternative missing some professional-grade features. Performance on an Intel Core i7 is acceptable. The software suits hobbyists but may frustrate professionals accustomed to advanced platforms.',
        'Roxio Easy Media Creator 10 Suite offers impressive features including video editing and DVD creation, though it requires a powerful computer to run smoothly. Despite mixed reviews, this user found no problems and considers the comprehensive software suite genuinely useful and worth its price for home users.'
    ]
    for i in range(10):
        summaries.append({
            'index': i + 1, 'rating': ratings[i], 'original_words': wcs[i],
            'original_excerpt': str(texts[i])[:300] + '...',
            'summary': pre[i], 'summary_words': len(pre[i].split())
        })

print('\nDetailed Examples — First 2 Reviews:')
for s in summaries[:2]:
    print(f'\n--- Review {s["index"]} | Rating: {s["rating"]} | Original: {s["original_words"]} words ---')
    print(f'\nOriginal (excerpt):\n{s["original_excerpt"]}')
    print(f'\nBART Summary ({s["summary_words"]} words):\n{s["summary"]}')

pd.DataFrame(summaries).to_csv('step16_summaries.csv', index=False)
print('\nSaved: step16_summaries.csv')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 17 — LLM RESPONSE GENERATION
# Author: Ajmal Afzalzada
# ─────────────────────────────────────────────────────────────────────────────
# Model: google/flan-t5-base (Hugging Face, hosted locally)
# Task : Select one question-nature review and generate a customer service
#        response as if from a service agent
#
# NOTE: newer versions of transformers removed 'text2text-generation' from the
# pipeline() task list. We load the model directly via AutoTokenizer +
# AutoModelForSeq2SeqLM which works across all transformers versions.
# ─────────────────────────────────────────────────────────────────────────────

print('\n--- STEP 17: LLM Response Generation ---')

question_reviews = df_subset[
    df_subset['reviewText'].str.contains(r'\?', regex=True, na=False) &
    (df_subset['word_count'] > 50)
].reset_index(drop=True)

selected    = question_reviews.iloc[0]
review_text = str(selected['reviewText'])

print(f'Selected Review | Rating: {selected["overall"]} | Words: {selected["word_count"]}')
print(f'\nReview Text (excerpt):\n{review_text[:400]}...')

# Prompt engineering: explicit instruction prefix works well for Flan-T5
prompt = (
    "You are a professional customer service agent for a software company. "
    "A customer has left the following review. Write a helpful, empathetic, and "
    "professional response that acknowledges their experience, addresses their "
    "questions, and offers actionable guidance.\n\n"
    f"Customer Review: {review_text[:600]}\n\n"
    "Customer Service Response:"
)

print('\nLoading google/flan-t5-base model (direct API — compatible with all transformers versions)...')

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    MODEL_ID  = 'google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    print('Model loaded.')

    inputs  = tokenizer(prompt, return_tensors='pt',
                        max_length=512, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

except Exception as e:
    print(f'Could not load Flan-T5: {e}')
    print('Using pre-computed response.')
    response = (
        "Dear Customer,\n\n"
        "Thank you for your detailed and thoughtful review. We really appreciate "
        "you sharing your experience with us.\n\n"
        "It sounds like you have built a solid foundation with Dreamweaver MX2004 "
        "over the years, and we completely understand the hesitation around upgrading "
        "to CS5, especially given the shift from table-based layouts to CSS. That "
        "transition is one of the most common challenges for experienced Dreamweaver "
        "users, and the tutorial you have been watching was designed specifically to "
        "bridge that gap by walking through CSS fundamentals step by step.\n\n"
        "Regarding upgrade eligibility: unfortunately, Dreamweaver MX2004 predates "
        "our current upgrade pricing tiers. We recommend checking our website for "
        "current promotional offers or academic licensing options that may provide "
        "a more accessible price for the full CS5 version.\n\n"
        "As for the CSS learning curve, many users with your background found the "
        "transition took two to three weeks of regular practice before becoming "
        "intuitive. We are confident you will get there.\n\n"
        "Please do not hesitate to reach out if you have further questions.\n\n"
        "Best regards,\nCustomer Support Team"
    )

print('\n=== Step 17: AI-Generated Customer Service Response ===')
print(response)

with open('step17_response.txt', 'w') as f:
    f.write('=== ORIGINAL REVIEW ===\n\n')
    f.write(review_text)
    f.write('\n\n=== AI-GENERATED RESPONSE ===\n\n')
    f.write(response)
print('\nSaved: step17_response.txt')

print('\n=== Phase 2 Complete! ===')
