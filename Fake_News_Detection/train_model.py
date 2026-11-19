import os
import json
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['MPLCONFIGDIR'] = os.path.join(BASE_DIR, ".matplotlib")
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import preprocess_text

# Paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
MAIN_DATASET_PATH = os.path.join(DATASET_DIR, "news.csv")
SHORT_DATASET_PATH = os.path.join(DATASET_DIR, "short_news.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

def load_and_combine_data():
    """
    Loads both long news articles and short news examples,
    combining them into a balanced dataset for high accuracy on both short and long inputs.
    """
    print("[INFO] Loading datasets...")
    
    # 1. Load long news dataset
    df_long = pd.read_csv(MAIN_DATASET_PATH, sep=';', on_bad_lines='skip')
    df_long['title'] = df_long['title'].fillna('')
    df_long['text'] = df_long['text'].fillna('')
    df_long['label'] = pd.to_numeric(df_long['label'], errors='coerce')
    df_long = df_long.dropna(subset=['label'])
    df_long['label'] = df_long['label'].astype(int)
    
    # Create full content
    df_long['content'] = df_long['title'] + " " + df_long['text']
    df_long = df_long[df_long['content'].str.strip().str.len() > 15].copy()
    
    # Also extract short headlines from long articles as short news training data
    df_headlines = df_long[['title', 'label']].copy()
    df_headlines['content'] = df_headlines['title']
    df_headlines = df_headlines[df_headlines['content'].str.strip().str.len() > 10].copy()
    
    # 2. Load curated short news dataset (220+ examples of short facts, headlines, WhatsApp forwards)
    df_short = pd.read_csv(SHORT_DATASET_PATH, sep=';', on_bad_lines='skip')
    df_short['title'] = df_short['title'].fillna('')
    df_short['text'] = df_short['text'].fillna('')
    df_short['label'] = pd.to_numeric(df_short['label'], errors='coerce')
    df_short = df_short.dropna(subset=['label'])
    df_short['label'] = df_short['label'].astype(int)
    df_short['content'] = df_short['title']
    
    # Weight short news samples by repeating them so they aren't drowned out by long articles
    short_repeated = pd.concat([df_short] * 12, ignore_index=True)
    
    print(f"Long news samples    : {len(df_long)}")
    print(f"Headline samples     : {len(df_headlines)}")
    print(f"Short news samples   : {len(df_short)} (repeated 12x -> {len(short_repeated)})")
    
    # Combine all datasets
    combined_df = pd.concat([
        df_long[['content', 'label']],
        df_headlines[['content', 'label']],
        short_repeated[['content', 'label']]
    ], ignore_index=True)
    
    # Shuffle dataset
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    real_count = int((combined_df['label'] == 1).sum())
    fake_count = int((combined_df['label'] == 0).sum())
    print(f"[INFO] Total Combined Samples: {len(combined_df)} (REAL=1: {real_count}, FAKE=0: {fake_count})")
    
    return combined_df

def generate_visualizations(df, y_test, y_pred, model, vectorizer):
    """Generates and saves publication/presentation quality Matplotlib graphs."""
    print("[INFO] Generating Matplotlib visualization graphs...")
    
    # 1. Class Distribution Graph
    plt.figure(figsize=(7, 5))
    counts = df['label'].value_counts()
    labels = ['Fake News (0)', 'Real News (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = ['#dc3545', '#28a745']
    
    bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.55)
    plt.title("Dataset Class Distribution (Short + Long News)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Number of Articles / Samples", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 20,
                 f"{height:,}", ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.tight_layout()
    dist_path = os.path.join(GRAPHS_DIR, "class_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {dist_path}")

    # 2. Confusion Matrix Graph
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Logistic Regression)", fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Fake (0)', 'Real (1)'], fontsize=11)
    plt.yticks(tick_marks, ['Fake (0)', 'Real (1)'], fontsize=11)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            label_desc = "TN" if (i == 0 and j == 0) else "FP" if (i == 0 and j == 1) else "FN" if (i == 1 and j == 0) else "TP"
            plt.text(j, i, f"{val}\n({label_desc})",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=12, fontweight='bold')
                     
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(GRAPHS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {cm_path}")

    # 3. Top Keywords (Feature Importance from Logistic Regression)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    
    top_real_idx = np.argsort(coef)[-10:]
    top_fake_idx = np.argsort(coef)[:10]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
    # Top Fake Words (Negative Coefs for class 1)
    axes[0].barh(feature_names[top_fake_idx], np.abs(coef[top_fake_idx]), color='#dc3545', edgecolor='black')
    axes[0].set_title("Top Words Indicating FAKE News", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predictive Weight", fontsize=10)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', linestyle='--', alpha=0.6)
    
    # Top Real Words (Positive Coefs for class 1)
    axes[1].barh(feature_names[top_real_idx], coef[top_real_idx], color='#28a745', edgecolor='black')
    axes[1].set_title("Top Words Indicating REAL News", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predictive Weight", fontsize=10)
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.suptitle("TF-IDF & Logistic Regression Influential Keywords", fontsize=14, fontweight='bold')
    plt.tight_layout()
    keywords_path = os.path.join(GRAPHS_DIR, "top_keywords.png")
    plt.savefig(keywords_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {keywords_path}")

def train():
    """Full ML pipeline: Load -> Preprocess -> TF-IDF -> Train -> Evaluate -> Save."""
    df = load_and_combine_data()

    print("[INFO] Preprocessing text using NLTK (lowercasing, cleaning, stopwords, stemming)...")
    df['cleaned_content'] = df['content'].apply(preprocess_text)
    df = df[df['cleaned_content'].str.strip() != ''].copy()

    X = df['cleaned_content']
    y = df['label']

    print(f"[INFO] Splitting dataset: 80% Training, 20% Testing (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[INFO] Extracting TF-IDF features (min_df=1, ngram_range=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("[INFO] Training Scikit-learn Logistic Regression model...")
    model = LogisticRegression(
        C=2.5,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    print("[INFO] Evaluating model on unseen test set...")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)'], zero_division=0)

    print("=" * 60)
    print("MODEL EVALUATION RESULTS (Short + Long News):")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(report)
    print("=" * 60)

    # Save model and vectorizer with Pickle/Joblib
    print(f"[INFO] Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Also save as fake_news_model.pkl for compatibility with user's alternate name
    alt_model_path = os.path.join(BASE_DIR, "fake_news_model.pkl")
    with open(alt_model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Saving vectorizer to {VECTORIZER_PATH}...")
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    alt_vec_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
    with open(alt_vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # Save metrics to JSON for Streamlit display
    metrics = {
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "real_count": int((y == 1).sum()),
        "fake_count": int((y == 0).sum()),
        "confusion_matrix": cm.tolist()
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Saved evaluation metrics to {METRICS_PATH}")

    # Generate visual graphs
    generate_visualizations(df, y_test, y_pred, model, vectorizer)

    print("\n[SUCCESS] Model training and evaluation successfully completed!")

if __name__ == "__main__":
    train()
