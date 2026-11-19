import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Set writable matplotlib config dir inside workspace to prevent sandbox access errors
os.environ['MPLCONFIGDIR'] = os.path.join(BASE_DIR, ".matplotlib")
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import joblib

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
DATASET_PATH = os.path.join(DATASET_DIR, "news.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

def load_and_prepare_data():
    """Loads CSV dataset and cleans labels."""
    print("[INFO] Loading dataset with Pandas...")
    
    filepath = DATASET_PATH
    if not os.path.exists(filepath):
        candidates = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
        if candidates:
            filepath = candidates[0]
        else:
            raise FileNotFoundError(f"No CSV dataset found in {DATASET_DIR}")
            
    try:
        df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
        if 'label' not in [c.lower() for c in df.columns] and len(df.columns) == 1:
            df = pd.read_csv(filepath, sep=',', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(filepath, sep=',', on_bad_lines='skip')

    print(f"Dataset Raw Shape: {df.shape}")

    col_mapping = {}
    for col in df.columns:
        c_low = col.lower().strip()
        if 'title' in c_low and 'title' not in col_mapping.values():
            col_mapping[col] = 'title'
        elif ('text' in c_low or 'body' in c_low or 'article' in c_low) and 'text' not in col_mapping.values():
            col_mapping[col] = 'text'
        elif ('label' in c_low or 'target' in c_low or 'class' in c_low) and 'label' not in col_mapping.values():
            col_mapping[col] = 'label'
            
    df = df.rename(columns=col_mapping)

    if 'text' not in df.columns and 'title' in df.columns:
        df['text'] = df['title']
    elif 'title' not in df.columns and 'text' in df.columns:
        df['title'] = ''

    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + " " + df['text']

    df = df[df['content'].str.strip().str.len() > 15].copy()

    # Map target: 1 = FAKE, 0 = REAL
    def map_label(val):
        val_str = str(val).strip().upper()
        if val_str in ['FAKE', 'FALSE', 'MISLEADING', '0', '0.0']:
            return 1
        elif val_str in ['REAL', 'TRUE', '1', '1.0']:
            return 0
        return None

    df['target'] = df['label'].apply(map_label)
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)

    # For fast and robust college demo, take balanced subset of 4,000 articles if dataset is large
    if len(df) > 4000:
        real_df = df[df['target'] == 0].sample(2000, random_state=42)
        fake_df = df[df['target'] == 1].sample(2000, random_state=42)
        df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    real_count = int((df['target'] == 0).sum())
    fake_count = int((df['target'] == 1).sum())
    print(f"[INFO] Labeled Data: Total={len(df)}, REAL (0)={real_count}, FAKE (1)={fake_count}")

    return df

def generate_visualizations(df, y_test, y_pred, model, vectorizer):
    """Generates and saves publication/presentation quality Matplotlib graphs."""
    print("[INFO] Generating Matplotlib visualization graphs...")
    
    # 1. Class Distribution Graph
    plt.figure(figsize=(7, 5))
    counts = df['target'].value_counts()
    labels = ['Real News (0)', 'Fake News (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = ['#28a745', '#dc3545']
    
    bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.55)
    plt.title("Dataset Class Distribution (Real vs Fake)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Number of Articles", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 15,
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
    plt.xticks(tick_marks, ['Real (0)', 'Fake (1)'], fontsize=11)
    plt.yticks(tick_marks, ['Real (0)', 'Fake (1)'], fontsize=11)
    
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
    
    top_fake_idx = np.argsort(coef)[-10:]
    top_real_idx = np.argsort(coef)[:10]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
    # Top Real Words (Negative Coefs)
    axes[0].barh(feature_names[top_real_idx], np.abs(coef[top_real_idx]), color='#28a745', edgecolor='black')
    axes[0].set_title("Top Words Indicating REAL News", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predictive Weight", fontsize=10)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', linestyle='--', alpha=0.6)
    
    # Top Fake Words (Positive Coefs)
    axes[1].barh(feature_names[top_fake_idx], coef[top_fake_idx], color='#dc3545', edgecolor='black')
    axes[1].set_title("Top Words Indicating FAKE News", fontsize=12, fontweight='bold')
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
    df = load_and_prepare_data()

    print("[INFO] Preprocessing text using NLTK (lowercasing, cleaning, stopwords, stemming)...")
    df['cleaned_content'] = df['content'].apply(preprocess_text)
    df = df[df['cleaned_content'].str.strip() != ''].copy()

    X = df['cleaned_content']
    y = df['target']

    print(f"[INFO] Splitting dataset: 80% Training, 20% Testing (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[INFO] Extracting TF-IDF features (max_features=5000, n-grams=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("[INFO] Training Scikit-learn Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("[INFO] Evaluating model on unseen test set...")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)'])

    print("=" * 60)
    print("🏆 MODEL EVALUATION RESULTS:")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(report)
    print("=" * 60)

    # Save model and vectorizer with Joblib
    print(f"[INFO] Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    print(f"[INFO] Saving vectorizer to {VECTORIZER_PATH}...")
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # Save metrics to JSON for instant Streamlit display
    metrics = {
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "real_count": int((y == 0).sum()),
        "fake_count": int((y == 1).sum()),
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
