import pandas as pd
import pickle
import json

from pathlib import Path
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

# 1. Load dataset
df = pd.read_csv("dataset/news.csv", sep=";", index_col=0)

# 2. Clean missing values
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# 3. Combine title and news text
df["content"] = df["title"] + " " + df["text"]

# 4. Preprocess news
print("Preprocessing news...")
X = df["content"].apply(preprocess_text)
y = df["label"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Convert text into TF-IDF features
print("Creating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 7. Train model
print("Training model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# 8. Evaluate model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

# 9. Save model and vectorizer
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

# 10. Save evaluation metrics
metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
    "classification_report": classification_report(
        y_test, predictions, output_dict=True,
        zero_division=0
    )
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel, vectorizer and metrics saved successfully!")