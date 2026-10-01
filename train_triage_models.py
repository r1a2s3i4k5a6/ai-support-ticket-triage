"""
train_triage_models.py

Trains two independent classifiers on the ticket dataset:
  1. category_classifier  - predicts which support queue a ticket belongs to
  2. urgency_classifier   - predicts how quickly it needs attention

Two separate models are used (rather than one model predicting both) because
category and urgency are driven by different signals in the text - category
comes from topic vocabulary ("refund", "crash"), urgency comes from tone
("URGENT", "no rush"). Keeping them separate also means either model can be
retrained or swapped independently later.

Also implements a confidence-based review queue: any ticket the category
model is under 60% sure about is routed to a human instead of being
auto-assigned. This mirrors how the brief's step 6 asks for low-confidence
predictions to be logged for review, and reflects a real limitation found
during testing - phrasing the model hasn't seen (e.g. "asap" instead of
"urgent") lowers its confidence even when a human would find it obvious.

Run:  python train_triage_models.py
Out:  models/category_classifier.joblib
      models/urgency_classifier.joblib
      reports/evaluation_results.txt
      reports/low_confidence_tickets.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

REVIEW_THRESHOLD = 0.60


def load_data():
    return pd.read_csv("data/tickets.csv")


def train_classifier(train_text, train_labels):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(train_text)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(features, train_labels)
    return vectorizer, classifier


def evaluate(name, vectorizer, classifier, test_text, test_labels, report_lines):
    features = vectorizer.transform(test_text)
    predictions = classifier.predict(features)
    acc = accuracy_score(test_labels, predictions)

    report_lines.append(f"\n{'=' * 60}\n{name.upper()} MODEL\n{'=' * 60}")
    report_lines.append(f"Accuracy: {acc:.3f}")
    report_lines.append(classification_report(test_labels, predictions, digits=3))

    labels_sorted = sorted(test_labels.unique())
    cm = confusion_matrix(test_labels, predictions, labels=labels_sorted)
    report_lines.append("Confusion matrix (rows = true, cols = predicted):")
    report_lines.append(str(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)))

    return predictions, acc


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    report_lines = [f"Dataset size: {len(df)} tickets ({len(train_df)} train / {len(test_df)} test)"]

    category_vectorizer, category_classifier = train_classifier(
        train_df["text"], train_df["category"]
    )
    category_predictions, category_acc = evaluate(
        "category", category_vectorizer, category_classifier,
        test_df["text"], test_df["category"], report_lines
    )

    urgency_vectorizer, urgency_classifier = train_classifier(
        train_df["text"], train_df["urgency"]
    )
    _, urgency_acc = evaluate(
        "urgency", urgency_vectorizer, urgency_classifier,
        test_df["text"], test_df["urgency"], report_lines
    )

    # Confidence-based human review queue (brief step 6)
    test_features = category_vectorizer.transform(test_df["text"])
    confidences = category_classifier.predict_proba(test_features).max(axis=1)

    review_df = test_df.copy()
    review_df["predicted_category"] = category_predictions
    review_df["confidence"] = confidences.round(3)
    low_confidence = review_df[review_df["confidence"] < REVIEW_THRESHOLD]

    review_note = (
        f"\n{len(low_confidence)} of {len(test_df)} test tickets fell below "
        f"the {REVIEW_THRESHOLD} confidence threshold and were routed to "
        f"manual review instead of being auto-assigned."
    )
    report_lines.append(review_note)
    print("\n".join(report_lines))

    low_confidence[["ticket_id", "text", "category", "predicted_category", "confidence"]] \
        .to_csv("reports/low_confidence_tickets.csv", index=False)

    with open("reports/evaluation_results.txt", "w") as f:
        f.write("\n".join(report_lines))

    joblib.dump({"vectorizer": category_vectorizer, "model": category_classifier},
                "models/category_classifier.joblib")
    joblib.dump({"vectorizer": urgency_vectorizer, "model": urgency_classifier},
                "models/urgency_classifier.joblib")

    print(f"\nSaved models to models/ and evaluation report to reports/")
    print(f"Final scores -> category: {category_acc:.3f}, urgency: {urgency_acc:.3f}")


if __name__ == "__main__":
    main()
