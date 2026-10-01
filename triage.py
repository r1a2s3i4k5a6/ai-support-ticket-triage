"""
triage.py

The live prediction tool: load the trained models and classify any ticket
text, either from the command line or imported as a function.

Usage:
    python triage.py "My payment failed and I need this fixed immediately"

    or interactively:
    python triage.py
"""

import sys
import joblib

REVIEW_THRESHOLD = 0.60


def load_models():
    category_bundle = joblib.load("models/category_classifier.joblib")
    urgency_bundle = joblib.load("models/urgency_classifier.joblib")
    return category_bundle, urgency_bundle


def triage_ticket(text, category_bundle, urgency_bundle):
    cat_vectorizer, cat_model = category_bundle["vectorizer"], category_bundle["model"]
    urg_vectorizer, urg_model = urgency_bundle["vectorizer"], urgency_bundle["model"]

    cat_features = cat_vectorizer.transform([text])
    predicted_category = cat_model.predict(cat_features)[0]
    confidence = cat_model.predict_proba(cat_features).max()

    urg_features = urg_vectorizer.transform([text])
    predicted_urgency = urg_model.predict(urg_features)[0]

    status = "Needs human review" if confidence < REVIEW_THRESHOLD else "Auto-routed"

    return {
        "text": text,
        "predicted_category": predicted_category,
        "confidence": round(float(confidence), 2),
        "predicted_urgency": predicted_urgency,
        "status": status,
    }


def main():
    category_bundle, urgency_bundle = load_models()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Type a support ticket: ")

    result = triage_ticket(text, category_bundle, urgency_bundle)

    print(f"\nTicket:    {result['text']}")
    print(f"Category:  {result['predicted_category']} (confidence: {result['confidence']})")
    print(f"Urgency:   {result['predicted_urgency']}")
    print(f"Status:    {result['status']}")


if __name__ == "__main__":
    main()
