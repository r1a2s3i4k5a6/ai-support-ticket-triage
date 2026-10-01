"""
app.py

A small Streamlit dashboard that exposes the trained triage models as a
usable web interface (brief step 6: "expose the model through a small
API/dashboard"). Lets anyone paste in a ticket and see the predicted
category, urgency, and whether it needs human review - no code required.

Run:  streamlit run app.py
"""

import streamlit as st
import joblib
import pandas as pd

REVIEW_THRESHOLD = 0.60

st.set_page_config(page_title="Ticket Triage", page_icon="🎫", layout="centered")


@st.cache_resource
def load_models():
    category_bundle = joblib.load("models/category_classifier.joblib")
    urgency_bundle = joblib.load("models/urgency_classifier.joblib")
    return category_bundle, urgency_bundle


def triage(text, category_bundle, urgency_bundle):
    cat_vectorizer, cat_model = category_bundle["vectorizer"], category_bundle["model"]
    urg_vectorizer, urg_model = urgency_bundle["vectorizer"], urgency_bundle["model"]

    cat_features = cat_vectorizer.transform([text])
    predicted_category = cat_model.predict(cat_features)[0]
    confidence = float(cat_model.predict_proba(cat_features).max())

    urg_features = urg_vectorizer.transform([text])
    predicted_urgency = urg_model.predict(urg_features)[0]

    needs_review = confidence < REVIEW_THRESHOLD
    return predicted_category, confidence, predicted_urgency, needs_review


st.title("🎫 AI Customer Support Ticket Triage")
st.caption("Paste a support ticket below to see its predicted category and urgency.")

category_bundle, urgency_bundle = load_models()

if "form_round" not in st.session_state:
    st.session_state.form_round = 0

# The text area's key changes each time "Check Another Ticket" is clicked,
# which gives it a fresh, empty widget instead of trying to overwrite the
# value of one already on screen (Streamlit does not allow that directly).
text_key = f"ticket_input_{st.session_state.form_round}"

ticket_text = st.text_area(
    "Ticket text",
    placeholder="e.g. My payment failed and I need this fixed immediately",
    height=100,
    key=text_key,
)

col_classify, col_clear = st.columns([1, 1])
classify_clicked = col_classify.button("Classify Ticket", type="primary")
clear_clicked = col_clear.button("🔄 Check Another Ticket")

if clear_clicked:
    st.session_state.form_round += 1
    st.rerun()

if classify_clicked and ticket_text.strip():
    category, confidence, urgency, needs_review = triage(ticket_text, category_bundle, urgency_bundle)

    col1, col2 = st.columns(2)
    col1.metric("Predicted category", category.title())
    col2.metric("Predicted urgency", urgency.title())

    st.progress(confidence, text=f"Category confidence: {confidence:.0%}")

    if needs_review:
        st.warning("⚠️ Low confidence — this ticket would be routed to a human for review, not auto-assigned.")
    else:
        st.success("✅ Confident prediction — this ticket would be auto-routed to the queue.")

st.divider()
st.caption(
    "Two independent models: one predicts category from topic vocabulary, "
    "the other predicts urgency from tone. Predictions below 60% confidence "
    "are flagged for manual review instead of being trusted automatically."
)

with st.expander("About this project"):
    st.write(
        "Trained on a 1,650-ticket synthetic dataset. Category accuracy: 95.2%, "
        "Urgency accuracy: 93.9% on a held-out test set. See the full project "
        "report for evaluation details and known limitations."
    )
