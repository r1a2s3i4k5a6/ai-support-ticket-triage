"""
generate_dataset.py

Builds a synthetic customer support ticket dataset for the triage project.

Why synthetic data: real historical support tickets were not available for
this project, so this script generates realistic ticket text instead. To
avoid an unrealistically easy dataset (an earlier version of this script
produced 100% accuracy, which is a red flag, not a result), it deliberately
includes:
  - varied phrasing per category
  - urgency signalled by tone words, independent of category
  - a batch of genuinely ambiguous tickets with randomly assigned labels,
    so the model is forced to handle real-world uncertainty

Output: data/tickets.csv
"""

import os
import random
import pandas as pd

random.seed(42)

CATEGORY_PHRASES = {
    "billing": [
        "I was charged twice for the same invoice",
        "Refund for my last order has not arrived",
        "My subscription payment failed but money was deducted",
        "Please explain this extra charge on my bill",
    ],
    "technical": [
        "The app crashes every time I open reports",
        "Getting an error when uploading a file",
        "The dashboard has been loading forever",
        "Data sync stopped working since yesterday",
    ],
    "account": [
        "I cannot log in, password reset email never arrives",
        "My account got locked after failed attempts",
        "Please change the email on my profile",
        "I need to add a team member to my account",
    ],
    "product": [
        "Does the plan include custom reports",
        "Can this integrate with our CRM tool",
        "What is the storage limit on my plan",
        "Requesting a dark mode option",
    ],
}

URGENCY_TONE = {
    "high": ["URGENT: ", "This is blocking our work. ", "Please escalate immediately. "],
    "medium": ["Hi team, ", "Following up, ", ""],
    "low": ["No rush, but ", "Whenever you get a chance, "],
}

# Vague tickets a human would also struggle to classify confidently.
# Their category/urgency is assigned randomly EACH time they appear -
# an earlier version fixed the label per phrase, which accidentally made
# these "ambiguous" tickets perfectly learnable and defeated their purpose.
AMBIGUOUS_PHRASES = [
    "it is not working please help",
    "same issue as last time, please check",
    "can someone call me back about this",
    "nothing works, please fix",
]


def build_core_tickets(n=1500):
    categories = list(CATEGORY_PHRASES)
    urgencies = list(URGENCY_TONE)
    rows = []
    for _ in range(n):
        category = random.choice(categories)
        urgency = random.choice(urgencies)
        text = random.choice(URGENCY_TONE[urgency]) + random.choice(CATEGORY_PHRASES[category])
        rows.append((text, category, urgency))
    return rows


def build_ambiguous_tickets(n=150):
    categories = list(CATEGORY_PHRASES)
    urgencies = list(URGENCY_TONE)
    rows = []
    for _ in range(n):
        text = random.choice(AMBIGUOUS_PHRASES)
        category = random.choice(categories)
        urgency = random.choice(urgencies)
        rows.append((text, category, urgency))
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    rows = build_core_tickets(1500) + build_ambiguous_tickets(150)
    df = pd.DataFrame(rows, columns=["text", "category", "urgency"])
    df.insert(0, "ticket_id", [f"T{1000 + i}" for i in range(len(df))])
    df.to_csv("data/tickets.csv", index=False)
    print(f"Wrote {len(df)} tickets to data/tickets.csv")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
