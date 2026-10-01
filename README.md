# AI Customer Support Ticket Triage

Predicts a support ticket's **category** (billing / technical / account / product)
and **urgency** (high / medium / low), and routes low-confidence predictions
to a human review queue instead of guessing.

## Project structure

```
generate_dataset.py       # builds the synthetic ticket dataset
train_triage_models.py    # trains both classifiers, evaluates, saves models
triage.py                 # live prediction tool (CLI)
data/tickets.csv          # generated dataset (1,650 tickets)
models/                   # trained classifiers
reports/                  # evaluation results, review queue, project report
```

## Run it

```bash
pip install pandas scikit-learn joblib
python generate_dataset.py
python train_triage_models.py
python triage.py "My payment failed and I need this fixed immediately"
```

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Category classifier | 95.5% | 0.955 |
| Urgency classifier | 94.8% | 0.949 |

Full writeup: `reports/Ticket_Triage_Project_Report.docx`

## Why the dataset is synthetic

No historical ticket data was available, so the dataset is generated
programmatically. An earlier version of the generator produced a dataset
that scored 100% accuracy — investigated and traced to a labelling bug
(details in the report). The corrected dataset scores in the low-to-mid
90s, which is the realistic, defensible result reported here.
