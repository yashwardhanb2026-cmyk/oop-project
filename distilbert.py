"""
Fine-tune DistilBERT: ingredient text -> Nutri-Score grade (A-E).
Run on Google Colab with a GPU (Runtime > Change runtime type > T4).

Steps:
  1. Run train.py first (it creates clean_data.csv.gz), upload that file to Colab.
  2. pip install transformers accelerate scikit-learn pandas torch
  3. python distilbert.py
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL = "distilbert-base-uncased"
MAX_LEN = 128
TRAIN_N, TEST_N = 100_000, 20_000     # subsample so training takes ~20-30 min on a T4
SEED = 42

df = pd.read_csv("clean_data.csv.gz", usecols=["text", "y"])
# same split as train.py
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["y"], random_state=SEED)
train_df = train_df.sample(min(TRAIN_N, len(train_df)), random_state=SEED)
test_df = test_df.sample(min(TEST_N, len(test_df)), random_state=SEED)

tok = AutoTokenizer.from_pretrained(MODEL)


class DS(torch.utils.data.Dataset):
    def __init__(self, d):
        self.enc = tok(d["text"].tolist(), truncation=True, max_length=MAX_LEN,
                       padding="max_length")
        self.y = d["y"].tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.y[i])
        return item


def compute_metrics(p):
    pred = p.predictions.argmax(-1)
    y = p.label_ids
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "within_1_grade": float(np.mean(np.abs(y - pred) <= 1))}


model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5)  # 5 grades

args = TrainingArguments(
    output_dir="distilbert_out",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
)

trainer = Trainer(model=model, args=args, train_dataset=DS(train_df),
                  eval_dataset=DS(test_df), compute_metrics=compute_metrics)
trainer.train()
print(trainer.evaluate())

trainer.save_model("distilbert_grade_model")
tok.save_pretrained("distilbert_grade_model")


def predict(text: str):
    """Grade one ingredient list with the fine-tuned model."""
    model.eval()
    enc = tok(text.lower(), return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, -1)[0].cpu().numpy()
    return "ABCDE"[int(probs.argmax())], probs.round(3)


print(predict("sugar, palm oil, hazelnuts, cocoa, skimmed milk powder"))
