"""
Train compact models and export them as JSON for the Android app.

Run AFTER train.py (it needs clean_data.csv.gz):
    python export_mobile.py

Creates:
    mobile_app/assets/ml/xgb_nutrition.json   (nutrition -> grade, XGBoost trees)
    mobile_app/assets/ml/text_model.json      (ingredients -> grade, TF-IDF + LogReg)
The app evaluates these files itself, so it works fully offline on the phone.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA = "clean_data.csv.gz"
OUT = Path("mobile_app/assets/ml")
SEED = 42
NUM = ["energy-kcal_100g", "sugars_100g", "saturated-fat_100g",
       "salt_100g", "fiber_100g", "proteins_100g"]
MIN_NUM = 4


def g9(x):                       # float32-safe, compact number
    return float(f"{x:.9g}")


# ----------------------------------------------------------------------------
# XGBoost -> JSON (+ a Python mirror of exactly what the app does)
# ----------------------------------------------------------------------------
def export_xgb(clf, path):
    raw = json.loads(bytes(clf.get_booster().save_raw("json")))
    model = raw["learner"]["gradient_booster"]["model"]
    n_class = clf.n_classes_
    keep = (clf.best_iteration + 1) * n_class if getattr(clf, "best_iteration", None) is not None \
        else len(model["trees"])
    trees = []
    for t in model["trees"][:keep]:
        trees.append({"l": t["left_children"], "r": t["right_children"],
                      "f": t["split_indices"], "c": [g9(v) for v in t["split_conditions"]],
                      "d": [int(v) for v in t["default_left"]]})
    obj = {"num_class": n_class, "features": NUM,
           "tree_class": model["tree_info"][:keep], "trees": trees}
    path.write_text(json.dumps(obj, separators=(",", ":")))
    return obj


def mirror_xgb(obj, x):
    """Same algorithm as lib/ml/xgb.dart."""
    f32 = lambda v: float(np.float32(v))
    margins = np.zeros(obj["num_class"])
    for t, cls in zip(obj["trees"], obj["tree_class"]):
        n = 0
        while t["l"][n] != -1:
            v = x[t["f"][n]]
            if v is None or np.isnan(v):
                n = t["l"][n] if t["d"][n] else t["r"][n]
            else:
                n = t["l"][n] if f32(v) < f32(t["c"][n]) else t["r"][n]
        margins[cls] += f32(t["c"][n])
    e = np.exp(margins - margins.max())
    return e / e.sum()


# ----------------------------------------------------------------------------
# TF-IDF + LogReg -> JSON (+ mirror of lib/ml/text_model.dart)
# ----------------------------------------------------------------------------
def export_text(tfidf, lr, path):
    vocab = {}
    for term, i in tfidf.vocabulary_.items():
        vocab[term] = [round(float(tfidf.idf_[i]), 5)] + [round(float(w), 5) for w in lr.coef_[:, i]]
    obj = {"bias": [round(float(b), 5) for b in lr.intercept_], "vocab": vocab}
    path.write_text(json.dumps(obj, separators=(",", ":")))
    return obj


def mirror_text(obj, text):
    toks = re.findall(r"[a-z]{2,}", text)
    terms = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    counts = {}
    for t in terms:
        counts[t] = counts.get(t, 0) + 1
    feats = {t: (1 + np.log(c)) * obj["vocab"][t][0] for t, c in counts.items() if t in obj["vocab"]}
    logits = np.array(obj["bias"], dtype=float)
    norm = np.sqrt(sum(v * v for v in feats.values()))
    if norm > 0:
        for t, v in feats.items():
            logits += v / norm * np.array(obj["vocab"][t][1:])
    e = np.exp(logits - logits.max())
    return e / e.sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["text"] = df["text"].fillna("")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["y"], random_state=SEED)

    # ---- nutrition model ------------------------------------------------------
    tr = train_df[train_df["has_num"]]
    te = test_df[test_df["has_num"]]
    fit, val = train_test_split(tr, test_size=0.1, stratify=tr["y"], random_state=SEED)
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.15, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", objective="multi:softprob",
                            eval_metric="mlogloss", early_stopping_rounds=20, n_jobs=-1,
                            random_state=SEED)
    clf.fit(fit[NUM], fit["y"], eval_set=[(val[NUM], val["y"])], verbose=False)
    pred = clf.predict(te[NUM])
    print(f"[nutrition] acc={accuracy_score(te['y'], pred):.3f} "
          f"macro-F1={f1_score(te['y'], pred, average='macro'):.3f}")
    xo = export_xgb(clf, OUT / "xgb_nutrition.json")

    sample = te[NUM].head(300)
    ref = clf.predict_proba(sample)
    mine = np.array([mirror_xgb(xo, [None if np.isnan(v) else v for v in row]) for row in sample.values])
    print(f"[nutrition] export check, max prob diff vs xgboost: {np.abs(ref - mine).max():.6f}")

    # ---- ingredients model ----------------------------------------------------
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20_000, sublinear_tf=True)
    X = tfidf.fit_transform(train_df["text"])
    lr = LogisticRegression(max_iter=300, C=4.0).fit(X, train_df["y"])
    pred = lr.predict(tfidf.transform(test_df["text"]))
    print(f"[ingredients] acc={accuracy_score(test_df['y'], pred):.3f} "
          f"macro-F1={f1_score(test_df['y'], pred, average='macro'):.3f}")
    to = export_text(tfidf, lr, OUT / "text_model.json")

    texts = test_df["text"].head(300).tolist()
    ref = lr.predict_proba(tfidf.transform(texts))
    mine = np.array([mirror_text(to, t) for t in texts])
    print(f"[ingredients] export check, max prob diff vs sklearn: {np.abs(ref - mine).max():.6f}")

    for f in OUT.glob("*.json"):
        print(f"{f}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
