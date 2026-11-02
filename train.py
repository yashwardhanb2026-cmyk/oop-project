"""
Food grade (Nutri-Score A-E) predictor trained on Open Food Facts.

Models compared (same test split):
  1. Baseline : TF-IDF (ingredients) + Logistic Regression
  2. XGBoost  : nutrition numbers only
  3. Hybrid   : nutrition numbers + TF-IDF/SVD of ingredients -> XGBoost

Run:  python train.py
Needs: pip install pandas numpy scikit-learn xgboost shap matplotlib joblib scipy
Data : en.openfoodfacts.org.products.csv.gz  (tab-separated, from world.openfoodfacts.org/data)
"""
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split

CSV_PATH = "en.openfoodfacts.org.products.csv.gz"
SAMPLE = 400_000          # max rows to use (None = everything). Lower it if RAM is tight.
SEED = 42
OUT = Path("models")
OUT.mkdir(exist_ok=True)

GRADES = list("abcde")
NUM = ["energy-kcal_100g", "sugars_100g", "saturated-fat_100g",
       "salt_100g", "fiber_100g", "proteins_100g"]
MIN_NUM = 4               # a product "has nutrition" if >= 4 of the 6 values exist


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d+([.,]\d+)?\s*%?", " ", s)      # drop numbers and percentages
    s = re.sub(r"[^a-z, ]", " ", s)                # keep letters and commas
    return re.sub(r"\s+", " ", s).strip()


def mostly_ascii(s: str) -> bool:
    return len(s) > 0 and sum(c.isascii() for c in s) / len(s) > 0.95


def load_data() -> pd.DataFrame:
    wanted = ["code", "product_name", "ingredients_text", "nutriscore_grade"] + NUM
    header = pd.read_csv(CSV_PATH, sep="\t", nrows=0).columns
    cols = [c for c in wanted if c in header]
    df = pd.read_csv(CSV_PATH, sep="\t", usecols=cols, low_memory=False,
                     on_bad_lines="skip")

    df = df[df["nutriscore_grade"].isin(GRADES)].drop_duplicates("code")
    df["ingredients_text"] = df["ingredients_text"].fillna("")
    df = df[df["ingredients_text"].map(mostly_ascii)]           # English/latin only (rough)
    df["text"] = df["ingredients_text"].map(clean_text)
    df = df[df["text"].str.len() >= 10]

    # remove impossible values
    for c in NUM:
        if c not in df:
            df[c] = np.nan
    df.loc[~df["energy-kcal_100g"].between(0, 900), "energy-kcal_100g"] = np.nan
    for c in NUM[1:]:
        df.loc[~df[c].between(0, 100), c] = np.nan

    df["has_num"] = df[NUM].notna().sum(axis=1) >= MIN_NUM
    df["y"] = df["nutriscore_grade"].map({g: i for i, g in enumerate(GRADES)})

    if SAMPLE and len(df) > SAMPLE:
        df = df.sample(SAMPLE, random_state=SEED)
    df = df.reset_index(drop=True)
    print(f"Rows: {len(df):,} | with nutrition: {df['has_num'].mean():.1%}")
    print(df["nutriscore_grade"].value_counts(normalize=True).sort_index().round(3).to_dict())
    return df


# ----------------------------------------------------------------------------
# 2. Metrics
# ----------------------------------------------------------------------------
def metrics(name, y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "within_1_grade": float(np.mean(np.abs(y_true - y_pred) <= 1))}


def xgb_model(**kw):
    return xgb.XGBClassifier(
        n_estimators=600, max_depth=7, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", objective="multi:softprob",
        eval_metric="mlogloss", early_stopping_rounds=30, n_jobs=-1,
        random_state=SEED, **kw)


# ----------------------------------------------------------------------------
# 3. Train + compare
# ----------------------------------------------------------------------------
def main():
    df = load_data()
    df.to_csv("clean_data.csv.gz", index=False)     # reused by distilbert.py

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["y"],
                                         random_state=SEED)
    tr_num = train_df[train_df["has_num"]]
    te_num = test_df[test_df["has_num"]]
    tr_fit, tr_val = train_test_split(tr_num, test_size=0.1, stratify=tr_num["y"],
                                      random_state=SEED)
    results = []

    # --- 1) Baseline: TF-IDF + Logistic Regression (ingredients only) ---------
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=100_000,
                            sublinear_tf=True)
    Xtr_t = tfidf.fit_transform(train_df["text"])
    logreg = LogisticRegression(max_iter=300, C=4.0)
    logreg.fit(Xtr_t, train_df["y"])
    p_all = logreg.predict(tfidf.transform(test_df["text"]))
    p_num = logreg.predict(tfidf.transform(te_num["text"]))
    results.append(metrics("Baseline TF-IDF+LogReg (all test rows)", test_df["y"], p_all))
    results.append(metrics("Baseline TF-IDF+LogReg", te_num["y"], p_num))

    # --- 2) XGBoost on nutrition numbers --------------------------------------
    xgb_num = xgb_model()
    xgb_num.fit(tr_fit[NUM], tr_fit["y"], eval_set=[(tr_val[NUM], tr_val["y"])],
                verbose=False)
    p_xgb = xgb_num.predict(te_num[NUM])
    results.append(metrics("XGBoost (nutrition)", te_num["y"], p_xgb))

    # --- 3) Hybrid: nutrition + ingredient SVD features -----------------------
    tfidf_small = TfidfVectorizer(min_df=5, max_features=20_000, sublinear_tf=True)
    svd = TruncatedSVD(n_components=100, random_state=SEED)
    svd.fit(tfidf_small.fit_transform(train_df["text"]))

    def hybrid_X(d):
        return np.hstack([d[NUM].values, svd.transform(tfidf_small.transform(d["text"]))])

    xgb_hyb = xgb_model()
    xgb_hyb.fit(hybrid_X(tr_fit), tr_fit["y"],
                eval_set=[(hybrid_X(tr_val), tr_val["y"])], verbose=False)
    p_hyb = xgb_hyb.predict(hybrid_X(te_num))
    results.append(metrics("Hybrid XGBoost (nutrition + ingredients)", te_num["y"], p_hyb))

    # --- Results table ---------------------------------------------------------
    table = pd.DataFrame(results).round(4)
    print("\n", table.to_string(index=False))
    table.to_csv("results.csv", index=False)

    best_name, best_pred = max(
        [("Baseline", p_num), ("XGBoost", p_xgb), ("Hybrid", p_hyb)],
        key=lambda t: f1_score(te_num["y"], t[1], average="macro"))
    print(f"\nBest model: {best_name}\n")
    print(classification_report(te_num["y"], best_pred, target_names=[g.upper() for g in GRADES]))

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        te_num["y"], best_pred, display_labels=[g.upper() for g in GRADES],
        normalize="true", cmap="Blues", ax=ax, values_format=".2f")
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)

    # --- XAI: SHAP on the nutrition model --------------------------------------
    sample = te_num[NUM].sample(min(2000, len(te_num)), random_state=0)
    sv = shap.TreeExplainer(xgb_num).shap_values(sample)
    if isinstance(sv, list):                       # older shap: list of (n, f) per class
        importance = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
    else:                                          # newer shap: (n, f, classes)
        importance = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(np.array(NUM)[order], importance[order], color="#0b6e4f")
    ax.set_xlabel("Mean |SHAP value| (impact on grade)")
    ax.set_title("Which nutrients drive the grade?")
    fig.tight_layout()
    fig.savefig("shap_importance.png", dpi=150)

    # --- Save everything for predict.py ---------------------------------------
    joblib.dump({"tfidf": tfidf, "logreg": logreg}, OUT / "text_model.joblib")
    joblib.dump(xgb_num, OUT / "xgb_nutrition.joblib")
    joblib.dump({"tfidf": tfidf_small, "svd": svd, "xgb": xgb_hyb}, OUT / "hybrid_model.joblib")
    print("\nSaved: models/, results.csv, confusion_matrix.png, shap_importance.png")


if __name__ == "__main__":
    main()
