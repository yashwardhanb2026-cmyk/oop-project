"""
Use the trained models to grade a product.   Run train.py first.

    from predict import predict_grade
    predict_grade(nutrition={"energy-kcal_100g": 520, "sugars_100g": 48,
                             "saturated-fat_100g": 11, "salt_100g": 0.2,
                             "fiber_100g": 1.5, "proteins_100g": 5})
    predict_grade(ingredients="sugar, palm oil, hazelnuts, cocoa, skimmed milk powder")
"""
import joblib
import numpy as np
import pandas as pd
import shap

from train import GRADES, MIN_NUM, NUM, clean_text

_text = joblib.load("models/text_model.joblib")
_xgb = joblib.load("models/xgb_nutrition.joblib")
_hyb = joblib.load("models/hybrid_model.joblib")
_explainer = shap.TreeExplainer(_xgb)
LETTERS = [g.upper() for g in GRADES]


def _reasons(row: pd.DataFrame, cls: int, top=3):
    """Top nutrients pushing the product toward its predicted grade."""
    sv = _explainer.shap_values(row)
    vals = sv[cls][0] if isinstance(sv, list) else sv[0, :, cls]
    idx = np.argsort(-np.abs(vals))[:top]
    return [f"{NUM[i]} = {row.iloc[0, i]:.1f} ({'pushes toward' if vals[i] > 0 else 'pushes away from'} {LETTERS[cls]})"
            for i in idx]


def predict_grade(nutrition: dict | None = None, ingredients: str | None = None) -> dict:
    nutrition = nutrition or {}
    row = pd.DataFrame([{c: nutrition.get(c, np.nan) for c in NUM}], dtype=float)
    has_num = row.notna().sum(axis=1).iloc[0] >= MIN_NUM
    text = clean_text(ingredients or "")
    has_text = len(text) >= 10

    if has_num and has_text:
        method = "hybrid (nutrition + ingredients)"
        X = np.hstack([row.values, _hyb["svd"].transform(_hyb["tfidf"].transform([text]))])
        proba = _hyb["xgb"].predict_proba(X)[0]
    elif has_num:
        method = "xgboost (nutrition)"
        proba = _xgb.predict_proba(row)[0]
    elif has_text:
        method = "tf-idf + logistic regression (ingredients only)"
        proba = _text["logreg"].predict_proba(_text["tfidf"].transform([text]))[0]
    else:
        raise ValueError("Give at least 4 nutrition values or an ingredients list.")

    cls = int(np.argmax(proba))
    return {
        "grade": LETTERS[cls],
        "confidence": round(float(proba[cls]), 3),
        "probabilities": {l: round(float(p), 3) for l, p in zip(LETTERS, proba)},
        "method": method,
        "reasons": _reasons(row, cls) if has_num else [],
    }


if __name__ == "__main__":
    print(predict_grade(nutrition={"energy-kcal_100g": 520, "sugars_100g": 48,
                                   "saturated-fat_100g": 11, "salt_100g": 0.2,
                                   "fiber_100g": 1.5, "proteins_100g": 5}))
    print(predict_grade(ingredients="oats, almonds, dried apricots, sunflower seeds"))
