# Food Grade Scanner (Nutri-Score A–E predictor)

Scan or photograph a food label and get an **A–E grade** (A = best, E = worst), like the Nutri-Score system used in Europe.
Machine-learning models trained on the [Open Food Facts](https://world.openfoodfacts.org/data) database predict the grade for products that have no official grade.
Everything runs **on your own device**, no internet needed after setup.

## How it works

1. **OCR** (RapidOCR) reads the nutrition table and ingredients from a photo of the label.
2. You can **correct** any value the OCR got wrong.
3. A trained model predicts the grade:
   - nutrition values available → **XGBoost** (or hybrid XGBoost + ingredients)
   - only ingredients available → **TF-IDF + Logistic Regression**
4. **SHAP** shows which nutrients pushed the grade up or down.

Models compared in `train.py`: TF-IDF + Logistic Regression (baseline), XGBoost (nutrition), Hybrid XGBoost (nutrition + ingredients). `distilbert.py` adds a fine-tuned DistilBERT on ingredients only.

## Project structure

| File | What it does |
|---|---|
| `train.py` | Loads Open Food Facts, trains and compares the models, saves results, plots and models |
| `predict.py` | `predict_grade()` function that loads the saved models and grades a product |
| `app.py` | Desktop GUI (photo or webcam → OCR → grade) |
| `distilbert.py` | Fine-tunes DistilBERT on ingredient text (run on Google Colab GPU) |
| `web/index.html` | Simple web page: type a barcode, get the official grade from Open Food Facts |
| `requirements.txt` | Python dependencies |

## Install

Needs **Python 3.10 or newer**.

```bash
git clone https://github.com/yashwardhanb2026-cmyk/oop-project.git
cd oop-project

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Linux only: if the GUI says Tkinter is missing, run `sudo apt install python3-tk`.

## Step 1: Get the data

Download the Open Food Facts CSV (a few GB) and put it in the project folder, **without unzipping**:

https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz

The file name must be `en.openfoodfacts.org.products.csv.gz`.

## Step 2: Train the models

```bash
python train.py
```

This creates:
- `models/` the saved models (used by the app)
- `results.csv` the model comparison table
- `confusion_matrix.png` and `shap_importance.png`
- `clean_data.csv.gz` the cleaned dataset (used by `distilbert.py`)

Short on RAM? Lower `SAMPLE` at the top of `train.py` (for example `100_000`).

## Step 3: Run the app

```bash
python app.py
```

1. Click **Open photo** (or **Capture from webcam**, press SPACE to take the picture).
2. Check the values the OCR found and fix any mistakes.
3. Click **Get grade**.

Test the model without the GUI:

```bash
python predict.py
```

## Optional: DistilBERT (ingredients only)

Upload `distilbert.py` and `clean_data.csv.gz` to Google Colab with a GPU, then:

```bash
pip install transformers accelerate scikit-learn pandas torch
python distilbert.py
```

## Optional: web page

Open `web/index.html` through a local server (the camera needs localhost or https):

```bash
cd web
python -m http.server 8000
```

Then open http://localhost:8000. It looks up the official Nutri-Score by barcode and needs internet.

## Limitations

- Ingredient lists have no quantities, so the ingredients-only model is less exact (neighbouring grades like B and C get confused). The nutrition model is much more accurate.
- Nutri-Score was designed for European diets and may grade some Indian foods (ghee, dal, paneer) unfairly.
- Labels often list values per serving and per 100 g. OCR takes the first number it finds, so always check the values.
- Grades are predictions, not official Nutri-Score ratings.

## Data

[Open Food Facts](https://world.openfoodfacts.org/), licensed under the Open Database License (ODbL).
