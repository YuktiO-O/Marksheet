# Student Performance Prediction – Interactive ML Dashboard (SML Final Project)

Upgrade of the mid-term Streamlit project into a Flask + Chart.js/Bootstrap web application.
Predictions always come from the trained scikit-learn pipelines in `models/`; every chart is computed from the real dataset
or from real test-set predictions. Nothing is hard-coded or simulated.

## Setup (5 minutes)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py             # optional: models are already included; this retrains and rewrites them
python app.py                     # open http://127.0.0.1:5000
```
The page loads Bootstrap, Chart.js and Google Fonts from CDNs, so the browser needs internet access. If `models/metrics.json` is missing, `app.py` trains automatically on first start.

## Folder structure
```
app.py                  Flask backend + JSON API, history (SQLite), CSV export
train_model.py          Trains/evaluates the 3 models, permutation importance, saves artifacts
data/StudentPerformanceFactors.csv
models/                 linear_regression / decision_tree / random_forest / best_model .joblib + metrics.json
templates/index.html    Dashboard page      static/css/style.css   static/js/app.js (all charts + form logic)
instance/history.db     Created at runtime (prediction history)
midterm_project7.ipynb  Your original mid-term notebook (unchanged)
```

## Features → where they live
| Feature | Frontend | API |
|---|---|---|
| KPIs, score distribution, correlations | Overview | `/api/summary`, `/api/eda` |
| EDA: scatter + trend, category means, correlation matrix, missing values | Data explorer | `/api/eda` |
| MAE / RMSE / R² table and charts, CV scores | Models | `/api/models` |
| Feature importance (permutation for all 3 models + RF impurity) | Models | `/api/importance` |
| Actual vs predicted + residuals (selectable model) | Models | `/api/actual-vs-predicted` |
| Prediction form → instant result, all-model comparison | Predict | `POST /api/predict` |
| History table, CSV download, clear | History | `/api/history`, `/api/history/download` |

## What was reused vs. changed (be upfront about this in the viva)
**Reused exactly from your mid-term `app.py`:** the preprocessing (median imputer + StandardScaler for numeric; most-frequent imputer + OneHotEncoder for categorical, in a `ColumnTransformer`), the `Pipeline` structure, `train_test_split(test_size=0.2, random_state=42)`, `LinearRegression`, the form fields/options, and the 80/60 score bands.

**Important:** the uploaded notebook only contained the Streamlit app with Linear Regression, and no Decision Tree / Random Forest code and no saved `.joblib` file. So all three models were (re)trained from your pipeline and split in `train_model.py`. Linear Regression is unchanged; the other two use these settings (my choice, change them if your mid-term used others):
* Decision Tree: `max_depth=6, min_samples_leaf=10` (limited depth to avoid heavy overfitting)
* Random Forest: `n_estimators=200, max_depth=12, min_samples_leaf=3, n_jobs=-1`

**Small changes:** column types are selected with `select_dtypes(include="number")` (robust across pandas versions); history is saved in SQLite; predictions are clipped to 0–100 (the raw model output is returned too); inputs are validated against the training ranges.

## Results (test set, 1,322 unseen students)
| Model | MAE | RMSE | R² | CV RMSE (5-fold) |
|---|---|---|---|---|
| **Linear Regression (best)** | 0.45 | 1.80 | 0.770 | 2.09 |
| Random Forest | 1.06 | 2.11 | 0.686 | 2.39 |
| Decision Tree | 1.60 | 2.56 | 0.537 | 2.79 |

(Exact values for all columns are in the Models section and `models/metrics.json`.)

## Viva cheat-sheet
* **Problem type:** supervised regression; target `Exam_Score`, 19 features (7 numeric, 12 categorical).
* **Why a Pipeline?** Imputation, scaling and encoding are fitted on training data only, so there is no data leakage, and the same object is used in training and in the web app.
* **Why is the best model chosen by CV and not by test score?** Choosing on the test set would leak information. The 20% test set is touched only for final reporting.
* **Why does Linear Regression beat Random Forest?** The relationship here is close to additive and linear (attendance, hours studied, previous scores). A linear model captures that with low variance, while the trees add variance without gaining anything. Train vs test R² in the table shows how much each model overfits.
* **MAE / RMSE / R²:** MAE = average absolute error in marks; RMSE = same but penalises big errors more; R² = share of score variance explained.
* **Why is permutation importance used?** It is model-agnostic (works for LR, DT and RF) and measured on unseen data. Impurity importance exists only for trees and is biased toward high-cardinality features.
* **Limitations:** the dataset looks synthetic/educational; a few students with very high scores (>80) are under-predicted by all models (visible in Actual vs Predicted); the model shows association, not causation; inputs outside the training ranges are rejected.

## Final-version additions
* **Dark mode UI** (Bootstrap dark theme + custom CSS, single amber accent).
* **What-if suggestions:** after each prediction the server re-runs the best pipeline with ONE actionable factor changed
  (e.g. +5 study hours, +10 attendance, motivation → High) and reports the real change in predicted score (`whatif()` in `app.py`).
  Non-actionable factors (income, gender, etc.) are deliberately excluded. These are model associations, not causal guarantees.
* **Likely range:** prediction ± test RMSE of the best model (clipped to 0–100).

## Google Colab (one cell)
Upload `student-performance-app.zip` when asked, then run:
```python
import os, re, sys, time, glob, zipfile, subprocess
if not glob.glob("student-performance-app*.zip") and not os.path.exists("student-performance-app/app.py"):
    from google.colab import files; files.upload()
z = glob.glob("student-performance-app*.zip")
if z and not os.path.exists("student-performance-app/app.py"): zipfile.ZipFile(z[0]).extractall(".")
os.chdir("student-performance-app")
subprocess.run([sys.executable,"-m","pip","install","-q","-r","requirements.txt"],check=True)
subprocess.run([sys.executable,"train_model.py"],check=True)
subprocess.Popen([sys.executable,"app.py"],stdout=open("flask.log","w"),stderr=subprocess.STDOUT); time.sleep(6)
subprocess.run("curl -sL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared",shell=True,check=True)
subprocess.Popen(["./cloudflared","tunnel","--url","http://localhost:5000"],stdout=open("tunnel.log","w"),stderr=subprocess.STDOUT)
for _ in range(40):
    time.sleep(1); m=re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",open("tunnel.log").read())
    if m: print("OPEN:",m.group(0)); break
```
