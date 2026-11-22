"""Flask backend: serves the dashboard and a JSON API on top of the trained scikit-learn pipelines."""
import datetime as dt, io, json, sqlite3
from pathlib import Path
import joblib, numpy as np, pandas as pd
from flask import Flask, Response, jsonify, render_template, request

BASE = Path(__file__).parent
DATA, MODELS, DB = BASE / "data" / "StudentPerformanceFactors.csv", BASE / "models", BASE / "instance" / "history.db"
TARGET = "Exam_Score"

if not (MODELS / "metrics.json").exists():          # first run: train automatically
    import train_model; train_model.main()

METRICS = json.loads((MODELS / "metrics.json").read_text())
BEST = METRICS["best_model"]
MODEL_FILES = {"Linear Regression": "linear_regression.joblib", "Decision Tree Regressor": "decision_tree.joblib",
               "Random Forest Regressor": "random_forest.joblib"}
PIPES = {n: joblib.load(MODELS / f) for n, f in MODEL_FILES.items()}   # the real trained pipelines
df = pd.read_csv(DATA)
NUM, CAT = METRICS["dataset"]["numeric"], METRICS["dataset"]["categorical"]
FEATURES = NUM + CAT
ORDER = {"Parental_Involvement": ["Low", "Medium", "High"], "Access_to_Resources": ["Low", "Medium", "High"],
         "Motivation_Level": ["Low", "Medium", "High"], "Family_Income": ["Low", "Medium", "High"],
         "Teacher_Quality": ["Low", "Medium", "High"], "Peer_Influence": ["Negative", "Neutral", "Positive"],
         "Parental_Education_Level": ["High School", "College", "Postgraduate"],
         "Distance_from_Home": ["Near", "Moderate", "Far"]}
pretty = lambda c: c.replace("_", " ")

SCHEMA = {"numeric": [{"name": c, "label": pretty(c), "min": int(df[c].min()), "max": int(df[c].max()),
                       "default": int(df[c].median())} for c in NUM],
          "categorical": [{"name": c, "label": pretty(c),
                           "options": ORDER.get(c) or sorted(df[c].dropna().unique().tolist()),
                           "default": df[c].mode()[0]} for c in CAT]}


def build_eda():
    y = df[TARGET]
    edges = np.arange(55, 106, 5)
    counts, _ = np.histogram(y, bins=edges)
    sample = df.sample(400, random_state=42)
    scatter, trend = {}, {}
    for c in NUM:
        scatter[c] = [{"x": float(a), "y": float(b)} for a, b in zip(sample[c], sample[TARGET])]
        k, b0 = np.polyfit(df[c], y, 1)
        lo, hi = float(df[c].min()), float(df[c].max())
        trend[c] = [{"x": lo, "y": float(k * lo + b0)}, {"x": hi, "y": float(k * hi + b0)}]
    cats = {}
    for c in CAT:
        g = df.groupby(c)[TARGET].agg(["mean", "count"])
        cats[c] = [{"label": k, "mean": round(float(g.loc[k, "mean"]), 2), "count": int(g.loc[k, "count"])}
                   for k in (ORDER.get(c) or sorted(g.index))]
    corr = df[NUM + [TARGET]].corr().round(3)
    return {"hist": {"labels": [f"{a}-{b-1}" for a, b in zip(edges[:-1], edges[1:])], "counts": counts.tolist()},
            "stats": {"mean": round(y.mean(), 2), "median": float(y.median()), "std": round(y.std(), 2),
                      "min": float(y.min()), "max": float(y.max())},
            "corr_target": [[k, float(v)] for k, v in corr[TARGET].drop(TARGET).sort_values(ascending=False).items()],
            "corr_matrix": {"labels": corr.columns.tolist(), "values": corr.values.tolist()},
            "scatter": scatter, "trend": trend, "categories": cats,
            "missing": {c: int(v) for c, v in df.isna().sum().items() if v}}


EDA = build_eda()
app = Flask(__name__)


def db():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
                   model TEXT, prediction REAL, inputs TEXT)""")
    return con


def band(score):                       # same thresholds as the mid-term app (80 / 60)
    if score >= 80: return "Outstanding", "Study habits and support indicators point to a top-tier result."
    if score >= 60: return "Stable", "Current profile points to a healthy, consistent standing."
    return "Needs support", "More study hours, tutoring or resource access would likely raise the projected score."


def parse_input(p):
    row, errors = {}, []
    for f in SCHEMA["numeric"]:
        try: v = float(p.get(f["name"]))
        except (TypeError, ValueError): errors.append(f"{f['label']} must be a number"); continue
        if not f["min"] <= v <= f["max"]: errors.append(f"{f['label']} must be between {f['min']} and {f['max']} (training range)")
        row[f["name"]] = v
    for f in SCHEMA["categorical"]:
        v = p.get(f["name"])
        if v not in f["options"]: errors.append(f"{f['label']} must be one of {f['options']}")
        row[f["name"]] = v
    return row, errors


# Actionable levers only (income, gender etc. are not things an advisor can change)
NUM_LEVERS = [("Hours_Studied", 5, "Study 5 more hours per week"), ("Attendance", 10, "Raise attendance by 10 points"),
              ("Tutoring_Sessions", 2, "Add 2 tutoring sessions per month"), ("Physical_Activity", 1, "1 more hour of physical activity")]
CAT_LEVERS = [("Motivation_Level", "High", "Raise motivation to High"), ("Parental_Involvement", "High", "Increase parental involvement to High"),
              ("Access_to_Resources", "High", "Improve access to resources to High"), ("Teacher_Quality", "High", "Teacher quality High"),
              ("Internet_Access", "Yes", "Provide internet access")]


def whatif(row, base_raw):
    """Re-run the best pipeline with ONE factor changed at a time; delta = new prediction - current prediction."""
    ranges = {f["name"]: f for f in SCHEMA["numeric"]}
    scen = []
    for col, d, text in NUM_LEVERS:
        new = min(row[col] + d, ranges[col]["max"])
        if new != row[col]: scen.append((text, {**row, col: new}))
    for col, val, text in CAT_LEVERS:
        if row[col] != val: scen.append((text, {**row, col: val}))
    if not scen: return []
    out = PIPES[BEST].predict(pd.DataFrame([r for _, r in scen])[FEATURES])
    res = [{"change": t, "delta": round(float(p - base_raw), 2)} for (t, _), p in zip(scen, out)]
    return sorted([r for r in res if r["delta"] >= 0.05], key=lambda r: -r["delta"])[:5]


@app.get("/")
def index(): return render_template("index.html")


@app.get("/api/summary")
def summary():
    m = METRICS["models"][BEST]
    return jsonify({"dataset": METRICS["dataset"], "best_model": BEST, "best": m, "criterion": METRICS["selection_criterion"],
                    "eda_stats": EDA["stats"], "missing": EDA["missing"], "sklearn_version": METRICS["sklearn_version"]})

@app.get("/api/schema")
def schema(): return jsonify(SCHEMA)

@app.get("/api/eda")
def eda(): return jsonify(EDA)

@app.get("/api/models")
def models(): return jsonify({"models": METRICS["models"], "best_model": BEST, "criterion": METRICS["selection_criterion"]})

@app.get("/api/importance")
def importance(): return jsonify(METRICS["importance"])

@app.get("/api/actual-vs-predicted")
def avp(): return jsonify(METRICS["test_predictions"])


@app.post("/api/predict")
def predict():
    row, errors = parse_input(request.get_json(silent=True) or {})
    if errors: return jsonify({"error": "; ".join(errors)}), 400
    X = pd.DataFrame([row])[FEATURES]
    preds = {n: float(p.predict(X)[0]) for n, p in PIPES.items()}
    raw = preds[BEST]
    score = float(np.clip(raw, 0, 100))      # exam scores are bounded; raw model output is also returned
    label, msg = band(score)
    with db() as con:
        cur = con.execute("INSERT INTO history (created_at, model, prediction, inputs) VALUES (?,?,?,?)",
                          (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), BEST, round(score, 2), json.dumps(row)))
    return jsonify({"id": cur.lastrowid, "score": round(score, 2), "raw_score": round(raw, 2), "band": label, "message": msg,
                    "model": BEST, "margin": round(METRICS["models"][BEST]["test"]["rmse"], 2),
                    "low": round(float(np.clip(raw - METRICS["models"][BEST]["test"]["rmse"], 0, 100)), 1),
                    "high": round(float(np.clip(raw + METRICS["models"][BEST]["test"]["rmse"], 0, 100)), 1),
                    "whatif": whatif(row, raw),
                    "all_models": {n: round(float(np.clip(v, 0, 100)), 2) for n, v in preds.items()}})


@app.get("/api/history")
def history():
    with db() as con: rows = con.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    return jsonify([{"id": r["id"], "created_at": r["created_at"], "model": r["model"], "prediction": r["prediction"],
                     "inputs": json.loads(r["inputs"])} for r in rows])


@app.delete("/api/history")
def clear_history():
    with db() as con: con.execute("DELETE FROM history")
    return jsonify({"ok": True})


@app.get("/api/history/download")
def download():
    with db() as con: rows = con.execute("SELECT * FROM history ORDER BY id").fetchall()
    out = pd.DataFrame([{"id": r["id"], "created_at": r["created_at"], "model": r["model"],
                         "predicted_exam_score": r["prediction"], **json.loads(r["inputs"])} for r in rows],
                       columns=["id", "created_at", "model", "predicted_exam_score"] + FEATURES)
    buf = io.StringIO(); out.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction_history.csv"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
