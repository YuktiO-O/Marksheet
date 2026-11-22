"""
Train + evaluate the three models from the mid-term project and save everything
the web app needs. Run:  python train_model.py

Reused from mid-term (app.py in project7.ipynb):
  * Same preprocessing: median-impute + StandardScaler (numeric),
    most-frequent-impute + OneHotEncoder(handle_unknown='ignore') (categorical)
  * Same split: train_test_split(test_size=0.2, random_state=42)
  * Same Linear Regression model
Added for the final project: Decision Tree + Random Forest (same pipeline),
cross-validation, permutation importance, and saved evaluation artifacts.
"""
import json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
DATA, MODELS = BASE / "data" / "StudentPerformanceFactors.csv", BASE / "models"
TARGET, SEED = "Exam_Score", 42
FILES = {"Linear Regression": "linear_regression.joblib",
         "Decision Tree Regressor": "decision_tree.joblib",
         "Random Forest Regressor": "random_forest.joblib"}


def build_pipeline(regressor, num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)])
    return Pipeline([("preprocessor", pre), ("regressor", regressor)])


def gini_by_feature(pipe, cols):
    """Aggregate Random Forest impurity importances of one-hot columns back to original features."""
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pipe.named_steps["regressor"].feature_importances_
    out = dict.fromkeys(cols, 0.0)
    for n, v in zip(names, imp):
        n = n.split("__", 1)[1] if "__" in n else n
        match = max((c for c in cols if n == c or n.startswith(c + "_")), key=len)
        out[match] += float(v)
    return out


def main():
    df = pd.read_csv(DATA)
    X, y = df.drop(TARGET, axis=1), df[TARGET]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    regs = {"Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=SEED),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3,
                                                             random_state=SEED, n_jobs=-1)}
    kf = KFold(5, shuffle=True, random_state=SEED)
    res = {"dataset": {"rows": len(df), "features": X.shape[1], "n_train": len(X_tr), "n_test": len(X_te),
                       "test_size": 0.2, "random_state": SEED, "numeric": num_cols, "categorical": cat_cols},
           "models": {}, "importance": {"permutation": {}}, "test_predictions": {"actual": y_te.round(2).tolist()},
           "sklearn_version": sklearn.__version__}
    MODELS.mkdir(exist_ok=True)
    fitted = {}
    for name, reg in regs.items():
        pipe = build_pipeline(reg, num_cols, cat_cols)
        cv = cross_validate(pipe, X_tr, y_tr, cv=kf, n_jobs=-1,
                            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        res["models"][name] = {
            "test": {"mae": mean_absolute_error(y_te, pred),
                     "rmse": float(np.sqrt(mean_squared_error(y_te, pred))), "r2": r2_score(y_te, pred)},
            "train_r2": r2_score(y_tr, pipe.predict(X_tr)),
            "cv": {"rmse_mean": float(-cv["test_rmse"].mean()), "rmse_std": float(cv["test_rmse"].std()),
                   "r2_mean": float(cv["test_r2"].mean()), "r2_std": float(cv["test_r2"].std())},
            "params": {k: v for k, v in reg.get_params().items() if v is not None}}
        res["test_predictions"][name] = np.round(pred, 2).tolist()
        pi = permutation_importance(pipe, X_te, y_te, scoring="r2", n_repeats=10, random_state=SEED, n_jobs=-1)
        res["importance"]["permutation"][name] = {c: {"mean": float(m), "std": float(s)}
                                                  for c, m, s in zip(X.columns, pi.importances_mean, pi.importances_std)}
        if name.startswith("Random"):
            res["importance"]["rf_gini"] = gini_by_feature(pipe, X.columns.tolist())
        joblib.dump(pipe, MODELS / FILES[name])
        fitted[name] = pipe
        print(f"{name:26s} test R2={res['models'][name]['test']['r2']:.4f}  "
              f"RMSE={res['models'][name]['test']['rmse']:.3f}  CV RMSE={res['models'][name]['cv']['rmse_mean']:.3f}")

    # Model selection uses 5-fold CV RMSE on the TRAIN split only, so the test set stays untouched for final reporting.
    best = min(res["models"], key=lambda n: res["models"][n]["cv"]["rmse_mean"])
    res["best_model"] = best
    res["selection_criterion"] = "lowest 5-fold cross-validated RMSE on the training split"
    joblib.dump(fitted[best], MODELS / "best_model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(res))
    print("Best model:", best)


if __name__ == "__main__":
    main()
