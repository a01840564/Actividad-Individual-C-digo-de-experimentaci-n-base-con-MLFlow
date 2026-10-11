"""Entrenamiento reproducible del baseline Titanic con MLflow Tracking."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess

# Fijar el paralelismo antes de importar bibliotecas numéricas.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "tmp" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings

from preprocessing import FareCapper

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "TC5061-Titanic-Experimentacion-Logistica"
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
METRICS = ["accuracy_test", "f1_test", "roc_auc_test", "precision_test", "recall_test"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "train.csv")
    parser.add_argument("--C", type=float, default=1.0, help="Inverso de regularización; > 0.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--solver", choices=["lbfgs", "liblinear"], default="lbfgs")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--fare-cap-quantile", type=float, default=0.99)
    parser.add_argument("--experiment-name", default=EXPERIMENT)
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()
    if args.C <= 0 or not np.isfinite(args.C):
        parser.error("--C debe ser finito y mayor que cero")
    if args.max_iter <= 0 or not 0 < args.test_size < 1:
        parser.error("--max-iter > 0 y --test-size en (0, 1)")
    if not 0 < args.fare_cap_quantile <= 1 or not 0 <= args.seed < 2**32:
        parser.error("--fare-cap-quantile en (0, 1] y --seed entre 0 y 2**32-1")
    return args


def load_data(path):
    data = pd.read_csv(path)
    required = NUMERIC + CATEGORICAL + ["Survived", "PassengerId"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    if data.empty or data["Survived"].isna().any() or set(data["Survived"]) != {0, 1}:
        raise ValueError("Survived debe contener ambas clases 0 y 1, sin nulos.")
    if data["PassengerId"].isna().any() or data["PassengerId"].duplicated().any():
        raise ValueError("PassengerId debe ser único y no nulo.")
    data[NUMERIC] = data[NUMERIC].astype(float)
    return data.drop_duplicates()


def build_pipeline(args):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric, NUMERIC),
                                     ("cat", categorical, CATEGORICAL)])
    classifier = LogisticRegression(C=args.C, solver=args.solver,
                                    class_weight=None if args.class_weight == "none" else "balanced",
                                    max_iter=args.max_iter, random_state=args.seed)
    return Pipeline([("fare_cap", FareCapper(args.fare_cap_quantile)),
                     ("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate(model, X_test, y_test):
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = dict(zip(METRICS, [accuracy_score(y_test, predicted),
                                f1_score(y_test, predicted, zero_division=0),
                                roc_auc_score(y_test, probabilities),
                                precision_score(y_test, predicted, zero_division=0),
                                recall_score(y_test, predicted, zero_division=0)]))
    return metrics, predicted, probabilities


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def provenance(data_path):
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ["train.py", "preprocessing.py", "requirements.txt"]}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "not-initialized", True
    return {"dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "source_sha256": hashes, "git_commit": commit, "git_dirty": dirty,
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {p: importlib.metadata.version(p)
                             for p in ["mlflow", "scikit-learn", "numpy", "pandas", "matplotlib", "skops", "scipy"]}}


def save_artifacts(directory, model, data, X_train, X_test, y_test,
                   metrics, predicted, probabilities, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "provenance.json", manifest)
    write_json(directory / "split.json", {
        "train_passenger_ids": data.loc[X_train.index, "PassengerId"].tolist(),
        "test_passenger_ids": data.loc[X_test.index, "PassengerId"].tolist()})
    write_json(directory / "classification_report.json",
               classification_report(y_test, predicted, output_dict=True, zero_division=0))
    pd.DataFrame({"PassengerId": data.loc[X_test.index, "PassengerId"],
                  "actual": y_test, "predicted": predicted,
                  "probability_survived": probabilities}).to_csv(directory / "predictions.csv", index=False)
    display = ConfusionMatrixDisplay.from_predictions(y_test, predicted,
                                                      display_labels=["No sobrevivió", "Sobrevivió"],
                                                      cmap="Blues", colorbar=False)
    display.ax_.set_title("Titanic - Matriz de confusión (prueba)")
    display.ax_.set_xlabel("Predicción")
    display.ax_.set_ylabel("Etiqueta real")
    display.figure_.tight_layout()
    display.figure_.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)
    display = RocCurveDisplay.from_predictions(y_test, probabilities)
    display.ax_.set_title("Titanic - Curva ROC (prueba)")
    display.figure_.tight_layout()
    display.figure_.savefig(directory / "roc_curve.png", dpi=160)
    plt.close(display.figure_)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_data(args.data_path)
    X = data[NUMERIC + CATEGORICAL]
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    manifest = provenance(args.data_path)
    params = {"C": args.C, "class_weight": args.class_weight, "solver": args.solver,
              "max_iter": args.max_iter, "seed": args.seed, "test_size": args.test_size,
              "fare_cap_quantile": args.fare_cap_quantile, "model": "LogisticRegression",
              "regularization": "L2", "decision_threshold": 0.5}
    with mlflow.start_run(run_name=args.run_name) as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.set_tags({"dataset_sha256": manifest["dataset_sha256"],
                         "source_sha256": json.dumps(manifest["source_sha256"], sort_keys=True),
                         "git_commit": manifest["git_commit"], "git_dirty": str(manifest["git_dirty"]),
                         "selection_metric": "f1_test"})
        model = build_pipeline(args)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train, y_train)
        metrics, predicted, probabilities = evaluate(model, X_test, y_test)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_param("fare_cap_learned", model.named_steps["fare_cap"].cap_)
        mlflow.log_metric("train_size_rows", len(X_train))
        mlflow.log_metric("test_size_rows", len(X_test))
        example = X_train.head(5)
        info = mlflow.sklearn.log_model(
            model, name="model", signature=infer_signature(example, model.predict(example)),
            input_example=example, code_paths=[str(ROOT / "preprocessing.py")],
            serialization_format="skops", skops_trusted_types=["numpy.dtype", "preprocessing.FareCapper"],
            pip_requirements=[f"{p}=={v}" for p, v in manifest["dependencies"].items()])
        mlflow.set_tag("model_uri", info.model_uri)
        manifest.update({"run_id": run.info.run_id, "run_name": args.run_name,
                         "model_uri": info.model_uri, "params": params})
        directory = args.output_dir / run.info.run_id
        save_artifacts(directory, model, data, X_train, X_test, y_test,
                       metrics, predicted, probabilities, manifest)
        for file in sorted(directory.iterdir()):
            mlflow.log_artifact(str(file), artifact_path="evaluation")
        for name in ["train.py", "preprocessing.py", "requirements.txt"]:
            mlflow.log_artifact(str(ROOT / name), artifact_path="source")
        print(json.dumps({"run_id": run.info.run_id, "model_uri": info.model_uri,
                          "params": params, "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
