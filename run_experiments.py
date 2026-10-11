"""Seis configuraciones controladas, réplica y selección documentada por F1."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from train import ROOT, EXPERIMENT, METRICS, load_data, write_json

CONFIGURATIONS = [
    ("01_baseline", 1.0, "none", "lbfgs"),
    ("02_C_0.01", 0.01, "none", "lbfgs"),
    ("03_C_0.1", 0.1, "none", "lbfgs"),
    ("04_C_10", 10.0, "none", "lbfgs"),
    ("05_balanced", 1.0, "balanced", "lbfgs"),
    ("06_liblinear", 1.0, "none", "liblinear"),
    ("07_baseline_replica", 1.0, "none", "lbfgs"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=EXPERIMENT)
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    logs = ROOT / "tmp" / "experiment_logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows, commands = [], []
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    for name, C, weight, solver in CONFIGURATIONS:
        command = [sys.executable, str(ROOT / "train.py"), "--C", str(C),
                   "--class-weight", weight, "--solver", solver, "--seed", "42",
                   "--run-name", name, "--experiment-name", args.experiment_name,
                   "--tracking-uri", args.tracking_uri]
        display = f"python train.py --C {C} --class-weight {weight} --solver {solver} --seed 42 --run-name {name}"
        print(display, flush=True)
        commands.append(display)
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (logs / f"{name}.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"Falló {name}: {completed.stderr}")
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        rows.append({"run_name": name, "run_id": payload["run_id"],
                     "C": C, "class_weight": weight, "solver": solver, "seed": 42,
                     "model_uri": payload["model_uri"], **payload["metrics"]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(results / "comparison.csv", index=False)
    (results / "commands.txt").write_text("\n".join(commands) + "\n")
    first, replica = rows[0], rows[-1]
    directories = [results / "runs" / r["run_id"] for r in [first, replica]]
    assert all(first[m] == replica[m] for m in METRICS), "Las métricas de la réplica difieren"
    for name in ["predictions.csv", "split.json"]:
        assert (directories[0] / name).read_bytes() == (directories[1] / name).read_bytes(), name
    # La réplica prueba reproducibilidad; se excluye de la selección de configuración.
    best = sorted(rows[:-1], key=lambda r: (-r["f1_test"], -r["roc_auc_test"], r["run_name"]))[0]
    model = mlflow.sklearn.load_model(best["model_uri"])
    predictions = pd.read_csv(results / "runs" / best["run_id"] / "predictions.csv")
    data = load_data(ROOT / "train.csv").set_index("PassengerId")
    X_test = data.loc[predictions["PassengerId"], ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]]
    assert np.array_equal(model.predict(X_test), predictions["predicted"].to_numpy())
    np.testing.assert_allclose(model.predict_proba(X_test)[:, 1], predictions["probability_survived"], rtol=0, atol=1e-15)
    split = json.loads((results / "runs" / best["run_id"] / "split.json").read_text())
    learned_cap = data.loc[split["train_passenger_ids"], "Fare"].quantile(0.99)
    assert model.named_steps["fare_cap"].cap_ == learned_cap, "El percentil no corresponde al entrenamiento"
    registry_name = "TC5061-Titanic-RegresionLogistica"
    version = mlflow.register_model(best["model_uri"], registry_name)
    client.set_registered_model_alias(registry_name, "candidate", version.version)
    client.set_model_version_tag(registry_name, version.version, "selection_metric", "f1_test")
    client.set_tag(best["run_id"], "selected_model", "true")
    client.set_tag(replica["run_id"], "purpose", "reproducibility_check")
    selected = results / "selected_run"
    selected.mkdir(exist_ok=True)
    for name in ["metrics.json", "classification_report.json", "confusion_matrix.png", "roc_curve.png", "provenance.json", "split.json", "predictions.csv"]:
        shutil.copy2(results / "runs" / best["run_id"] / name, selected / name)
    verification = {"baseline_run_id": first["run_id"], "replica_run_id": replica["run_id"],
                    "exact_metrics_match": True, "exact_predictions_csv_match": True,
                    "exact_split_match": True, "model_reload_predictions_match": True,
                    "fare_cap_fit_on_train_only": True,
                    "baseline_metrics": {m: first[m] for m in METRICS},
                    "replica_metrics": {m: replica[m] for m in METRICS}}
    write_json(results / "reproducibility.json", verification)
    write_json(results / "experiment_summary.json", {
        "experiment_name": args.experiment_name, "experiment_id": client.get_run(best["run_id"]).info.experiment_id,
        "runs": rows, "selection_metric": "f1_test", "tie_breaker": "roc_auc_test, run_name",
        "best": best, "registry_name": registry_name, "registry_version": version.version,
        "registry_alias": "candidate", "verification": verification})
    print(comparison[["run_name", "C", "class_weight", "solver"] + METRICS].to_string(index=False))
    print(f"Mejor por F1: {best['run_name']}; versión {version.version}; reproducibilidad verificada.")


if __name__ == "__main__":
    main()
