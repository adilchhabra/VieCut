#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


FEATURE_COLUMNS = [
    "feature_n",
    "feature_m",
    "feature_avg_degree",
    "feature_density",
    "feature_min_degree",
    "feature_degree_p50",
    "feature_degree_p90",
    "feature_degree_p99",
    "feature_degree_ratio_p99_p50",
    "feature_degree_ratio_max_p90",
    "feature_degree_tail_hill_alpha",
    "feature_degree_stddev",
    "feature_degree_cv",
    "feature_max_degree",
    "feature_leaf_fraction",
    "feature_isolated_fraction",
    "feature_kcore_max",
    "feature_kcore_mean",
    "feature_kcore_top_fraction",
    "feature_kcore_ge_2_fraction",
    "feature_kcore_ge_4_fraction",
    "feature_kcore_ge_8_fraction",
    "feature_component_count",
    "feature_largest_component_fraction",
    "feature_second_component_fraction",
    "feature_clustering_sampled_mean",
    "feature_clustering_samples_used",
    "feature_transitivity_sampled",
    "feature_degree_assortativity_sampled",
    "feature_bfs_mean_distance",
    "feature_bfs_p90_distance",
    "feature_bfs_diameter_proxy",
    "feature_bfs_reachable_fraction",
]


@dataclass
class Dataset:
    x: np.ndarray
    y_class: np.ndarray
    y_preset: np.ndarray
    groups: np.ndarray
    feature_names: List[str]


def to_float(v: str) -> float:
    if v is None:
        return 0.0
    v = v.strip()
    if v == "":
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def load_preset_map(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            graph_class = (row.get("graph_class") or "").strip()
            preset = (row.get("recommended_config") or "").strip()
            if graph_class and preset:
                mapping[graph_class] = preset
    if not mapping:
        raise RuntimeError(f"No class->preset mapping loaded from {path}")
    return mapping


def load_dataset(features_csv: str, preset_map: Dict[str, str]) -> Dataset:
    x_rows: List[List[float]] = []
    y_class: List[str] = []
    y_preset: List[str] = []
    groups: List[str] = []

    with open(features_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("exit_code") or "").strip() != "0":
                continue
            graph_class = (row.get("graph_class_true") or "").strip()
            graph_path = (row.get("graph_path") or "").strip()
            if not graph_class or not graph_path:
                continue
            if graph_class not in preset_map:
                continue

            feat = [to_float(row.get(c, "")) for c in FEATURE_COLUMNS]
            x_rows.append(feat)
            y_class.append(graph_class)
            y_preset.append(preset_map[graph_class])
            groups.append(graph_path)

    if not x_rows:
        raise RuntimeError("No usable rows in features CSV.")

    return Dataset(
        x=np.asarray(x_rows, dtype=np.float64),
        y_class=np.asarray(y_class),
        y_preset=np.asarray(y_preset),
        groups=np.asarray(groups),
        feature_names=list(FEATURE_COLUMNS),
    )


def build_model(random_state: int):
    try:
        from xgboost import XGBClassifier  # type: ignore

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=4,
        )
        backend = "xgboost"
    except Exception:
        model = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.06,
            max_iter=400,
            random_state=random_state,
        )
        backend = "sklearn_hist_gradient_boosting"
    return model, backend


def split_indices(groups: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def train_one(
    x: np.ndarray,
    y_text: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> Dict:
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y_text)

    train_idx, test_idx = split_indices(groups, test_size, random_state)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model, backend = build_model(random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model": model,
        "encoder": y_encoder,
        "backend": backend,
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(f1m),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "labels": [str(x) for x in y_encoder.classes_],
            "confusion_matrix": cm,
        },
    }


def maybe_export_onnx(model, feature_dim: int, onnx_path: str) -> str:
    try:
        from skl2onnx import to_onnx  # type: ignore

        dummy = np.zeros((1, feature_dim), dtype=np.float32)
        onx = to_onnx(model, dummy)
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        return "ok"
    except Exception as e:
        return f"failed: {e}"


def dump_obj(obj, path: str) -> None:
    if joblib is not None:
        joblib.dump(obj, path)
        return
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train graph class/preset guesser models")
    ap.add_argument("--features-csv", required=True, help="CSV from run_graph_class_guessing.py")
    ap.add_argument(
        "--recommended-presets",
        required=True,
        help="ablation_recommended_presets.csv mapping graph_class -> recommended_config",
    )
    ap.add_argument("--outdir", default="results/models", help="Output directory")
    ap.add_argument("--test-size", type=float, default=0.25, help="Group holdout ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--export-onnx", action="store_true", help="Try ONNX export for sklearn backends")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    preset_map = load_preset_map(args.recommended_presets)
    ds = load_dataset(args.features_csv, preset_map)

    class_train = train_one(ds.x, ds.y_class, ds.groups, args.test_size, args.seed)
    preset_train = train_one(ds.x, ds.y_preset, ds.groups, args.test_size, args.seed)

    class_model_path = os.path.join(args.outdir, "graph_class_model.joblib")
    preset_model_path = os.path.join(args.outdir, "preset_model.joblib")
    class_encoder_path = os.path.join(args.outdir, "graph_class_encoder.joblib")
    preset_encoder_path = os.path.join(args.outdir, "preset_encoder.joblib")

    dump_obj(class_train["model"], class_model_path)
    dump_obj(preset_train["model"], preset_model_path)
    dump_obj(class_train["encoder"], class_encoder_path)
    dump_obj(preset_train["encoder"], preset_encoder_path)

    onnx_status = {}
    if args.export_onnx:
        onnx_status["graph_class_model.onnx"] = maybe_export_onnx(
            class_train["model"], len(ds.feature_names), os.path.join(args.outdir, "graph_class_model.onnx")
        )
        onnx_status["preset_model.onnx"] = maybe_export_onnx(
            preset_train["model"], len(ds.feature_names), os.path.join(args.outdir, "preset_model.onnx")
        )

    manifest = {
        "feature_names": ds.feature_names,
        "rows": int(ds.x.shape[0]),
        "class_model_backend": class_train["backend"],
        "preset_model_backend": preset_train["backend"],
        "class_metrics": class_train["metrics"],
        "preset_metrics": preset_train["metrics"],
        "artifacts": {
            "graph_class_model": class_model_path,
            "preset_model": preset_model_path,
            "graph_class_encoder": class_encoder_path,
            "preset_encoder": preset_encoder_path,
        },
        "onnx_export": onnx_status,
    }

    manifest_path = os.path.join(args.outdir, "training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Rows used: {manifest['rows']}")
    print(f"Class model backend: {class_train['backend']}")
    print(
        "Class metrics: "
        f"acc={class_train['metrics']['accuracy']:.4f} "
        f"macro_f1={class_train['metrics']['macro_f1']:.4f}"
    )
    print(f"Preset model backend: {preset_train['backend']}")
    print(
        "Preset metrics: "
        f"acc={preset_train['metrics']['accuracy']:.4f} "
        f"macro_f1={preset_train['metrics']['macro_f1']:.4f}"
    )
    print(f"Wrote model artifacts + metrics manifest to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
