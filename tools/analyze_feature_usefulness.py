#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import f_oneway, kruskal  # type: ignore
except Exception:
    f_oneway = None
    kruskal = None


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


def to_float(v: str) -> float:
    if v is None:
        return 0.0
    v = v.strip()
    if v == "":
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values)))


def load_preset_map(path: str) -> Dict[str, str]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = (r.get("graph_class") or "").strip()
            p = (r.get("recommended_config") or "").strip()
            if c and p:
                out[c] = p
    return out


def load_rows(features_csv: str, preset_map: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = []
    y_class = []
    y_preset = []
    groups = []
    with open(features_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("exit_code") or "").strip() != "0":
                continue
            c = (r.get("graph_class_true") or "").strip()
            g = (r.get("graph_path") or "").strip()
            if not c or not g or c not in preset_map:
                continue
            x.append([to_float(r.get(k, "")) for k in FEATURE_COLUMNS])
            y_class.append(c)
            y_preset.append(preset_map[c])
            groups.append(g)
    if not x:
        raise RuntimeError("No usable rows from features CSV")
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y_class),
        np.asarray(y_preset),
        np.asarray(groups),
    )


def split_groups(groups: np.ndarray, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def fit_eval(x, y_text, groups, seed, test_size):
    enc = LabelEncoder()
    y = enc.fit_transform(y_text)
    tr, te = split_groups(groups, test_size=test_size, seed=seed)
    clf = HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.06, max_iter=400, random_state=seed
    )
    clf.fit(x[tr], y[tr])
    pred = clf.predict(x[te])
    return {
        "clf": clf,
        "enc": enc,
        "train_idx": tr,
        "test_idx": te,
        "accuracy": float(accuracy_score(y[te], pred)),
        "macro_f1": float(f1_score(y[te], pred, average="macro")),
    }


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def separability_stats(x: np.ndarray, y_text: np.ndarray, feature_names: List[str]):
    rows = []
    classes = sorted(set(y_text.tolist()))
    for j, feat in enumerate(feature_names):
        by_class = defaultdict(list)
        for i, c in enumerate(y_text):
            by_class[c].append(float(x[i, j]))
        arrays = [by_class[c] for c in classes if by_class[c]]
        stat_f = p_f = stat_k = p_k = None
        if len(arrays) >= 2 and f_oneway is not None:
            try:
                stat_f, p_f = f_oneway(*arrays)
            except Exception:
                pass
        if len(arrays) >= 2 and kruskal is not None:
            try:
                stat_k, p_k = kruskal(*arrays)
            except Exception:
                pass
        overall = [float(v) for v in x[:, j]]
        q10 = float(np.quantile(overall, 0.10))
        q50 = float(np.quantile(overall, 0.50))
        q90 = float(np.quantile(overall, 0.90))
        rows.append(
            {
                "feature": feat,
                "overall_q10": q10,
                "overall_median": q50,
                "overall_q90": q90,
                "anova_f": stat_f if stat_f is not None else "",
                "anova_p": p_f if p_f is not None else "",
                "kruskal_h": stat_k if stat_k is not None else "",
                "kruskal_p": p_k if p_k is not None else "",
            }
        )
    rows.sort(key=lambda r: (r["kruskal_p"] if r["kruskal_p"] != "" else 1.0))
    return rows


def correlation_rows(x: np.ndarray, feature_names: List[str]):
    corr = np.corrcoef(x, rowvar=False)
    out = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            out.append(
                {
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "pearson_corr": float(corr[i, j]),
                    "abs_corr": float(abs(corr[i, j])),
                }
            )
    out.sort(key=lambda r: r["abs_corr"], reverse=True)
    return out


def permutation_rows(clf, x_test, y_test, feature_names, seed):
    imp = permutation_importance(
        clf, x_test, y_test, n_repeats=10, random_state=seed, scoring="f1_macro"
    )
    rows = []
    for i, feat in enumerate(feature_names):
        rows.append(
            {
                "feature": feat,
                "importance_mean": float(imp.importances_mean[i]),
                "importance_std": float(imp.importances_std[i]),
            }
        )
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    return rows


def feature_set_ablation(x, y_text, groups, feature_names, seed, test_size):
    idx = {k: i for i, k in enumerate(feature_names)}
    sets = {
        "all": feature_names,
        "size_only": ["feature_n", "feature_m"],
        "degree_core": [
            "feature_avg_degree",
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
            "feature_kcore_max",
            "feature_kcore_mean",
            "feature_kcore_top_fraction",
            "feature_kcore_ge_2_fraction",
            "feature_kcore_ge_4_fraction",
            "feature_kcore_ge_8_fraction",
        ],
        "sparsity_shape": [
            "feature_density",
            "feature_leaf_fraction",
            "feature_isolated_fraction",
        ],
        "connectivity_clustering": [
            "feature_component_count",
            "feature_largest_component_fraction",
            "feature_second_component_fraction",
            "feature_clustering_sampled_mean",
            "feature_clustering_samples_used",
            "feature_transitivity_sampled",
            "feature_degree_assortativity_sampled",
        ],
        "distance_profile": [
            "feature_bfs_mean_distance",
            "feature_bfs_p90_distance",
            "feature_bfs_diameter_proxy",
            "feature_bfs_reachable_fraction",
        ],
        "all_without_tail": [
            f for f in feature_names
            if f not in (
                "feature_degree_p99",
                "feature_degree_cv",
                "feature_degree_ratio_p99_p50",
                "feature_degree_ratio_max_p90",
                "feature_degree_tail_hill_alpha",
            )
        ],
    }
    rows = []
    for name, feats in sets.items():
        cols = [idx[f] for f in feats]
        res = fit_eval(x[:, cols], y_text, groups, seed, test_size)
        rows.append(
            {
                "feature_set": name,
                "num_features": len(cols),
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
            }
        )
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    return rows


def make_plots(outdir: str, separability: List[Dict], perm_rows: List[Dict]):
    if plt is None:
        return

    # Top separability (smallest Kruskal p-values).
    top_sep = [r for r in separability if r["kruskal_p"] != ""][:10]
    if top_sep:
        names = [r["feature"] for r in top_sep]
        vals = [-np.log10(float(r["kruskal_p"]) + 1e-16) for r in top_sep]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(names[::-1], vals[::-1], color="#1f77b4")
        ax.set_xlabel("-log10(Kruskal p-value)")
        ax.set_title("Top Univariate-Separable Features")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "feature_separability_top.png"), dpi=200)
        plt.close(fig)

    # Permutation importance.
    top_imp = perm_rows[:10]
    if top_imp:
        names = [r["feature"] for r in top_imp]
        vals = [r["importance_mean"] for r in top_imp]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(names[::-1], vals[::-1], color="#2ca02c")
        ax.set_xlabel("Permutation importance (macro-F1 drop)")
        ax.set_title("Top Feature Importances")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "feature_importance_top.png"), dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Analyze usefulness of graph guesser features")
    ap.add_argument("--features-csv", required=True, help="results/graph_class_guess.csv")
    ap.add_argument("--recommended-presets", required=True, help="ablation_recommended_presets.csv")
    ap.add_argument("--outdir", default="results/feature_analysis", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of repeated grouped splits for stability analysis (default: 5)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    preset_map = load_preset_map(args.recommended_presets)
    x, y_class, y_preset, groups = load_rows(args.features_csv, preset_map)

    class_eval = fit_eval(x, y_class, groups, args.seed, args.test_size)
    preset_eval = fit_eval(x, y_preset, groups, args.seed, args.test_size)

    split_summaries = []
    perm_per_split = []
    ablation_per_split = []
    for i in range(args.num_splits):
        split_seed = args.seed + i
        ce = fit_eval(x, y_class, groups, split_seed, args.test_size)
        pe = fit_eval(x, y_preset, groups, split_seed, args.test_size)
        y_class_enc = ce["enc"].transform(y_class)
        y_class_test = y_class_enc[ce["test_idx"]]
        x_class_test = x[ce["test_idx"]]
        perm_rows_i = permutation_rows(
            ce["clf"], x_class_test, y_class_test, FEATURE_COLUMNS, split_seed
        )
        perm_per_split.append(perm_rows_i)
        ab_rows_i = feature_set_ablation(
            x, y_class, groups, FEATURE_COLUMNS, split_seed, args.test_size
        )
        ablation_per_split.append(ab_rows_i)
        split_summaries.append(
            {
                "split_id": i,
                "seed": split_seed,
                "class_accuracy": ce["accuracy"],
                "class_macro_f1": ce["macro_f1"],
                "preset_accuracy": pe["accuracy"],
                "preset_macro_f1": pe["macro_f1"],
            }
        )

    # Aggregate permutation importance across splits.
    by_feat = defaultdict(list)
    for split_rows in perm_per_split:
        for r in split_rows:
            by_feat[r["feature"]].append(r["importance_mean"])
    perm_class = []
    for feat in FEATURE_COLUMNS:
        vals = by_feat.get(feat, [])
        perm_class.append(
            {
                "feature": feat,
                "importance_mean": float(np.mean(vals)) if vals else 0.0,
                "importance_std": float(np.std(vals)) if vals else 0.0,
                "importance_median": float(np.median(vals)) if vals else 0.0,
                "non_positive_rate": float(np.mean([1.0 if v <= 0.0 else 0.0 for v in vals]))
                if vals
                else 1.0,
            }
        )
    perm_class.sort(key=lambda r: r["importance_mean"], reverse=True)

    sep_class = separability_stats(x, y_class, FEATURE_COLUMNS)
    corr_rows = correlation_rows(x, FEATURE_COLUMNS)
    # Aggregate feature-set ablation across splits.
    ab_aggr = defaultdict(list)
    for split_rows in ablation_per_split:
        for r in split_rows:
            ab_aggr[r["feature_set"]].append((r["accuracy"], r["macro_f1"]))
    ablation_rows = []
    for fset, vals in ab_aggr.items():
        accs = [x[0] for x in vals]
        f1s = [x[1] for x in vals]
        ablation_rows.append(
            {
                "feature_set": fset,
                "num_features": next(
                    r["num_features"] for rows_i in ablation_per_split for r in rows_i if r["feature_set"] == fset
                ),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
            }
        )
    ablation_rows.sort(key=lambda r: r["macro_f1_mean"], reverse=True)

    write_csv(
        os.path.join(args.outdir, "class_feature_importance.csv"),
        ["feature", "importance_mean", "importance_std", "importance_median", "non_positive_rate"],
        perm_class,
    )
    write_csv(
        os.path.join(args.outdir, "class_univariate_separability.csv"),
        [
            "feature",
            "overall_q10",
            "overall_median",
            "overall_q90",
            "anova_f",
            "anova_p",
            "kruskal_h",
            "kruskal_p",
        ],
        sep_class,
    )
    write_csv(
        os.path.join(args.outdir, "feature_correlation_pairs.csv"),
        ["feature_a", "feature_b", "pearson_corr", "abs_corr"],
        corr_rows,
    )
    write_csv(
        os.path.join(args.outdir, "feature_set_ablation.csv"),
        ["feature_set", "num_features", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"],
        ablation_rows,
    )
    write_csv(
        os.path.join(args.outdir, "split_metrics.csv"),
        ["split_id", "seed", "class_accuracy", "class_macro_f1", "preset_accuracy", "preset_macro_f1"],
        split_summaries,
    )

    make_plots(args.outdir, sep_class, perm_class)

    summary = {
        "rows_used": int(x.shape[0]),
        "num_splits": int(args.num_splits),
        "class_model_accuracy": class_eval["accuracy"],
        "class_model_macro_f1": class_eval["macro_f1"],
        "preset_model_accuracy": preset_eval["accuracy"],
        "preset_model_macro_f1": preset_eval["macro_f1"],
        "class_accuracy_mean_over_splits": float(np.mean([x["class_accuracy"] for x in split_summaries])),
        "class_macro_f1_mean_over_splits": float(np.mean([x["class_macro_f1"] for x in split_summaries])),
        "preset_accuracy_mean_over_splits": float(np.mean([x["preset_accuracy"] for x in split_summaries])),
        "preset_macro_f1_mean_over_splits": float(np.mean([x["preset_macro_f1"] for x in split_summaries])),
        "top_class_features_by_importance": perm_class[:5],
        "top_class_features_by_separability": sep_class[:5],
    }
    with open(os.path.join(args.outdir, "feature_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Rows used: {summary['rows_used']}")
    print(
        "Class model: "
        f"acc={summary['class_model_accuracy']:.4f} "
        f"macro_f1={summary['class_model_macro_f1']:.4f}"
    )
    print(
        "Preset model: "
        f"acc={summary['preset_model_accuracy']:.4f} "
        f"macro_f1={summary['preset_model_macro_f1']:.4f}"
    )
    print(f"Wrote feature usefulness analysis to: {args.outdir}")


if __name__ == "__main__":
    main()
