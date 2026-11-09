"""
Consolidation and Scientific Visualization for Phase 7 Multi-Seed FL Experiment.
Processes all five research seeds: 42, 123, 2026, 3407, 7777.
Generates:
  - experiments/results/multiseed_consolidated_summary.json
  - experiments/results/multiseed_consolidated_summary.csv
  - reports/figures/FIG-P7-01_test_accuracy_comparison.png
  - reports/figures/FIG-P7-02_macro_f1_comparison.png
  - reports/figures/FIG-P7-03_balanced_accuracy_comparison.png
  - reports/figures/FIG-P7-04_validation_learning_curves.png
  - reports/figures/FIG-P7-05_aggregate_metrics_mean_std.png
  - reports/figures/FIG-P7-06_per_class_f1_comparison.png
"""

import os
import json
import shutil
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = r"C:\Users\mohit\.gemini\antigravity\brain\6c9e6809-31d9-4d77-b974-cccbb65cf12c"

SEED_DIRS = [
    (42, os.path.join(PROJ_ROOT, "experiments", "seed_42_attempt_3")),
    (123, os.path.join(PROJ_ROOT, "experiments", "seed_123_attempt_2")),
    (2026, os.path.join(PROJ_ROOT, "experiments", "seed_2026")),
    (3407, os.path.join(PROJ_ROOT, "experiments", "seed_3407")),
    (7777, os.path.join(PROJ_ROOT, "experiments", "seed_7777")),
]

CLASS_NAMES = {
    0: "Adipose",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth Muscle",
    6: "Normal Mucosa",
    7: "Stroma",
    8: "Adenocarcinoma"
}


def load_all_seeds():
    summaries = []
    round_logs = {}
    for seed, sdir in SEED_DIRS:
        sum_file = os.path.join(sdir, "results", "seed_summary.json")
        with open(sum_file, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))
        
        # Load round-level logs for learning curves
        avg_log = glob.glob(os.path.join(sdir, "logs", "fedavg*metrics.jsonl"))[0]
        prox_log = glob.glob(os.path.join(sdir, "logs", "fedprox*metrics.jsonl"))[0]
        
        with open(avg_log, "r", encoding="utf-8") as f:
            avg_rounds = [json.loads(l) for l in f]
        with open(prox_log, "r", encoding="utf-8") as f:
            prox_rounds = [json.loads(l) for l in f]
            
        round_logs[seed] = {
            "fedavg": avg_rounds,
            "fedprox": prox_rounds
        }
    return summaries, round_logs


def calc_stats(vals):
    arr = np.array(vals, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1 if n > 1 else 0))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    range_val = float(max_val - min_val)
    sem_val = float(std_val / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": round(mean_val, 5),
        "std": round(std_val, 5),
        "sem": round(sem_val, 5),
        "median": round(median_val, 5),
        "min": round(min_val, 5),
        "max": round(max_val, 5),
        "range": round(range_val, 5),
        "raw": [round(float(x), 5) for x in vals]
    }


def compute_paired_tests(avg_vals, prox_vals):
    diffs = np.array(prox_vals) - np.array(avg_vals)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    sem_diff = float(std_diff / np.sqrt(n))
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(prox_vals, avg_vals)
    
    # 95% Confidence Interval with Student-t (df = n - 1)
    df = n - 1
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci_lower = mean_diff - t_crit * sem_diff
    ci_upper = mean_diff + t_crit * sem_diff
    
    # Wilcoxon signed rank test
    try:
        w_res = stats.wilcoxon(prox_vals, avg_vals)
        w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_pval = None, None
        
    return {
        "mean_difference": round(mean_diff, 5),
        "std_difference": round(std_diff, 5),
        "sem_difference": round(sem_diff, 5),
        "t_statistic": round(float(t_stat), 4),
        "degrees_of_freedom": df,
        "p_value_two_sided": round(float(p_val), 5),
        "ci_95_percent": [round(ci_lower, 5), round(ci_upper, 5)],
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_pval,
        "statistically_significant_alpha_0_05": bool(p_val < 0.05)
    }


def main():
    summaries, round_logs = load_all_seeds()
    seeds = [s["seed"] for s in summaries]
    print(f"Loaded summaries for {len(seeds)} seeds: {seeds}")

    # Extract metrics
    metrics = {
        "fedavg": {
            "test_accuracy": [s["fedavg"]["test_accuracy"] for s in summaries],
            "test_macro_f1": [s["fedavg"]["test_macro_f1"] for s in summaries],
            "test_balanced_accuracy": [s["fedavg"]["test_balanced_accuracy"] for s in summaries],
            "test_loss": [s["fedavg"]["test_loss"] for s in summaries],
            "val_accuracy": [s["fedavg"]["best_validation_accuracy"] for s in summaries],
            "val_loss": [s["fedavg"]["best_validation_loss"] for s in summaries],
            "wall_clock_minutes": [s["fedavg"]["wall_clock_minutes"] for s in summaries]
        },
        "fedprox": {
            "test_accuracy": [s["fedprox"]["test_accuracy"] for s in summaries],
            "test_macro_f1": [s["fedprox"]["test_macro_f1"] for s in summaries],
            "test_balanced_accuracy": [s["fedprox"]["test_balanced_accuracy"] for s in summaries],
            "test_loss": [s["fedprox"]["test_loss"] for s in summaries],
            "val_accuracy": [s["fedprox"]["best_validation_accuracy"] for s in summaries],
            "val_loss": [s["fedprox"]["best_validation_loss"] for s in summaries],
            "wall_clock_minutes": [s["fedprox"]["wall_clock_minutes"] for s in summaries]
        }
    }

    # Paired differences
    paired_deltas = {
        "test_accuracy": [p - a for a, p in zip(metrics["fedavg"]["test_accuracy"], metrics["fedprox"]["test_accuracy"])],
        "test_macro_f1": [p - a for a, p in zip(metrics["fedavg"]["test_macro_f1"], metrics["fedprox"]["test_macro_f1"])],
        "test_balanced_accuracy": [p - a for a, p in zip(metrics["fedavg"]["test_balanced_accuracy"], metrics["fedprox"]["test_balanced_accuracy"])],
        "test_loss": [p - a for a, p in zip(metrics["fedavg"]["test_loss"], metrics["fedprox"]["test_loss"])],
        "val_accuracy": [p - a for a, p in zip(metrics["fedavg"]["val_accuracy"], metrics["fedprox"]["val_accuracy"])],
        "wall_clock_minutes": [p - a for a, p in zip(metrics["fedavg"]["wall_clock_minutes"], metrics["fedprox"]["wall_clock_minutes"])]
    }

    # Build consolidated dict
    consolidation = {
        "study": "IndiaAI Fellowship 2026-27 PathMNIST FL Research",
        "protocol": "FedAvg vs FedProx (mu=0.01), 5 clients, Dirichlet alpha=0.3, Client 5 40% label noise, 30 rounds",
        "num_completed_seeds": len(seeds),
        "completed_seeds": seeds,
        "per_seed_results": summaries,
        "statistical_summary": {
            "fedavg": {k: calc_stats(v) for k, v in metrics["fedavg"].items()},
            "fedprox": {k: calc_stats(v) for k, v in metrics["fedprox"].items()},
            "paired_differences_fedprox_minus_fedavg": {k: calc_stats(v) for k, v in paired_deltas.items()}
        },
        "hypothesis_tests_paired": {
            "test_accuracy": compute_paired_tests(metrics["fedavg"]["test_accuracy"], metrics["fedprox"]["test_accuracy"]),
            "test_macro_f1": compute_paired_tests(metrics["fedavg"]["test_macro_f1"], metrics["fedprox"]["test_macro_f1"]),
            "test_balanced_accuracy": compute_paired_tests(metrics["fedavg"]["test_balanced_accuracy"], metrics["fedprox"]["test_balanced_accuracy"]),
            "test_loss": compute_paired_tests(metrics["fedavg"]["test_loss"], metrics["fedprox"]["test_loss"])
        }
    }

    # Save JSON
    out_json = os.path.join(PROJ_ROOT, "experiments", "results", "multiseed_consolidated_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(consolidation, f, indent=2)
    print(f"Saved: {out_json}")

    # Save CSV
    out_csv = os.path.join(PROJ_ROOT, "experiments", "results", "multiseed_consolidated_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("seed,algorithm,best_val_round,val_acc,val_loss,test_acc,test_macro_f1,test_bal_acc,test_loss,wall_clock_min\n")
        for s in summaries:
            sd = s["seed"]
            a = s["fedavg"]
            p = s["fedprox"]
            f.write(f"{sd},FedAvg,{a['best_validation_round']},{a['best_validation_accuracy']:.5f},{a['best_validation_loss']:.5f},{a['test_accuracy']:.5f},{a['test_macro_f1']:.5f},{a['test_balanced_accuracy']:.5f},{a['test_loss']:.5f},{a['wall_clock_minutes']:.2f}\n")
            f.write(f"{sd},FedProx,{p['best_validation_round']},{p['best_validation_accuracy']:.5f},{p['best_validation_loss']:.5f},{p['test_accuracy']:.5f},{p['test_macro_f1']:.5f},{p['test_balanced_accuracy']:.5f},{p['test_loss']:.5f},{p['wall_clock_minutes']:.2f}\n")
    print(f"Saved: {out_csv}")

    # -------------------------------------------------------------
    # GENERATE PUBLICATION-QUALITY FIGURES
    # -------------------------------------------------------------
    fig_dir = os.path.join(PROJ_ROOT, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    artifact_fig_dir = os.path.join(ARTIFACT_DIR, "figures")
    os.makedirs(artifact_fig_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "fedavg": "#1f77b4",
        "fedprox": "#ff7f0e",
        "delta_pos": "#2ca02c",
        "delta_neg": "#d62728",
        "neutral": "#7f7f7f"
    }

    # 1. FIG-P7-01: Test Accuracy Comparison across 5 seeds
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x = np.arange(len(seeds))
    width = 0.35

    avg_accs = [m * 100 for m in metrics["fedavg"]["test_accuracy"]]
    prox_accs = [m * 100 for m in metrics["fedprox"]["test_accuracy"]]

    rects1 = ax.bar(x - width/2, avg_accs, width, label="FedAvg", color=colors["fedavg"], alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width/2, prox_accs, width, label="FedProx (μ=0.01)", color=colors["fedprox"], alpha=0.85, edgecolor="black")

    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Test Accuracy Comparison Across All 5 Seeds (Held-Out Test Split)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.set_ylim(0, 90)

    for i in range(len(seeds)):
        delta = prox_accs[i] - avg_accs[i]
        d_color = colors["delta_pos"] if delta >= 0 else colors["delta_neg"]
        ax.annotate(f"{avg_accs[i]:.2f}%", (x[i] - width/2, avg_accs[i] + 1.0), ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{prox_accs[i]:.2f}%", (x[i] + width/2, prox_accs[i] + 1.0), ha='center', va='bottom', fontsize=9)
        ax.annotate(f"Δ: {delta:+.2f}%", (x[i], max(avg_accs[i], prox_accs[i]) + 4.5), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=d_color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=d_color, alpha=0.9))

    plt.tight_layout()
    f1_path = os.path.join(fig_dir, "FIG-P7-01_test_accuracy_comparison.png")
    plt.savefig(f1_path, dpi=300)
    shutil.copy(f1_path, os.path.join(artifact_fig_dir, "FIG-P7-01_test_accuracy_comparison.png"))
    shutil.copy(f1_path, os.path.join(ARTIFACT_DIR, "FIG-P7-01_test_accuracy_comparison.png"))
    plt.close()
    print("Saved FIG-P7-01")

    # 2. FIG-P7-02: Macro-F1 Comparison across 5 seeds
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    avg_f1s = metrics["fedavg"]["test_macro_f1"]
    prox_f1s = metrics["fedprox"]["test_macro_f1"]

    rects1 = ax.bar(x - width/2, avg_f1s, width, label="FedAvg", color=colors["fedavg"], alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width/2, prox_f1s, width, label="FedProx (μ=0.01)", color=colors["fedprox"], alpha=0.85, edgecolor="black")

    ax.set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Test Macro-F1 Score Across All 5 Seeds", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.set_ylim(0, 0.85)

    for i in range(len(seeds)):
        delta = prox_f1s[i] - avg_f1s[i]
        d_color = colors["delta_pos"] if delta >= 0 else colors["delta_neg"]
        ax.annotate(f"{avg_f1s[i]:.4f}", (x[i] - width/2, avg_f1s[i] + 0.01), ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{prox_f1s[i]:.4f}", (x[i] + width/2, prox_f1s[i] + 0.01), ha='center', va='bottom', fontsize=9)
        ax.annotate(f"Δ: {delta:+.4f}", (x[i], max(avg_f1s[i], prox_f1s[i]) + 0.04), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=d_color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=d_color, alpha=0.9))

    plt.tight_layout()
    f2_path = os.path.join(fig_dir, "FIG-P7-02_macro_f1_comparison.png")
    plt.savefig(f2_path, dpi=300)
    shutil.copy(f2_path, os.path.join(artifact_fig_dir, "FIG-P7-02_macro_f1_comparison.png"))
    shutil.copy(f2_path, os.path.join(ARTIFACT_DIR, "FIG-P7-02_macro_f1_comparison.png"))
    plt.close()
    print("Saved FIG-P7-02")

    # 3. FIG-P7-03: Balanced Accuracy Comparison across 5 seeds
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    avg_bals = [m * 100 for m in metrics["fedavg"]["test_balanced_accuracy"]]
    prox_bals = [m * 100 for m in metrics["fedprox"]["test_balanced_accuracy"]]

    rects1 = ax.bar(x - width/2, avg_bals, width, label="FedAvg", color=colors["fedavg"], alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width/2, prox_bals, width, label="FedProx (μ=0.01)", color=colors["fedprox"], alpha=0.85, edgecolor="black")

    ax.set_ylabel("Balanced Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Test Balanced Accuracy Across All 5 Seeds", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.set_ylim(0, 85)

    for i in range(len(seeds)):
        delta = prox_bals[i] - avg_bals[i]
        d_color = colors["delta_pos"] if delta >= 0 else colors["delta_neg"]
        ax.annotate(f"{avg_bals[i]:.2f}%", (x[i] - width/2, avg_bals[i] + 1.0), ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{prox_bals[i]:.2f}%", (x[i] + width/2, prox_bals[i] + 1.0), ha='center', va='bottom', fontsize=9)
        ax.annotate(f"Δ: {delta:+.2f}%", (x[i], max(avg_bals[i], prox_bals[i]) + 4.0), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=d_color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=d_color, alpha=0.9))

    plt.tight_layout()
    f3_path = os.path.join(fig_dir, "FIG-P7-03_balanced_accuracy_comparison.png")
    plt.savefig(f3_path, dpi=300)
    shutil.copy(f3_path, os.path.join(artifact_fig_dir, "FIG-P7-03_balanced_accuracy_comparison.png"))
    shutil.copy(f3_path, os.path.join(ARTIFACT_DIR, "FIG-P7-03_balanced_accuracy_comparison.png"))
    plt.close()
    print("Saved FIG-P7-03")

    # 4. FIG-P7-04: Validation Learning Curves across all 30 rounds for all 5 seeds
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes_flat = axes.flatten()
    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (seed, color) in enumerate(zip(seeds, seed_colors)):
        ax = axes_flat[idx]
        rounds = list(range(1, 31))
        avg_accs = [r["val/global_accuracy"] * 100 for r in round_logs[seed]["fedavg"]]
        prox_accs = [r["val/global_accuracy"] * 100 for r in round_logs[seed]["fedprox"]]

        ax.plot(rounds, avg_accs, label="FedAvg", color="#1f77b4", linewidth=2.0)
        ax.plot(rounds, prox_accs, label="FedProx (μ=0.01)", color="#ff7f0e", linewidth=2.0, linestyle="--")

        ax.set_title(f"Seed {seed}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Communication Round", fontsize=10)
        ax.set_ylabel("Global Val Accuracy (%)", fontsize=10)
        ax.set_ylim(10, 80)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)

    # 6th panel: Mean across all 5 seeds
    ax_mean = axes_flat[5]
    all_avg_curves = np.array([[r["val/global_accuracy"] * 100 for r in round_logs[s]["fedavg"]] for s in seeds])
    all_prox_curves = np.array([[r["val/global_accuracy"] * 100 for r in round_logs[s]["fedprox"]] for s in seeds])

    mean_avg_curve = np.mean(all_avg_curves, axis=0)
    std_avg_curve = np.std(all_avg_curves, axis=0)
    mean_prox_curve = np.mean(all_prox_curves, axis=0)
    std_prox_curve = np.std(all_prox_curves, axis=0)

    rounds = list(range(1, 31))
    ax_mean.plot(rounds, mean_avg_curve, label="FedAvg (Mean)", color="#1f77b4", linewidth=2.5)
    ax_mean.fill_between(rounds, mean_avg_curve - std_avg_curve, mean_avg_curve + std_avg_curve, color="#1f77b4", alpha=0.2)

    ax_mean.plot(rounds, mean_prox_curve, label="FedProx (Mean)", color="#ff7f0e", linewidth=2.5, linestyle="--")
    ax_mean.fill_between(rounds, mean_prox_curve - std_prox_curve, mean_prox_curve + std_prox_curve, color="#ff7f0e", alpha=0.2)

    ax_mean.set_title("5-Seed Aggregate Mean ± 1 Std Dev", fontsize=12, fontweight="bold")
    ax_mean.set_xlabel("Communication Round", fontsize=10)
    ax_mean.set_ylabel("Global Val Accuracy (%)", fontsize=10)
    ax_mean.set_ylim(10, 80)
    ax_mean.grid(True, linestyle=":", alpha=0.6)
    ax_mean.legend(loc="lower right", fontsize=9)

    fig.suptitle("Validation Accuracy Convergence Trajectories Across 30 Rounds", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    f4_path = os.path.join(fig_dir, "FIG-P7-04_validation_learning_curves.png")
    plt.savefig(f4_path, dpi=300)
    shutil.copy(f4_path, os.path.join(artifact_fig_dir, "FIG-P7-04_validation_learning_curves.png"))
    shutil.copy(f4_path, os.path.join(ARTIFACT_DIR, "FIG-P7-04_validation_learning_curves.png"))
    plt.close()
    print("Saved FIG-P7-04")

    # 5. FIG-P7-05: Aggregate Mean and Std Dev comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    metric_keys = ["Test Accuracy", "Macro-F1", "Balanced Accuracy", "Best Val Accuracy"]
    x_pos = np.arange(len(metric_keys))

    # Normalized percentages for plotting
    avg_means = [
        consolidation["statistical_summary"]["fedavg"]["test_accuracy"]["mean"] * 100,
        consolidation["statistical_summary"]["fedavg"]["test_macro_f1"]["mean"] * 100,
        consolidation["statistical_summary"]["fedavg"]["test_balanced_accuracy"]["mean"] * 100,
        consolidation["statistical_summary"]["fedavg"]["val_accuracy"]["mean"] * 100
    ]
    avg_stds = [
        consolidation["statistical_summary"]["fedavg"]["test_accuracy"]["std"] * 100,
        consolidation["statistical_summary"]["fedavg"]["test_macro_f1"]["std"] * 100,
        consolidation["statistical_summary"]["fedavg"]["test_balanced_accuracy"]["std"] * 100,
        consolidation["statistical_summary"]["fedavg"]["val_accuracy"]["std"] * 100
    ]

    prox_means = [
        consolidation["statistical_summary"]["fedprox"]["test_accuracy"]["mean"] * 100,
        consolidation["statistical_summary"]["fedprox"]["test_macro_f1"]["mean"] * 100,
        consolidation["statistical_summary"]["fedprox"]["test_balanced_accuracy"]["mean"] * 100,
        consolidation["statistical_summary"]["fedprox"]["val_accuracy"]["mean"] * 100
    ]
    prox_stds = [
        consolidation["statistical_summary"]["fedprox"]["test_accuracy"]["std"] * 100,
        consolidation["statistical_summary"]["fedprox"]["test_macro_f1"]["std"] * 100,
        consolidation["statistical_summary"]["fedprox"]["test_balanced_accuracy"]["std"] * 100,
        consolidation["statistical_summary"]["fedprox"]["val_accuracy"]["std"] * 100
    ]

    rects1 = ax.bar(x_pos - width/2, avg_means, width, yerr=avg_stds, capsize=5, label="FedAvg", color=colors["fedavg"], alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x_pos + width/2, prox_means, width, yerr=prox_stds, capsize=5, label="FedProx (μ=0.01)", color=colors["fedprox"], alpha=0.85, edgecolor="black")

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("5-Seed Aggregate Benchmark Metrics (Mean ± 1 Sample Std Dev)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_keys, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.set_ylim(0, 95)

    for i in range(len(metric_keys)):
        d = prox_means[i] - avg_means[i]
        d_color = colors["delta_pos"] if d >= 0 else colors["delta_neg"]
        ax.annotate(f"{avg_means[i]:.2f}±{avg_stds[i]:.2f}", (x_pos[i] - width/2, avg_means[i] + avg_stds[i] + 2.0), ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{prox_means[i]:.2f}±{prox_stds[i]:.2f}", (x_pos[i] + width/2, prox_means[i] + prox_stds[i] + 2.0), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    f5_path = os.path.join(fig_dir, "FIG-P7-05_aggregate_metrics_mean_std.png")
    plt.savefig(f5_path, dpi=300)
    shutil.copy(f5_path, os.path.join(artifact_fig_dir, "FIG-P7-05_aggregate_metrics_mean_std.png"))
    shutil.copy(f5_path, os.path.join(ARTIFACT_DIR, "FIG-P7-05_aggregate_metrics_mean_std.png"))
    plt.close()
    print("Saved FIG-P7-05")

    # 6. FIG-P7-06: Per-class performance comparison across 9 PathMNIST classes
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    x_classes = np.arange(9)
    width_c = 0.38

    avg_class_f1s = []
    prox_class_f1s = []
    for c in range(9):
        c_str = str(c)
        avg_c = [s["fedavg"]["per_class"][c_str]["f1_score"] for s in summaries]
        prox_c = [s["fedprox"]["per_class"][c_str]["f1_score"] for s in summaries]
        avg_class_f1s.append(float(np.mean(avg_c)))
        prox_class_f1s.append(float(np.mean(prox_c)))

    rects1 = ax.bar(x_classes - width_c/2, avg_class_f1s, width_c, label="FedAvg", color=colors["fedavg"], alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x_classes + width_c/2, prox_class_f1s, width_c, label="FedProx (μ=0.01)", color=colors["fedprox"], alpha=0.85, edgecolor="black")

    ax.set_ylabel("Mean F1-Score Across 5 Seeds", fontsize=12, fontweight="bold")
    ax.set_title("Per-Class Generalization Comparison (PathMNIST 9 Histopathology Tissue Classes)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_classes)
    ax.set_xticklabels([f"C{c}: {CLASS_NAMES[c]}" for c in range(9)], rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.set_ylim(0, 1.05)

    for i in range(9):
        d = prox_class_f1s[i] - avg_class_f1s[i]
        d_color = colors["delta_pos"] if d >= 0 else colors["delta_neg"]
        ax.annotate(f"{avg_class_f1s[i]:.3f}", (x_classes[i] - width_c/2, avg_class_f1s[i] + 0.015), ha='center', va='bottom', fontsize=8)
        ax.annotate(f"{prox_class_f1s[i]:.3f}", (x_classes[i] + width_c/2, prox_class_f1s[i] + 0.015), ha='center', va='bottom', fontsize=8)
        ax.annotate(f"{d:+.3f}", (x_classes[i], max(avg_class_f1s[i], prox_class_f1s[i]) + 0.055), ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=d_color,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=d_color, alpha=0.85))

    plt.tight_layout()
    f6_path = os.path.join(fig_dir, "FIG-P7-06_per_class_f1_comparison.png")
    plt.savefig(f6_path, dpi=300)
    shutil.copy(f6_path, os.path.join(artifact_fig_dir, "FIG-P7-06_per_class_f1_comparison.png"))
    shutil.copy(f6_path, os.path.join(ARTIFACT_DIR, "FIG-P7-06_per_class_f1_comparison.png"))
    plt.close()
    print("Saved FIG-P7-06")

    print("\nAll 6 scientific comparison figures generated and verified!")


if __name__ == "__main__":
    main()
