"""
Per-class and aggregate performance comparison: MobileNetV2 vs ResNet-50.

Reads pre-computed classification reports from results/ and produces:
  results/per_class_f1_comparison.png
  results/f1_difference_plot.png
  results/tier_analysis.png
  results/macro_metrics_comparison.png
  results/per_class_comparison.csv
"""

import csv
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

MOBILENET_COLOR = "#1f77b4"
RESNET_COLOR    = "#ff7f0e"
REPORT_DIR      = "results"
OUT_DIR         = "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def abbreviate(name: str) -> str:
    """'Apple___Apple_scab' -> 'Apple/Apple_scab' for readable axis labels."""
    return name.replace("___", "/")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_classification_report(path: str) -> Dict:
    """
    Parse a sklearn classification_report text file.

    Handles multi-word class names (e.g. 'Corn_(maize)___Cercospora Gray_leaf_spot')
    by requiring >= 2 spaces between the class name and the first metric value.

    Returns:
        {
          'classes':         {name: {'precision': f, 'recall': f, 'f1': f, 'support': i}},
          'accuracy':        float,
          'macro_precision': float,
          'macro_recall':    float,
          'macro_f1':        float,
        }
    """
    result: Dict = {
        "classes": {},
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
    }

    # accuracy line has only 2 trailing numbers (no precision/recall split)
    re_accuracy = re.compile(r"^\s+accuracy\s+(\d+\.\d+)\s+(\d+)\s*$")
    # macro/weighted avg lines have 4 numbers
    re_macro    = re.compile(r"^\s+macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*$")
    # class lines: name (any chars) + 2+ spaces + 4 numbers
    re_class    = re.compile(r"^(.+?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*$")

    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("weighted"):
                continue

            m = re_accuracy.match(line)
            if m:
                result["accuracy"] = float(m.group(1))
                continue

            m = re_macro.match(line)
            if m:
                result["macro_precision"] = float(m.group(1))
                result["macro_recall"]    = float(m.group(2))
                result["macro_f1"]        = float(m.group(3))
                continue

            m = re_class.match(line)
            if m:
                name = m.group(1).strip()
                if name in ("macro avg", "weighted avg"):
                    continue
                result["classes"][name] = {
                    "precision": float(m.group(2)),
                    "recall":    float(m.group(3)),
                    "f1":        float(m.group(4)),
                    "support":   int(m.group(5)),
                }

    return result


def assign_tier(mob_f1: float, res_f1: float) -> Tuple[int, str]:
    """
    Assign a performance tier (mutually exclusive, priority order):

      1 - "Both >=0.99"    : both models excel
      4 - "Both <0.95"     : both models struggle (checked before 2/3 to catch hard classes)
      2 - "MobileNet wins" : MobileNet F1 - ResNet F1 > 0.02
      3 - "ResNet wins"    : ResNet F1 - MobileNet F1 > 0.02
      5 - "Close"          : no significant difference
    """
    diff = mob_f1 - res_f1
    if mob_f1 >= 0.99 and res_f1 >= 0.99:
        return 1, "Both >=0.99"
    if mob_f1 < 0.95 and res_f1 < 0.95:
        return 4, "Both <0.95"
    if diff > 0.02:
        return 2, "MobileNet wins"
    if diff < -0.02:
        return 3, "ResNet wins"
    return 5, "Close"


# ---------------------------------------------------------------------------
# Plot 1: Per-class grouped bar chart
# ---------------------------------------------------------------------------

def plot_comparison(
    class_names: List[str],
    mob_f1s: List[float],
    res_f1s: List[float],
    out_path: str,
) -> None:
    """Horizontal grouped bar chart: per-class F1 for both models."""
    labels = [abbreviate(c) for c in class_names]
    n      = len(labels)
    y      = np.arange(n)
    bar_h  = 0.38

    fig, ax = plt.subplots(figsize=(14, 18))
    ax.barh(y + bar_h / 2, mob_f1s, bar_h, label="MobileNetV2", color=MOBILENET_COLOR, zorder=3)
    ax.barh(y - bar_h / 2, res_f1s, bar_h, label="ResNet-50",   color=RESNET_COLOR,   zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0.70, 1.03)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score: MobileNetV2 vs ResNet-50", fontsize=13, pad=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.axvline(x=1.0, color="grey", linewidth=0.6, linestyle="--", zorder=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: F1 difference bar chart
# ---------------------------------------------------------------------------

def plot_difference(
    class_names: List[str],
    differences: List[float],
    out_path: str,
) -> None:
    """Horizontal bar chart: F1 difference (MobileNet - ResNet). Green=MobileNet better."""
    labels = [abbreviate(c) for c in class_names]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in differences]

    fig, ax = plt.subplots(figsize=(12, 16))
    y = np.arange(len(labels))
    ax.barh(y, differences, color=colors, edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvline(x=0, color="black", linewidth=1.2, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("F1 Difference (MobileNetV2 - ResNet-50)", fontsize=11)
    ax.set_title("F1 Score Difference (MobileNetV2 - ResNet-50)", fontsize=13, pad=12)

    green_patch = mpatches.Patch(color="#2ca02c", label="MobileNet better")
    red_patch   = mpatches.Patch(color="#d62728", label="ResNet better")
    ax.legend(handles=[green_patch, red_patch], loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Tier distribution pie chart
# ---------------------------------------------------------------------------

def plot_tiers(tier_counts: Dict[str, int], out_path: str) -> None:
    """Pie chart: how many classes fall in each performance tier."""
    ordered_tiers = [
        ("Both >=0.99",    "#2ca02c"),
        ("MobileNet wins", MOBILENET_COLOR),
        ("ResNet wins",    RESNET_COLOR),
        ("Both <0.95",     "#d62728"),
        ("Close",          "#9467bd"),
    ]

    # Keep only tiers that have classes
    slices = [
        (label, count, color)
        for label, color in ordered_tiers
        if (count := tier_counts.get(label, 0)) > 0
    ]
    if not slices:
        return
    labels_s, counts_s, colors_s = zip(*slices)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        counts_s,
        labels=[f"{l}\n({c} classes)" for l, c in zip(labels_s, counts_s)],
        colors=colors_s,
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(10)

    ax.set_title("Class Distribution Across Performance Tiers", fontsize=13, pad=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Aggregate metrics grouped bar chart
# ---------------------------------------------------------------------------

def plot_macro_metrics(mob_report: Dict, res_report: Dict, out_path: str) -> None:
    """Grouped bar chart for accuracy, macro precision, recall, F1."""
    metrics  = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    mob_vals = [
        mob_report["accuracy"],
        mob_report["macro_precision"],
        mob_report["macro_recall"],
        mob_report["macro_f1"],
    ]
    res_vals = [
        res_report["accuracy"],
        res_report["macro_precision"],
        res_report["macro_recall"],
        res_report["macro_f1"],
    ]

    x     = np.arange(len(metrics))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_mob = ax.bar(x - bar_w / 2, mob_vals, bar_w, label="MobileNetV2", color=MOBILENET_COLOR)
    bars_res = ax.bar(x + bar_w / 2, res_vals, bar_w, label="ResNet-50",   color=RESNET_COLOR)

    for bar in list(bars_mob) + list(bars_res):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0015,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.88, 1.01)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Aggregate Performance Metrics Comparison", fontsize=13, pad=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(
    class_names: List[str],
    mob_f1s: List[float],
    res_f1s: List[float],
    differences: List[float],
    tiers: List[Tuple[int, str]],
    out_path: str,
) -> None:
    """Write per-class comparison to CSV with winner and tier columns."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "mobilenet_f1", "resnet_f1", "difference", "winner", "tier"])
        for name, mob, res, diff, (_, tier_label) in zip(
            class_names, mob_f1s, res_f1s, differences, tiers
        ):
            winner = "MobileNet" if diff > 0.02 else ("ResNet" if diff < -0.02 else "Tie")
            writer.writerow([name, f"{mob:.4f}", f"{res:.4f}", f"{diff:+.4f}", winner, tier_label])
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_tier_report(
    class_names: List[str],
    mob_f1s: List[float],
    res_f1s: List[float],
    differences: List[float],
    tiers: List[Tuple[int, str]],
) -> None:
    """Print structured tier summary to terminal."""
    buckets: Dict[str, list] = defaultdict(list)
    for name, mob, res, diff, (_, label) in zip(class_names, mob_f1s, res_f1s, differences, tiers):
        buckets[label].append((name, mob, res, diff))

    sep = "=" * 62

    print(f"\n{sep}")
    print("  TIER ANALYSIS SUMMARY")
    print(sep)

    # Tier 1
    t1 = buckets.get("Both >=0.99", [])
    print(f"\nTier 1 - Both >= 0.99  ({len(t1)} classes):")
    for name, mob, res, _ in sorted(t1, key=lambda x: x[1], reverse=True):
        print(f"  {abbreviate(name):42s}  M={mob:.4f}  R={res:.4f}")

    # Tier 2
    t2 = sorted(buckets.get("MobileNet wins", []), key=lambda x: x[3], reverse=True)
    print(f"\nTier 2 - MobileNet clearly better  ({len(t2)} classes, top 5 by margin):")
    for name, mob, res, diff in t2[:5]:
        print(f"  {abbreviate(name):42s}  diff={diff:+.4f}  (M={mob:.4f}, R={res:.4f})")

    # Tier 3
    t3 = sorted(buckets.get("ResNet wins", []), key=lambda x: x[3])
    print(f"\nTier 3 - ResNet clearly better  ({len(t3)} classes):")
    for name, mob, res, diff in t3:
        print(f"  {abbreviate(name):42s}  diff={diff:+.4f}  (M={mob:.4f}, R={res:.4f})")

    # Tier 4
    t4 = sorted(buckets.get("Both <0.95", []), key=lambda x: min(x[1], x[2]))
    print(f"\nTier 4 - Both < 0.95  ({len(t4)} classes):")
    for name, mob, res, _ in t4:
        print(f"  {abbreviate(name):42s}  M={mob:.4f}  R={res:.4f}")

    # Tier 5
    t5 = buckets.get("Close", [])
    print(f"\nTier 5 - Close, no clear winner  ({len(t5)} classes)")

    print(f"\n{sep}")
    mob_wins = len(t2)
    res_wins = len(t3)
    winner   = "MobileNetV2" if mob_wins >= res_wins else "ResNet-50"
    print(f"  MobileNet wins in {mob_wins} classes | ResNet wins in {res_wins} classes")
    print(f"  Overall winner by class count: {winner}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    mob_path = os.path.join(REPORT_DIR, "classification_report_mobilenet.txt")
    res_path = os.path.join(REPORT_DIR, "classification_report_resnet.txt")

    print("Parsing classification reports...")
    mob_report = parse_classification_report(mob_path)
    res_report = parse_classification_report(res_path)

    # Both reports must contain the same 38 classes (sorted for consistent ordering)
    class_names = sorted(mob_report["classes"].keys())
    missing = [c for c in class_names if c not in res_report["classes"]]
    if missing:
        raise ValueError(f"Classes missing from ResNet report: {missing}")
    print(f"Parsed {len(class_names)} classes from both reports.")

    mob_f1s = [mob_report["classes"][c]["f1"] for c in class_names]
    res_f1s = [res_report["classes"][c]["f1"] for c in class_names]
    diffs   = [m - r for m, r in zip(mob_f1s, res_f1s)]
    tiers   = [assign_tier(m, r) for m, r in zip(mob_f1s, res_f1s)]
    tier_counts = dict(Counter(label for _, label in tiers))

    print("\nGenerating plots...")
    plot_comparison(
        class_names, mob_f1s, res_f1s,
        os.path.join(OUT_DIR, "per_class_f1_comparison.png"),
    )
    plot_difference(
        class_names, diffs,
        os.path.join(OUT_DIR, "f1_difference_plot.png"),
    )
    plot_tiers(
        tier_counts,
        os.path.join(OUT_DIR, "tier_analysis.png"),
    )
    plot_macro_metrics(
        mob_report, res_report,
        os.path.join(OUT_DIR, "macro_metrics_comparison.png"),
    )
    save_csv(
        class_names, mob_f1s, res_f1s, diffs, tiers,
        os.path.join(OUT_DIR, "per_class_comparison.csv"),
    )

    print_tier_report(class_names, mob_f1s, res_f1s, diffs, tiers)

    print("\n=== Comparison complete ===")


if __name__ == "__main__":
    main()
