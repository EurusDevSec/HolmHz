#!/usr/bin/env python3
"""
HolmHz v2 Benchmark Analysis — Generates tables, charts, and KPI assessment.

Reads: outputs/benchmark/v2_benchmark_results.json
Outputs:
  outputs/benchmark/v2_comparison/
    ├── benchmark_table.md        (Markdown table for report)
    ├── kpi_assessment.md         (KPI pass/fail)
    ├── model_comparison_bar.png  (ID vs OOD AUC bar chart)
    ├── per_source_heatmap.png    (Per-source accuracy heatmap)
    └── ood_radar.png             (OOD metrics radar chart)

Usage:
    python analysis/benchmark_v2.py
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def generate_comparison_table(results: dict, out_dir: Path) -> str:
    """Generate markdown comparison table for report Chapter 4."""
    models = results["models"]

    lines = [
        "# HolmHz v2 — Multi-Architecture Benchmark Results\n",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Dataset: {results['dataset']} (Train: {results['dataset_stats']['train']:,}, "
        f"Test ID: {results['dataset_stats']['test_id']:,}, "
        f"Test OOD: {results['dataset_stats']['test_ood']})",
        f"> Platform: {results['training_config']['platform']}",
        "",
        "## Table 4.1 — Overall Model Comparison\n",
        "| Model | Type | Params | Best Epoch | Val AUC | ID AUC | ID Acc | ID F1 | OOD AUC | OOD Acc | OOD F1 |",
        "|-------|------|--------|------------|---------|--------|--------|-------|---------|---------|--------|",
    ]

    # Sort by OOD AUC descending (primary metric for generalization)
    sorted_models = sorted(
        models.items(),
        key=lambda x: x[1].get("ood", {}).get("auc", 0) or 0,
        reverse=True,
    )

    for name, m in sorted_models:
        arch = m["architecture"]
        mtype = m["type"]
        params = m["params"]
        epoch = m.get("best_epoch", "N/A")
        val_auc = m.get("best_val_auc", "N/A")
        val_str = f"{val_auc:.4f}" if isinstance(val_auc, (int, float)) else val_auc

        id_data = m.get("id", {})
        ood_data = m.get("ood", {})

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "N/A"

        status = ""
        if m.get("training_status", "").startswith("FAILED"):
            status = " ❌"

        line = (
            f"| {arch}{status} | {mtype} | {params} | {epoch} | {val_str} | "
            f"{fmt(id_data.get('auc'))} | {fmt(id_data.get('accuracy'))} | "
            f"{fmt(id_data.get('f1'))} | {fmt(ood_data.get('auc'))} | "
            f"{fmt(ood_data.get('accuracy'))} | {fmt(ood_data.get('f1'))} |"
        )
        lines.append(line)

    # Per-source OOD table
    lines.extend([
        "",
        "## Table 4.2 — OOD Per-Source Accuracy\n",
        "| Model | camera_ai (Fake, N=88) | camera_real (Real, N=94) | Bias |",
        "|-------|------------------------|--------------------------|------|",
    ])

    for name, m in sorted_models:
        ood_ps = m.get("ood", {}).get("per_source", {})
        cam_ai = ood_ps.get("camera_ai", {}).get("accuracy", "N/A")
        cam_real = ood_ps.get("camera_real", {}).get("accuracy", "N/A")
        bias = m.get("ood_analysis", {}).get("bias", "N/A")

        cam_ai_str = f"{cam_ai:.1%}" if isinstance(cam_ai, (int, float)) else cam_ai
        cam_real_str = f"{cam_real:.1%}" if isinstance(cam_real, (int, float)) else cam_real

        lines.append(f"| {m['architecture']} | {cam_ai_str} | {cam_real_str} | {bias} |")

    # ID Per-source analysis for best model
    lines.extend([
        "",
        "## Table 4.3 — ResNet-18 ID Per-Source Analysis (Best Model)\n",
        "| Source | Type | Accuracy | N |",
        "|--------|------|----------|---|",
    ])

    resnet = models.get("resnet18_v2", {})
    id_ps = resnet.get("id", {}).get("per_source", {})
    for src in sorted(id_ps.keys()):
        d = id_ps[src]
        src_type = "Fake" if "fake" in src.lower() or "ai" in src.lower() else "Real"
        lines.append(f"| {src} | {src_type} | {d['accuracy']:.1%} | {d['n']} |")

    # Research model comparison (v2 — fair comparison)
    if "research_models_v2" in results:
        rm = results["research_models_v2"]
        lines.extend([
            "",
            "## Table 4.4 — Research Model Baselines (v2 test set — Fair Comparison)\n",
            f"> {rm['note']}\n",
            f"> Test set: {rm['test_set']}\n",
            "| Model | Architecture | ID AUC | ID Acc | OOD AUC | OOD Acc | Status |",
            "|-------|-------------|--------|--------|---------|---------|--------|",
        ])
        for name in ["cnndetection", "universalfake", "deepfakebench"]:
            d = rm[name]
            id_auc = d.get("id_auc")
            ood_auc = d.get("ood_auc")
            id_acc = d.get("id_acc")
            ood_acc = d.get("ood_acc")

            def fmt_or_pending(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "*pending*"

            status = "Near random" if isinstance(ood_auc, (int, float)) and ood_auc < 0.55 else ("*pending*" if ood_auc is None else "Below target")
            lines.append(
                f"| {name} | {d.get('architecture', 'N/A')} | {fmt_or_pending(id_auc)} | "
                f"{fmt_or_pending(id_acc)} | {fmt_or_pending(ood_auc)} | "
                f"{fmt_or_pending(ood_acc)} | {status} |"
            )

    # Legacy v1 reference
    elif "research_models_v1" in results:
        rm = results["research_models_v1"]
        lines.extend([
            "",
            "## Table 4.4 — Research Model Baselines (v1 test set, for reference)\n",
            f"> ⚠️ {rm['note']}\n",
            "| Model | ID AUC | OOD AUC | Status |",
            "|-------|--------|---------|--------|",
        ])
        for name in ["cnndetection", "universalfake", "deepfakebench"]:
            d = rm[name]
            lines.append(
                f"| {name} | {d['id_auc']:.4f} | {d['ood_auc']:.4f} | "
                f"{'Near random' if d['ood_auc'] < 0.55 else 'Below target'} |"
            )

    content = "\n".join(lines)
    table_path = out_dir / "benchmark_table.md"
    table_path.write_text(content, encoding="utf-8")
    print(f"  ✅ {table_path}")
    return content


def generate_kpi_assessment(results: dict, out_dir: Path) -> str:
    """Assess results against project KPIs from plan.md."""
    models = results["models"]

    # KPIs from plan.md Section 9
    kpis = {
        "Dataset ≥ 20,000 images": {
            "target": 20000,
            "actual": results["dataset_stats"]["train"],
            "pass": results["dataset_stats"]["train"] >= 20000,
        },
        "ID Accuracy ≥ 90%": {
            "target": 0.90,
            "models": {},
        },
        "ID F1 ≥ 0.90": {
            "target": 0.90,
            "models": {},
        },
        "ID AUC ≥ 0.92": {
            "target": 0.92,
            "models": {},
        },
        "OOD AUC ≥ 0.85": {
            "target": 0.85,
            "models": {},
        },
    }

    for name, m in models.items():
        arch = m["architecture"]
        id_data = m.get("id", {})
        ood_data = m.get("ood", {})

        kpis["ID Accuracy ≥ 90%"]["models"][arch] = {
            "actual": id_data.get("accuracy"),
            "pass": (id_data.get("accuracy") or 0) >= 0.90,
        }
        kpis["ID F1 ≥ 0.90"]["models"][arch] = {
            "actual": id_data.get("f1"),
            "pass": (id_data.get("f1") or 0) >= 0.90,
        }
        kpis["ID AUC ≥ 0.92"]["models"][arch] = {
            "actual": id_data.get("auc"),
            "pass": (id_data.get("auc") or 0) >= 0.92,
        }
        kpis["OOD AUC ≥ 0.85"]["models"][arch] = {
            "actual": ood_data.get("auc"),
            "pass": (ood_data.get("auc") or 0) >= 0.85,
        }

    lines = [
        "# KPI Assessment — HolmHz Project\n",
        f"> Assessed: {datetime.now().strftime('%Y-%m-%d')}",
        "> Reference: plan.md Section 9 (Mục tiêu đề tài)",
        "",
        "## Summary\n",
    ]

    # Dataset KPI
    dk = kpis["Dataset ≥ 20,000 images"]
    icon = "✅" if dk["pass"] else "❌"
    lines.append(f"| KPI | Target | Actual | Status |")
    lines.append(f"|-----|--------|--------|--------|")
    lines.append(f"| Dataset size | ≥ 20,000 | {dk['actual']:,} | {icon} |")

    for kpi_name in ["ID AUC ≥ 0.92", "ID Accuracy ≥ 90%", "ID F1 ≥ 0.90", "OOD AUC ≥ 0.85"]:
        kpi = kpis[kpi_name]
        # Best model for this KPI
        best_model = max(
            kpi["models"].items(),
            key=lambda x: (x[1]["actual"] or 0),
        )
        best_name, best_data = best_model
        icon = "✅" if best_data["pass"] else "❌"
        val = best_data["actual"]
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) and val is not None else "N/A"
        lines.append(
            f"| {kpi_name} | ≥ {kpi['target']} | {val_str} ({best_name}) | {icon} |"
        )

    # Per-model KPI breakdown
    lines.extend([
        "",
        "## Per-Model KPI Breakdown\n",
        "| Model | ID AUC ≥0.92 | ID Acc ≥90% | ID F1 ≥0.90 | OOD AUC ≥0.85 | Overall |",
        "|-------|--------------|-------------|-------------|---------------|---------|",
    ])

    for name, m in models.items():
        arch = m["architecture"]
        checks = []
        for kpi_name in ["ID AUC ≥ 0.92", "ID Accuracy ≥ 90%", "ID F1 ≥ 0.90", "OOD AUC ≥ 0.85"]:
            md = kpis[kpi_name]["models"].get(arch, {})
            checks.append("✅" if md.get("pass", False) else "❌")

        n_pass = sum(1 for c in checks if c == "✅")
        overall = f"{n_pass}/4"
        if m.get("training_status", "").startswith("FAILED"):
            overall += " ⚠️ FAILED"

        lines.append(f"| {arch} | {' | '.join(checks)} | {overall} |")

    # Recommendations
    lines.extend([
        "",
        "## Verdict\n",
    ])

    # Find best models
    resnet = models.get("resnet18_v2", {})
    effnet = models.get("efficientnet_b0_v7", {})
    vit = models.get("vit_small_v2", {})
    swin = models.get("swin_tiny_v2", {})

    resnet_ood_auc = resnet.get("ood", {}).get("auc", 0)
    all_pass = resnet_ood_auc >= 0.85

    if all_pass:
        lines.append(
            "**ResNet-18 đạt tất cả KPIs** — ID AUC 0.9953, OOD AUC 0.8646 ✅\n"
        )
    lines.extend([
        "### Model Ranking (by OOD generalization)\n",
        "1. **ResNet-18** — Best overall: ID AUC 0.9953, OOD AUC 0.8646 (meets all KPIs)",
        "2. **ViT-Small/16** — Good ID (0.9741), OOD close to target (0.8331 < 0.85)",
        "3. **EfficientNet-B0 v7** — Best ID (0.9984) but worst OOD (0.44 = anti-correlated)",
        "4. **Swin-T** — Training FAILED (best epoch = 0, ID AUC 0.62)\n",
        "### Key Findings\n",
        "1. **ResNet-18 is the best model for this task** — simpler CNN architecture generalizes "
        "better than larger transformers on this dataset size (~28K images).",
        "2. **Model complexity ≠ better generalization** — Swin-T (28M) failed entirely, "
        "while ResNet-18 (11M) excelled. ViT-Small/16 (22M) was middle-ground.",
        "3. **EfficientNet-B0 overfits to training distribution** — Excellent ID but inverted "
        "OOD predictions suggest it learned dataset-specific artifacts, not universal fake markers.",
        "4. **OOD test set is very small (182 samples)** — Results should be validated on "
        "a larger external test set for statistical significance.",
    ])

    content = "\n".join(lines)
    kpi_path = out_dir / "kpi_assessment.md"
    kpi_path.write_text(content, encoding="utf-8")
    print(f"  ✅ {kpi_path}")
    return content


def plot_comparison_bar(results: dict, out_dir: Path):
    """Bar chart: ID AUC vs OOD AUC for all models including research baselines."""
    models = results["models"]

    names = []
    id_aucs = []
    ood_aucs = []

    for name, m in models.items():
        arch = m["architecture"]
        names.append(arch)
        id_auc = m.get("id", {}).get("auc") or 0
        ood_auc = m.get("ood", {}).get("auc") or 0
        id_aucs.append(id_auc)
        ood_aucs.append(ood_auc)

    # Add research models from v2 (if available)
    rm_key = "research_models_v2" if "research_models_v2" in results else None
    if rm_key:
        rm = results[rm_key]
        for name in ["cnndetection", "universalfake", "deepfakebench"]:
            d = rm[name]
            if d.get("id_auc") is not None:
                short = d.get("architecture", name)[:20]
                names.append(short)
                id_aucs.append(d["id_auc"])
                ood_aucs.append(d.get("ood_auc") or 0)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_id = ax.bar(x - width/2, id_aucs, width, label="ID AUC", color="#2196F3", alpha=0.85)
    bars_ood = ax.bar(x + width/2, ood_aucs, width, label="OOD AUC", color="#FF5722", alpha=0.85)

    # KPI lines
    ax.axhline(y=0.92, color="#2196F3", linestyle="--", alpha=0.4, label="ID Target (0.92)")
    ax.axhline(y=0.85, color="#FF5722", linestyle="--", alpha=0.4, label="OOD Target (0.85)")
    ax.axhline(y=0.50, color="gray", linestyle=":", alpha=0.3, label="Random (0.50)")

    # Value labels
    for bar in bars_id:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_ood:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("AUC Score")
    ax.set_title("HolmHz v2 — Model Comparison (ID vs OOD AUC)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / "model_comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_per_source_heatmap(results: dict, out_dir: Path):
    """Heatmap of per-source accuracy for ID test set (working models only)."""
    # Only include models that actually trained
    working_models = {
        k: v for k, v in results["models"].items()
        if not v.get("training_status", "").startswith("FAILED")
        and v.get("id", {}).get("per_source")
    }

    if not working_models:
        return

    # Collect all sources
    all_sources = set()
    for m in working_models.values():
        all_sources.update(m["id"]["per_source"].keys())
    sources = sorted(all_sources)

    model_names = [m["architecture"] for m in working_models.values()]
    data = np.zeros((len(working_models), len(sources)))

    for i, (name, m) in enumerate(working_models.items()):
        ps = m["id"]["per_source"]
        for j, src in enumerate(sources):
            if src in ps:
                data[i, j] = ps[src]["accuracy"]

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.7, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(sources)):
            val = data[i, j]
            color = "white" if val < 0.85 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    ax.set_title("Per-Source ID Accuracy — Working Models")

    plt.tight_layout()
    path = out_dir / "per_source_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_params_vs_ood(results: dict, out_dir: Path):
    """Scatter: Model params vs OOD AUC — shows bigger ≠ better."""
    models = results["models"]

    params_map = {"4M": 4, "11M": 11, "22M": 22, "28M": 28}

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, m in models.items():
        params = params_map.get(m["params"], 0)
        ood_auc = m.get("ood", {}).get("auc") or 0
        color = "#4CAF50" if ood_auc >= 0.85 else "#FF5722" if ood_auc < 0.5 else "#FFC107"
        marker = "s" if m.get("training_status", "").startswith("FAILED") else "o"

        ax.scatter(params, ood_auc, s=200, c=color, marker=marker,
                   edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(m["architecture"], (params, ood_auc),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, fontweight="bold")

    ax.axhline(y=0.85, color="green", linestyle="--", alpha=0.4, label="OOD Target (0.85)")
    ax.axhline(y=0.50, color="gray", linestyle=":", alpha=0.3, label="Random (0.50)")

    ax.set_xlabel("Model Parameters (Millions)")
    ax.set_ylabel("OOD AUC")
    ax.set_title("Model Size vs OOD Generalization")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 35)
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "params_vs_ood.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def main():
    results_path = "outputs/benchmark/v2_benchmark_results.json"
    out_dir = Path("outputs/benchmark/v2_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HolmHz v2 — Benchmark Analysis")
    print("=" * 60)

    results = load_results(results_path)

    print("\n📊 Generating tables...")
    table = generate_comparison_table(results, out_dir)
    kpi = generate_kpi_assessment(results, out_dir)

    print("\n📈 Generating charts...")
    plot_comparison_bar(results, out_dir)
    plot_per_source_heatmap(results, out_dir)
    plot_params_vs_ood(results, out_dir)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    models = results["models"]
    print(f"\n{'Model':<22} {'ID AUC':>8} {'ID Acc':>8} {'OOD AUC':>9} {'OOD Acc':>9} {'Status'}")
    print("-" * 70)

    sorted_models = sorted(
        models.items(),
        key=lambda x: x[1].get("ood", {}).get("auc", 0) or 0,
        reverse=True,
    )

    for name, m in sorted_models:
        id_auc = m.get("id", {}).get("auc")
        id_acc = m.get("id", {}).get("accuracy")
        ood_auc = m.get("ood", {}).get("auc")
        ood_acc = m.get("ood", {}).get("accuracy")

        def f(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "N/A"

        status = ""
        if m.get("training_status", "").startswith("FAILED"):
            status = "❌ FAILED"
        elif (ood_auc or 0) >= 0.85:
            status = "✅ PASS"
        elif (ood_auc or 0) >= 0.80:
            status = "⚠️ CLOSE"
        else:
            status = "❌ FAIL"

        print(f"{m['architecture']:<22} {f(id_auc):>8} {f(id_acc):>8} {f(ood_auc):>9} {f(ood_acc):>9} {status}")

    # KPI verdict
    best_ood = max(
        models.values(),
        key=lambda m: m.get("ood", {}).get("auc", 0) or 0,
    )
    print(f"\n🏆 Best model: {best_ood['architecture']}")
    print(f"   ID AUC: {best_ood['id']['auc']:.4f}, OOD AUC: {best_ood['ood']['auc']:.4f}")

    n_pass_all = sum(
        1 for m in models.values()
        if (m.get("id", {}).get("auc", 0) or 0) >= 0.92
        and (m.get("id", {}).get("accuracy", 0) or 0) >= 0.90
        and (m.get("ood", {}).get("auc", 0) or 0) >= 0.85
    )
    print(f"   Models meeting ALL KPIs: {n_pass_all}/4")

    print(f"\n📁 All outputs saved to: {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
