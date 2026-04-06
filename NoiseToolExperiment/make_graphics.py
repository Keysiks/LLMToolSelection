from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


DATASETS = ["stabletoolbench", "toollinkos", "ultratool", "LiveMcpBench"]
NOISE_TYPES = [("random", "Random Noise"), ("similarity", "Similarity Noise")]
TIERS = ["Tier 1", "Tier 2"]
NOISE_LEVELS = [1, 3, 5, 7, 9]

MODEL_ORDER = [
    "qwen/qwen2.5-coder-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "anthropic/claude-haiku-4.5",
    "x-ai/grok-code-fast-1",
    "openai/gpt-5-mini",
]

MODEL_LABELS = {
    "qwen/qwen2.5-coder-7b-instruct": "Qwen2.5 Coder 7B",
    "qwen/qwen-2.5-coder-32b-instruct": "Qwen2.5 Coder 32B",
    "qwen/qwen3-235b-a22b-thinking-2507": "Qwen3-235B",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "x-ai/grok-code-fast-1": "Grok Code Fast 1",
    "openai/gpt-5-mini": "GPT-5 mini",
}

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"


def load_noise_csv(dataset_name: str, noise_type: str) -> pd.DataFrame:
    csv_path = DATA_ROOT / dataset_name / "results" / f"summary_noise_{noise_type}.csv"
    if not csv_path.exists():
        print(f"⚠️ Файл не найден: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    required_cols = {"model", "tier", "noise_level", "f1"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ В {csv_path} нет нужных колонок: {required_cols}")
        return pd.DataFrame()

    df = df.copy()
    df["dataset"] = dataset_name
    df["noise_type"] = noise_type
    df["noise_level"] = pd.to_numeric(df["noise_level"], errors="coerce")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

    df = df[df["tier"].isin(TIERS)]
    df = df[df["noise_level"].isin(NOISE_LEVELS)]
    df = df.dropna(subset=["model", "tier", "noise_level", "f1"])
    return df


def resolve_model_order(models: list[str]) -> list[str]:
    ordered = [m for m in MODEL_ORDER if m in models]
    leftovers = sorted(m for m in models if m not in MODEL_ORDER)
    return ordered + leftovers


def build_style_map(models: list[str]) -> dict[str, dict[str, str]]:
    palette = sns.color_palette("tab10", n_colors=max(6, len(models)))
    markers = ["o", "s", "^", "D", "P", "X", "v", ">", "<", "*"]
    style = {}
    for idx, model in enumerate(models):
        style[model] = {
            "color": palette[idx % len(palette)],
            "marker": markers[idx % len(markers)],
        }
    return style


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def plot_benchmark_panels(
    dataset_name: str,
    dataset_df: pd.DataFrame,
    model_order: list[str],
    style_map: dict[str, dict[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    fig.suptitle(f"{dataset_name}: F1 vs Noise Level", fontsize=16, fontweight="bold")

    for row_idx, (noise_type, noise_title) in enumerate(NOISE_TYPES):
        for col_idx, tier in enumerate(TIERS):
            ax = axes[row_idx, col_idx]
            panel_df = dataset_df[
                (dataset_df["noise_type"] == noise_type)
                & (dataset_df["tier"] == tier)
            ]

            for model_name in model_order:
                model_df = panel_df[panel_df["model"] == model_name].sort_values("noise_level")
                if model_df.empty:
                    continue

                style = style_map[model_name]
                ax.plot(
                    model_df["noise_level"],
                    model_df["f1"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2.0,
                    markersize=6,
                )

            ax.set_title(f"{noise_title} | {tier}", fontsize=12)
            ax.set_xticks(NOISE_LEVELS)
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, linestyle="--", alpha=0.35)

            if row_idx == 1:
                ax.set_xlabel("Noise Level n")
            if col_idx == 0:
                ax.set_ylabel("F1")

            if panel_df.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style_map[m]["color"],
            marker=style_map[m]["marker"],
            linewidth=2.0,
            markersize=6,
            label=model_label(m),
        )
        for m in model_order
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.93))

    output_path = FIGURES_DIR / f"{dataset_name}_noise_degradation_2x2.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✅ Сохранено: {output_path}")


def prepare_delta_tables(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n9_df = all_df[all_df["noise_level"] == 9].copy()
    pivot = (
        n9_df.pivot_table(
            index=["dataset", "model", "tier"],
            columns="noise_type",
            values="f1",
            aggfunc="mean",
        )
        .reset_index()
    )

    for col in ("random", "similarity"):
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["delta"] = pivot["random"] - pivot["similarity"]
    summary = (
        pivot.groupby(["dataset", "model"], as_index=False)["delta"]
        .mean()
        .sort_values(["dataset", "model"])
    )
    return pivot, summary


def plot_delta_summary(
    delta_summary_df: pd.DataFrame,
    model_order: list[str],
    style_map: dict[str, dict[str, str]],
) -> None:
    datasets_present = [d for d in DATASETS if d in set(delta_summary_df["dataset"])]
    x = np.arange(len(datasets_present))
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 7.5))

    for idx, model_name in enumerate(model_order):
        offsets = (idx - (len(model_order) - 1) / 2) * width
        values = []
        for ds in datasets_present:
            row = delta_summary_df[
                (delta_summary_df["dataset"] == ds)
                & (delta_summary_df["model"] == model_name)
            ]
            values.append(float(row["delta"].iloc[0]) if not row.empty else np.nan)

        ax.bar(
            x + offsets,
            values,
            width=width,
            color=style_map[model_name]["color"],
            label=model_label(model_name),
            alpha=0.9,
        )

    ax.axhline(0.0, color="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_present)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("ΔF1 = F1(random) - F1(similarity) at n=9")
    ax.set_title("Overall Delta by Benchmark (averaged over Tier 1/Tier 2)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style_map[m]["color"],
            marker="s",
            linewidth=0,
            markersize=10,
            label=model_label(m),
        )
        for m in model_order
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.24)
    output_path = FIGURES_DIR / "delta_n9_by_benchmark.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✅ Сохранено: {output_path}")


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Arial",
        "Helvetica",
        "Liberation Sans",
    ]
    sns.set_theme(style="whitegrid", context="talk")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_frames = {}
    for dataset_name in DATASETS:
        random_df = load_noise_csv(dataset_name, "random")
        similarity_df = load_noise_csv(dataset_name, "similarity")
        dataset_df = pd.concat([random_df, similarity_df], ignore_index=True)
        dataset_df = dataset_df.dropna(subset=["model", "tier", "noise_level", "f1"])
        if dataset_df.empty:
            print(f"⚠️ Пропускаю {dataset_name}: нет данных.")
            continue
        dataset_frames[dataset_name] = dataset_df

    if not dataset_frames:
        print("❌ Нет данных для построения графиков.")
        return

    all_df = pd.concat(dataset_frames.values(), ignore_index=True)
    all_models = sorted(all_df["model"].dropna().unique().tolist())
    model_order = resolve_model_order(all_models)
    style_map = build_style_map(model_order)

    for dataset_name in DATASETS:
        if dataset_name not in dataset_frames:
            continue
        plot_benchmark_panels(
            dataset_name=dataset_name,
            dataset_df=dataset_frames[dataset_name],
            model_order=model_order,
            style_map=style_map,
        )

    delta_by_tier_df, delta_summary_df = prepare_delta_tables(all_df)
    delta_by_tier_df.to_csv(FIGURES_DIR / "delta_n9_by_tier.csv", index=False)
    delta_summary_df.to_csv(FIGURES_DIR / "delta_n9_summary.csv", index=False)
    print(f"✅ Сохранено: {FIGURES_DIR / 'delta_n9_by_tier.csv'}")
    print(f"✅ Сохранено: {FIGURES_DIR / 'delta_n9_summary.csv'}")

    plot_delta_summary(
        delta_summary_df=delta_summary_df,
        model_order=model_order,
        style_map=style_map,
    )


if __name__ == "__main__":
    main()
