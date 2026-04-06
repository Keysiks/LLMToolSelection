import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from config import OPENROUTER_API_KEY

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

try:
    from config import MODELS_TO_TEST, OPENROUTER_API_KEY, MAX_CONCURRENT_REQUESTS, MAX_RETRIES
except Exception:
    MODELS_TO_TEST = [
        "qwen/qwen2.5-coder-7b-instruct",
        "qwen/qwen-2.5-coder-32b-instruct",
        "qwen/qwen3-235b-a22b-thinking-2507",
        "anthropic/claude-haiku-4.5",
        "x-ai/grok-code-fast-1",
        "openai/gpt-5-mini",
    ]
    OPENROUTER_API_KEY = ""
    MAX_CONCURRENT_REQUESTS = 20
    MAX_RETRIES = 3


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR / "results" / "position_bias_all_models"

DEFAULT_BENCHMARKS = ["stabletoolbench", "toollinkos"]
DEFAULT_BASE_MODEL = "qwen/qwen2.5-coder-7b-instruct"


def stable_int_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def normalize_tool_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"(_tool|_function|_api)$", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def parse_selected_tools(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return [str(x) for x in loaded]
    except Exception:
        pass

    for match in re.finditer(r"\[[\s\S]*?\]", text):
        chunk = match.group(0)
        try:
            loaded = json.loads(chunk)
            if isinstance(loaded, list):
                return [str(x) for x in loaded]
        except Exception:
            chunk_fixed = chunk.replace("'", '"')
            try:
                loaded = json.loads(chunk_fixed)
                if isinstance(loaded, list):
                    return [str(x) for x in loaded]
            except Exception:
                continue

    return []


def format_tool_for_prompt(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "input_schema": tool.get("arguments", {}),
    }


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_valid_queries(
    benchmark: str,
    context_size: int,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path = DATA_ROOT / benchmark / "benchmarks_enriched.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_data = load_dataset(dataset_path)
    valid_queries: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_data):
        question = item.get("question", "")
        if not isinstance(question, str) or not question.strip():
            continue

        gt_names = {r.get("tool") for r in item.get("reference", []) if r.get("tool")}
        if len(gt_names) != context_size:
            continue

        toolset = item.get("toolset", [])
        if not isinstance(toolset, list) or not toolset:
            continue

        gt_tools = [t for t in toolset if t.get("name") in gt_names]
        gt_tool_names = [t.get("name") for t in gt_tools if t.get("name")]

        if len(gt_tool_names) != context_size:
            continue
        if set(gt_tool_names) != gt_names:
            continue

        valid_queries.append(
            {
                "benchmark": benchmark,
                "query_id": idx,
                "query": question,
                "gt_tools": gt_tools,
                "gt_tool_names": gt_tool_names,
            }
        )

    rng = random.Random(seed + stable_int_hash(benchmark))
    if len(valid_queries) > sample_size:
        selected = rng.sample(valid_queries, sample_size)
    else:
        selected = valid_queries

    selected = sorted(selected, key=lambda x: x["query_id"])

    meta = {
        "benchmark": benchmark,
        "dataset_path": str(dataset_path),
        "valid_queries_available": len(valid_queries),
        "selected_queries": len(selected),
        "context_size": context_size,
    }
    return selected, meta


def generate_unique_permutations(
    tools: list[dict[str, Any]],
    n_permutations: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    permutations: list[list[dict[str, Any]]] = []
    seen_orders: set[tuple[str, ...]] = set()

    attempts = 0
    max_attempts = n_permutations * 50
    base_tools = list(tools)

    while len(permutations) < n_permutations and attempts < max_attempts:
        current = list(base_tools)
        rng.shuffle(current)
        key = tuple(t.get("name", "") for t in current)
        if key not in seen_orders:
            seen_orders.add(key)
            permutations.append(current)
        attempts += 1

    while len(permutations) < n_permutations:
        permutations.append(list(base_tools))

    return permutations


def make_task_key(benchmark: str, model: str, query_id: int, perm_id: int) -> str:
    return f"{benchmark}::{model}::{query_id}::{perm_id}"


def load_completed_ok_keys(raw_jsonl_path: Path) -> set[str]:
    completed: set[str] = set()
    if not raw_jsonl_path.exists():
        return completed

    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("status") != "ok":
                continue

            try:
                key = make_task_key(
                    str(obj["benchmark"]),
                    str(obj["model"]),
                    int(obj["query_id"]),
                    int(obj["perm_id"]),
                )
                completed.add(key)
            except Exception:
                continue

    return completed


def load_latest_results(raw_jsonl_path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not raw_jsonl_path.exists():
        return []

    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            try:
                key = make_task_key(
                    str(obj["benchmark"]),
                    str(obj["model"]),
                    int(obj["query_id"]),
                    int(obj["perm_id"]),
                )
            except Exception:
                continue
            latest[key] = obj

    return list(latest.values())


async def process_single_call(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    task: dict[str, Any],
    max_retries: int,
    timeout: float,
) -> dict[str, Any]:
    async with sem:
        tools_json = json.dumps(
            [format_tool_for_prompt(t) for t in task["tools_permutation"]],
            indent=2,
            ensure_ascii=False,
        )

        system_prompt = (
            "You are a helpful assistant.\n"
            "Given available tools and a user request, select all required tools.\n"
            "Return ONLY a JSON list of tool names.\n\n"
            f"Available tools:\n{tools_json}\n"
        )

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=task["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task["query"]},
                    ],
                    temperature=0.0,
                    timeout=timeout,
                )
                content = response.choices[0].message.content if response.choices else ""
                selected_raw = parse_selected_tools(content or "")
                selected_norm = sorted(
                    {
                        normalize_tool_name(name)
                        for name in selected_raw
                        if normalize_tool_name(name)
                    }
                )

                return {
                    "status": "ok",
                    "benchmark": task["benchmark"],
                    "model": task["model"],
                    "query_id": task["query_id"],
                    "perm_id": task["perm_id"],
                    "query": task["query"],
                    "selected_raw": selected_raw,
                    "selected_norm": selected_norm,
                    "selected_count": len(selected_norm),
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    return {
                        "status": "error",
                        "benchmark": task["benchmark"],
                        "model": task["model"],
                        "query_id": task["query_id"],
                        "perm_id": task["perm_id"],
                        "query": task["query"],
                        "selected_raw": [],
                        "selected_norm": [],
                        "selected_count": 0,
                        "error": str(e)[:500],
                    }


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pairwise_jaccard(sets_list: list[set[str]]) -> list[float]:
    scores: list[float] = []
    n = len(sets_list)
    for i in range(n):
        for j in range(i + 1, n):
            a = sets_list[i]
            b = sets_list[j]
            union = len(a.union(b))
            if union == 0:
                scores.append(1.0)
            else:
                scores.append(len(a.intersection(b)) / union)
    return scores


def compute_query_level_metrics(results: list[dict[str, Any]]) -> Any:
    if np is None or pd is None:
        raise RuntimeError("Для расчета метрик нужны numpy и pandas.")

    grouped: dict[tuple[str, str, int], list[set[str]]] = defaultdict(list)

    for row in results:
        if row.get("status") != "ok":
            continue
        key = (str(row["benchmark"]), str(row["model"]), int(row["query_id"]))
        grouped[key].append(set(row.get("selected_norm", [])))

    rows = []
    for (benchmark, model, query_id), selected_sets in grouped.items():
        if len(selected_sets) < 2:
            continue

        jac_scores = pairwise_jaccard(selected_sets)
        sizes = [len(s) for s in selected_sets]

        rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "query_id": query_id,
                "permutations_used": len(selected_sets),
                "mean_jaccard": float(np.mean(jac_scores)) if jac_scores else np.nan,
                "std_jaccard": float(np.std(jac_scores)) if jac_scores else np.nan,
                "variance_selected_size": float(np.var(sizes)),
                "mean_selected_size": float(np.mean(sizes)),
            }
        )

    return pd.DataFrame(rows)


def summarize_model_benchmark(query_df: Any) -> Any:
    if pd is None:
        raise RuntimeError("Для агрегации нужна pandas.")
    if query_df.empty:
        return pd.DataFrame()

    summary = (
        query_df.groupby(["benchmark", "model"], as_index=False)
        .agg(
            queries_used=("query_id", "count"),
            mean_jaccard=("mean_jaccard", "mean"),
            std_jaccard=("mean_jaccard", "std"),
            mean_variance_selected_size=("variance_selected_size", "mean"),
            mean_selected_size=("mean_selected_size", "mean"),
            mean_permutations_used=("permutations_used", "mean"),
        )
        .round(4)
    )
    return summary


def run_wilcoxon_vs_base(query_df: Any, base_model: str) -> Any:
    if np is None or pd is None:
        raise RuntimeError("Для статистики нужны numpy и pandas.")
    rows = []
    if query_df.empty:
        return pd.DataFrame(rows)

    for benchmark in sorted(query_df["benchmark"].unique()):
        base_df = query_df[
            (query_df["benchmark"] == benchmark) & (query_df["model"] == base_model)
        ][["query_id", "mean_jaccard"]].rename(columns={"mean_jaccard": "base_jaccard"})

        if base_df.empty:
            continue

        other_models = sorted(
            m
            for m in query_df[query_df["benchmark"] == benchmark]["model"].unique()
            if m != base_model
        )

        for model in other_models:
            other_df = query_df[
                (query_df["benchmark"] == benchmark) & (query_df["model"] == model)
            ][["query_id", "mean_jaccard"]].rename(columns={"mean_jaccard": "other_jaccard"})

            merged = base_df.merge(other_df, on="query_id", how="inner")
            n_pairs = len(merged)
            if n_pairs == 0:
                continue

            base_vals = merged["base_jaccard"].to_numpy()
            other_vals = merged["other_jaccard"].to_numpy()

            stat = np.nan
            p_value = np.nan
            if wilcoxon is not None and n_pairs >= 10:
                try:
                    stat, p_value = wilcoxon(base_vals, other_vals, alternative="two-sided")
                except Exception:
                    stat, p_value = np.nan, np.nan

            rows.append(
                {
                    "benchmark": benchmark,
                    "base_model": base_model,
                    "compared_model": model,
                    "n_pairs": n_pairs,
                    "base_mean_jaccard": float(np.mean(base_vals)),
                    "compared_mean_jaccard": float(np.mean(other_vals)),
                    "delta_compared_minus_base": float(np.mean(other_vals) - np.mean(base_vals)),
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                    "significant_0_05": bool(pd.notna(p_value) and p_value < 0.05),
                }
            )

    return pd.DataFrame(rows).round(6)


def build_analysis_text(summary_df: Any, wilcoxon_df: Any) -> str:
    lines = []
    lines.append("# PositionBias Analysis")
    lines.append("")
    lines.append(
        "Интерпретация: чем выше `mean_jaccard`, тем ниже чувствительность модели к порядку инструментов."
    )
    lines.append(
        "Чем выше `mean_variance_selected_size`, тем нестабильнее количество выбранных инструментов при перестановках."
    )
    lines.append("")

    if summary_df.empty:
        lines.append("Недостаточно данных для анализа.")
        return "\n".join(lines)

    for benchmark in sorted(summary_df["benchmark"].unique()):
        sub = summary_df[summary_df["benchmark"] == benchmark].sort_values(
            "mean_jaccard", ascending=False
        )
        best = sub.iloc[0]
        worst = sub.iloc[-1]
        lines.append(f"## {benchmark}")
        lines.append(
            f"- Самая устойчивая модель: `{best['model']}` (mean_jaccard={best['mean_jaccard']:.4f})."
        )
        lines.append(
            f"- Самая чувствительная модель: `{worst['model']}` (mean_jaccard={worst['mean_jaccard']:.4f})."
        )
        lines.append("")

    lines.append("## Wilcoxon vs 7B")
    if wilcoxon_df.empty:
        lines.append("- Тест не выполнен (нет данных или недоступен `scipy`).")
    else:
        sig = wilcoxon_df[wilcoxon_df["significant_0_05"] == True]
        if sig.empty:
            lines.append("- Значимых различий (p < 0.05) не обнаружено.")
        else:
            for _, row in sig.iterrows():
                lines.append(
                    f"- `{row['benchmark']}`: `{row['compared_model']}` отличается от 7B "
                    f"(p={row['p_value']:.6f}, Δ={row['delta_compared_minus_base']:.4f})."
                )

    return "\n".join(lines)


def save_sample_description(
    sample_meta: list[dict[str, Any]],
    sampled_queries: dict[str, list[dict[str, Any]]],
    out_path: Path,
) -> None:
    payload = {
        "benchmarks": sample_meta,
        "sampled_query_ids": {
            bench: [int(q["query_id"]) for q in queries]
            for bench, queries in sampled_queries.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


async def run_experiment(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_jsonl_path = OUTPUT_DIR / "raw_calls.jsonl"

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    sampled_queries: dict[str, list[dict[str, Any]]] = {}
    sample_meta: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        selected, meta = extract_valid_queries(
            benchmark=benchmark,
            context_size=args.context_size,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        sampled_queries[benchmark] = selected
        sample_meta.append(meta)

    save_sample_description(
        sample_meta=sample_meta,
        sampled_queries=sampled_queries,
        out_path=OUTPUT_DIR / "sampled_queries.json",
    )

    tasks = []
    for benchmark in benchmarks:
        for query_obj in sampled_queries[benchmark]:
            perm_seed = args.seed + stable_int_hash(f"{benchmark}:{query_obj['query_id']}")
            permutations = generate_unique_permutations(
                tools=query_obj["gt_tools"],
                n_permutations=args.permutations,
                seed=perm_seed,
            )

            for model in models:
                for perm_id, tools_perm in enumerate(permutations):
                    tasks.append(
                        {
                            "benchmark": benchmark,
                            "model": model,
                            "query_id": query_obj["query_id"],
                            "perm_id": perm_id,
                            "query": query_obj["query"],
                            "tools_permutation": tools_perm,
                        }
                    )

    print("=== План эксперимента ===")
    for meta in sample_meta:
        print(
            f"{meta['benchmark']}: доступно {meta['valid_queries_available']} | "
            f"взято {meta['selected_queries']} запросов (|GT|={meta['context_size']})"
        )
    print(f"Моделей: {len(models)}")
    print(f"Перестановок на запрос: {args.permutations}")
    print(f"Всего потенциальных API-вызовов: {len(tasks)}")

    if args.prepare_only:
        print("Режим --prepare-only: API-вызовы не выполнялись.")
        return

    if AsyncOpenAI is None:
        raise RuntimeError("В окружении не найден пакет openai. Установи зависимости и запусти снова.")

    if np is None or pd is None:
        raise RuntimeError(
            "В окружении не найдены numpy/pandas. "
            "Установи зависимости и запусти снова."
        )

    completed_ok = load_completed_ok_keys(raw_jsonl_path)
    pending_tasks = [
        t
        for t in tasks
        if make_task_key(t["benchmark"], t["model"], t["query_id"], t["perm_id"])
        not in completed_ok
    ]

    print(f"Уже готово (ok): {len(completed_ok)}")
    print(f"Осталось выполнить: {len(pending_tasks)}")

    api_key = OPENROUTER_API_KEY
    if not api_key or "sk-or" not in api_key:
        raise RuntimeError("Не найден валидный OPENROUTER_API_KEY.")

    if pending_tasks:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        sem = asyncio.Semaphore(args.max_concurrent_requests)
        write_lock = asyncio.Lock()
        done_counter = 0
        total = len(pending_tasks)

        async def run_task(task_obj: dict[str, Any]) -> None:
            nonlocal done_counter
            result = await process_single_call(
                sem=sem,
                client=client,
                task=task_obj,
                max_retries=args.max_retries,
                timeout=args.timeout,
            )
            async with write_lock:
                append_jsonl(raw_jsonl_path, result)
                done_counter += 1
                if done_counter % 50 == 0 or done_counter == total:
                    print(f"Прогресс API: {done_counter}/{total}")

        await asyncio.gather(*(run_task(t) for t in pending_tasks))
        print("API-прогон завершен.")
    else:
        print("Новых API-вызовов не требуется.")

    all_latest = load_latest_results(raw_jsonl_path)
    query_df = compute_query_level_metrics(all_latest)
    summary_df = summarize_model_benchmark(query_df)
    wilcoxon_df = run_wilcoxon_vs_base(query_df, args.base_model)

    query_df.to_csv(OUTPUT_DIR / "query_level_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary_position_bias.csv", index=False)
    wilcoxon_df.to_csv(OUTPUT_DIR / "wilcoxon_vs_7b.csv", index=False)

    analysis_text = build_analysis_text(summary_df, wilcoxon_df)
    with open(OUTPUT_DIR / "analysis.md", "w", encoding="utf-8") as f:
        f.write(analysis_text)

    print("=== Готово ===")
    print(f"Сырые вызовы: {raw_jsonl_path}")
    print(f"Метрики по запросам: {OUTPUT_DIR / 'query_level_metrics.csv'}")
    print(f"Сводная таблица: {OUTPUT_DIR / 'summary_position_bias.csv'}")
    print(f"Wilcoxon: {OUTPUT_DIR / 'wilcoxon_vs_7b.csv'}")
    print(f"Аналитический текст: {OUTPUT_DIR / 'analysis.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 7: PositionBias для 6 моделей на 2 бенчмарках."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(DEFAULT_BENCHMARKS),
        help="CSV-список бенчмарков (например: stabletoolbench,toollinkos)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS_TO_TEST),
        help="CSV-список моделей OpenRouter.",
    )
    parser.add_argument("--context-size", type=int, default=10, help="Размер GT-контекста.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Целевое число запросов на бенчмарк (~100).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10,
        help="Число перестановок на запрос.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости.")
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help="Максимум одновременных API-вызовов.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help="Максимум ретраев на вызов.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Таймаут одного API-вызова (сек).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Базовая модель для Wilcoxon-сравнения.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Только отбор запросов и план эксперимента, без API-вызовов.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_experiment(args))
