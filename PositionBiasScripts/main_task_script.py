import json
import asyncio
import pandas as pd
import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI


# ================= КОНФИГУРАЦИЯ =================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = "qwen/qwen2.5-coder-7b-instruct"

# Настройки эксперимента
NUM_PERMUTATIONS = 5
NUM_SAMPLES = 10
TEMPERATURE = 0.7

MAX_CONCURRENT_REQUESTS = 20
MAX_RETRIES = 3
target_count = 8 # число тулов в запросе на которых тестим

DATASET_NAME = "toollinkos"

current_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(current_dir, "..", "data", DATASET_NAME, "benchmarks_enriched.json"))
# Папка результатов
RESULTS_DIR = os.path.join(current_dir, "..", "data", DATASET_NAME, "results", "PositionBias")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ================= 1. ПОДГОТОВКА ДАННЫХ =================

def load_and_filter_data():
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Считаем распределение (Задача 2.1)
    length_map = {}  # {кол-во_тулов: [список_задач]}

    for item in data:
        # Извлекаем Ground Truth
        gt_names = set()
        if "reference" in item:
            for ref in item["reference"]:
                if "tool" in ref:
                    gt_names.add(ref["tool"])

        count = len(gt_names)
        if count > 0:
            if count not in length_map:
                length_map[count] = []

            # Находим определения тулов
            tool_defs = []
            toolset = {t['name']: t for t in item.get('toolset', [])}
            for name in gt_names:
                if name in toolset:
                    tool_defs.append(toolset[name])

            # Сохраняем задачу, если нашли все определения
            if len(tool_defs) == count:
                length_map[count].append({
                    "query": item["question"],
                    "tools": tool_defs,  # Список объектов инструментов
                    "gt_names": gt_names
                })

    # Вывод распределения
    print("\n=== Распределение референсных тулов (Задача 2.1) ===")
    sorted_counts = sorted(length_map.keys())
    for c in sorted_counts:
        print(f"Запросов с {c} тулами: {len(length_map[c])}")

    selected_tasks = length_map[target_count]

    print(f"\nВыбрано {len(selected_tasks)} задач с количеством тулов = {target_count}")
    return selected_tasks, target_count


# ================= 2. ЛОГИКА ЭКСПЕРИМЕНТА =================

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def format_tools_to_string(tools_list):
    """Превращает список тулов в JSON строку для промпта"""
    simplified = []
    for t in tools_list:
        simplified.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("arguments", {})
        })
    return json.dumps(simplified, indent=2)


async def run_single_sample(sem, query, tools_permutation, perm_id, sample_id, task_id):
    async with sem:
        # Формируем контекст (порядок тулов важен!)
        tools_str = format_tools_to_string(tools_permutation)

        system_prompt = (
            "You are a helpful assistant.\n"
            f"Here is a list of available tools:\n{tools_str}\n\n"
            "Select ALL tools needed to answer the user request.\n"
            "Return ONLY a JSON list of tool names strings.\n"
            "Example: [\"tool_a\", \"tool_b\"]"
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=TEMPERATURE,  # > 0 для вариативности
                    timeout=45.0
                )

                content = response.choices[0].message.content

                selected_tools = set()
                try:
                    import re
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        json_str = match.group(0).replace("'", '"')
                        loaded = json.loads(json_str)
                        if isinstance(loaded, list):
                            selected_tools = set(str(x) for x in loaded)
                except:
                    pass

                position_data = []
                for idx, tool in enumerate(tools_permutation):
                    is_selected = tool['name'] in selected_tools
                    position_data.append({
                        "task_id": task_id,
                        "perm_id": perm_id,
                        "sample_id": sample_id,
                        "position_index": idx,  # Абсолютная позиция
                        "relative_pos": idx / len(tools_permutation),  # 0.0 - начало, 1.0 - конец
                        "tool_name": tool['name'],
                        "is_selected": 1 if is_selected else 0
                    })

                return {
                    "status": "ok",
                    "position_data": position_data,
                    "selected_set": selected_tools,
                    "count_selected": len(selected_tools)
                }

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2)
                else:
                    return {"status": "error"}


async def main():
    # 1. Готовим данные
    tasks, tools_count = load_and_filter_data()
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    all_tasks_coroutines = []

    # Структура для хранения результатов
    # List of coroutines

    print(f"\n🚀 Запуск эксперимента: {len(tasks)} задач x {NUM_PERMUTATIONS} перестановок x {NUM_SAMPLES} сэмплов")

    # Генерируем все задачи асинхронно
    for task_idx, task in enumerate(tasks):
        gt_tools = task['tools']  # Это список объектов

        for perm_i in range(NUM_PERMUTATIONS):
            # Задача 2.3: Случайная перестановка
            # Копируем список и мешаем
            shuffled_tools = list(gt_tools)
            random.shuffle(shuffled_tools)

            for sample_i in range(NUM_SAMPLES):
                coro = run_single_sample(
                    sem,
                    task['query'],
                    shuffled_tools,
                    perm_i,
                    sample_i,
                    task_idx
                )
                all_tasks_coroutines.append(coro)

    # Запускаем все запросы
    results = await tqdm_asyncio.gather(*all_tasks_coroutines)

    # ================= 3. АГРЕГАЦИЯ И МЕТРИКИ =================

    flat_position_data = []  # Для heatmap и корреляции

    # Группируем результаты по (task_id, perm_id) для расчета Variance и Jaccard
    groups = {}

    for res in results:
        if res['status'] != 'ok': continue

        flat_position_data.extend(res['position_data'])

        meta = res['position_data'][0]
        key = (meta['task_id'], meta['perm_id'])

        if key not in groups:
            groups[key] = []
        groups[key].append(res['selected_set'])  # Сохраняем множество выбранных

    metrics_rows = []

    for (t_id, p_id), sets_list in groups.items():
        if len(sets_list) < 2: continue

        counts = [len(s) for s in sets_list]
        variance = np.var(counts)

        jaccards = []
        for i in range(len(sets_list)):
            for j in range(i + 1, len(sets_list)):
                s1, s2 = sets_list[i], sets_list[j]
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                if union > 0:
                    jaccards.append(intersection / union)
                else:
                    jaccards.append(1.0)  # оба пустые

        avg_jaccard = np.mean(jaccards) if jaccards else 0.0

        metrics_rows.append({
            "task_id": t_id,
            "perm_id": p_id,
            "variance": variance,
            "avg_jaccard": avg_jaccard,
            "num_tools": tools_count
        })

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "variance_jaccard_metrics.csv"), index=False)

    print("\n=== Variance & Jaccard Stats ===")
    print(df_metrics.describe())

    df_pos = pd.DataFrame(flat_position_data)
    df_pos.to_csv(os.path.join(RESULTS_DIR, "position_bias_raw.csv"), index=False)

    pos_stats = df_pos.groupby("position_index")["is_selected"].mean().reset_index()

    corr, p_value = spearmanr(pos_stats["position_index"], pos_stats["is_selected"])
    print(f"\nSpearman Correlation (Position vs Selection Rate): {corr:.4f} (p={p_value:.4f})")


    def get_bucket(idx, total):
        if idx < total / 3:
            return "Start"
        elif idx < 2 * total / 3:
            return "Middle"
        else:
            return "End"

    df_pos['bucket'] = df_pos.apply(lambda x: get_bucket(x['position_index'], tools_count), axis=1)
    bucket_stats = df_pos.groupby("bucket")["is_selected"].mean()
    print("\nSelection Rate by Bucket:")
    print(bucket_stats)


    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_pos, x="position_index", y="is_selected", marker="o")
    plt.title(f"Position Bias Analysis (Model: Qwen 7B, GT_Count={tools_count})")
    plt.xlabel("Position in Prompt (Index)")
    plt.ylabel("Selection Probability")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "position_bias_plot.png"))
    print(f"\nГрафик сохранен в {RESULTS_DIR}")


if __name__ == "__main__":
    if "sk-or" not in OPENROUTER_API_KEY:
        print("Вставь ключ!")
    else:
        asyncio.run(main())