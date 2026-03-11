import json
import asyncio
import pandas as pd
import numpy as np
import re
import os
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= КОНФИГУРАЦИЯ =================

from config import *

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.normpath(os.path.join(current_dir, "..", "data", DATASET_NAME, "results"))

if not os.path.exists(results_path):
    os.makedirs(results_path)

# ================= 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

print("Загрузка модели эмбеддингов...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_tool_name(name):
    """
    Очищает имя для сравнения метрик.
    Пример: "Travel_Planner_Tool" -> "travelplanner"
    """
    if not isinstance(name, str): return ""
    name = name.lower()
    # Убираем частые суффиксы, которые модели любят добавлять
    name = re.sub(r'(_tool|_function|_api)$', '', name)
    # Оставляем только буквы и цифры
    name = re.sub(r'[^a-z0-9]', '', name)
    return name


def extract_tools_from_text(text):
    """Парсит JSON-список из текстового ответа модели"""
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            if "'" in json_str and '"' not in json_str:
                json_str = json_str.replace("'", '"')
            loaded = json.loads(json_str)
            if isinstance(loaded, list):
                # Возвращаем список строк (оригинальных)
                return [str(x) for x in loaded]
    except:
        pass
    return []


def format_tool_for_prompt(tool_def):
    """Формирует описание инструмента для промпта"""
    output_desc = "Not specified"
    if "results" in tool_def and isinstance(tool_def["results"], dict):
        output_desc = tool_def["results"].get("output_structure", "Not specified")

    return {
        "name": tool_def["name"],
        "description": tool_def.get("description", ""),
        "input_schema": tool_def.get("arguments", {}),
        "output_description": output_desc
    }


# ================= 2. ПОДГОТОВКА ДАТАСЕТА =================

def prepare_dataset_for_model(data):
    print("Подготовка данных и сортировка инструментов...")
    prepared_items = []

    for i, item in enumerate(data):
        query = item.get("question", "")

        # Ground Truth (Оригинальные имена)
        gt_names = set()
        if "reference" in item:
            for ref in item["reference"]:
                if "tool" in ref:
                    gt_names.add(ref["tool"])

        if not gt_names:
            continue

        # Ищем определения инструментов
        available_tools_map = {t["name"]: t for t in item.get("toolset", [])}
        gt_tool_defs = []
        descriptions = []

        for name in gt_names:
            if name in available_tools_map:
                tool = available_tools_map[name]
                gt_tool_defs.append(tool)
                descriptions.append(f"{tool['name']}: {tool.get('description', '')}")

        # Сортировка по Cosine Similarity (Task 4.1 Requirement)
        sorted_tools = []
        if gt_tool_defs:
            query_emb = embedder.encode([query])
            tools_emb = embedder.encode(descriptions)
            scores = cosine_similarity(query_emb, tools_emb)[0]
            sorted_pairs = sorted(zip(scores, gt_tool_defs), key=lambda x: x[0], reverse=True)
            sorted_tools = [pair[1] for pair in sorted_pairs]

        # Определение Tier
        count = len(gt_names)
        if count <= 2:
            tier = "Tier 1"
        elif 3 <= count <= 5:
            tier = "Tier 2"
        elif 6 <= count <= 10:
            tier = "Tier 3"
        else:
            tier = "Tier 4+"

        prepared_items.append({
            "id": i,
            "query": query,
            "sorted_tools": sorted_tools,
            "gt_names": list(gt_names),  # Сохраняем как список оригинальных имен
            "tier": tier
        })

    return prepared_items


# ================= 3. ЛОГИКА ЗАПРОСОВ =================

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


async def process_single_item(sem, model_id, item):
    async with sem:
        tools_list_dicts = [format_tool_for_prompt(t) for t in item["sorted_tools"]]
        tools_json_str = json.dumps(tools_list_dicts, indent=2)

        system_prompt = (
            "You are an expert AI assistant designed to select the correct tools for a user request.\n"
            "You are provided with a list of available tools below.\n\n"
            f"### AVAILABLE TOOLS ###\n{tools_json_str}\n\n"
            "### INSTRUCTION ###\n"
            "1. Analyze the User Request.\n"
            "2. Select ALL tools from the list above that are needed to solve the request.\n"
            "3. Output ONLY a JSON list of strings with the tool names.\n"
            "4. Do NOT output any code, explanations, or extra text.\n\n"
            "Example response: [\"tool_name_1\", \"tool_name_2\"]"
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"User Request: {item['query']}"}
                    ],
                    temperature=0.0,
                    timeout=60
                )

                content = response.choices[0].message.content

                selected_original = extract_tools_from_text(content)

                # 2. Нормализуем для расчета метрик
                gt_norm = {normalize_tool_name(t) for t in item["gt_names"]}
                pred_norm = {normalize_tool_name(t) for t in selected_original}

                tp = len(gt_norm.intersection(pred_norm))
                fp = len(pred_norm - gt_norm)
                fn = len(gt_norm - pred_norm)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                result_obj = {
                    "query": item['query'],
                    "ref_tools": item['gt_names'],  # Оригинальные Ground Truth
                    "selected": selected_original,  # Оригинальные ответы модели
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "num_ref_tools": len(item['gt_names']),
                    "num_selected": len(selected_original)
                }

                # Добавляем служебные поля для CSV (потом удалим перед сохранением в JSON)
                return {
                    **result_obj,
                    "model": model_id,
                    "tier": item["tier"],
                    "status": "ok"
                }

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    # Возвращаем пустой результат с ошибкой
                    return {
                        "query": item['query'],
                        "ref_tools": item['gt_names'],
                        "selected": [],
                        "precision": 0.0, "recall": 0.0, "f1": 0.0,
                        "num_ref_tools": len(item['gt_names']),
                        "num_selected": 0,
                        "model": model_id, "tier": item["tier"], "status": "error"
                    }


async def run_model_eval(model_id, dataset):
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_single_item(sem, model_id, item) for item in dataset]
    return await tqdm_asyncio.gather(*tasks, desc=f"Testing {model_id}")


# ================= 4. ЗАПУСК =================

async def main():
    # Загрузка
    raw_data = load_data(INPUT_FILE)
    if not raw_data: return

    # Подготовка (CPU)
    prepared_data = prepare_dataset_for_model(raw_data)
    print(f"Подготовлено {len(prepared_data)} задач.")

    # Список для сбора данных в CSV
    all_summary_data = []

    for model_name in MODELS_TO_TEST:
        print(f"\n🚀 Запуск модели: {model_name}")

        # Получаем результаты
        model_results = await run_model_eval(model_name, prepared_data)

        # --- 1. Сохранение JSON для конкретной модели ---
        # Очищаем результаты от служебных полей (model, tier, status)
        json_output = []
        for res in model_results:
            clean_item = {
                k: v for k, v in res.items()
                if k not in ["model", "tier", "status"]
            }
            json_output.append(clean_item)

        # Формируем имя файла (заменяем / на _)
        safe_name = model_name.replace("/", "_")
        json_path = os.path.join(RESULTS_DIR, f"{DATASET_NAME[:-1]}_{safe_name}.json")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)
        print(f"✅ JSON сохранен: {json_path}")

        # --- 2. Сбор данных для CSV ---
        # Добавляем только успешные результаты
        all_summary_data.extend([r for r in model_results if r["status"] == "ok"])

    # --- 3. Сохранение итогового CSV ---
    df = pd.DataFrame(all_summary_data)
    if not df.empty:
        summary = df.groupby(['model', 'tier']).agg(
            number_of_queries=('query', 'count'),
            avg_tools_in_tier=('num_ref_tools', 'mean'),
            precision=('precision', 'mean'),
            recall=('recall', 'mean'),
            f1=('f1', 'mean')
        ).reset_index().round(3)

        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"\n📊 Сводная таблица сохранена: {SUMMARY_CSV}")
        print(summary.head())
    else:
        print("Нет данных для таблицы.")


if __name__ == "__main__":
    if "sk-or" not in OPENROUTER_API_KEY:
        print("⚠️ ОШИБКА: Вставь API ключ.")
    else:
        asyncio.run(main())