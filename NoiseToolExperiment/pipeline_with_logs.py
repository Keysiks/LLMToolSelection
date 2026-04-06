import json
import asyncio
import pandas as pd
import numpy as np
import random
import os
import re
import torch
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util

from config import *

logs_filename = "results/logs/stabletoolbench_examples.json"

if torch.backends.mps.is_available():
    device = "mps"
    print("✅ ИСПОЛЬЗУЕТСЯ УСКОРЕНИЕ: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✅ ИСПОЛЬЗУЕТСЯ УСКОРЕНИЕ: CUDA")
else:
    device = "cpu"
    print("⚠️ Работаем на CPU")

embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def load_data(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_tool(t):
    # Берем расширенное описание, если оно есть, иначе обычное
    desc = t.get("description_expanded", t.get("description", ""))
    return {
        "name": t["name"],
        "description": desc,
        "input_schema": t.get("arguments", {})
    }

def get_similarity_distractors(gt_tools, all_tools, n_needed):
    gt_names = {t['name'] for t in gt_tools}
    candidates = [t for t in all_tools if t['name'] not in gt_names]

    if not candidates: return []

    # НОВОЕ: Используем description_expanded для генерации векторов (улучшит качество шума)
    gt_texts = [f"{t['name']} {t.get('description_expanded', t.get('description', ''))}" for t in gt_tools]
    cand_texts = [f"{t['name']} {t.get('description_expanded', t.get('description', ''))}" for t in candidates]

    gt_emb = embedder.encode(gt_texts, convert_to_tensor=True)
    cand_emb = embedder.encode(cand_texts, convert_to_tensor=True)

    similarity_matrix = util.cos_sim(gt_emb, cand_emb)
    max_scores_per_candidate, _ = torch.max(similarity_matrix, dim=0)
    scores_np = max_scores_per_candidate.cpu().numpy()

    sorted_candidates = sorted(zip(scores_np, candidates), key=lambda x: x[0], reverse=True)
    return [pair[1] for pair in sorted_candidates[:n_needed]]


# --- 1. ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ ПОИСКА ---
def get_similarity_distractors(gt_tools, all_tools, global_embs, n_needed):
    gt_names = {t['name'] for t in gt_tools}

    # Кодируем только 1-5 правильных тулзов (мгновенно)
    gt_texts = [f"{t['name']} {t.get('description_expanded', t.get('description', ''))}" for t in gt_tools]
    gt_emb = embedder.encode(gt_texts, convert_to_tensor=True)

    # Сравниваем правильные тулзы со ВСЕЙ предвычисленной базой (матричное умножение = доли миллисекунды)
    similarity_matrix = util.cos_sim(gt_emb, global_embs)
    max_scores_per_candidate, _ = torch.max(similarity_matrix, dim=0)
    scores_np = max_scores_per_candidate.cpu().numpy()

    # Сортируем все тулзы по схожести
    scored_tools = sorted(zip(scores_np, all_tools), key=lambda x: x[0], reverse=True)

    # Отбираем нужные, пропуская те, что являются правильными ответами
    distractors = []
    for score, tool in scored_tools:
        if tool['name'] not in gt_names:
            distractors.append(tool)
            if len(distractors) == n_needed:
                break

    return distractors


# --- 2. ОПТИМИЗИРОВАННАЯ ГЕНЕРАЦИЯ ДАТАСЕТА ---
def prepare_dataset(data, global_tools_map, samples_per_tier=10):
    print("Отбор задач и генерация шума (ТОЛЬКО для выбранных примеров)...")

    all_tools_pool = list(global_tools_map.values())

    valid_tier1 = []
    valid_tier2 = []

    for i, item in enumerate(data):
        gt_names = set()
        if "reference" in item:
            for ref in item["reference"]:
                if "tool" in ref: gt_names.add(ref["tool"])

        count = len(gt_names)
        gt_tools_objs = [global_tools_map[name] for name in gt_names if name in global_tools_map]

        if len(gt_tools_objs) != count or count == 0:
            continue

        item["gt_tools_objs"] = gt_tools_objs
        item["gt_names"] = gt_names
        item["original_id"] = i

        if 1 <= count <= 2:
            item["tier_label"] = "Tier 1"
            valid_tier1.append(item)
        elif 3 <= count <= 5:
            item["tier_label"] = "Tier 2"
            valid_tier2.append(item)

    sampled_t1 = random.sample(valid_tier1, min(samples_per_tier, len(valid_tier1)))
    sampled_t2 = random.sample(valid_tier2, min(samples_per_tier, len(valid_tier2)))
    selected_items = sampled_t1 + sampled_t2

    print(f"Отобрано {len(sampled_t1)} (Tier 1) и {len(sampled_t2)} (Tier 2).")

    # НОВОЕ: Предвычисляем векторы для всех 2479 тулзов (выполнится 1 раз за ~1-2 секунды)
    print("Векторизация базы инструментов (около 1-2 секунд)...")
    all_texts = [f"{t['name']} {t.get('description_expanded', t.get('description', ''))}" for t in all_tools_pool]
    global_embs = embedder.encode(all_texts, convert_to_tensor=True)

    prepared_items = []

    for item in selected_items:
        gt_names = item["gt_names"]
        gt_tools_objs = item["gt_tools_objs"]
        count = len(gt_names)
        tier_label = item["tier_label"]
        query = item.get("question", "")

        for n in NOISE_LEVELS:
            if n == 0: continue

            num_noise_needed = count * n

            # --- Random ---
            candidates = [t for t in all_tools_pool if t['name'] not in gt_names]
            k = min(len(candidates), num_noise_needed)
            random_noise = random.sample(candidates, k)
            combined_rnd = gt_tools_objs + random_noise
            random.shuffle(combined_rnd)

            prepared_items.append({
                "id": item["original_id"], "query": query, "gt_names": gt_names,
                "tools_context": combined_rnd, "noise_type": "random",
                "noise_level": n, "tier": tier_label
            })

            # --- Similarity ---
            # Передаем предвычисленные глобальные векторы (global_embs)
            sim_noise = get_similarity_distractors(gt_tools_objs, all_tools_pool, global_embs, num_noise_needed)
            combined_sim = gt_tools_objs + sim_noise
            random.shuffle(combined_sim)

            prepared_items.append({
                "id": item["original_id"], "query": query, "gt_names": gt_names,
                "tools_context": combined_sim, "noise_type": "similarity",
                "noise_level": n, "tier": tier_label
            })

    print(f"Готово. Итоговый датасет для прогона API (с учетом уровней шума): {len(prepared_items)} запросов к LLM.")
    return prepared_items


client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def extract_tools(text):
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            json_str = match.group(0).replace("'", '"')
            loaded = json.loads(json_str)
            if isinstance(loaded, list): return set(str(x) for x in loaded)
    except:
        pass
    return set()


def normalize(name):
    name = name.lower()
    name = re.sub(r'(_tool|_function)$', '', name)
    return re.sub(r'[^a-z0-9]', '', name)


def format_tool(t):
    return {"name": t["name"], "description": t.get("description", ""), "input_schema": t.get("arguments", {})}


async def process_item(sem, model_id, item):
    async with sem:
        tools_str = json.dumps([format_tool(t) for t in item["tools_context"]], indent=2)

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
                    model=model_id,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": item["query"]}],
                    temperature=0.0, timeout=30.0
                )

                content = response.choices[0].message.content
                pred_raw = extract_tools(content)

                gt_norm = {normalize(x) for x in item["gt_names"]}
                pred_norm = {normalize(x) for x in pred_raw}

                tp = len(gt_norm.intersection(pred_norm))
                fp = len(pred_norm - gt_norm)
                fn = len(gt_norm - pred_norm)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                # --- Формируем список шумовых тулзов ---
                noise_tools = [t["name"] for t in item["tools_context"] if t["name"] not in item["gt_names"]]

                return {
                    "model": model_id, "noise_type": item["noise_type"],
                    "noise_level": item["noise_level"], "tier": item["tier"],
                    "f1": f1, "precision": precision, "recall": recall, "status": "ok",
                    # --- Данные для требуемого лога ---
                    "query": item["query"],
                    "ref_tools": list(item["gt_names"]),
                    "noise_tools": noise_tools,
                    "selected": list(pred_raw)
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2)
                else:
                    return {"status": "error"}


async def run_model(model_id, dataset):
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_item(sem, model_id, item) for item in dataset]
    return await tqdm_asyncio.gather(*tasks, desc=f"Testing {model_id}")


async def main():
    SAMPLES_PER_TIER = 10  # Задаем количество примеров тут

    raw_data = load_data(INPUT_FILE)
    if not raw_data: return

    raw_tools = load_data(INPUT_TOOLS)
    global_tools_map = {t["name"]: t for t in raw_tools}
    print(f"📦 Загружено {len(global_tools_map)} инструментов из базы.")

    dataset = prepare_dataset(raw_data, global_tools_map, samples_per_tier=SAMPLES_PER_TIER)
    all_results = []
    for model in MODELS_TO_TEST:
        print(f"\n🚀 {model} ...")
        res = await run_model(model, dataset)
        all_results.extend([r for r in res if r["status"] == "ok"])

    if not all_results:
        print("Нет успешных результатов для записи.")
        return

    formatted_logs = []
    for item in all_results:
        formatted_logs.append({
            "model": item["model"],
            "tier": item["tier"],
            "noise_level": item["noise_level"],
            "noise_type": item["noise_type"],
            "query": item["query"],
            "ref_tools": item["ref_tools"],
            "noise_tools": item["noise_tools"],
            "selected": item["selected"],
            "precision": round(item["precision"], 3),
            "recall": round(item["recall"], 3),
            "f1": round(item["f1"], 3)
        })

    logs_filename = os.path.join(RESULTS_DIR, f"logs_{SAMPLES_PER_TIER}_run.json")
    with open(logs_filename, "w", encoding="utf-8") as f:
        json.dump(formatted_logs, f, indent=4, ensure_ascii=False)

    print(f"\n📝 Логи прогона успешно сохранены в: {logs_filename}")

if __name__ == "__main__":
    asyncio.run(main())