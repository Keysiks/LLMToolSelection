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


def get_similarity_distractors(gt_tools, all_tools, n_needed):
    gt_names = {t['name'] for t in gt_tools}
    candidates = [t for t in all_tools if t['name'] not in gt_names]

    if not candidates: return []

    gt_texts = [f"{t['name']} {t.get('description', '')}" for t in gt_tools]
    cand_texts = [f"{t['name']} {t.get('description', '')}" for t in candidates]

    gt_emb = embedder.encode(gt_texts, convert_to_tensor=True)
    cand_emb = embedder.encode(cand_texts, convert_to_tensor=True)

    similarity_matrix = util.cos_sim(gt_emb, cand_emb)
    max_scores_per_candidate, _ = torch.max(similarity_matrix, dim=0)
    scores_np = max_scores_per_candidate.cpu().numpy()

    sorted_candidates = sorted(zip(scores_np, candidates), key=lambda x: x[0], reverse=True)
    return [pair[1] for pair in sorted_candidates[:n_needed]]


def prepare_dataset(data):
    print("Генерация шума:")
    prepared_items = []

    tier1_count = 0
    tier2_count = 0

    for i, item in enumerate(data):
        query = item.get("question", "")

        gt_names = set()
        if "reference" in item:
            for ref in item["reference"]:
                if "tool" in ref: gt_names.add(ref["tool"])

        count = len(gt_names)

        tier_label = ""

        if 1 <= count <= 2:
            tier_label = "Tier 1"
            tier1_count += 1
        elif 3 <= count <= 5:
            tier_label = "Tier 2"
            tier2_count += 1
        else:
            continue

        available_tools = item.get("toolset", [])
        available_map = {t['name']: t for t in available_tools}

        gt_tools_objs = []
        for name in gt_names:
            if name in available_map: gt_tools_objs.append(available_map[name])

        if len(gt_tools_objs) != count: continue

        for n in NOISE_LEVELS:
            # Сколько шума добавить (N * количество правильных)
            num_noise_needed = count * n

            # --- Random ---
            candidates = [t for t in available_tools if t['name'] not in gt_names]
            k = min(len(candidates), num_noise_needed)
            random_noise = random.sample(candidates, k)
            combined_rnd = gt_tools_objs + random_noise
            random.shuffle(combined_rnd)

            prepared_items.append({
                "id": i, "query": query, "gt_names": gt_names,
                "tools_context": combined_rnd, "noise_type": "random",
                "noise_level": n, "tier": tier_label
            })

            # --- Similarity ---
            sim_noise = get_similarity_distractors(gt_tools_objs, available_tools, num_noise_needed)
            combined_sim = gt_tools_objs + sim_noise
            random.shuffle(combined_sim)

            prepared_items.append({
                "id": i, "query": query, "gt_names": gt_names,
                "tools_context": combined_sim, "noise_type": "similarity",
                "noise_level": n, "tier": tier_label
            })

    print(f"Готово. Статистика по исходным задачам:")
    print(f"Tier 1 (1-2 tools): {tier1_count} задач")
    print(f"Tier 2 (3-5 tools): {tier2_count} задач")
    print(f"Всего тестовых сэмплов (с учетом уровней шума): {len(prepared_items)}")

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

                return {
                    "model": model_id, "noise_type": item["noise_type"],
                    "noise_level": item["noise_level"], "tier": item["tier"],
                    "f1": f1, "precision": precision, "recall": recall, "status": "ok"
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
    raw_data = load_data(INPUT_FILE)
    if not raw_data: return

    dataset = prepare_dataset(raw_data)

    all_results = []
    for model in MODELS_TO_TEST:
        print(f"\n🚀 {model} ...")
        res = await run_model(model, dataset)
        all_results.extend([r for r in res if r["status"] == "ok"])

    df = pd.DataFrame(all_results)
    if df.empty:
        print("Нет данных.")
        return

    for noise_type in ["random", "similarity"]:
        df_sub = df[df["noise_type"] == noise_type]
        summary = df_sub.groupby(['model', 'tier', 'noise_level'])[
            ['precision', 'recall', 'f1']].mean().reset_index().round(3)

        summary.sort_values(by=['model', 'tier', 'noise_level'], inplace=True)

        filename = os.path.join(RESULTS_DIR, f"summary_noise_qwen7b_{noise_type}.csv")
        summary.to_csv(filename, index=False)
        print(f"\n📊 Таблица ({noise_type}) сохранена: {filename}")
        print(summary.head(10))


if __name__ == "__main__":
    asyncio.run(main())