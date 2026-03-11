import json
import os
import pandas as pd
import glob

# ================= КОНФИГУРАЦИЯ =================

from config import *


# ================= 1. ЛОГИКА ОПРЕДЕЛЕНИЯ ТИРА =================

def calculate_tier(ref_tools_list):
    """
    ЗДЕСЬ ТЫ МОЖЕШЬ ИЗМЕНИТЬ ЛОГИКУ ТИРОВ.

    На вход подается список правильных инструментов (ref_tools).
    Сейчас логика стандартная (по количеству уникальных тулов).
    """

    # Считаем количество уникальных инструментов
    # (set убирает дубликаты, если один тул вызывается дважды)
    count = len(set(ref_tools_list))
    if count <= 2:
        tier = "Tier 1"
    elif 3 <= count <= 5:
        tier = "Tier 2"
    elif 6 <= count <= 10:
        tier = "Tier 3"
    else:
        tier = "Tier 4+"
    return tier

# ================= 2. СКРИПТ ПЕРЕСЧЕТА =================

def main():
    # Ищем все .json файлы в папке
    json_pattern = os.path.join(RESULTS_DIR, "*.json")
    files = glob.glob(json_pattern)

    if not files:
        print(f"❌ Файлы не найдены в папке: {RESULTS_DIR}")
        return

    print(f"Найдено файлов: {len(files)}")

    all_data = []

    for filepath in files:
        filename = os.path.basename(filepath)

        # Пытаемся красиво извлечь имя модели из имени файла
        # Ожидаем формат: ultratool_имя_модели.json
        model_name = filename.replace("ultratool_", "").replace(".json", "")
        # Возвращаем слэши на место, если они были заменены на _ (опционально, для красоты)
        # model_name = model_name.replace("_", "/")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Ошибка чтения {filename}: {e}")
            continue

        print(f"Обработка {model_name} ({len(data)} запросов)...")

        for item in data:
            # Получаем список правильных инструментов
            ref_tools = item.get("ref_tools", [])

            # === ПЕРЕСЧИТЫВАЕМ ТИР ===
            new_tier = calculate_tier(ref_tools)

            # Собираем данные
            all_data.append({
                "model": model_name,
                "tier": new_tier,
                "query": item.get("query"),
                "precision": item.get("precision", 0.0),
                "recall": item.get("recall", 0.0),
                "f1": item.get("f1", 0.0),
                "num_ref_tools": item.get("num_ref_tools", 0)
            })

    # ================= 3. АГРЕГАЦИЯ И СОХРАНЕНИЕ =================

    if not all_data:
        print("Данных нет.")
        return

    df = pd.DataFrame(all_data)

    # Группировка
    summary = df.groupby(['model', 'tier']).agg(
        number_of_queries=('query', 'count'),
        avg_tools_in_tier=('num_ref_tools', 'mean'),
        precision=('precision', 'mean'),
        recall=('recall', 'mean'),
        f1=('f1', 'mean')
    ).reset_index().round(3)

    # Сортировка для красоты (Model -> Tier 1 -> Tier 2...)
    summary.sort_values(by=['model', 'tier'], inplace=True)

    summary.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Готово!")
    print(f"Новая таблица сохранена в: {OUTPUT_CSV}")
    print("-" * 30)
    print(summary.head(10))


if __name__ == "__main__":
    main()