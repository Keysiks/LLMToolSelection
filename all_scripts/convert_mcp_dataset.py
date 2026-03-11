import json
import re
import os

# ================= КОНФИГУРАЦИЯ =================

TOOLS_INPUT = "../data/LiveMcpBench/tools.json"  # Твой файл с инструментами
TASKS_INPUT = "../data/LiveMcpBench/tasks.json"  # Твой файл с задачами
OUTPUT_FILE = "../data/LiveMcpBench/benchmarks_enriched.json"  # Результат


# ================= 1. ПАРСИНГ ИНСТРУМЕНТОВ =================

def parse_tools(tools_data):
    """
    Превращает сложную вложенную структуру tools.json в плоский список инструментов.
    """
    flat_tools = []

    # tools_data может быть списком (если инструментов много) или словарем (если один)
    if isinstance(tools_data, dict):
        tools_data = [tools_data]

    for item in tools_data:
        # Проверяем верхний уровень (категории/группы)
        if "tools" in item and isinstance(item["tools"], dict):
            # Проходим по внутренним ключам (например, "bing-cn-mcp")
            for server_key, server_data in item["tools"].items():
                # Ищем список реальных функций внутри
                if "tools" in server_data and isinstance(server_data["tools"], list):
                    for tool_def in server_data["tools"]:
                        # Формируем структуру под наш бенчмарк
                        new_tool = {
                            "name": tool_def.get("name"),
                            "description": tool_def.get("description"),
                            # В твоем файле inputSchema, в бенчмарке часто arguments/parameters
                            # Мы сохраним как arguments, скрипт оценки это поймет (я добавлял адаптер)
                            "arguments": tool_def.get("inputSchema", {})
                        }

                        # Добавляем output_structure, если есть (опционально)
                        # В твоем примере его нет, но скрипт оценки это переживет.

                        flat_tools.append(new_tool)

    print(f"Извлечено инструментов: {len(flat_tools)}")
    return flat_tools


# ================= 2. ПАРСИНГ ЗАДАЧ (TASKS) =================

def parse_ground_truth_tools(metadata_tools_str):
    """
    Парсит строку вида:
    "1. get-weread-rank\n2. generate_word_cloud_chart"
    в список имен:
    ["get-weread-rank", "generate_word_cloud_chart"]
    """
    if not metadata_tools_str:
        return []

    tool_names = []
    # Разбиваем по новой строке
    lines = metadata_tools_str.split('\n')
    for line in lines:
        # Удаляем нумерацию (цифра + точка + пробел) и лишние пробелы
        # Пример: "1. tool_name " -> "tool_name"
        clean_name = re.sub(r'^\d+\.\s*', '', line).strip()
        if clean_name:
            tool_names.append(clean_name)

    return tool_names


def convert_dataset():
    # 1. Загружаем исходники
    try:
        with open(TOOLS_INPUT, 'r', encoding='utf-8') as f:
            tools_raw = json.load(f)
        with open(TASKS_INPUT, 'r', encoding='utf-8') as f:
            tasks_raw = json.load(f)
    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл {e.filename}")
        return

    # 2. Готовим плоский список всех инструментов
    # Это будет наше глобальное "меню", из которого модель должна выбирать
    all_tools_flat = parse_tools(tools_raw)

    # Создаем множество имен для проверки валидности (есть ли такой тул вообще)
    available_tool_names = set(t["name"] for t in all_tools_flat)

    converted_data = []

    # 3. Обрабатываем задачи
    for task in tasks_raw:
        question = task.get("Question")

        # Достаем список правильных инструментов из Metadata
        meta = task.get("Annotator Metadata", {})
        tools_str = meta.get("Tools", "")

        gt_tool_names = parse_ground_truth_tools(tools_str)

        # Формируем поле reference
        reference = []
        for name in gt_tool_names:
            # Проверка: есть ли такой инструмент в базе?
            if name not in available_tool_names:
                print(
                    f"⚠️ Предупреждение: В задаче {task.get('task_id')} указан тул '{name}', которого нет в tools.json!")

            reference.append({"tool": name})

        # 4. Собираем финальный объект
        # Важно: Скрипт оценки ожидает, что 'toolset' находится внутри каждого объекта.
        # Мы кладем туда ВСЕ доступные инструменты (all_tools_flat).
        # Это создает "Retrieval" задачу: найти нужные среди кучи ненужных.

        new_item = {
            "question": question,
            "toolset": all_tools_flat,  # Весь набор инструментов
            "reference": reference,
            # Доп поля (не мешают скрипту, но полезны для отладки)
            "task_id": task.get("task_id"),
            "category": task.get("category")
        }

        converted_data.append(new_item)

    # 5. Сохраняем
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Успешно сконвертировано {len(converted_data)} задач.")
    print(f"Файл сохранен как: {OUTPUT_FILE}")
    print("Теперь можно указывать этот файл в переменной INPUT_FILE в скрипте оценки.")


if __name__ == "__main__":
    convert_dataset()