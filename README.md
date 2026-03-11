# ComplexitySelection

Скрипты для оценки моделей (через OpenRouter) в задаче *tool selection*: по каждому запросу модель должна выбрать правильные инструменты из заданного `toolset`. На выходе получается:

- JSON с результатами по каждой модели
- CSV со сводной статистикой (precision/recall/f1) по тирам сложности

## Структура репозитория

- `all_scripts/`
  - `config.py` — единый конфиг для пайплайна
  - `queries_pipeline.py` — основной пайплайн: прогон датасета через модели + подсчёт метрик + сохранение результатов
  - `recalculate_metric.py` — пересчёт тиров/агрегаций по уже сохранённым JSON
  - `convert_mcp_dataset.py` — конвертация датасета LiveMcpBench (`tools.json` + `tasks.json`) в формат `benchmarks_enriched.json`
  - `merge_top_and_without.py` — утилита для склейки частей датасета
- `data/` — датасеты и результаты

## Конфиг (`all_scripts/config.py`)

Все параметры импортируются в пайплайн как `from config import *`.

- `MODELS_TO_TEST`
  - Список моделей в формате OpenRouter (например `openai/gpt-5-mini`, `qwen/qwen2.5-coder-7b-instruct`).
  - Пайплайн прогоняет датасет последовательно по каждой модели и сохраняет отдельный JSON на модель.

- `MAX_CONCURRENT_REQUESTS`
  - Максимум параллельных запросов к API (через `asyncio.Semaphore`).

- `MAX_RETRIES`
  - Количество ретраев при ошибках API.

- `DATA_PATH`
  - Базовый путь к папке `data/`.
  - **Важно:** сейчас указано `"../data/"`, т.е. пути рассчитаны на запуск скриптов **из папки `all_scripts/`**.

- `DATASET_NAME`
  - Имя датасета (папка внутри `data/`). Например: `toolret`, `LiveMcpBench`, `ultratool` и т.д.

- `RESULTS_DIR`
  - Папка с результатами: `DATA_PATH + DATASET_NAME + "/results/"`.

- `INPUT_FILE`
  - JSON датасета, который будет читаться пайплайном: `.../benchmarks_enriched.json`.

- `OUTPUT_CSV`, `SUMMARY_CSV`
  - Путь для итоговой сводной таблицы.
  - Сейчас обе переменные указывают на один и тот же файл: `summary_task_<DATASET_NAME без последнего символа>.csv`.

## Переменные окружения (.env)

Скрипт `all_scripts/queries_pipeline.py` читает:

- `OPENROUTER_API_KEY`
  - API ключ OpenRouter.
  - Скрипт дополнительно проверяет, что ключ содержит подстроку `sk-or`.

Пример (создай файл `.env` где удобно и экспортируй переменные, либо просто экспортируй в shell):

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

> В репозитории сейчас **нет** загрузчика `.env` (типа `python-dotenv`), поэтому переменная должна быть в окружении процесса.

## Подготовка данных

Ожидаемый формат датасета для `queries_pipeline.py`:

- `data/<DATASET_NAME>/benchmarks_enriched.json`
- внутри каждого элемента должны быть поля:
  - `question` — текст запроса
  - `toolset` — список доступных инструментов (каждый с полями минимум `name`, `description`)
  - `reference` — список правильных инструментов, формат: `[ {"tool": "tool_name"}, ... ]`

В репозитории уже есть примеры:

- `data/toolret/benchmarks_enriched.json`
- `data/LiveMcpBench/benchmarks_enriched.json`

### (Опционально) Конвертация LiveMcpBench

Если есть исходники:

- `data/LiveMcpBench/tools.json`
- `data/LiveMcpBench/tasks.json`

то можно собрать `benchmarks_enriched.json`:

```bash
cd all_scripts
python convert_mcp_dataset.py
```

## Установка зависимостей

В репозитории нет `requirements.txt`/`pyproject.toml`, поэтому ставь руками (желательно в venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install openai pandas numpy tqdm sentence-transformers scikit-learn
```

## Запуск

### 1) Основной пайплайн оценки моделей

1. Проверь `all_scripts/config.py`:
   - `DATASET_NAME`
   - `MODELS_TO_TEST`
   - (при необходимости) `MAX_CONCURRENT_REQUESTS`
2. Убедись, что существует файл `data/<DATASET_NAME>/benchmarks_enriched.json`.
3. Экспортируй `OPENROUTER_API_KEY`.

Запуск (важно запускать из `all_scripts/`, потому что в конфиге относительные пути `../data/...`):

```bash
cd all_scripts
python queries_pipeline.py
```

Выходные артефакты:

- JSON по каждой модели: `data/<DATASET_NAME>/results/<DATASET_NAME без последнего символа>_<model_name с заменой / на _>.json`
- CSV сводка: `data/<DATASET_NAME>/results/summary_task_<DATASET_NAME без последнего символа>.csv`

### 2) Пересчёт тиров/сводки по готовым JSON

Если ты менял логику тиров или хочешь заново собрать агрегаты по уже сохранённым JSON в `RESULTS_DIR`:

```bash
cd all_scripts
python recalculate_metric.py
```

## Примечания

- `queries_pipeline.py` сортирует инструменты в промпте по cosine similarity между эмбеддингом запроса и эмбеддингом описаний инструментов (`SentenceTransformer('all-MiniLM-L6-v2')`).
- Тиры сложности сейчас определяются по количеству уникальных инструментов в ground truth:
  - `<=2` — Tier 1
  - `3..5` — Tier 2
  - `6..10` — Tier 3
  - `>10` — Tier 4+
