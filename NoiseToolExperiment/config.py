MODELS_TO_TEST = [
    "qwen/qwen2.5-coder-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",

    "qwen/qwen3-235b-a22b-thinking-2507",
    "anthropic/claude-haiku-4.5",
    "x-ai/grok-code-fast-1",
    "openai/gpt-5-mini"
]

OPENROUTER_API_KEY = ""


MAX_CONCURRENT_REQUESTS = 20  # Сколько запросов слать одновременно
MAX_RETRIES = 3  # Сколько раз повторять при ошибке API
DATA_PATH = "../data/"
DATASET_NAME = "toollinkos"
RESULTS_DIR = DATA_PATH +  DATASET_NAME + "/results/"

INPUT_FILE = DATA_PATH +  DATASET_NAME + "/benchmarks_enriched.json"
NOISE_LEVELS = [1, 3, 5, 7, 9]
