import pandas as pd

# Пути к файлам
input_csv = "/Users/kiriill/Documents/Python/ComplexitySelection/data/toollinkos/results/summary_noise_qwen7b_similarity.csv"
output_excel = "output.xlsx"

df = pd.read_csv(
    input_csv,
    names=["model", "tier", "noise_level", "precision", "recall", "f1"]
)


pivot = df.pivot_table(
    index=["model", "tier"],
    columns="noise_level",
    values=["precision", "recall", "f1"],
    aggfunc="first"
)

pivot = pivot.swaplevel(0, 1, axis=1)

metric_order = ["precision", "recall", "f1"]
pivot = pivot.reindex(
    columns=sorted(pivot.columns, key=lambda x: (x[0], metric_order.index(x[1])))
)

pivot.columns = [f"noise_{noise}_{metric}" for noise, metric in pivot.columns]

result = pivot.reset_index()

# Нужный порядок моделей
model_order = [
    "qwen/qwen2.5-coder-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",

    "qwen/qwen3-235b-a22b-thinking-2507",
    "anthropic/claude-haiku-4.5",
    "x-ai/grok-code-fast-1",
    "openai/gpt-5-mini"
]

tier_order = ["Tier 1", "Tier 2"]

# Превращаем в категориальные столбцы с заданным порядком
result["model"] = pd.Categorical(result["model"], categories=model_order, ordered=True)
result["tier"] = pd.Categorical(result["tier"], categories=tier_order, ordered=True)

# Сортируем
result = result.sort_values(["tier", "model"]).reset_index(drop=True)

# Если хочешь, можно вернуть обратно строковый тип
result["model"] = result["model"].astype(str)
result["tier"] = result["tier"].astype(str)

result.to_excel(output_excel, index=False)

print(f"Готово: файл сохранён в {output_excel}")