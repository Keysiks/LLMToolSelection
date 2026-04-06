# PositionBias Analysis

Интерпретация: чем выше `mean_jaccard`, тем ниже чувствительность модели к порядку инструментов.
Чем выше `mean_variance_selected_size`, тем нестабильнее количество выбранных инструментов при перестановках.

## stabletoolbench
- Самая устойчивая модель: `x-ai/grok-code-fast-1` (mean_jaccard=0.9527).
- Самая чувствительная модель: `qwen/qwen2.5-coder-7b-instruct` (mean_jaccard=0.6416).

## toollinkos
- Самая устойчивая модель: `x-ai/grok-code-fast-1` (mean_jaccard=0.9359).
- Самая чувствительная модель: `qwen/qwen2.5-coder-7b-instruct` (mean_jaccard=0.6028).

## Wilcoxon vs 7B
- `stabletoolbench`: `anthropic/claude-haiku-4.5` отличается от 7B (p=0.000000, Δ=0.2953).
- `stabletoolbench`: `openai/gpt-5-mini` отличается от 7B (p=0.000000, Δ=0.2865).
- `stabletoolbench`: `qwen/qwen-2.5-coder-32b-instruct` отличается от 7B (p=0.000000, Δ=0.2984).
- `stabletoolbench`: `qwen/qwen3-235b-a22b-thinking-2507` отличается от 7B (p=0.000000, Δ=0.3003).
- `stabletoolbench`: `x-ai/grok-code-fast-1` отличается от 7B (p=0.000000, Δ=0.3111).
- `toollinkos`: `anthropic/claude-haiku-4.5` отличается от 7B (p=0.000000, Δ=0.1927).
- `toollinkos`: `openai/gpt-5-mini` отличается от 7B (p=0.000000, Δ=0.2076).
- `toollinkos`: `qwen/qwen-2.5-coder-32b-instruct` отличается от 7B (p=0.000000, Δ=0.2588).
- `toollinkos`: `qwen/qwen3-235b-a22b-thinking-2507` отличается от 7B (p=0.000000, Δ=0.3064).
- `toollinkos`: `x-ai/grok-code-fast-1` отличается от 7B (p=0.000000, Δ=0.3331).