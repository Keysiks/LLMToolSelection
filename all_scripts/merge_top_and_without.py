import json
datasets = ["toolret"]


for dataset in datasets:
    with open(f"../data/{dataset}/top_benchmarks_enriched.json", "r") as f:
        top_benchmarks = json.load(f)
    with open(f"../data/{dataset}/without_top_benchmarks_enriched.json", "r") as f:
        without_benchmarks = json.load(f)
    with open(f"../data/{dataset}/benchmarks_enriched.json", "w") as f:
        json.dump(top_benchmarks + without_benchmarks, f)