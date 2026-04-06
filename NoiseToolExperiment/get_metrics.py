import json
from config import *


path_json = "../data/stabletoolbench/results/logs_10_run.json"
with open(path_json, "r") as f:
    logs = json.load(f)


model = 0
res = {}
res_model = {}
for i in range(len(logs)):
    model = logs[i]["model"]
    tier = logs[i]["tier"]
    if (model, tier) in res.keys():
        res[(model, tier)]["f1"] += logs[i]["f1"]
        res[(model, tier)]["precision"] += logs[i]["precision"]
        res[(model, tier)]["recall"] += logs[i]["recall"]
        res_model[(model, tier)] += 1
    else:
        res[(model, tier)] = {"f1": logs[i]["f1"], "precision": logs[i]["precision"], "recall": logs[i]["recall"]}
        res_model[(model ,tier)] = 1

for key, value in res.items():
    value["f1"] /= 100
    value["precision"] /= 100
    value["recall"] /= 100
    value["f1"] = round(value["f1"], 3)
    value["precision"] = round(value["precision"], 3)
    value["recall"] = round(value["recall"], 3)
    print(key, value)

print(res_model)
