import pandas as pd
import os

DATASET_NAME = "stabletoolbench"
current_dir = os.path.dirname(os.path.abspath(__file__))
# Папка результатов

RESULTS_DIR = f"/Users/kiriill/Documents/Python/ComplexitySelection/data/{DATASET_NAME}/results/PositionBias/variance_jaccard_metrics.csv"


df = pd.read_csv(RESULTS_DIR)
print("Mean Varience: ", df["variance"].mean())
print("Mean jaccard: ", df["avg_jaccard"].mean())
print("Std Varience:", df["variance"].std())
print("Std jaccard:", df["avg_jaccard"].std())
