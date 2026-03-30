import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics_by_tier(data, metric='f1', output_filename=None, file_mode=None):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    tiers = df['tier'].unique()

    sns.set_theme(style="whitegrid")

    for tier in sorted(tiers):
        tier_data = df[df['tier'] == tier]
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=tier_data,
            x='noise_level',
            y=metric,
            hue='model',
            marker='o',
            linewidth=2,  # Толщина линии
            markersize=8  # Размер точек
        )

        plt.title(f'Зависимость {metric.upper()} от уровня шума ({tier})', fontsize=16, pad=15)
        plt.xlabel('Уровень шума (noise_level)', fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)

        plt.xticks(sorted(df['noise_level'].unique()))

        plt.legend(
            title='Модели',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.
        )

        output_filename_by_tier = f"results/plot_{dataset_name}_{tier}_{file_mode}.png"
        plt.savefig(output_filename_by_tier, dpi=300, bbox_inches='tight')
        print(f"График успешно сохранен в файл: {output_filename}")
        plt.close()


datasets_names = ["stabletoolbench", "toollinkos", "ultratool", "LiveMcpBench"]
for dataset_name in datasets_names:
    data_path_random = "../data/" + dataset_name + "/results/summary_noise_random.csv"
    data_path_similarity = "../data/" + dataset_name + "/results/summary_noise_similarity.csv"

    plot_metrics_by_tier(data_path_random, metric='f1', output_filename=dataset_name, file_mode="random")
    plot_metrics_by_tier(data_path_similarity, metric='f1', output_filename=dataset_name, file_mode="similarity")
