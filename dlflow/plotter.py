import matplotlib.pyplot as plt
import torch
import pandas as pd


class SimplePlotter:
    def __init__(self, metrics_path):
        self.metrics_path = metrics_path

    def extract_metrics(self):
        metrics = torch.load(self.metrics_path)

        data = []
        for m in metrics:
            data.append({
                'epoch': m.epoch,
                'metric_name': m.metric_name,
                'metric': m.metric
            })

        df = pd.DataFrame(data)
        grouped_df = df.groupby(['epoch', 'metric_name']).mean().reset_index()
        df_pivot = grouped_df.pivot(
            index='epoch', columns='metric_name', values='metric')

        return df_pivot

    def plot_metrics(self, by="loss"):
        df = self.extract_metrics()

        if by == "loss":
            df = df[["train_loss", "val_loss"]]
        elif by == "accuracy":
            df = df[["train_accuracy", "val_accuracy"]]

        df.plot(kind='line', marker='o')

        # Ensure the x-axis (epoch) is displayed only as integers
        plt.xticks(torch.arange(
            df.index.min(), df.index.max() + 1, step=1))

        plt.title('Metrics over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.grid(True)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        plt.show()
