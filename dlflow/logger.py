from unique_names_generator import get_random_name
from pathlib import Path
import json
import torch


class SimpleLogger:
    def __init__(self, logs_path, experiment_name, config):
        self.logs_path = logs_path
        self.experiment_name = experiment_name
        self.config = config

    def init(self):
        self.simple_logger_path = self.logs_path / "SimpleLogger"

        self.simple_logger_path.mkdir(exist_ok=True)
        self.experiment = Experiment(
            self.experiment_name, self.simple_logger_path)
        self.experiment.run.save_conifg(self.config)

    def log_metric(self, *args):
        self.experiment.run.log_metric(*args)

    def log_model(self, model, epoch):
        self.experiment.run.save_model(model, epoch)

    def __repr__(self):
        return f"Logger(path={self.logs_path.as_posix()})"


class Experiment:
    def __init__(self, experiment_name, simple_logger_path):
        self.experiment_name = experiment_name
        self.simple_logger_path = simple_logger_path

        self.init()

    def init(self):
        self.experiment_path = self.simple_logger_path / self.experiment_name
        self.experiment_path.mkdir(exist_ok=True)
        self.run_name = self.generate_unique_run_name()
        self.run = Run(self.experiment_path/self.run_name)

    def generate_unique_run_name(self):
        meta_file_path = self.experiment_path / "meta_data.pt"
        run_names = torch.load(
            meta_file_path) if meta_file_path.exists() else set()

        def get_unique_name():
            new_name = get_random_name(separator='-', style='lowercase')
            while new_name in run_names:
                new_name = get_random_name(separator='-', style='lowercase')
            return new_name

        # Generate and save the unique name
        unique_name = get_unique_name()
        run_names.add(unique_name)

        torch.save(run_names, meta_file_path)

        return unique_name


class Run:
    def __init__(self, run_path):
        self.run_path = run_path
        self.metrics_file_path = self.run_path / "metrics.pt"
        self.init()

    def init(self):
        if not self.run_path.exists():
            self.run_path.mkdir()

    def save_conifg(self, config):
        self.config = config
        with open(self.run_path/"config.json", "w") as f:
            json.dump(self.config, f, indent=4)

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), self.run_path /
                   f"model_(epochs={epoch}).pt")

    def log_metric(self, *args):
        metrics = torch.load(
            self.metrics_file_path) if self.metrics_file_path.exists() else []

        data_row = DataRow(*args)
        metrics.append(data_row)
        torch.save(metrics, self.metrics_file_path)


class DataRow:
    def __init__(self, *args):
        self.metric, self.metric_name, self.epoch = args
