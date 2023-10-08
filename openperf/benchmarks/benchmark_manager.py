from openperf.data_management.data_loader import DataLoader

from openperf.benchmarks.data_science.clustering import ClusteringBenchmark


class BenchmarkManager:
    def __init__(self, data_path):
        self.data_loader = DataLoader(data_path)
        self.benchmarks = []
        self.results = {}

    def load_data(self):
        self.data_loader.load_data()

    def register_benchmark(self, benchmark):
        self.benchmarks.append(benchmark)

    def run_all(self):
        # Ensure data is loaded once and get the needed data splits directly
        train_features, train_labels, test_features, test_labels = self.data_loader.get_data_splits()

        for benchmark in self.benchmarks:
            if isinstance(benchmark, ClusteringBenchmark):
                # Clustering benchmarks only require training data
                result = benchmark.run(train_features)
            else:
                # Classification and Regression benchmarks require both training and testing data
                result = benchmark.run(train_features, train_labels, test_features, test_labels)
            self.results[benchmark.name] = result

    def get_all_metrics(self):
        metrics = {}
        for name, result in self.results.items():
            for metric, value in result.get_metrics().items():
                metrics[f"{name}_{metric}"] = value
        return metrics
