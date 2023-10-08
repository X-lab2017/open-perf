import click
from openperf.benchmarks.benchmark_manager import BenchmarkManager
import sys
from pathlib import Path

from openperf.user_interface.data_viewer import DataViewer
from openperf.user_interface.exporter import Exporter
from openperf.user_interface.results_visualizer import ResultsVisualizer

# Assuming main.py is at the level of openperf directory
sys.path.insert(0, str(Path(__file__).parent))

@click.group()
def main():
    """OpenPerf Benchmark Suite CLI"""
    pass

@main.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--benchmark_group', type=click.Choice(['index', 'data_science']), required=True, help="Benchmark group to run.")
@click.option('--benchmark_type', type=click.Choice(['activity', 'influence', 'classification', 'clustering', 'regression']), required=True, help="Specific benchmark type to run.")
@click.option('--format', type=click.Choice(['csv', 'json']), default='csv', help="Data file type.")
def benchmark(data_path, benchmark_group, benchmark_type, format):
    """Run benchmarks for given type."""
    manager = BenchmarkManager(data_path)

    # Retrieve DataLoader instance from BenchmarkManager
    data_loader = manager.get_data_loader()
    train_features, train_labels = data_loader.get_train_data()
    test_features, test_labels = data_loader.get_test_data()

    if benchmark_group == 'index':
        if benchmark_type == 'activity':
            from openperf.benchmarks.index import ActivityBenchmark
            activity_model = ActivityModel()
            manager.register_benchmark(ActivityBenchmark("Activity Benchmark", activity_model, train_features, train_labels))

        elif benchmark_type == 'influence':
            from openperf.benchmarks.index import InfluenceBenchmark
            influence_model = InfluenceModel()
            manager.register_benchmark(InfluenceBenchmark("Influence Benchmark", influence_model, test_features, test_labels))

    elif benchmark_group == 'data_science':
        if benchmark_type == 'classification':
            from openperf.benchmarks.data_science import ClassificationBenchmark
            classification_model = ClassificationModel()
            manager.register_benchmark(ClassificationBenchmark("Classification Benchmark", classification_model, train_features, train_labels))

        elif benchmark_type == 'clustering':
            from openperf.benchmarks.data_science import ClusteringBenchmark
            clustering_model = ClusteringModel()
            manager.register_benchmark(ClusteringBenchmark("Clustering Benchmark", clustering_model, train_features))

        elif benchmark_type == 'regression':
            from openperf.benchmarks.data_science import RegressionBenchmark
            regression_model = RegressionModel()
            manager.register_benchmark(RegressionBenchmark("Regression Benchmark", regression_model, train_features, train_labels))

    manager.run_all()

@main.command()
@click.argument('data_path', type=click.Path(exists=True))
def view(data_path):
    """View imported data."""
    viewer = DataViewer()
    viewer.display_data(data_path)

@main.command()
@click.argument('results_path', type=click.Path(exists=True))
def visualize(results_path):
    """Visualize benchmark results."""
    visualizer = ResultsVisualizer()
    visualizer.visualize(results_path)

@main.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['csv', 'json']), default='csv', help="Export format.")
def export(data_path, format):
    """Export data to desired format."""
    exporter = Exporter()
    if format == 'csv':
        exporter.to_csv(data_path, f"exported_data.{format}")

if __name__ == "__main__":
    main()
