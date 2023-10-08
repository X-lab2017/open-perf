import click
from data_management import DataImporter
from user_interface import DataViewer, ResultsVisualizer, Exporter
import sys
from pathlib import Path

# Assuming main.py is at the level of openperf directory
sys.path.insert(0, str(Path(__file__).parent))

@click.group()
def main():
    """OpenPerf Benchmark Suite CLI"""
    pass

@main.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--type', type=click.Choice(['csv', 'json']), default='csv', help="Data file type.")
def import_data(data_path, type):
    """Import data for benchmarking."""
    importer = DataImporter()
    if type == 'csv':
        data = importer.from_csv(data_path)
        # ...其他逻辑...
    # ...处理其他数据类型...

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
    # ...处理其他导出格式...

if __name__ == "__main__":
    manager = BenchmarkManager()

    # 假设我们有两个benchmark：classification_benchmark 和 clustering_benchmark
    manager.register_benchmark()
    manager.register_benchmark()

    manager.run_all()
    main()
