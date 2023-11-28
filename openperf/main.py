import sys
from pathlib import Path

# Assuming main.py is at the level of openperf directory
sys.path.insert(0, str(Path(__file__).parent))

import click
from openperf.benchmarks.data_science.bot_detection import bench 
from openperf.user_interface.data_viewer import DataViewer
from openperf.user_interface.exporter import Exporter
from openperf.user_interface.results_visualizer import ResultsVisualizer



@click.group()
def main():
    """OpenPerf Benchmark Suite CLI"""
    pass

@click.group()
def data_science():
    """Data Science Benchmarks"""
    pass

@data_science.command()
def bot_detection():
    """Run bot detection benchmark."""
    print("Running bot detection benchmark. Please wait...")
    result_df = bench.run()
    print(result_df.to_string())

main.add_command(data_science)
print("Available commands:", main.list_commands(ctx=None))

if __name__ == "__main__":
    main()

   



