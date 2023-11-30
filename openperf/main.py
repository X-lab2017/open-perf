import sys
from pathlib import Path

# Assuming main.py is at the level of openperf directory
sys.path.insert(0, str(Path(__file__).parent))

import click
from openperf.benchmarks.data_science.bot_detection import bench 
from openperf.benchmarks.standard.company import activity as c_a, influence as c_i
from openperf.benchmarks.standard.developer import activity as d_a, influence as d_i
from openperf.benchmarks.standard.project import activity as p_a, influence as p_i
from openperf.benchmarks.index.rank import activity as index_activity, influence as index_influence
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

@click.group()
def index():
    """index Benchmarks"""
    pass

@index.command()
def activity():
    print(index_activity.run())
    
@index.command()
def influence():
    print(index_influence.run())

@data_science.command()
def bot_detection():
    """Run bot detection benchmark."""
    print("Running bot detection benchmark. Please wait...")
    result_df = bench.run()
    print(result_df.to_string())

@click.group()
def standard():
    """Standard Benchmarks"""
    pass

@standard.command()
def company():
    """Run company benchmark."""
    print("activity:")
    print(c_a.run())
    print("influence:")
    print(c_i.run())

@standard.command()
def developer():
    """Run developer benchmark."""
    print("activity:")
    print(d_a.run())
    print("influence:")
    print(d_i.run())

@standard.command()
def project():
    """Run project benchmark."""
    print("activity:")
    print(p_a.run())
    print("influence:")
    print(p_i.run())

main.add_command(index)
main.add_command(standard)
main.add_command(data_science)
print("Available commands:", main.list_commands(ctx=None))

if __name__ == "__main__":
    main()

   



