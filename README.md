# OpenPerf

OpenPerf is a benchmarking suite tailored for the sustainable management of open-source projects. It assesses key metrics and standards vital for the successful development of open-source ecosystems.

## Features

- **Data Science Benchmarks**: Focus on analyzing and predicting behaviors that impact the sustainability of open-source projects, such as bot detection mechanisms.
- **Standard Benchmarks**: Includes a wide range of benchmarks that measure company, developer, and project impacts on open-source community health and growth.
- **Index Benchmarks**: Provides tools for evaluating and ranking different entities based on metrics critical to open-source sustainability, such as activity levels and influence.
- **Modular CLI**: A robust command-line interface that allows for straightforward interaction with all available benchmarks, facilitating ease of use and integration into other tools.
- **Extensible Framework**: Designed to be flexible and expandable, allowing researchers and developers to add new benchmarks and features as the field evolves.

## Installation

To get started with OpenPerf, clone the repository to your local machine:

```bash
git clone https://github.com/yourgithubusername/openperf.git
cd openperf
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

OpenPerf is equipped with a CLI for easy execution of benchmarks. Here’s how you can run different types of benchmarks:

### Running Data Science Benchmarks
To run the bot detection benchmark, which helps understand automated interactions in project management:
```bash
openperf data_science bot_detection
```

### Running Standard Benchmarks
Evaluate the impact of companies, developers, and projects on open-source sustainability:
```bash
openperf standard company
openperf standard developer
openperf standard project
```

### Running Index Benchmarks
To assess activity and influence indices, crucial for understanding leadership and contributions in open-source projects:
```bash
openperf index activity
openperf index influence
```

### Extending OpenPerf
To add a new benchmark, create a new module under the appropriate directory and update the main.py to include this benchmark in the CLI.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Thanks to all the contributors who have helped to expand and maintain OpenPerf.
Special thanks to the community for the continuous feedback that enriches the project's scope and functionality.
