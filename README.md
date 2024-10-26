# Gold Price Forecasting using Monte Carlo Simulation

This project implements a Monte Carlo simulation approach to forecast gold prices using historical data. The analysis is performed using Python and Jupyter Notebook, providing insights into potential future price movements of gold.

## Overview

The project uses Monte Carlo methods to simulate possible future price paths for gold based on historical price data. This statistical approach helps in understanding potential price ranges and risk assessment in gold price movements.

## Prerequisites

- Python 3.x
- Git
- Required Python packages:
  - pandas
  - numpy
  - matplotlib

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib
```

## Getting Started

1. Clone the repository:

```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Source

The analysis uses historical gold price data from:
https://huggingface.co/datasets/mltrev23/gold-price/blob/main/FINAL_USO.csv

## Project Structure

- `gold-pricing-montecarlo.ipynb`: Main Jupyter notebook containing the analysis and simulation code
- `FINAL_USO.csv`: Historical gold price dataset
- `README.md`: Project documentation
- `.gitignore`: Specifies which files Git should ignore

## Usage

1. Ensure all prerequisites are installed
2. Open `gold-pricing-montecarlo.ipynb` in Jupyter Notebook or JupyterLab
3. Run the cells sequentially to perform the analysis

## Analysis Steps

1. Data loading and preprocessing
2. Statistical analysis of historical price movements
3. Monte Carlo simulation implementation
4. Visualization of results and price forecasts

## Git Setup

This repository includes a `.gitignore` file configured for Python projects. It ignores:

- Python cache files and virtual environments
- Jupyter Notebook checkpoints
- IDE-specific files
- Operating system files
- Environment files

To start tracking your changes:

```bash
git add .
git commit -m "Initial commit"
```

## License

This project is available for educational and research purposes.
