# Gold Price Monte Carlo Simulation

This project implements a Monte Carlo simulation to forecast gold prices using historical data. The analysis provides insights into potential future price movements through statistical modeling and simulation techniques.

## Project Description

The Monte Carlo simulation is implemented step-by-step to:

1. Analyze historical gold price data
2. Model price movements and volatility
3. Generate multiple simulated price paths
4. Visualize potential future price ranges

## Prerequisites

Required Python packages:

```
pandas
numpy
matplotlib
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Data

The project uses historical gold price data from `FINAL_USO.csv`, sourced from:
https://huggingface.co/datasets/mltrev23/gold-price/blob/main/FINAL_USO.csv

## Project Structure

- `gold-pricing-montecarlo.ipynb`: Jupyter notebook containing the step-by-step Monte Carlo simulation implementation
- `FINAL_USO.csv`: Historical gold price dataset
- `requirements.txt`: Python package dependencies
- `.gitignore`: Git ignore rules for Python projects
- `README.md`: Project documentation

## Implementation Steps

The analysis is implemented in `gold-pricing-montecarlo.ipynb` with the following steps:

1. **Data Preparation**

   - Import required libraries (pandas, numpy, matplotlib)
   - Load and preprocess historical gold price data

2. **Monte Carlo Simulation**
   - Calculate historical price movements
   - Generate multiple price path simulations
   - Analyze and visualize results

## Usage

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `gold-pricing-montecarlo.ipynb` in Jupyter Notebook
4. Follow the step-by-step implementation in the notebook

## Git Setup

Initialize git repository and start tracking changes:

```bash
git init
git add .
git commit -m "Initial commit"
```

The included `.gitignore` is configured for Python projects, excluding:

- Python cache and compiled files
- Virtual environments
- Jupyter checkpoints
- IDE and OS-specific files

## License

This project is available for educational and research purposes.
