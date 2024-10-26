# Gold Price Monte Carlo Simulation Analysis

This project implements a Monte Carlo simulation to forecast gold prices using historical data. The analysis uses 1000 simulations over a 252-day trading period (1 year) to generate potential future price paths and provide statistical insights into possible price movements.

## Technical Overview

The simulation implements the following key components:

- Daily returns calculation using percentage changes
- Normal distribution sampling based on historical return statistics
- 1000 independent price path simulations
- 252-day (1 trading year) forecast period
- Statistical analysis of final price distributions

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

## Data Source

The analysis uses historical gold price data from `FINAL_USO.csv`, available at:
https://huggingface.co/datasets/mltrev23/gold-price/blob/main/FINAL_USO.csv

## Implementation Details

The analysis is implemented in `gold-pricing-montecarlo.ipynb` with the following detailed steps:

1. **Data Preparation**

   - Import required libraries (pandas, numpy, matplotlib)
   - Load historical price data from FINAL_USO.csv
   - Convert dates to datetime format
   - Sort data chronologically

2. **Return Calculation**

   - Calculate daily returns using adjusted closing prices
   - Compute return statistics (mean, standard deviation)
   - Remove any NA values from the return series

3. **Simulation Parameters**

   - Number of simulations: 1000
   - Forecast period: 252 trading days
   - Starting price: Latest adjusted closing price
   - Return distribution: Normal distribution based on historical statistics

4. **Monte Carlo Simulation**

   - Initialize price paths matrix
   - Generate random daily returns
   - Calculate cumulative price paths
   - Store results in numpy array

5. **Visualization and Analysis**
   - Plot all simulated price paths
   - Calculate key statistics:
     - Expected (mean) price
     - Median price
     - Price standard deviation
     - 5th and 95th percentiles for confidence intervals

## Usage Instructions

1. Ensure all prerequisites are installed:

   ```bash
   pip install -r requirements.txt
   ```

2. Place the FINAL_USO.csv file in the project directory

3. Open `gold-pricing-montecarlo.ipynb` in Jupyter Notebook

4. Execute cells sequentially to:
   - Load and prepare data
   - Run simulations
   - Generate visualizations
   - View statistical analysis

## Project Structure

- `gold-pricing-montecarlo.ipynb`: Main analysis notebook
- `FINAL_USO.csv`: Historical price dataset
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore rules
- `README.md`: Project documentation

## Git Configuration

The repository includes a `.gitignore` file configured for Python projects. To initialize:

```bash
git init
git add .
git commit -m "Initial commit"
```

## Output Analysis

The simulation provides:

- Visual representation of potential price paths
- Statistical measures of price distribution:
  - Expected future price
  - Price range with 90% confidence interval
  - Volatility measures through standard deviation

## License

This project is available for educational and research purposes. The implementation is designed for academic understanding of Monte Carlo methods in financial forecasting.
