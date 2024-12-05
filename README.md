# Gold Price Monte Carlo Simulation Analysis

This project implements an interactive Monte Carlo simulation to forecast gold prices using historical data. The analysis includes both a Jupyter notebook implementation and a Streamlit web application that provides real-time visualization and parameter adjustment capabilities.

## Features

- Interactive web interface for simulation parameters
- Real-time visualization of price paths
- Technical indicators (RSI, Moving Averages)
- Risk metrics (VaR, Expected Shortfall)
- Multiple simulation models (Random Walk, Geometric Brownian Motion)
- Downloadable simulation results
- Statistical analysis with price distribution visualization

## Technical Overview

The simulation implements the following key components:

- Daily returns calculation using percentage changes
- Customizable number of simulations and forecast period
- Adjustable volatility and drift parameters
- Technical analysis indicators
- Comprehensive risk metrics
- Interactive visualization using Plotly

## Prerequisites

Required Python packages:

```
pandas
numpy
matplotlib
plotly
streamlit
scipy
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Data Source

The analysis uses historical gold price data from `FINAL_USO.csv`, available at:
https://huggingface.co/datasets/mltrev23/gold-price/blob/main/FINAL_USO.csv

## Implementation Details

### Jupyter Notebook (`gold-pricing-montecarlo.ipynb`)

Basic implementation with the following steps:

1. **Data Preparation**

   - Import required libraries
   - Load historical price data
   - Convert dates to datetime format
   - Sort data chronologically

2. **Return Calculation**

   - Calculate daily returns using adjusted closing prices
   - Compute return statistics (mean, standard deviation)
   - Remove any NA values from the return series

3. **Monte Carlo Simulation**
   - Initialize price paths matrix
   - Generate random daily returns
   - Calculate cumulative price paths
   - Store results in numpy array

### Streamlit App (`gold_monte_carlo_app.py`)

Interactive web application with advanced features:

1. **Simulation Parameters**

   - Adjustable number of simulations
   - Customizable forecast period
   - Volatility and drift adjustments
   - Model selection (Random Walk/Geometric Brownian Motion)

2. **Technical Analysis**

   - Moving Averages (customizable periods)
   - RSI (Relative Strength Index)
   - Price trend visualization

3. **Risk Analytics**

   - Value at Risk (VaR)
   - Expected Shortfall
   - Maximum Drawdown
   - Price distribution statistics

4. **Visualization**
   - Real-time price path plotting
   - Confidence intervals
   - Technical indicators overlay
   - Price distribution histogram
   - Interactive Plotly charts

## Usage Instructions

### Jupyter Notebook

1. Install prerequisites:

   ```bash
   pip install -r requirements.txt
   ```

2. Place the FINAL_USO.csv file in the project directory

3. Open `gold-pricing-montecarlo.ipynb` in Jupyter Notebook

### Streamlit App

1. Install prerequisites as above

2. Run the Streamlit app:

   ```bash
   streamlit run gold_monte_carlo_app.py
   ```

3. Use the sidebar controls to:
   - Adjust simulation parameters
   - Toggle technical indicators
   - Customize visualization options
   - Download simulation results

## Project Structure

- `gold-pricing-montecarlo.ipynb`: Basic analysis notebook
- `gold_monte_carlo_app.py`: Interactive Streamlit application
- `FINAL_USO.csv`: Historical price dataset
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore rules
- `README.md`: Project documentation

## Output Analysis

The simulation provides:

- Visual representation of potential price paths
- Statistical measures of price distribution
- Technical analysis indicators
- Risk metrics including:
  - Value at Risk (VaR)
  - Expected Shortfall
  - Maximum Drawdown
  - Price range and volatility measures

## License

This project is available for educational and research purposes. The implementation is designed for academic understanding of Monte Carlo methods in financial forecasting.
