import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page config
st.set_page_config(
    page_title="Gold Price Monte Carlo Simulation",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("🏆 Gold Price Monte Carlo Simulation")
st.markdown("""
This application performs Monte Carlo simulation to forecast potential gold prices.
The simulation generates multiple possible price paths based on historical data and volatility patterns.

### How it works:
1. Historical gold price data is used to calculate daily returns
2. Multiple price paths are simulated using random walks
3. Statistical analysis is performed on the simulated paths
""")

# Load and process data
@st.cache_data
def load_data():
    data = pd.read_csv("FINAL_USO.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data['Daily Return'] = data['Adj Close'].pct_change()
    return data

# Load data
data = load_data()
daily_returns = data['Daily Return'].dropna()
starting_price = data['Adj Close'].iloc[-1]

# Sidebar controls with better descriptions
st.sidebar.header("📊 Simulation Parameters")

st.sidebar.markdown("### Number of Simulations")
st.sidebar.markdown("More simulations provide better statistical accuracy but take longer to compute.")
num_simulations = st.sidebar.slider("Select number of paths", 100, 1000, 500, 100)

st.sidebar.markdown("### Forecast Horizon")
st.sidebar.markdown("Choose how many trading days to forecast into the future.")
num_days = st.sidebar.slider("Trading days to simulate", 30, 365, 252, 
                           help="252 trading days ≈ 1 year")

# Display initial price
st.sidebar.markdown("### Current Price Information")
st.sidebar.markdown(f"**Starting Price:** ${starting_price:.2f}")
st.sidebar.markdown(f"**Historical Volatility:** {daily_returns.std() * np.sqrt(252):.1%} annually")

# Create placeholder for simulation progress
progress_container = st.container()
with progress_container:
    progress_bar = st.progress(0)
    status_text = st.empty()

# Initialize simulation
simulated_prices = np.zeros((num_days, num_simulations))
simulated_prices[0, :] = starting_price

# Create figures placeholder
fig_placeholder = st.empty()

# Create two columns for statistics
col1, col2 = st.columns(2)

# Run simulation with animation
for day in range(1, num_days):
    # Update progress
    progress = day / (num_days - 1)
    progress_bar.progress(progress)
    status_text.text(f"Simulating day {day}/{num_days-1}")
    
    # Calculate prices for this day
    random_daily_returns = np.random.normal(daily_returns.mean(), daily_returns.std(), num_simulations)
    simulated_prices[day, :] = simulated_prices[day-1, :] * (1 + random_daily_returns)
    
    if day % 5 == 0 or day == num_days-1:  # Update plot every 5 days
        fig = go.Figure()
        
        # Add all simulation paths
        for sim in range(0, num_simulations, max(1, num_simulations//100)):
            fig.add_trace(
                go.Scatter(
                    y=simulated_prices[:day+1, sim],
                    mode='lines',
                    line=dict(width=1, color='rgba(0,100,255,0.1)'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        # Add mean line
        mean_line = np.mean(simulated_prices[:day+1, :], axis=1)
        fig.add_trace(
            go.Scatter(
                y=mean_line,
                mode='lines',
                line=dict(width=3, color='red', dash='dash'),
                name='Mean Path'
            )
        )
        
        # Add confidence intervals
        percentile_5 = np.percentile(simulated_prices[:day+1, :], 5, axis=1)
        percentile_95 = np.percentile(simulated_prices[:day+1, :], 95, axis=1)
        
        fig.add_trace(
            go.Scatter(
                y=percentile_95,
                mode='lines',
                line=dict(width=0),
                showlegend=True,
                name='95% Confidence Interval',
                fillcolor='rgba(0,100,255,0.2)',
                fill='tonexty'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                y=percentile_5,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                fillcolor='rgba(0,100,255,0.2)',
                fill='tonexty'
            )
        )
        
        fig.update_layout(
            title={
                'text': 'Monte Carlo Simulation of Gold Prices',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Trading Days',
            yaxis_title='Price ($)',
            height=600,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        # Update the plot
        fig_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display statistics
        current_prices = simulated_prices[day, :]
        
        with col1:
            st.subheader("📈 Current Statistics")
            st.write(f"Mean Price: ${np.mean(current_prices):.2f}")
            st.write(f"Median Price: ${np.median(current_prices):.2f}")
            st.write(f"Standard Deviation: ${np.std(current_prices):.2f}")
        
        with col2:
            st.subheader("🎯 Price Ranges")
            st.write(f"5th Percentile: ${np.percentile(current_prices, 5):.2f}")
            st.write(f"95th Percentile: ${np.percentile(current_prices, 95):.2f}")
            st.write(f"Price Range: ${np.min(current_prices):.2f} - ${np.max(current_prices):.2f}")
        
        time.sleep(0.1)  # Small delay for animation effect

# Final status
status_text.text("✅ Simulation completed!")
progress_bar.progress(1.0)

# Add histogram of final prices
st.subheader("📊 Distribution of Final Prices")
fig_hist = go.Figure()
fig_hist.add_trace(
    go.Histogram(
        x=simulated_prices[-1, :],
        nbinsx=50,
        name='Final Prices',
        marker_color='rgba(0,100,255,0.5)'
    )
)
fig_hist.update_layout(
    title='Distribution of Simulated Final Prices',
    xaxis_title='Price ($)',
    yaxis_title='Frequency',
    height=400,
    template='plotly_dark'
)
st.plotly_chart(fig_hist, use_container_width=True)

# Display summary statistics
st.subheader("📑 Final Simulation Summary")
final_prices = simulated_prices[-1, :]
summary_data = {
    "Metric": [
        "Expected Price (Mean)",
        "Median Price",
        "Standard Deviation",
        "5th Percentile",
        "95th Percentile",
        "Minimum Price",
        "Maximum Price",
        "Price Range"
    ],
    "Value": [
        f"${np.mean(final_prices):.2f}",
        f"${np.median(final_prices):.2f}",
        f"${np.std(final_prices):.2f}",
        f"${np.percentile(final_prices, 5):.2f}",
        f"${np.percentile(final_prices, 95):.2f}",
        f"${np.min(final_prices):.2f}",
        f"${np.max(final_prices):.2f}",
        f"${np.max(final_prices) - np.min(final_prices):.2f}"
    ]
}
st.table(pd.DataFrame(summary_data))

# Add explanatory notes
st.markdown("""
### 📝 Notes:
- The simulation uses historical volatility to generate possible future price paths
- The confidence interval shows the range where prices are expected to fall 90% of the time
- The mean path (red dashed line) represents the average of all simulated paths
- The histogram shows the distribution of possible final prices
""")
