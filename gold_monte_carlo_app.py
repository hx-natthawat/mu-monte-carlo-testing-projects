import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from scipy import stats

# Initialize session state variables
if 'show_ma' not in st.session_state:
    st.session_state.show_ma = False
if 'show_rsi' not in st.session_state:
    st.session_state.show_rsi = True
if 'ma_short' not in st.session_state:
    st.session_state.ma_short = 50
if 'ma_long' not in st.session_state:
    st.session_state.ma_long = 200
if 'rsi_period' not in st.session_state:
    st.session_state.rsi_period = 14

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
3. Statistical analysis and risk metrics are computed
""")

# Load and process data
@st.cache_data
def load_data():
    data = pd.read_csv("FINAL_USO.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data['Daily Return'] = data['Adj Close'].pct_change()
    return data

# Calculate technical indicators
def calculate_moving_averages(prices, window_short=50, window_long=200):
    ma_short = prices.rolling(window=window_short).mean()
    ma_long = prices.rolling(window=window_long).mean()
    return ma_short, ma_long

# Calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

# Calculate Expected Shortfall (ES)
def calculate_es(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

# Load data
data = load_data()
daily_returns = data['Daily Return'].dropna()
starting_price = data['Adj Close'].iloc[-1]

# Sidebar controls
st.sidebar.header("📊 Simulation Parameters")

# Basic Parameters
st.sidebar.subheader("Basic Parameters")
num_simulations = st.sidebar.slider("Number of simulations", 100, 1000, 500, 100)
num_days = st.sidebar.slider("Trading days to forecast", 30, 365, 252)

# Advanced Parameters
st.sidebar.subheader("Advanced Parameters")
custom_volatility = st.sidebar.slider(
    "Volatility Adjustment",
    0.5,
    2.0,
    1.0,
    0.1,
    help="Multiply historical volatility by this factor"
)

custom_drift = st.sidebar.slider(
    "Return Adjustment (Annual %)",
    -20.0,
    20.0,
    0.0,
    1.0,
    help="Adjust expected return"
)

model_type = st.sidebar.selectbox(
    "Simulation Model",
    ["Standard Random Walk", "Geometric Brownian Motion"],
    help="Choose the mathematical model for price simulation"
)

# Technical Analysis Parameters
st.sidebar.subheader("Technical Analysis")
st.session_state.show_ma = st.sidebar.checkbox("Show Moving Averages", value=st.session_state.show_ma)
if st.session_state.show_ma:
    st.session_state.ma_short = st.sidebar.slider("Short MA Window", 10, 100, st.session_state.ma_short)
    st.session_state.ma_long = st.sidebar.slider("Long MA Window", 50, 300, st.session_state.ma_long)

st.session_state.show_rsi = st.sidebar.checkbox("Show RSI", value=st.session_state.show_rsi)
if st.session_state.show_rsi:
    st.session_state.rsi_period = st.sidebar.slider("RSI Period", 5, 30, st.session_state.rsi_period)

# Display initial information
st.sidebar.markdown("### Current Price Information")
st.sidebar.markdown(f"**Starting Price:** ${starting_price:.2f}")
annual_vol = daily_returns.std() * np.sqrt(252)
st.sidebar.markdown(f"**Historical Volatility:** {annual_vol:.1%} annually")

# Progress tracking
progress_container = st.container()
with progress_container:
    progress_bar = st.progress(0)
    status_text = st.empty()

# Initialize simulation
simulated_prices = np.zeros((num_days, num_simulations))
simulated_prices[0, :] = starting_price

# Create figures placeholder
fig_placeholder = st.empty()

# Create columns for statistics
col1, col2 = st.columns(2)

# Run simulation with animation
adjusted_vol = daily_returns.std() * custom_volatility
adjusted_drift = (custom_drift / 100) / 252  # Convert annual % to daily

for day in range(1, num_days):
    progress = day / (num_days - 1)
    progress_bar.progress(progress)
    status_text.text(f"Simulating day {day}/{num_days-1}")
    
    if model_type == "Standard Random Walk":
        random_daily_returns = np.random.normal(
            daily_returns.mean() + adjusted_drift,
            adjusted_vol,
            num_simulations
        )
    else:  # Geometric Brownian Motion
        random_daily_returns = np.random.normal(
            (daily_returns.mean() + adjusted_drift - 0.5 * adjusted_vol**2),
            adjusted_vol,
            num_simulations
        )
    
    simulated_prices[day, :] = simulated_prices[day-1, :] * np.exp(random_daily_returns)
    
    if day % 5 == 0 or day == num_days-1:
        # Create subplots - 2 rows if RSI is shown
        fig = make_subplots(
            rows=2 if st.session_state.show_rsi else 1,
            cols=1,
            row_heights=[0.7, 0.3] if st.session_state.show_rsi else [1],
            vertical_spacing=0.1
        )
        
        # Add simulation paths
        for sim in range(0, num_simulations, max(1, num_simulations//100)):
            fig.add_trace(
                go.Scatter(
                    y=simulated_prices[:day+1, sim],
                    mode='lines',
                    line=dict(width=1, color='rgba(0,100,255,0.1)'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
        
        # Add mean line
        mean_line = np.mean(simulated_prices[:day+1, :], axis=1)
        fig.add_trace(
            go.Scatter(
                y=mean_line,
                mode='lines',
                line=dict(width=3, color='red', dash='dash'),
                name='Mean Path'
            ),
            row=1, col=1
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
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=percentile_5,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                fillcolor='rgba(0,100,255,0.2)',
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Add moving averages if enabled
        if st.session_state.show_ma:
            ma_short_values, ma_long_values = calculate_moving_averages(
                pd.Series(mean_line),
                st.session_state.ma_short,
                st.session_state.ma_long
            )
            fig.add_trace(
                go.Scatter(
                    y=ma_short_values,
                    mode='lines',
                    line=dict(width=2, color='yellow'),
                    name=f'{st.session_state.ma_short}-day MA'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=ma_long_values,
                    mode='lines',
                    line=dict(width=2, color='purple'),
                    name=f'{st.session_state.ma_long}-day MA'
                ),
                row=1, col=1
            )
        
        # Add RSI if enabled
        if st.session_state.show_rsi:
            rsi_values = calculate_rsi(pd.Series(mean_line), st.session_state.rsi_period)
            fig.add_trace(
                go.Scatter(
                    y=rsi_values,
                    mode='lines',
                    line=dict(width=2, color='orange'),
                    name=f'RSI ({st.session_state.rsi_period})'
                ),
                row=2, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            title={
                'text': 'Monte Carlo Simulation of Gold Prices',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=800 if st.session_state.show_rsi else 600,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if st.session_state.show_rsi:
            fig.update_yaxes(title_text="RSI", row=2, col=1)
        
        fig_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display statistics
        current_prices = simulated_prices[day, :]
        current_returns = np.log(current_prices / starting_price)
        
        with col1:
            st.subheader("📈 Price Statistics")
            st.write(f"Mean Price: ${np.mean(current_prices):.2f}")
            st.write(f"Median Price: ${np.median(current_prices):.2f}")
            st.write(f"Standard Deviation: ${np.std(current_prices):.2f}")
            st.write(f"Skewness: {stats.skew(current_prices):.3f}")
            st.write(f"Kurtosis: {stats.kurtosis(current_prices):.3f}")
        
        with col2:
            st.subheader("🎯 Risk Metrics")
            st.write(f"Value at Risk (95%): ${-calculate_var(current_returns, 0.95) * starting_price:.2f}")
            st.write(f"Expected Shortfall (95%): ${-calculate_es(current_returns, 0.95) * starting_price:.2f}")
            st.write(f"Maximum Drawdown: ${starting_price - np.min(current_prices):.2f}")
            st.write(f"Price Range: ${np.min(current_prices):.2f} - ${np.max(current_prices):.2f}")
        
        time.sleep(0.1)

# Final status
status_text.text("✅ Simulation completed!")
progress_bar.progress(1.0)

# Distribution of final prices
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

# Final summary statistics
st.subheader("📑 Final Simulation Summary")
final_prices = simulated_prices[-1, :]
final_returns = np.log(final_prices / starting_price)

summary_data = {
    "Metric": [
        "Expected Price (Mean)",
        "Median Price",
        "Standard Deviation",
        "Skewness",
        "Kurtosis",
        "Value at Risk (95%)",
        "Expected Shortfall (95%)",
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
        f"{stats.skew(final_prices):.3f}",
        f"{stats.kurtosis(final_prices):.3f}",
        f"${-calculate_var(final_returns, 0.95) * starting_price:.2f}",
        f"${-calculate_es(final_returns, 0.95) * starting_price:.2f}",
        f"${np.percentile(final_prices, 5):.2f}",
        f"${np.percentile(final_prices, 95):.2f}",
        f"${np.min(final_prices):.2f}",
        f"${np.max(final_prices):.2f}",
        f"${np.max(final_prices) - np.min(final_prices):.2f}"
    ]
}

# Create a DataFrame for the summary and display it
summary_df = pd.DataFrame(summary_data)
st.table(summary_df)

# Download button for simulation results
final_results = pd.DataFrame({
    'Simulation': range(1, num_simulations + 1),
    'Final Price': final_prices,
    'Return': (final_prices - starting_price) / starting_price * 100
})

st.download_button(
    label="Download Simulation Results",
    data=final_results.to_csv(index=False).encode('utf-8'),
    file_name='simulation_results.csv',
    mime='text/csv'
)

# Add explanatory notes
st.markdown("""
### 📝 Notes:
- The simulation uses historical volatility adjusted by user-defined parameters
- Value at Risk (VaR) represents the potential loss at the 95% confidence level
- Expected Shortfall (ES) is the average loss beyond VaR
- Moving averages (if enabled) help identify potential trends in the simulated paths
- RSI (Relative Strength Index) helps identify overbought (>70) and oversold (<30) conditions
- The histogram shows the distribution of possible final prices
- Skewness measures the asymmetry of the price distribution
- Kurtosis indicates the "tailedness" of the distribution
""")
