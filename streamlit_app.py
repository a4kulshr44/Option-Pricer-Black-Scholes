import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 12px;
    width: auto;
    margin: 0 auto;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-call {
    background-color: #e6f3ff;
    color: #0066cc;
}

.metric-put {
    background-color: #fff0e6;
    color: #cc6600;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    margin: 0;
}

.metric-label {
    font-size: 1.2rem;
    margin-bottom: 4px;
}

.stButton > button {
    width: 100%;
}

.sidebar .stButton > button {
    background-color: #4CAF50;
    color: white;
}

</style>
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        d1 = (log(self.current_price / self.strike) + 
              (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (
                  self.volatility * sqrt(self.time_to_maturity)
              )
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)

        self.call_price = self.current_price * norm.cdf(d1) - (
            self.strike * exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(d2)
        )
        self.put_price = (
            self.strike * exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(-d2)
        ) - self.current_price * norm.cdf(-d1)

        # Greeks
        self.call_delta = norm.cdf(d1)
        self.put_delta = -norm.cdf(-d1)
        self.gamma = norm.pdf(d1) / (self.current_price * self.volatility * sqrt(self.time_to_maturity))
        self.vega = self.current_price * norm.pdf(d1) * sqrt(self.time_to_maturity)
        self.call_theta = -(self.current_price * norm.pdf(d1) * self.volatility / (2 * sqrt(self.time_to_maturity))) - \
                          self.interest_rate * self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        self.put_theta = -(self.current_price * norm.pdf(d1) * self.volatility / (2 * sqrt(self.time_to_maturity))) + \
                         self.interest_rate * self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)
        self.call_rho = self.strike * self.time_to_maturity * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        self.put_rho = -self.strike * self.time_to_maturity * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)

        return self.call_price, self.put_price

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/mprudhvi/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Akshat Kulshreshtha`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=1.00, min_value=0.01, step=0.01)
    strike = st.number_input("Strike Price", value=1.00, min_value=0.01, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=0.02, min_value=0.01, step=0.01)
    volatility = st.number_input("Volatility (σ)", value=0.20, min_value=0.01, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, min_value=0.00, step=0.01)
    call_purchase_price = st.number_input("Call Purchase Price", value=0.02, min_value=0.00, step=0.01)
    put_purchase_price = st.number_input("Put Purchase Price", value=0.02, min_value=0.00, step=0.01)

    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=max(current_price*0.8, 0.01), step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=max(current_price*1.2, 0.02), step=0.01)
    vol_min = st.slider('Min Volatility', min_value=0.01, max_value=1.0, value=max(volatility*0.5, 0.01), step=0.01)
    vol_max = st.slider('Max Volatility', min_value=0.01, max_value=1.0, value=min(volatility*1.5, 1.0), step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 20)
    vol_range = np.linspace(vol_min, vol_max, 20)

def plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, option_type, purchase_price):
    pnl_values = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            if option_type == 'call':
                pnl_values[i, j] = bs_temp.call_price - purchase_price
            else:
                pnl_values[i, j] = bs_temp.put_price - purchase_price
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a custom colormap
    colors = ['red', 'yellow', 'green']
    n_bins = 100
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Determine the maximum absolute value for symmetrical color scaling
    max_abs_pnl = max(abs(np.min(pnl_values)), abs(np.max(pnl_values)))
    
    sns.heatmap(pnl_values, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), 
                annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax, 
                vmin=-max_abs_pnl, vmax=max_abs_pnl)
    
    ax.set_title(f'{option_type.upper()} Option PnL')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    
    return fig

# Main Page for Output Display
st.title("Black-Scholes Option Pricing Model")

# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display input parameters
st.subheader("Input Parameters")
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
    "Call Purchase Price": [call_purchase_price],
    "Put Purchase Price": [put_purchase_price],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Display Call and Put Values
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Display Greeks
st.subheader("Option Greeks")
greeks_data = {
    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Call": [bs_model.call_delta, bs_model.gamma, bs_model.vega, bs_model.call_theta, bs_model.call_rho],
    "Put": [bs_model.put_delta, bs_model.gamma, bs_model.vega, bs_model.put_theta, bs_model.put_rho]
}
greeks_df = pd.DataFrame(greeks_data)

# Apply formatting only to numeric columns
st.table(greeks_df.style.format({
    "Call": "{:.4f}",
    "Put": "{:.4f}"
}))

st.title("Options PnL - Interactive Heatmap")
st.info("Explore how option PnL changes with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, while maintaining a constant 'Strike Price'.")

# Interactive Heatmaps for Call and Put Options PnL
col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Option PnL Heatmap")
    heatmap_fig_call = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, 'call', call_purchase_price)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Option PnL Heatmap")
    heatmap_fig_put = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, 'put', put_purchase_price)
    st.pyplot(heatmap_fig_put)

# Add explanations
st.markdown("""
## How to Interpret the Heatmaps

The heatmaps show the Profit and Loss (PnL) for call and put options across different spot prices and volatility levels:

- **Green areas**: Represent positive PnL (profit)
- **Red areas**: Represent negative PnL (loss)
- **Yellow areas**: Represent PnL close to zero (break-even)

The intensity of the color indicates the magnitude of the PnL. Darker shades represent larger profits or losses.

## Key Observations

1. **Call Options**: 
   - Tend to be more profitable (greener) as the spot price increases
   - Higher volatility generally increases the potential for profit, but also the potential for loss

2. **Put Options**:
   - Tend to be more profitable (greener) as the spot price decreases
   - Like calls, higher volatility increases both profit and loss potential

3. **Break-even Points**: 
   - Look for the yellow areas in the heatmap to identify break-even scenarios

4. **Risk Assessment**:
   - Areas with intense red or green indicate higher risk and reward scenarios
   - More neutral colors suggest more stable, lower-risk positions

Use these heatmaps to visualize how changes in the underlying asset price and market volatility might affect your option positions.
""")
