# Black-Scholes Option Pricing Model Web App

## Overview

This Streamlit web application implements the Black-Scholes Option Pricing Model, a fundamental tool in quantitative finance for pricing European-style options.

### What is the Black-Scholes Model?

The Black-Scholes model, developed by Fischer Black, Myron Scholes, and Robert Merton in 1973, is a mathematical model used to estimate the theoretical price of European-style options. In simple terms, it helps determine the fair price of an option contract based on several key factors.

#### The Black-Scholes Formula:

For a call option:
C = S * N(d1) - K * e^(-r * T) * N(d2)

For a put option:
P = K * e^(-r * T) * N(-d2) - S * N(-d1)

Where:
- C = Call option price
- P = Put option price
- S = Current stock price
- K = Strike price
- r = Risk-free interest rate
- T = Time to maturity
- N = Cumulative standard normal distribution function
- d1 and d2 are calculated using volatility and other parameters

### Importance in Quantitative Finance

The Black-Scholes model is crucial in the world of quantitative finance for:
1. Pricing options and other derivatives
2. Risk management and hedging strategies
3. Developing more complex financial models
4. Analyzing market behavior and implied volatility

## Web App Features

This application allows users to:

1. Input parameters for the Black-Scholes model
2. Calculate option prices and Greeks (Delta, Gamma, Vega, Theta, Rho)
3. Visualize the Profit and Loss (PnL) for both call and put options using interactive heatmaps

### PnL Heatmap Interpretation

- Green areas: Positive PnL (profit)
- Red areas: Negative PnL (loss)
- Yellow areas: Break-even points

The intensity of the color indicates the magnitude of profit or loss.

## How to Use

1. Enter the required parameters in the sidebar:
   - Current Asset Price
   - Strike Price
   - Time to Maturity
   - Volatility
   - Risk-Free Interest Rate
   - Call/Put Purchase Price
2. View the calculated option prices and Greeks
3. Explore the PnL heatmaps to understand how changes in spot price and volatility affect option profitability

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/black-scholes-web-app.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dependencies

- streamlit
- pandas
- numpy
- scipy
- plotly
- matplotlib
- seaborn

## Created By

Akshat Kulshreshtha

[LinkedIn Profile](https://www.linkedin.com/in/mprudhvi/)
