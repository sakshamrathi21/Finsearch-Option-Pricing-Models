import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

# Define the ticker symbol
tickerSymbol = 'AAPL'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the options data for calls
options_data = tickerData.option_chain()
implied_volatilities = options_data.calls['impliedVolatility'].values
strike_prices = options_data.calls['strike'].values
last_prices = options_data.calls['lastPrice'].values

# Define the parameters
S = 100  # underlying price
T = 1    # time to expiration in years
r = 0.0  # risk-free interest rate
q = 0    # dividend yield

def black_scholes_call_price(S, K, T, r, q, sigma):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

# Calculate model option prices using Black-Scholes and the real implied volatilities
model_option_prices = np.zeros_like(strike_prices)
for i, vol in enumerate(implied_volatilities):
    model_option_prices[i] = black_scholes_call_price(S, strike_prices[i], T, r, q, implied_volatilities[i])

# Plotting
plt.plot(strike_prices, last_prices, label='Real Data')
plt.plot(strike_prices, model_option_prices, label='Model')
plt.xlabel('Strike Price')
plt.ylabel('Call Option Price')
plt.title('Accuracy of Black-Scholes Model on Real Data')
plt.legend()
plt.show()