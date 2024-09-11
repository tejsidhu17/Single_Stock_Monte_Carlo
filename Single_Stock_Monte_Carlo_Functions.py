import pandas as pd
import yfinance as yf
import numpy as np
from collections import Counter

def get_percentage_changes(ticker: str, period: str):
    ticker = ticker.upper()
    data_ticker = yf.download(ticker, period=period)
    percentage_changes = data_ticker["Close"].pct_change() * 100
    percentage_changes = percentage_changes.dropna()
    previous_close = data_ticker["Close"].tail(1).values[0]
    return percentage_changes, previous_close

def get_probability_distribution(percentage_changes):
    rounded_percentage_changes = ((percentage_changes * 2).round())/200
    frequency = rounded_percentage_changes.value_counts(normalize=True)
    frequency = frequency.sort_index()

    values = frequency.index.to_numpy()
    prob_distribution = frequency.values
    return values, prob_distribution

def run_monte_carlo(values, prob_distribution, time_to_run, interations, previous_close):
    prices = []
    prices.append(previous_close)
    for i in range(0, interations):
        np.random.seed(i)
        curr_price = previous_close
        for j in range(1, time_to_run):
            percent_change = np.random.choice(values, p=prob_distribution)
            curr_price *= (1 + percent_change)
        prices.append(curr_price)
    return prices

def get_price_distribution(prices):
    rounded_prices = [round(price) for price in prices]
    freq = Counter(rounded_prices)
    price_distribution = dict(sorted(freq.items()))
    return price_distribution

def single_stock_monte_carlo(ticker: str, period: str, time_to_run: int, iterations: int):
    percentage_changes, previous_close = get_percentage_changes(ticker, period)
    values, prob_distribution = get_probability_distribution(percentage_changes)
    prices = run_monte_carlo(values=values, prob_distribution=prob_distribution, time_to_run=time_to_run, interations=iterations, previous_close=previous_close)
    price_distribution = get_price_distribution(prices)

    for key, value in price_distribution.items():
        print(f"{key}: {value}")
