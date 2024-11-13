# Quant-Tools

Tooling for quantitative research and trading strategies.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Quant-Tools is a Python-based toolkit designed for quantitative research and trading strategy development. It provides a framework for backtesting trading strategies, analyzing performance, and optimizing parameters.

## Features

- Multiple trading strategies
- Backtesting framework
- Performance metrics calculation
- Parameter optimization
- Logging and error handling

## Installation

To install Quant-Tools, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/quant-tools.git
cd quant-tools
pip install -r requirements.txt
```

## Usage

```python
from backtest.run_bt import run_train_test_analysis
from strategies.strategies import LiquidityImbStrategy

symbol = 'BTC-USD'
start_date = '2018-01-01'
end_date = '2024-10-31'
strategy_class = LiquidityImbStrategy
timeframes = ['1h']
train_length_days = 365
test_length_days = 365
param_ranges = {
    'vol_window': [12, 24, 36],
    'dev_threshold': [1.5, 2.0, 2.5],
    'roc_window': [3, 6, 9],
    'volume_ma': [12, 24, 36],
    'rsi_period': [14, 21, 28],
    'rsi_thresh_low': [30, 35, 40],
    'rsi_thresh_high': [70, 75, 80],
    'sl_coef': [1.0, 1.5, 2.0],
    'tp_coef': [2.0, 2.5, 3.0]
}
n_samples = 10
n_trials = 10
gap_days = 0
lookback_period_days = 10
starting_cash = 100000
commission = 0.001
limit = 10000

results_df, best_params = run_train_test_analysis(
    symbol, start_date, end_date, strategy_class, timeframes,
    train_length_days, test_length_days, param_ranges,
    n_samples, n_trials, gap_days, lookback_period_days,
    starting_cash, commission, limit
)
```
