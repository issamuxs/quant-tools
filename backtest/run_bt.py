import backtrader as bt
import pandas as pd
import numpy as np
from utils.fetch_data import fetch_crypto_data
from utils.prep_data import align_data_periods, format_data_cerebro

def generate_bootstrap_periods(start, end, period_length_days, n_samples):
   """
   Generate bootstrap samples of backtest periods
   """
   start = pd.Timestamp(start)
   end = pd.Timestamp(end)
   period_delta = pd.Timedelta(days=period_length_days)
   
   # Adjust end to account for period length
   max_start = end - period_delta
   
   # Generate potential start dates
   date_range = pd.date_range(start=start, end=max_start, freq='D')
   
   # Bootstrap sampling
   sampled_dates = np.random.choice(date_range, size=n_samples, replace=True)
   
   # Convert numpy datetime64 to string directly
   ordered_dates = sorted(pd.DatetimeIndex(sampled_dates))
   return [d.strftime('%Y-%m-%d') for d in ordered_dates]

def run_backtest(start_date, strategy_class, stf, ltf, period_length_days, starting_cash, limit, commission):
    """Run a single backtest with given parameters"""
    cerebro = bt.Cerebro()
    
    # Calculate end date based on period length
    start = pd.Timestamp(start_date)
    if period_length_days != None:
        end = start + pd.Timedelta(days=period_length_days)
        end_date = end.strftime('%Y-%m-%d')
    else:
        end_date = None
    
    # Get and prepare data
    df_stf = fetch_crypto_data(symbol='BTC/USDT', timeframe=stf, start_date=start_date, end_date=end_date, limit=limit)
    df_ltf = fetch_crypto_data(symbol='BTC/USDT', timeframe=ltf, start_date=start_date, end_date=end_date, limit=limit)
    df_stf, df_ltf = align_data_periods(df_stf, df_ltf)
    [data_stf, data_ltf] = format_data_cerebro([df_stf, df_ltf])
    
    # Setup cerebro
    cerebro.adddata(data_stf)
    cerebro.adddata(data_ltf)
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    trades = strategy.analyzers.trades.get_analysis()
    
    # Extract metrics
    return {
        'total_return': strategy.total_return,
        'cagr': strategy.cagr,
        'max_drawdown': strategy.max_drawdown,
        'ret_mdd_ratio': strategy.ret_mdd_ratio,
        'n_trades': trades.total.total if hasattr(trades, 'total') else 0,
        'win_trades': trades.won.total if hasattr(trades, 'won') else 0,
        'loss_trades': trades.lost.total if hasattr(trades, 'lost') else 0,
        'start_date': start_date,
        'end_date': end_date
    }

def run_stability_analysis(start_dates, strategy_class, stf, ltf, period_length_days, starting_cash=100000, limit=10000, commission=0.001):
    """Run multiple backtests and analyze stability"""
    results = []
    
    for start_date in start_dates:
        try:
            result = run_backtest(start_date, strategy_class, stf, ltf, period_length_days, 
                                  starting_cash=starting_cash, limit=limit, commission=commission)
            results.append(result)
            print(f"Completed backtest for start date: {start_date}")
        except Exception as e:
            print(f"Error running backtest for {start_date}: {str(e)}")
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    # Calculate statistics for each metric
    metrics = ['total_return', 'cagr', 'max_drawdown', 'ret_mdd_ratio', 
              'n_trades', 'win_trades', 'loss_trades']
    
    stats = {}
    for metric in metrics:
        values = df_results[metric]
        stats[metric] = {
            'mean': values.mean(),
            'std': values.std(),
            'ci_lower': values.mean() - 1.96 * values.std() / np.sqrt(len(values)),
            'ci_upper': values.mean() + 1.96 * values.std() / np.sqrt(len(values)),
            'min': values.min(),
            'max': values.max()
        }
    
    return df_results, stats

def print_stability_stats(stability_stats):
    # Define which metrics are percentages
    pct_metrics = {'total_return', 'cagr', 'max_drawdown'}
    
    for metric, stat in stability_stats.items():
        print(f"\n{metric.upper()}:")
        if metric in pct_metrics:
            print(f"Mean: {stat['mean']:.2%}")
            print(f"95% CI: [{stat['ci_lower']:.2%}, {stat['ci_upper']:.2%}]")
            print(f"Min: {stat['min']:.2%}")
            print(f"Max: {stat['max']:.2%}")
            print(f"Std Dev: {stat['std']:.2%}")
        else:
            print(f"Mean: {stat['mean']:.2f}")
            print(f"95% CI: [{stat['ci_lower']:.2f}, {stat['ci_upper']:.2f}]")
            print(f"Min: {stat['min']:.2f}")
            print(f"Max: {stat['max']:.2f}")
            print(f"Std Dev: {stat['std']:.2f}")