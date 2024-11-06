import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
from utils.fetch_data import fetch_crypto_data
from utils.prep_data import align_data_periods, format_data_cerebro

def execute_backtest(strategy_class, df_list, strategy_params, starting_cash, commission):
    """
    Core backtest execution function with prepared data
    """
    cerebro = bt.Cerebro()
    
    # Format and add data
    data_list = format_data_cerebro(df_list)
    for data in data_list:
        cerebro.adddata(data)
    
    # Setup strategy and broker
    cerebro.addstrategy(strategy_class, **strategy_params)
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    trades = strategy.analyzers.trades.get_analysis()

    # Log print
    print(f"Start date: {df_list[0].index[0]}")
    print(f"End date: {df_list[0].index[-1]}")
    print(f"Duration days: {(df_list[0].index[-1] - df_list[0].index[0]).days}")
    print(f"Total return: {strategy.total_return:.2%}")
    print(f"CAGR: {strategy.cagr:.2%}")
    print(f"CAGR: {strategy.max_drawdown:.2%}")
    print(f"Return/drawdown ratio: {strategy.ret_mdd_ratio:.2f}")



    return {
        'total_return': strategy.total_return,
        'cagr': strategy.cagr,
        'max_drawdown': strategy.max_drawdown,
        'ret_mdd_ratio': strategy.ret_mdd_ratio,
        'n_trades': trades.total.total if hasattr(trades, 'total') else 0,
        'win_trades': trades.won.total if hasattr(trades, 'won') else 0,
        'loss_trades': trades.lost.total if hasattr(trades, 'lost') else 0,
        'start_date': df_list[0].index[0].strftime('%Y-%m-%d'),
        'end_date': df_list[0].index[-1].strftime('%Y-%m-%d')
    }

def run_backtest(symbol, start_date, strategy_class, timeframes, period_length_days, 
                starting_cash, commission, limit=10000, **strategy_params):  
    """
    Run backtest with data fetching and preparation
    """
    # Calculate end date
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=period_length_days)
    end_date = end.strftime('%Y-%m-%d')
    
    # Fetch and prepare data
    df_list = [fetch_crypto_data(symbol=symbol, timeframe=tf, 
                               start_date=start_date, end_date=end_date,
                               limit=limit) 
               for tf in timeframes]
    df_list = align_data_periods(df_list)
    
    return execute_backtest(
        strategy_class=strategy_class,
        df_list=df_list,
        strategy_params=strategy_params,  # Pass the strategy parameters
        starting_cash=starting_cash,
        commission=commission
    )

def optimize_strategy_random(strategy_class, df_list, param_ranges, starting_cash, commission, 
                          n_trials=20, random_state=None):
    """
    Optimize strategy parameters using random search
    """
    if random_state is not None:
        np.random.seed(random_state)

    results = []
   
   # Generate random parameter combinations
    for trial in range(n_trials):
        try:
            # Randomly sample one value from each parameter range
            params = {}
            for param, values in param_ranges.items():
                if isinstance(values, range):
                    params[param] = int(np.random.choice(list(values)))
                elif isinstance(values, list):
                    params[param] = float(np.random.choice(values))
                else:
                    raise ValueError(f"Unsupported parameter range type for {param}")
            
            print(f"\nTrial {trial + 1}/{n_trials}...")
            
            result = execute_backtest(
                strategy_class=strategy_class,
                df_list=df_list,
                strategy_params=params,
                starting_cash=starting_cash,
                commission=commission
            )
            
            # Add parameters to results
            result.update(params)
            results.append(result)
            
        except Exception as e:
            print(f"Error in trial {trial + 1}: {str(e)}")
            continue
   
    if not results:
        print("Debug info:")
        print(f"Parameter ranges: {param_ranges}")
        print(f"Data shapes: {[df.shape for df in df_list]}")
        print(f"Data index range: {df_list[0].index[0]} to {df_list[0].index[-1]}")
        raise ValueError("No successful optimization trials")
   
    # Convert to DataFrame and find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['ret_mdd_ratio'].idxmax()
    best_params = {col: results_df.loc[best_idx, col] 
                    for col in param_ranges.keys()}
   
    print(f"\nBest parameters found for Return/drawdown: {best_params}")
    print(f"Return: {results_df.loc[best_idx, 'total_return']:.2%}")
    print(f"CAGR: {results_df.loc[best_idx, 'cagr']:.2%}")
    print(f"MDD: {results_df.loc[best_idx, 'max_drawdown']:.2%}")
    print(f"Return/drawdown ratio: {results_df.loc[best_idx, 'ret_mdd_ratio']:.2f}")

   
    return results_df, best_params

def generate_train_test_periods(start, end, train_length_days, test_length_days, gap_days=0, n_samples=30):
    """
    Generate bootstrap samples of aligned train-test periods
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    total_period = train_length_days + test_length_days + gap_days
    
    # Adjust end to account for total period length
    max_start = end - pd.Timedelta(days=total_period)
    
    # Generate potential start dates
    date_range = pd.date_range(start=start, end=max_start, freq='D')
    
    # Bootstrap sampling
    sampled_starts = np.random.choice(date_range, size=n_samples, replace=True)
    sampled_starts = sorted(pd.DatetimeIndex(sampled_starts))
    
    # Generate aligned train-test periods
    periods = []
    for start_date in sampled_starts:
        train_start = start_date
        train_end = train_start + pd.Timedelta(days=train_length_days)
        test_start = train_end + pd.Timedelta(days=gap_days)
        test_end = test_start + pd.Timedelta(days=test_length_days)
        
        periods.append({
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d')
        })
    
    return periods

def calculate_stability_stats(df_results):
    """Calculate statistical metrics for backtest results"""
    metrics = ['total_return', 'cagr', 'max_drawdown', 'ret_mdd_ratio', 
              'n_trades', 'win_trades', 'loss_trades']
    
    stats = {}
    for metric in metrics:
        if metric in df_results.columns:
            values = df_results[metric]
            stats[metric] = {
                'mean': values.mean(),
                'std': values.std(),
                'ci_lower': values.mean() - 1.96 * values.std() / np.sqrt(len(values)),
                'ci_upper': values.mean() + 1.96 * values.std() / np.sqrt(len(values)),
                'min': values.min(),
                'max': values.max()
            }
    
    return stats

def run_train_test_analysis(symbol, start_date, end_date, strategy_class, timeframes,
                           train_length_days, test_length_days, param_ranges,
                           n_samples=30, n_trials=20, gap_days=0, starting_cash=100000, 
                           commission=0.001, limit=10000):
    """Run walk-forward optimization and out-of-sample testing with random search"""
    
    # Convert dates to timestamps
    global_start = pd.Timestamp(start_date)
    global_end = pd.Timestamp(end_date)
    
    # Calculate the available range for sampling complete train+test periods
    total_period = train_length_days + test_length_days + gap_days
    max_start = global_end - pd.Timedelta(days=total_period)
    
    # Generate potential sample dates ensuring complete periods fit within global range
    date_range = pd.date_range(start=global_start, end=max_start, freq='D')
    
    if len(date_range) == 0:
        raise ValueError("Global period too short to fit train+test period")
    
    # Sample start dates
    sampled_starts = sorted(pd.Timestamp(d) for d in 
                          np.random.choice(date_range, size=n_samples, replace=True))
    
    results = []
    
    for idx, sample_start in enumerate(sampled_starts):
        try:
            # For each sampled start, calculate the aligned train+test periods
            train_start = sample_start  # Training starts at sampled date
            train_end = train_start + pd.Timedelta(days=train_length_days)
            test_start = train_end + pd.Timedelta(days=gap_days)
            test_end = test_start + pd.Timedelta(days=test_length_days)
            
            print(f"\nGlobal period: {global_start.strftime('%Y-%m-%d')} to {global_end.strftime('%Y-%m-%d')}")
            print(f"Train period sample: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"Test period sample:  {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            # Fetch training and testing data
            train_dfs = [fetch_crypto_data(symbol=symbol, timeframe=tf, 
                                         start_date=train_start.strftime('%Y-%m-%d'), 
                                         end_date=train_end.strftime('%Y-%m-%d'))
                        for tf in timeframes]
            test_dfs = [fetch_crypto_data(symbol=symbol, timeframe=tf, 
                                    start_date=test_start.strftime('%Y-%m-%d'), 
                                    end_date=test_end.strftime('%Y-%m-%d'),
                                    limit=limit)
                    for tf in timeframes]
            
            # Optimize on training data
            _, best_params = optimize_strategy_random(
                strategy_class=strategy_class,
                df_list=train_dfs,
                param_ranges=param_ranges,
                starting_cash=starting_cash,
                commission=commission,
                n_trials=n_trials,
                random_state=idx  # Different seed for each sample
            )
                        
            # Test on out-of-sample data
            print("\nRunning backtest on out-of-sample data...")
            test_result = execute_backtest(
                strategy_class=strategy_class,
                df_list=test_dfs,
                strategy_params=best_params,
                starting_cash=starting_cash,
                commission=commission
            )
            
            test_result.update({
                'sample_start': train_start.strftime('%Y-%m-%d'),
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'optimized_params': best_params
            })
            
            results.append(test_result)
            print(f"Completed train-test cycle starting at {train_start.strftime('%Y-%m-%d')}")
            print("*" * 47)
            
        except Exception as e:
            print(f"Error in train-test cycle: {type(e).__name__}: {str(e)}")
            print("*" * 47)
    
    df_results = pd.DataFrame(results)
    stats = calculate_stability_stats(df_results)
    
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

def analyze_test_results(df_results):
    """
    Find most recurring complete parameter set from test samples
    """
    # Get all optimal parameters
    optimal_params = df_results['optimized_params'].tolist()
    
    # Convert parameter dictionaries to tuples for counting
    param_tuples = [tuple(sorted(p.items())) for p in optimal_params]
    param_counts = pd.Series(param_tuples).value_counts()
    
    print("\n=== PARAMETER SETS FREQUENCY ANALYSIS ===")
    for param_set, count in param_counts.items():
        print(f"\nParameter set occurred {count/len(param_tuples):.1%}:")
        params_dict = dict(param_set)
        for param, value in params_dict.items():
            print(f"{param}: {value}")
        print("-" * 40)
    
    # Performance statistics
    print("\n=== AVERAGE PERFORMANCE STATISTICS ON TEST SAMPLES ===")
    performance_stats = calculate_stability_stats(df_results)
    print_stability_stats(performance_stats)

    # Get most frequent parameter set
    most_common_params = dict(param_counts.index[0])
    print("\n=== MOST RECURRING PARAMETER SET ON TEST SAMPLES ===\n")
    return most_common_params
