import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
from utils.fetch_data import fetch_crypto_data
from utils.prep_data import align_data_periods, format_data_cerebro, lookback_test_data

def execute_backtest(strategy_class, df_list, lookback_reset_idx, strategy_params, starting_cash, commission):
    """
    Executes a backtest for a given trading strategy.
    Parameters:
    strategy_class (class): The trading strategy class to be tested.
    df_list (list): List of dataframes containing the historical data.
    lookback_reset_idx (int): Index to reset the lookback period.
    strategy_params (dict): Parameters for the trading strategy.
    starting_cash (float): Initial cash for the backtest.
    commission (float): Commission rate for the broker.
    Returns:
    dict: A dictionary containing the backtest results including total return, CAGR, max drawdown, 
          CAGR/MDD ratio, number of trades, winning trades, losing trades, lookback start date, 
          test start date, and test end date.
    """

    try:
        print(f"Starting backtest for strategy {strategy_class.__name__}...")
        cerebro = bt.Cerebro()
        
        # Format and add data
        data_list = format_data_cerebro(df_list)
        for data in data_list:
            cerebro.adddata(data)
        
        # Add test_start_idx to strategy parameters
        strategy_params['lookback_reset_idx']=lookback_reset_idx

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

        # Log print including lookback info
        print("*"*47)
        print(f"Start date with lookback: {df_list[0].index[0]}")
        print(f"Start date w/o lookback: {df_list[0].index[lookback_reset_idx]}")
        print(f"Start date of trade logs: {strategy.start_date}")
        print(f"End date: {df_list[0].index[-1]}")
        print("*"*47)
        print(f"Total return: {strategy.total_return:.2%}")
        print(f"CAGR: {strategy.cagr:.2%}")
        print(f"MDD: {strategy.max_drawdown:.2%}")
        print(f"CAGR/MDD: {strategy.cagr_mdd_ratio:.2f}")
        print("Backtest completed successfully.")
        print("*"*47)

        return {
            'total_return': strategy.total_return,
            'cagr': strategy.cagr,
            'max_drawdown': strategy.max_drawdown,
            'cagr_mdd_ratio': strategy.cagr_mdd_ratio,
            'n_trades': trades.total.total if hasattr(trades, 'total') else 0,
            'win_trades': trades.won.total if hasattr(trades, 'won') else 0,
            'loss_trades': trades.lost.total if hasattr(trades, 'lost') else 0,
            'lookback_start': df_list[0].index[0].strftime('%Y-%m-%d'),
            'test_start': df_list[0].index[lookback_reset_idx].strftime('%Y-%m-%d'),
            'test_end': df_list[0].index[-1].strftime('%Y-%m-%d')
        }
    
    except Exception as e:
        print(f"Error during backtest execution: {e}")
        raise


def optimize_strategy_random(strategy_class, df_list, lookback_reset_idx, param_ranges, starting_cash, commission, n_trials, random_state):
    """
    Optimize a trading strategy using random parameter search.
    Parameters:
    strategy_class (class): The trading strategy class to be optimized.
    df_list (list): List of dataframes containing historical data.
    lookback_reset_idx (int): Index to reset lookback period.
    param_ranges (dict): Dictionary of parameter ranges to sample from.
    starting_cash (float): Initial cash for the backtest.
    commission (float): Commission rate for trades.
    n_trials (int): Number of random trials to perform.
    random_state (int, optional): Seed for random number generator.
    Returns:
    tuple: A tuple containing the DataFrame of results and the best parameters found.
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
                lookback_reset_idx=lookback_reset_idx,
                strategy_params=params,
                starting_cash=starting_cash,
                commission=commission
            )
            
            # Add parameters to results
            result.update(params)
            results.append(result)
            
        except Exception as e:
            print(f"Error in trial {trial + 1}: {type(e).__name__}: {str(e)}")
            continue
   
    if not results:
        raise ValueError("No successful optimization trials")
   
    # Convert to DataFrame and find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['cagr_mdd_ratio'].idxmax()
    best_params = {col: results_df.loc[best_idx, col] 
                    for col in param_ranges.keys()}
   
    print("\nBest parameters found:")
    print("*"*47)
    print(f"{best_params}")
    print("*"*47)
    print(f"Return: {results_df.loc[best_idx, 'total_return']:.2%}")
    print(f"CAGR: {results_df.loc[best_idx, 'cagr']:.2%}")
    print(f"MDD: {results_df.loc[best_idx, 'max_drawdown']:.2%}")
    print(f"CAGR/MDD: {results_df.loc[best_idx, 'cagr_mdd_ratio']:.2f}")
    print("*"*47)
   
    return results_df, best_params

def generate_train_test_periods(start, end, train_length_days, test_length_days, gap_days, n_samples):
    """
    Generate train and test periods for backtesting.
    Parameters:
    start (str): The start date of the overall period in 'YYYY-MM-DD' format.
    end (str): The end date of the overall period in 'YYYY-MM-DD' format.
    train_length_days (int): The length of the training period in days.
    test_length_days (int): The length of the testing period in days.
    gap_days (int): The number of days between the training and testing periods.
    n_samples (int): The number of train-test periods to generate.
    Returns:
    list: A list of dictionaries, each containing 'train_start', 'train_end', 'test_start', and 'test_end' keys with corresponding date values in 'YYYY-MM-DD' format.
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

def geometric_mean(returns):
    """
    Calculate the geometric mean of a series of returns.
    The geometric mean is calculated by transforming the returns using the natural logarithm,
    computing the mean of the transformed returns, and then exponentiating back to the original scale.
    Parameters:
    returns (array-like): A series of returns.
    Returns:
    float: The geometric mean of the returns.
    """

    # Ensure the returns are in the form of 1 + return (e.g., 10% return is 1.1)
    transformed_returns = np.log1p(returns)  # log(1 + return)
    
    # Compute the average of the transformed returns
    mean_log_return = np.mean(transformed_returns)
    
    # Exponentiate back to the original scale and subtract 1 to get the geometric mean
    gmean_return = np.expm1(mean_log_return)
    
    return gmean_return

def calculate_stability_stats(df_results):
    """
    Calculate statistical metrics for a given set of values.
    
    Args:
        values (array-like): Array of numerical values.
    
    Returns:
        dict: Dictionary containing statistical metrics.
    """
    metrics = ['total_return', 'cagr', 'max_drawdown', 
               'n_trades', 'win_trades', 'loss_trades']
    
    stats = {}
    for metric in metrics:
        if metric in df_results.columns:
            values = df_results[metric]
            
            # For multiplicative metrics like returns, calculate geometric mean
            if metric in ['total_return', 'cagr']: 
                stats[metric] = {
                    'gmean': geometric_mean(values),  # Geometric mean for returns
                    'mean': values.mean(),
                    'std': values.std(),
                    'ci_lower': np.percentile(values, 2.5),
                    'ci_upper': np.percentile(values, 97.5),
                    'median': np.median(values),
                    'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                    'min': values.min(),
                    'max': values.max()
                }
            else:
                # For other metrics, handle the usual statistics (mean, median, etc.)
                stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'ci_lower': values.mean() - 1.96 * values.std() / np.sqrt(len(values)),
                    'ci_upper': values.mean() + 1.96 * values.std() / np.sqrt(len(values)),
                    'median': np.median(values),
                    'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                    'min': values.min(),
                    'max': values.max()
                }
    
    # After calculating the statistics, compute the CAGR-to-MDD ratio if available
    if 'cagr' in stats and 'max_drawdown' in stats:
        stats['cagr_mdd_ratio'] = {
            'mean': stats['cagr']['gmean'] / stats['max_drawdown']['mean'] if stats['max_drawdown']['mean'] != 0 else np.nan,
            'median': stats['cagr']['median'] / stats['max_drawdown']['median'] if stats['max_drawdown']['median'] != 0 else np.nan
        }
    
    return stats

def run_train_test_analysis(symbol, start_date, end_date, strategy_class, timeframes,
                           train_length_days, test_length_days, param_ranges,
                           n_samples=10, n_trials=10, gap_days=0, lookback_period_days=None, starting_cash=100000, 
                           commission=0.001, limit=10000):
    """
    Run walk-forward optimization and out-of-sample testing with random search.
    
    Args:
        symbol (str): Trading symbol.
        start_date (str): Start date for the analysis.
        end_date (str): End date for the analysis.
        strategy_class (class): Strategy class to be tested.
        timeframes (list): List of timeframes to be used.
        train_length_days (int): Number of days for the training period.
        test_length_days (int): Number of days for the testing period.
        param_ranges (dict): Dictionary of parameter ranges for optimization.
        n_samples (int): Number of samples for walk-forward analysis.
        n_trials (int): Number of trials for random search.
        gap_days (int): Number of gap days between training and testing periods.
        lookback_period_days (int): Number of days for the lookback period.
        starting_cash (float): Starting cash for the backtest.
        commission (float): Commission rate for the backtest.
        limit (int): Limit for the number of data points.
    
    Returns:
        tuple: DataFrame of results and best parameters.
    """
    
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
    
    test_results = []
    
    for idx, sample_start in enumerate(sampled_starts):
        try:
            # For each sampled start, calculate the aligned train+test periods
            train_start = sample_start  # Training starts at sampled date
            train_end = train_start + pd.Timedelta(days=train_length_days)
            test_start = train_end + pd.Timedelta(days=gap_days)
            test_end = test_start + pd.Timedelta(days=test_length_days)
            
            print("*"*94)
            print(f"Global period: {global_start.strftime('%Y-%m-%d')} to {global_end.strftime('%Y-%m-%d')}")
            
            # Fetch training and testing data
            train_dfs = [fetch_crypto_data(symbol=symbol, timeframe=tf, 
                                         start_date=train_start.strftime('%Y-%m-%d %H:%M:%S'), 
                                         end_date=train_end.strftime('%Y-%m-%d %H:%M:%S'))
                        for tf in timeframes]
            test_dfs = [fetch_crypto_data(symbol=symbol, timeframe=tf, 
                                    start_date=test_start.strftime('%Y-%m-%d %H:%M:%S'), 
                                    end_date=test_end.strftime('%Y-%m-%d %H:%M:%S'),
                                    limit=limit)
                    for tf in timeframes]
            
            common_train_start, common_train_end, aligned_train_dfs = align_data_periods(train_dfs)
            common_test_start, common_test_end, aligned_test_dfs = align_data_periods(test_dfs)
            print("*"*47)
            print(f"Aligned train period sample: {common_train_start.strftime('%Y-%m-%d %H:%M:%S')} to {common_train_end.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Aligned test period sample:  {common_test_start.strftime('%Y-%m-%d %H:%M:%S')} to {common_test_end.strftime('%Y-%m-%d %H:%M:%S')}")
            print("*"*47)

            # Determine maximum lookback needed

            test_dfs_with_lookback = [lookback_test_data(original_train_df=train_dfs[i],
                                                        original_test_df=test_dfs[i],
                                                        aligned_test_df=aligned_test_dfs[i],
                                                        lookback_period_days=lookback_period_days) for i, tf in enumerate(timeframes)
                                                        if not print(f"Processing timeframe: {tf}")]
            
            test_dfs = [df_tuple[0] for df_tuple in test_dfs_with_lookback]
            lookback_reset_idx = test_dfs_with_lookback[0][1] # Same lookback period for all test samples
            
            # Optimize on training data
            _, best_params = optimize_strategy_random(
                strategy_class=strategy_class,
                df_list=aligned_train_dfs,
                lookback_reset_idx=0, # No need for lookback on train periods
                param_ranges=param_ranges,
                starting_cash=starting_cash,
                commission=commission,
                n_trials=n_trials,
                random_state=idx  # Different seed for each sample
            )
                        
            # Test on out-of-sample data
            print("\nRunning backtest on out-of-sample data with sample best parameters...")
            test_result = execute_backtest(
                strategy_class=strategy_class,
                df_list=test_dfs,
                lookback_reset_idx=lookback_reset_idx,
                strategy_params=best_params,
                starting_cash=starting_cash,
                commission=commission
            )
            
            test_result.update({
                'optimized_params': best_params
            })
            
            test_results.append(test_result)

            print(f"Completed train-test cycle starting at {train_start.strftime('%Y-%m-%d')}")
            print("*"*94)
            
        except Exception as e:
            print(f"Error in train-test cycle: {type(e).__name__}: {str(e)}")
            print("*"*47)
    
    df_test_results = pd.DataFrame(test_results)
    stats = calculate_stability_stats(df_test_results)
    
    return df_test_results, stats

def print_stability_stats(stability_stats):
    """
    Print stability statistics for various metrics.

    Parameters:
    stability_stats (dict): A dictionary where keys are metric names and values are dictionaries containing statistical data.

    The function handles different metrics such as 'total_return', 'cagr', 'max_drawdown', 'cagr_mdd_ratio', and others,
    and prints their geometric mean, mean, standard deviation, 95% confidence interval, median, interquartile range, minimum, and maximum values.
    """

    # Define which metrics are percentages
    pct_metrics = {'total_return', 'cagr', 'max_drawdown'}

    for metric, stat in stability_stats.items():
        print(f"\n{metric.upper()}:")

        # Handle metrics like total_return and cagr which involve geometric mean
        if metric in ['total_return', 'cagr']:
            print(f"Geometric Mean: {stat['gmean']:.2%}")
            print(f"Mean: {stat['mean']:.2%}")
            print(f"Std Dev: {stat['std']:.2%}")
            print(f"95% CI: [{stat['ci_lower']:.2%}, {stat['ci_upper']:.2%}]")
            print(f"Median: {stat['median']:.2%}")
            print(f"IQR: {stat['iqr']:.2%}")
            print(f"Min: {stat['min']:.2%}")
            print(f"Max: {stat['max']:.2%}")
        # Special handling for max_drawdown
        elif metric == 'max_drawdown':
            print(f"Mean: {stat['mean']:.2%}")
            print(f"Std Dev: {stat['std']:.2%}")
            print(f"95% CI: [{stat['ci_lower']:.2%}, {stat['ci_upper']:.2%}]")
            print(f"Median: {stat['median']:.2%}")
            print(f"IQR: {stat['iqr']:.2%}")
            print(f"Min: {stat['min']:.2%}")
            print(f"Max: {stat['max']:.2%}")
        # Special handling for cagr_mdd_ratio
        elif metric == 'cagr_mdd_ratio':
            print(f"Mean: {stat['mean']:.2f}")
            print(f"Median: {stat['median']:.2f}")
        # Handle other metrics like n_trades, win_trades, loss_trades
        else:
            print(f"Mean: {stat['mean']:.2f}")
            print(f"Std Dev: {stat['std']:.2f}")
            print(f"95% CI: [{stat['ci_lower']:.2f}, {stat['ci_upper']:.2f}]")
            print(f"Median: {stat['median']:.2f}")
            print(f"IQR: {stat['iqr']:.2f}")
            print(f"Min: {stat['min']:.2f}")
            print(f"Max: {stat['max']:.2f}")



def analyze_test_results(df_results):
    """
    Analyze the test results from a backtest.
    This function performs the following tasks:
    1. Extracts and counts the frequency of optimized parameter sets.
    2. Prints the frequency analysis of parameter sets.
    3. Calculates and prints average performance statistics on test samples.
    4. Identifies and returns the most recurring parameter set.
    Args:
        df_results (pd.DataFrame): DataFrame containing the backtest results with an 'optimized_params' column.
    Returns:
        dict: The most recurring parameter set from the test samples.
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
