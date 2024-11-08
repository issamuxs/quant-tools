import backtrader as bt
import pandas as pd

def align_data_periods(df_list):
    if not df_list:  # Check if list is empty
        raise ValueError("DataFrame list is empty")
    
    # Find common start and end dates
    common_start = max(df.index[0] for df in df_list)
    common_end = min(df.index[-1] for df in df_list)
    
    # Create new list with aligned dataframes
    aligned_df_list = [df.loc[common_start:common_end].copy() for df in df_list]
    
    return common_start, common_end, aligned_df_list

def format_data_cerebro(df_list):

    data_df_list = []

    for df in df_list:
        data_df = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open=0,
        high=1,
        low=2,
        close=3,
        volume=4,
        openinterest=-1
        )
        data_df_list.append(data_df)

    return data_df_list

def lookback_test_data(train_df, test_df, lookback_period_days):
    """
    Prepare test data with initialization period using days
    Returns both full dataframe and index where actual test period starts
    """
    # Calculate lookback start date
    test_start = test_df.index[0]
    lookback_start = test_start - pd.Timedelta(days=lookback_period_days)
    
    # Get lookback data using date range
    lookback_data = train_df[train_df.index >= lookback_start].copy()
    
    # Concatenate with test data
    full_test_df = pd.concat([lookback_data, test_df])
    
    # Mark where actual test period starts
    lookback_reset_idx = full_test_df.index.get_loc(test_start) + 1
    print("*"*47)
    print(f"Lookback data from: {lookback_data.index[0]}")
    print(f"Lookback data to: {lookback_data.index[-1]}")
    print(f"Test data starts at: {test_start}")
    print(f"Calculated lookback_reset_idx: {lookback_reset_idx}")
    print("*"*47)
    
    return full_test_df, lookback_reset_idx
