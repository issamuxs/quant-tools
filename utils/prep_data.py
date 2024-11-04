import backtrader as bt

def align_data_periods(df_list):
    if not df_list:  # Check if list is empty
        raise ValueError("DataFrame list is empty")
    
    # Find common start and end dates
    common_start = max(df.index[0] for df in df_list)
    common_end = min(df.index[-1] for df in df_list)
    
    # Create new list with aligned dataframes
    aligned_df_list = [df.loc[common_start:common_end].copy() for df in df_list]
    
    # Print aligned periods
    print("\nAligned Backtest Period:")
    print(f"Start: {common_start.strftime('%Y-%m-%d')}")
    print(f"End: {common_end.strftime('%Y-%m-%d')}")
    print(f"Total Days: {(common_end - common_start).days}")
    print(f"\nCandles per DataFrame: {[len(df) for df in aligned_df_list]}")
    
    return aligned_df_list

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
