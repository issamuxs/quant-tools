import backtrader as bt
import pandas as pd

def align_data_periods(df_list):
    """
    Align data periods for multiple timeframes, ensuring proper overlap
    """
    if not df_list:
        raise ValueError("DataFrame list is empty")
    
    # Find common start and end dates
    common_start = max(df.index[0] for df in df_list)
    common_end = min(df.index[-1] for df in df_list)
    
    # Create new list with aligned dataframes
    aligned_df_list = [df.loc[common_start:common_end].copy() for df in df_list]
    
    return common_start, common_end, aligned_df_list

def format_data_cerebro(df_list):

    data_df_list = []
    try:
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
    except Exception as e:
        print(f"Error while formatting data for Cerebro: {type(e).__name__}: {str(e)}")

    return data_df_list

def lookback_test_data(original_train_df, original_test_df, aligned_test_df, lookback_period_days):
    """
    Prepare test data with initialization period, using both original datasets
    """
    # Get dates from aligned data
    aligned_test_start = aligned_test_df.index[0]

    # Calculate lookback start date
    lookback_start = aligned_test_start - pd.Timedelta(days=lookback_period_days)

    # Get lookback data from both original datasets
    lookback_train = original_train_df[
        (original_train_df.index >= lookback_start) &
        (original_train_df.index < aligned_test_start)
        ].copy()
    lookback_test = original_test_df[
        (original_test_df.index >= lookback_start) & 
        (original_test_df.index < aligned_test_start)
    ].copy()

    # Combine lookback data from both sources and sort
    lookback_data = pd.concat([lookback_train, lookback_test]).sort_index()
    # Remove any duplicates if they exist
    lookback_data = lookback_data[~lookback_data.index.duplicated(keep='first')]

    # Concatenate lookback and aligned test data
    full_test_df = pd.concat([lookback_data, aligned_test_df])
    full_test_df = full_test_df[~full_test_df.index.duplicated(keep='first')].sort_index()

    # Mark where actual test period starts
    lookback_start_idx = len(full_test_df) - len(aligned_test_df)

    print("*"*47)
    print(f"Lookback period needed: {lookback_start} to {aligned_test_start}")
    print(f"Lookback data collected: {lookback_data.index[0]} to {lookback_data.index[-1]} ({len(lookback_data)} bars)")
    print(f"Test data period: {aligned_test_df.index[0]} to {aligned_test_df.index[-1]} ({len(aligned_test_df)} bars)")
    print(f"Reset index at: {lookback_start_idx} bars")
    print("*"*47)
    
    return full_test_df, lookback_start_idx