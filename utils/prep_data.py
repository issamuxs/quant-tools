import backtrader as bt
import pandas as pd

def align_data_periods(df_list):
    """
    Aligns the periods of a list of DataFrames to a common start and end date.
    Parameters:
    df_list (list of pd.DataFrame): List of DataFrames to be aligned.
    Returns:
    tuple: A tuple containing the common start date, common end date, and a list of aligned DataFrames.
    Raises:
    ValueError: If the input list is empty.
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
    """
    Formats a list of pandas DataFrames for use with the Cerebro engine in Backtrader.

    Args:
        df_list (list): A list of pandas DataFrames to be formatted.

    Returns:
        list: A list of Backtrader PandasData objects.

    Raises:
        Exception: If there is an error during the formatting process.
    """

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
    Prepares test data with a lookback period.
    This function combines a lookback period of data from the original training and test datasets
    with the aligned test dataset. It ensures that there are no duplicate indices and marks the 
    point where the actual test period starts.
    Parameters:
    original_train_df (pd.DataFrame): The original training dataset.
    original_test_df (pd.DataFrame): The original test dataset.
    aligned_test_df (pd.DataFrame): The aligned test dataset.
    lookback_period_days (int): The number of days to look back from the start of the aligned test data.
    Returns:
    pd.DataFrame: The combined dataset with the lookback period and aligned test data.
    int: The index position where the actual test period starts.
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