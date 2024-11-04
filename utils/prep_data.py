import backtrader as bt

def align_data_periods(df1, df2):
    # Align the periods
    common_start = max(df1.index[0], df2.index[0])
    common_end = min(df1.index[-1], df2.index[-1])

    df1 = df1[common_start:common_end]
    df2 = df2[common_start:common_end]

    # Print aligned periods
    print("\nAligned Backtest Period:")
    print(f"Start: {common_start.strftime('%Y-%m-%d')}")
    print(f"End: {common_end.strftime('%Y-%m-%d')}")
    print(f"Total Days: {(common_end - common_start).days}")
    print(f"\nSTF Candles: {len(df1)}")
    print(f"LTF Candles: {len(df2)}")

    return df1, df2

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
