import pandas as pd
import ccxt

def fetch_crypto_data(symbol, timeframe, start_date, end_date, limit=1000):
    exchange = ccxt.binance()
   
    from_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)

    if end_date is None:
        to_timestamp = int(pd.Timestamp.now().timestamp() * 1000)
    else:
        to_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)

    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, from_timestamp, limit=limit)
        if not ohlcv or from_timestamp >= to_timestamp:
            break
            
        # Only add data points before end_date
        if end_date:
            ohlcv = [candle for candle in ohlcv if candle[0] <= to_timestamp]
            
        all_ohlcv.extend(ohlcv)
        
        if not ohlcv:  # No more data
            break
            
        from_timestamp = ohlcv[-1][0] + 1

    df = pd.DataFrame(all_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)

    # Filter data between start and end dates
    if end_date:
        df = df[start_date:end_date]
    else:
        df = df[start_date:]
        
    return df
