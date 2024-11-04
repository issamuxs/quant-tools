import numpy as np 

def get_bt_results(starting_cash, cerebro, strategy, trades):
    print('\nBacktest Results:')
    print(f'Starting Portfolio Value: ${starting_cash:,.2f}')
    print(f'Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    print(f'Total Return: {strategy.total_return:.2%}')
    print(f'CAGR: {strategy.cagr:.2%}')
    print(f'Max Drawdown: {strategy.max_drawdown:.2%}')
    print(f'Risk-adjusted return: {strategy.ret_mdd_ratio:.2f}')
    print(f'Number of Trades: {trades.total.total}')
    if trades.total.total > 1: 
        print(f'Winning Trades: {trades.won.total}')
        print(f'Losing Trades: {trades.lost.total}')

    if len(strategy.returns) > 0:
        returns_array = np.array(strategy.returns)
        print('\nTrade Statistics:')
        print(f'Average Trade Return: {np.mean(returns_array):.2%}')
        print(f'Trade Return Std Dev: {np.std(returns_array):.2%}')
        print(f'Best Trade: {np.max(returns_array):.2%}')
        print(f'Worst Trade: {np.min(returns_array):.2%}')

    print('\nDetailed Trades:')
    for i, trade in enumerate(strategy.trade_list, 1):
        print(f"\nTrade {i}:")
        print(f"Position type: {trade['position_type']}")
        print(f"Position size: {trade['trade_size']}")
        print(f"Entry Date: {trade['entry_date']}")
        print(f"Exit Date: {trade['exit_date']}")
        print(f"Duration: {trade['duration']} bars")
        print(f"Entry Price: ${trade['entry_price']:.2f}")
        print(f"Exit Price: ${trade['exit_price']:.2f}")
        print(f"P&L: ${trade['pnl']:.2f}")
        print(f"Return: {trade['return']:.2%}")