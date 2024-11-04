import backtrader as bt
from datetime import datetime

class BuyHold(bt.Strategy):

    params = dict(
        risk_perc=0.99  
    )

    def __init__(self):

        #Initialize trade size 
        self.trade_size = None  

        # Track performance metrics
        self.start_value = self.broker.getvalue()
        self.max_value = self.start_value
        self.max_drawdown = 0
        self.total_return = 0
        self.ret_mdd_ratio = 0
        self.cagr = 0
        self.start_date = None

        # Track trades and returns
        self.trade_list = []
        self.returns = []

    def stop(self):
    # This method is called when the backtest ends
        if self.position:  # If position is still open
            self.close()  # Close the position

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_price = trade.price
            exit_price = self.data_stf.open[0]
            size = trade.size
            
            position_type = 'Long' 
            
            # Calculate PnL and return
            trade_return = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
            pnl = (exit_price - entry_price) * abs(size)
            
            # Store trade information
            self.trade_list.append({
                'position_type': position_type,
                'trade_size': size,
                'entry_date': self.data_stf.datetime.datetime(-trade.barlen),
                'exit_date': self.data_stf.datetime.datetime(0),
                'duration': trade.barlen,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'return': trade_return
            })
            
            # Track returns
            self.returns.append(trade_return)

            # Reset the trade size
            self.trade_size = None

    def next(self):
        # Update dates and values
        if self.start_date is None:
            self.start_date = self.data.datetime.datetime(0)

        current_value = self.broker.getvalue()
        current_date = self.data.datetime.datetime(0)

        # Update metrics on each candle
        self.max_value = max(self.max_value, current_value)
        self.total_return = (current_value - self.start_value) / self.start_value

        # Update drawdown
        drawdown = (self.max_value - current_value) / self.max_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Update risk-adjusted-return
        if self.max_drawdown != 0:
            self.ret_mdd_ratio = self.total_return / self.max_drawdown
        else:
            self.ret_mdd_ratio = 0

        # Update CAGR
        years = (current_date - self.start_date).days / 365.25
        if years > 0:  # Avoid division by zero
            self.cagr = (current_value / self.start_value) ** (1/years) - 1

        # Trading logic
        if not self.position:  
            size = (self.broker.getcash() * self.p.risk_perc) / self.data.open[0]
            self.buy(size=size)  


class SmaSimpleCrossL(bt.Strategy):

    params = dict(
        pfast=10,  
        pslow=30,  
        risk_perc=0.99  
    )

    def __init__(self):

        #Initialize trade size 
        self.trade_size = None
        
        self.sma1 = bt.ind.SMA(self.data.open, period=self.p.pfast)
        self.sma2 = bt.ind.SMA(self.data.open, period=self.p.pslow)
        self.crossover = bt.ind.CrossOver(self.sma1, self.sma2)

        # Track performance metrics
        self.start_value = self.broker.getvalue()
        self.max_value = self.start_value
        self.max_drawdown = 0
        self.total_return = 0
        self.ret_mdd_ratio = 0
        self.cagr = 0
        self.start_date = None

        # Track trades and returns
        self.trade_list = []
        self.returns = []

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_price = trade.price
            exit_price = self.data.open[0]
            
            position_type = 'Long' 
            trade_return = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
            
            # Store trade information
            self.trade_list.append({
                'position_type': position_type,
                'trade_size': self.trade_size,
                'entry_date': self.datas[0].datetime.datetime(-trade.barlen),
                'exit_date': self.datas[0].datetime.datetime(0),
                'duration': trade.barlen,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': trade.pnl,
                'return': trade_return
            })
            
            # Track returns
            self.returns.append(trade_return)

    def next(self):
        # Update dates and values
        if self.start_date is None:
            self.start_date = self.data.datetime.datetime(0)

        current_value = self.broker.getvalue()
        current_date = self.data.datetime.datetime(0)

        # Update metrics on each candle
        self.max_value = max(self.max_value, current_value)
        self.total_return = (current_value - self.start_value) / self.start_value

        # Update drawdown
        drawdown = (self.max_value - current_value) / self.max_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Update risk-adjusted-return
        if self.max_drawdown != 0:
            self.ret_mdd_ratio = self.total_return / self.max_drawdown
        else:
            self.ret_mdd_ratio = 0

        # Update CAGR
        years = (current_date - self.start_date).days / 365.25
        if years > 0:  # Avoid division by zero
            self.cagr = (current_value / self.start_value) ** (1/years) - 1

        # Trading logic
        if not self.position:  
            if self.crossover > 0:  
                size = (self.broker.getcash() * self.p.risk_perc) / self.data.open[0]
                self.trade_size = size
                self.buy(size=size)  
        elif self.crossover < 0:  
            self.close()


class SmaConfCrossLS(bt.Strategy):
    params = dict(
        stf_pfast=20,  
        stf_pslow=200, 
        ltf_pfast=7,  
        ltf_pslow=21, 
        risk_perc=0.99,
        stop_loss_perc=0.01
    )

    def __init__(self):

        #Initialize trade size 
        self.trade_size = None  
        
        # Initialize variables for tracking entry prices and stop losses
        self.entry_price = None
        self.stop_loss_price = None
        
        # Get both timeframes (STF = Short TimeFrame, LTF = Long TimeFrame)
        self.data_stf = self.datas[0]
        self.data_ltf = self.datas[1]
        
        # Calculate SMAs for both timeframes
        self.sma1_stf = bt.ind.SMA(self.data_stf.open, period=self.p.stf_pfast)
        self.sma2_stf = bt.ind.SMA(self.data_stf.open, period=self.p.stf_pslow)
        self.crossover_stf = bt.ind.CrossOver(self.sma1_stf, self.sma2_stf)
        
        self.sma1_ltf = bt.ind.SMA(self.data_ltf.open, period=self.p.ltf_pfast)
        self.sma2_ltf = bt.ind.SMA(self.data_ltf.open, period=self.p.ltf_pslow)
        
        # Volume indicators
        self.vol_sma = bt.ind.SMA(self.data_stf.volume(-1), period=20)
        self.vol_std = bt.ind.StdDev(self.data_stf.volume(-1), period=20)
        self.price_change = self.data_stf.open - self.data_stf.close(-1)
        self.vol_delta = (self.data_stf.volume(-1) - self.vol_sma) / self.vol_std
        
        # Track performance metrics
        self.start_value = self.broker.getvalue()
        self.max_value = self.start_value
        self.max_drawdown = 0
        self.total_return = 0
        self.ret_mdd_ratio = 0
        self.cagr = 0
        self.start_date = None
        
        # Track trades and returns
        self.trade_list = []
        self.returns = []

    def volume_confirmation(self, last_to_avg_volume_ratio=1.2, vol_delta_lb=0.2):
        # Current volume metrics
        last_volume = self.data_stf.volume[-1]
        avg_volume = self.vol_sma[0]
        vol_delta = self.vol_delta[0]
        price_change = self.price_change[0]
        
        # Volume conditions
        above_average = last_volume > avg_volume * last_to_avg_volume_ratio 
        strong_momentum = vol_delta > vol_delta_lb  
        price_aligned = (price_change > 0 and self.crossover_stf > 0) or \
                        (price_change < 0 and self.crossover_stf < 0)
        
        return above_average and strong_momentum and price_aligned
        
    def notify_trade(self, trade):
        if trade.isclosed:
            entry_price = trade.price
            exit_price = self.data_stf.open[0] # Approximation that might have strong impact on performance
            
            # Determine if the trade was long or short based on trade.size
            position_type = 'Long' if self.trade_size > 0 else 'Short'
            
            # Calculate PnL and return based on trade direction
            if self.trade_size >= 0:  # Long position
                trade_return = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
            else:  # Short position
                trade_return = (entry_price - exit_price) / entry_price if entry_price != 0 else 0
            
            # Store trade information
            self.trade_list.append({
                'position_type': position_type,
                'trade_size': self.trade_size,
                'entry_date': self.data_stf.datetime.datetime(-trade.barlen),
                'exit_date': self.data_stf.datetime.datetime(0),
                'duration': trade.barlen,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': trade.pnl,
                'return': trade_return
            })
            
            # Track returns
            self.returns.append(trade_return)

            # Reset the trade size
            self.trade_size = None


    def next(self):
        # Update dates and values
        if self.start_date is None:
            self.start_date = self.data_stf.datetime.datetime(0)
        
        current_value = self.broker.getvalue()
        current_date = self.data_stf.datetime.datetime(0)
        
        # Update metrics on each candle
        self.max_value = max(self.max_value, current_value)
        self.total_return = (current_value - self.start_value) / self.start_value
        
        # Update drawdown
        drawdown = (self.max_value - current_value) / self.max_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Update risk-adjusted-return
        if self.max_drawdown != 0:
            self.ret_mdd_ratio = self.total_return / self.max_drawdown
        else:
            self.ret_mdd_ratio = 0
        
        # Update CAGR
        years = (current_date - self.start_date).days / 365.25
        if years > 0:
            self.cagr = (current_value / self.start_value) ** (1/years) - 1
        
        # Check if we have data for both timeframes
        if len(self.data_ltf) > 0:
            current_price = self.data_stf.open[0]
            vol_confirmed = self.volume_confirmation()
            
            # Trading logic
            if not self.position:
                # Long signal: STF crossover and LTF trend up
                if (self.crossover_stf > 0 and 
                    self.sma1_ltf > self.sma2_ltf and 
                    current_price > self.sma1_ltf and
                    vol_confirmed):
                    size = (self.broker.getcash() * self.p.risk_perc) / self.data_stf.open[0]
                    self.trade_size = size
                    self.buy(size=size)
                    self.entry_price = current_price
                    self.stop_loss_price = self.entry_price * (1 - self.p.stop_loss_perc)  # Set stop loss

                # Short signal: STF crossover and LTF trend down
                elif (self.crossover_stf < 0 and 
                        self.sma1_ltf < self.sma2_ltf and 
                        current_price < self.sma1_ltf and
                        vol_confirmed):
                    size = (self.broker.getcash() * self.p.risk_perc) / self.data_stf.open[0]
                    self.trade_size = -size
                    self.sell(size=size)
                    self.entry_price = current_price
                    self.stop_loss_price = self.entry_price * (1 + self.p.stop_loss_perc)  # Set stop loss for short

            else:
                # Close long if STF crosses down or LTF trend changes
                if self.trade_size > 0:
                    if current_price <= self.stop_loss_price or (self.crossover_stf < 0 or self.sma1_ltf < self.sma2_ltf or current_price < self.sma1_ltf):
                        self.close()

                # Close short if STF crosses up or LTF trend changes
                elif self.trade_size < 0:
                    if current_price >= self.stop_loss_price or (self.crossover_stf > 0 or self.sma1_ltf > self.sma2_ltf or current_price > self.sma1_ltf):
                        self.close()
