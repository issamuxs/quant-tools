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
        self.cagr_mdd_ratio = 0
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
            self.cagr_mdd_ratio = self.cagr / self.max_drawdown
        else:
            self.cagr_mdd_ratio = 0

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
        risk_perc=0.99,
        atr_period=14,
        sl_coef=1,
        tp_coef=1
    )

    def __init__(self):
        #Initialize trade size and price tracking
        self.trade_size = None
        self.entry_price = None
        
        # Initialize ATR-based stops
        self.data.atr = bt.indicators.ATR(
            self.data,
            period=self.p.atr_period
        )
        self.sl_atr = None
        self.tp_atr = None
        
        # SMA indicators
        self.sma1 = bt.ind.SMA(self.data.open, period=self.p.pfast)
        self.sma2 = bt.ind.SMA(self.data.open, period=self.p.pslow)
        self.crossover = bt.ind.CrossOver(self.sma1, self.sma2)

        # Track performance metrics
        self.start_value = self.broker.getvalue()
        self.max_value = self.start_value
        self.max_drawdown = 0
        self.total_return = 0
        self.cagr_mdd_ratio = 0
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
            
            # Reset trade size
            self.trade_size = None

    def next(self):
        # Update dates and values
        if self.start_date is None:
            self.start_date = self.data.datetime.datetime(0)

        current_value = self.broker.getvalue()
        current_date = self.data.datetime.datetime(0)
        
        # Update ATR values
        self.sl_atr = self.p.sl_coef*self.data.atr[-1]
        self.tp_atr = self.p.tp_coef*self.data.atr[-1]

        # Update metrics on each candle
        self.max_value = max(self.max_value, current_value)
        self.total_return = (current_value - self.start_value) / self.start_value

        # Update drawdown
        drawdown = (self.max_value - current_value) / self.max_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Update risk-adjusted-return
        if self.max_drawdown != 0:
            self.cagr_mdd_ratio = self.cagr / self.max_drawdown
        else:
            self.cagr_mdd_ratio = 0

        # Update CAGR
        years = (current_date - self.start_date).days / 365.25
        if years > 0:  # Avoid division by zero
            self.cagr = (current_value / self.start_value) ** (1/years) - 1

        current_price = self.data.open[0]

        # Trading logic
        if not self.position:  
            if self.crossover > 0:  
                size = (self.broker.getcash() * self.p.risk_perc) / self.data.open[0]
                self.trade_size = size
                self.buy(size=size)
                self.entry_price = current_price
                self.stop_loss_price = self.entry_price - self.sl_atr
                self.take_profit_price = self.entry_price + self.tp_atr
        else:
            if (current_price <= self.stop_loss_price) or \
               (current_price >= self.take_profit_price) or \
               (self.crossover < 0):  
                self.close()


class SmaConfCrossLS(bt.Strategy):
    params = dict(
        stf_pfast=10,  
        stf_pslow=30, 
        ltf_pfast=20,  
        ltf_pslow=50, 
        last_to_avg_volume_ratio=1.2,
        vol_delta_lb=0.2,
        risk_perc=0.99,
        atr_period=14,
        sl_coef = 1,
        tp_coef = 1,
        lookback_reset_idx=0
    )

    def __init__(self):

        #Initialize trade size 
        self.trade_size = None  
        
        # Initialize variables for tracking entry prices
        self.entry_price = None
        
        # Get both timeframes (STF = Short TimeFrame, LTF = Long TimeFrame)
        self.data_stf = self.datas[0]
        self.data_ltf = self.datas[1]

        # Initialize variables for stop losses and take profits
        self.data_stf.atr = bt.indicators.ATR(
            self.data_stf,
            period=self.p.atr_period)
        self.sl_atr = None
        self.tp_atr = None
        
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
        
        # Initialize performance tracking immediately for training
        # Will be reset at test_start_idx for testing
        self.start_value = self.broker.getvalue()
        self.max_value = self.start_value
        self.max_drawdown = 0
        self.total_return = 0
        self.cagr_mdd_ratio = 0
        self.cagr = 0
        self.start_date = None
        
        # Track trades and returns
        self.trade_list = []
        self.returns = []

    def volume_confirmation(self):
        # Current volume metrics
        last_volume = self.data_stf.volume[-1]
        avg_volume = self.vol_sma[0]
        vol_delta = self.vol_delta[0]
        price_change = self.price_change[0]
        
        # Volume conditions
        above_average = last_volume > avg_volume * self.p.last_to_avg_volume_ratio 
        strong_momentum = vol_delta > self.p.vol_delta_lb  
        price_aligned = (price_change > 0 and self.crossover_stf > 0) or \
                        (price_change < 0 and self.crossover_stf < 0)
        
        return above_average and strong_momentum and price_aligned
        
    def notify_trade(self, trade):
        if trade.isclosed:
            if (self.p.lookback_reset_idx == 0 or len(self) >= self.p.lookback_reset_idx): # If training or past lookback period
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

        # For test dataset, reset metrics at test_start_idx
        if self.p.lookback_reset_idx != 0 and len(self) == self.p.lookback_reset_idx + 1:
            self.start_value = self.broker.getvalue()
            self.max_value = self.start_value
            self.max_drawdown = 0
            self.total_return = 0
            self.cagr_mdd_ratio = 0
            self.cagr = 0
            self.start_date = self.data_stf.datetime.datetime(0)
            self.trade_list = []
            self.returns = []

        # Update dates and values
        if self.start_date is None:
            self.start_date = self.data_stf.datetime.datetime(0)
        
        current_value = self.broker.getvalue()
        current_date = self.data_stf.datetime.datetime(0)

        self.sl_atr = self.p.sl_coef*self.data_stf.atr[-1]
        self.tp_atr = self.p.tp_coef*self.data_stf.atr[-1]
                
        # Update metrics on each candle
        self.max_value = max(self.max_value, current_value)
        self.total_return = (current_value - self.start_value) / self.start_value
        
        # Update drawdown
        drawdown = (self.max_value - current_value) / self.max_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Update risk-adjusted-return
        if self.max_drawdown != 0:
            self.cagr_mdd_ratio = self.cagr / self.max_drawdown
        else:
            self.cagr_mdd_ratio = 0
        
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
                    self.stop_loss_price = self.entry_price - self.sl_atr
                    self.take_profit_price = self.entry_price + self.tp_atr

                # Short signal: STF crossover and LTF trend down
                elif (self.crossover_stf < 0 and 
                        self.sma1_ltf < self.sma2_ltf and 
                        current_price < self.sma1_ltf and 
                        vol_confirmed):
                    size = (self.broker.getcash() * self.p.risk_perc) / self.data_stf.open[0]
                    self.trade_size = -size
                    self.sell(size=size)
                    self.entry_price = current_price
                    self.stop_loss_price = self.entry_price + self.sl_atr
                    self.take_profit_price = self.entry_price - self.tp_atr
            else:
                # Close long if STF crosses down or LTF trend changes
                if self.trade_size > 0:
                    if (current_price <= self.stop_loss_price) or (current_price >= self.take_profit_price) or (self.crossover_stf < 0 or self.sma1_ltf < self.sma2_ltf or current_price < self.sma1_ltf):
                        self.close()

                # Close short if STF crosses up or LTF trend changes
                elif self.trade_size < 0:
                    if (current_price >= self.stop_loss_price) or (current_price <= self.take_profit_price) or (self.crossover_stf > 0 or self.sma1_ltf > self.sma2_ltf or current_price > self.sma1_ltf):
                        self.close()

class RSIBBStrategy(bt.Strategy):
    params = dict(
        rsi_period=14,
        bb_period=30,
        bb_devfactor=2,
        rsi_threshold_low=30,
        rsi_threshold_high=70,
        bb_width_threshold=0.1,
        risk_perc=0.99,
        atr_period=14,
        sl_coef=1,
        tp_coef=1
    )

    def __init__(self):
        # Initialize trade tracking
        self.trade_size = None
        self.entry_price = None
        
        # Initialize ATR-based stops
        self.data.atr = bt.indicators.ATR(
            self.data,
            period=self.p.atr_period
        )
        self.sl_atr = None
        self.tp_atr = None
        
        # Calculate RSI
        self.rsi = bt.indicators.RSI(
            self.data.close, 
            period=self.params.rsi_period
        )
        
        # Calculate Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.params.bb_period,
            devfactor=self.params.bb_devfactor
        )
        
        # Calculate BB Width
        self.bb_width = (self.bb.lines.top - self.bb.lines.bot) / self.bb.lines.mid
        
        # Track performance metrics
        self.start_value = self.broker.getvalue()
        self.max_value = self.start_value
        self.max_drawdown = 0
        self.total_return = 0
        self.cagr_mdd_ratio = 0
        self.cagr = 0
        self.start_date = None
        
        # Track trades and returns
        self.trade_list = []
        self.returns = []

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_price = trade.price
            exit_price = self.data.open[0]
            
            position_type = 'Long' if self.trade_size > 0 else 'Short'
            
            if self.trade_size >= 0:  # Long position
                trade_return = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
            else:  # Short position
                trade_return = (entry_price - exit_price) / entry_price if entry_price != 0 else 0
            
            self.trade_list.append({
                'position_type': position_type,
                'trade_size': self.trade_size,
                'entry_date': self.data.datetime.datetime(-trade.barlen),
                'exit_date': self.data.datetime.datetime(0),
                'duration': trade.barlen,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': trade.pnl,
                'return': trade_return
            })
            
            self.returns.append(trade_return)
            self.trade_size = None

    def next(self):
        # Update dates and values
        if self.start_date is None:
            self.start_date = self.data.datetime.datetime(0)
        
        current_value = self.broker.getvalue()
        current_date = self.data.datetime.datetime(0)

        # Update ATR values
        self.sl_atr = self.p.sl_coef*self.data.atr[-1]
        self.tp_atr = self.p.tp_coef*self.data.atr[-1]
        
        # Update metrics
        self.max_value = max(self.max_value, current_value)
        self.total_return = (current_value - self.start_value) / self.start_value
        
        # Update drawdown
        drawdown = (self.max_value - current_value) / self.max_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Update risk-adjusted-return
        if self.max_drawdown != 0:
            self.cagr_mdd_ratio = self.cagr / self.max_drawdown
        else:
            self.cagr_mdd_ratio = 0
        
        # Update CAGR
        years = (current_date - self.start_date).days / 365.25
        if years > 0:
            self.cagr = (current_value / self.start_value) ** (1/years) - 1

        current_price = self.data.open[0]

        if not self.position:
            # BUY Signal
            if (self.data.close[-2] < self.bb.lines.bot[-2] and
                self.rsi[-2] < self.params.rsi_threshold_low and
                self.data.close[-1] > self.data.high[-2] and
                self.bb_width[-1] > self.params.bb_width_threshold):
                
                size = (self.broker.getcash() * self.p.risk_perc) / self.data.open[0]
                self.trade_size = size
                self.buy(size=size)
                self.entry_price = current_price
                self.stop_loss_price = self.entry_price - self.sl_atr
                self.take_profit_price = self.entry_price + self.tp_atr

            # SELL Signal
            elif (self.data.close[-2] > self.bb.lines.top[-2] and
                  self.rsi[-2] > self.params.rsi_threshold_high and
                  self.data.close[-1] < self.data.low[-2] and
                  self.bb_width[-1] > self.params.bb_width_threshold):
                
                size = (self.broker.getcash() * self.p.risk_perc) / self.data.open[0]
                self.trade_size = -size
                self.sell(size=size)
                self.entry_price = current_price
                self.stop_loss_price = self.entry_price + self.sl_atr
                self.take_profit_price = self.entry_price - self.tp_atr

        else:
            # Exit long position
            if self.trade_size > 0:
                if (current_price <= self.stop_loss_price) or \
                   (current_price >= self.take_profit_price) or \
                   (self.rsi[-1] > self.params.rsi_threshold_high):
                    self.close()

            # Exit short position
            elif self.trade_size < 0:
                if (current_price >= self.stop_loss_price) or \
                   (current_price <= self.take_profit_price) or \
                   (self.rsi[-1] < self.params.rsi_threshold_low):
                    self.close()

class LiquidityImbStrategy(bt.Strategy):
    params = dict(
        vol_window=24,        
        dev_threshold=2.0,    
        roc_window=6,         
        volume_ma=12,         
        rsi_period=14,
        rsi_thresh_low=30,
        rsi_thresh_high=70,
        atr_period=14,
        sl_coef=1.5,
        tp_coef=2.0,
        risk_perc=0.99,
        lookback_reset_idx=0
    )

    def __init__(self):
        # Initialize performance tracking
        self.start_value = self.broker.getvalue()
        self.max_value = self.start_value
        self.max_drawdown = 0
        self.total_return = 0
        self.cagr_mdd_ratio = 0
        self.cagr = 0
        self.start_date = None
        
        # Track trades and returns
        self.trade_list = []
        self.trade_returns = []
        
        # Initialize trade tracking
        self.trade_size = None
        self.entry_price = None

        # Indicators
        self.price_returns = bt.ind.ROC(self.data.open, period=1)  
        self.vol = bt.ind.StdDev(self.price_returns, period=self.p.vol_window)  
        self.roc = bt.ind.ROC(self.data.open, period=self.p.roc_window)
        self.vol_ma = bt.ind.SMA(self.data.volume, period=self.p.volume_ma)
        self.vol_ratio = self.data.volume / self.vol_ma
        self.rsi = bt.ind.RSI(self.data.open, period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

    def notify_trade(self, trade):
        if trade.isclosed:
            if (self.p.lookback_reset_idx == 0 or len(self) >= self.p.lookback_reset_idx):
                entry_price = trade.price
                exit_price = self.data.open[0]
                
                position_type = 'Long' if self.trade_size > 0 else 'Short'
                
                if self.trade_size >= 0:  # Long position
                    trade_return = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
                else:  # Short position
                    trade_return = (entry_price - exit_price) / entry_price if entry_price != 0 else 0
                
                self.trade_list.append({
                    'position_type': position_type,
                    'trade_size': self.trade_size,
                    'entry_date': self.data.datetime.datetime(-trade.barlen),
                    'exit_date': self.data.datetime.datetime(0),
                    'duration': trade.barlen,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': trade.pnl,
                    'return': trade_return
                })
                
                self.trade_returns.append(trade_return)  # Updated from self.returns
                self.trade_size = None
    
    def next(self):
        # Wait for indicators to have valid values
        if len(self) < max(self.p.vol_window, self.p.roc_window, 
                      self.p.volume_ma, self.p.rsi_period, 
                      self.p.atr_period):
            return

        # Reset metrics at lookback_reset_idx for testing
        if self.p.lookback_reset_idx != 0 and len(self) == self.p.lookback_reset_idx:
            self.start_value = self.broker.getvalue()
            self.max_value = self.start_value
            self.max_drawdown = 0
            self.total_return = 0
            self.cagr_mdd_ratio = 0
            self.cagr = 0
            self.start_date = self.data.datetime.datetime(0)
            self.trade_list = []
            self.trade_returns = []

        # Update dates and values
        if self.start_date is None:
            self.start_date = self.data.datetime.datetime(0)
        
        current_value = self.broker.getvalue()
        current_date = self.data.datetime.datetime(0)

        # Update metrics
        self.max_value = max(self.max_value, current_value)
        self.total_return = (current_value - self.start_value) / self.start_value
        
        drawdown = (self.max_value - current_value) / self.max_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        if self.max_drawdown != 0:
            self.cagr_mdd_ratio = self.cagr / self.max_drawdown
        
        years = (current_date - self.start_date).days / 365.25
        if years > 0:
            self.cagr = (current_value / self.start_value) ** (1/years) - 1

        # Wait for indicators to have valid values
        if not all([self.vol[-1], self.vol[-self.p.vol_window], self.roc[-1], 
                    self.vol_ratio[-1], self.rsi[-1], self.atr[-1]]):
            return

        # Debug prints for entry conditions
        if not self.position:
            
            vol_condition = self.vol[-1] >= self.vol[-self.p.vol_window] * self.p.dev_threshold

            if not vol_condition:
                return

            # Long setup: Sharp down move with high volume
            if (self.roc[-1] < -self.vol[-1] and 
                self.vol_ratio[-1] > 1.5 and
                self.rsi[-1] < self.p.rsi_thresh_low):

                size = (self.broker.getcash() * self.p.risk_perc) / self.data.open[0]
                self.trade_size = size
                self.buy(size=size)
                self.entry_price = self.data.open[0]
                self.stop_loss_price = self.entry_price - self.p.sl_coef * self.atr[0]
                self.take_profit_price = self.entry_price + self.p.tp_coef * self.atr[0]

            # Short setup: Sharp up move with high volume
            elif (self.roc[-1] > self.vol[-1] and
                self.vol_ratio[-1] > 1.5 and
                self.rsi[-1] > self.p.rsi_thresh_high):

                size = (self.broker.getcash() * self.p.risk_perc) / self.data.open[0]
                self.trade_size = -size
                self.sell(size=size)
                self.entry_price = self.data.open[0]
                self.stop_loss_price = self.entry_price + self.p.sl_coef * self.atr[0]
                self.take_profit_price = self.entry_price - self.p.tp_coef * self.atr[0]

        else:
            # Exit logic
            if self.trade_size > 0:  # Long position
                if (self.data.open[0] <= self.stop_loss_price or
                    self.data.open[0] >= self.take_profit_price or
                    self.rsi[0] > self.p.rsi_thresh_high):
                    self.close()

            else:  # Short position
                if (self.data.open[0] >= self.stop_loss_price or
                    self.data.open[0] <= self.take_profit_price or
                    self.rsi[0] < self.p.rsi_thresh_low):
                    self.close()
