# trading_bot/backtesting/backtester.py

import pandas as pd
import numpy as np
# For importing strategies, we'll assume the script is run in an environment
# where trading_bot package is accessible.
# Ensure trading_bot.utils.indicators is also available for strategies.
from trading_bot.strategies import BaseStrategy
from trading_bot.strategies import EMACrossStrategy, AdvancedExampleStrategy
# Import for type hinting if needed, and for __main__
from trading_bot.utils.indicators import get_ema_signals # For EMACrossStrategy's potential direct use or reference


class Backtester:
    """
    A simple event-driven backtester for trading strategies.
    """
    def __init__(self,
                 strategy: BaseStrategy,
                 historical_data: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 commission_rate: float = 0.001,  # 0.1%
                 trade_size: float = 1.0):        # e.g., 1 unit of asset like 1 BTC
        """
        Initializes the Backtester.

        Args:
            strategy (BaseStrategy): An instance of a strategy class.
            historical_data (pd.DataFrame): DataFrame with OHLCV data, indexed by Datetime.
                                            Must contain 'open', 'high', 'low', 'close', 'volume'.
            initial_balance (float): Starting capital.
            commission_rate (float): Commission rate per trade (e.g., 0.001 for 0.1%).
            trade_size (float): The fixed amount of asset to trade per signal.
        """
        if not isinstance(strategy, BaseStrategy):
            raise ValueError("Strategy must be an instance of BaseStrategy.")
        if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
            raise ValueError("Historical data must be a non-empty pandas DataFrame.")
        required_cols = ['open', 'high', 'low', 'close'] # Volume is used by AdvancedStrategy, good to have
        if not all(col in historical_data.columns for col in required_cols):
            raise ValueError(f"Historical data must contain at least columns: {required_cols}")
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            raise ValueError("Historical data must have a DatetimeIndex.")
        if 'volume' not in historical_data.columns:
            print("Warning: 'volume' column not found in historical_data. Some strategies might need it.")


        self.strategy = strategy
        self.data = historical_data.copy() # Work on a copy
        self.initial_balance = float(initial_balance)
        self.commission_rate = float(commission_rate)
        self.trade_size = float(trade_size)

        # State variables to be reset by run()
        self.balance = self.initial_balance
        self.positions = 0.0  # Current holding of the asset (units)
        self.trades = []      # List to store details of each trade
        self.equity_curve = pd.Series(dtype='float64') # Will be indexed like self.data.index
        self._last_buy_price = 0.0 # Helper to calculate P&L for simple buy/sell logic

    def _reset_state(self):
        """Resets the backtester's state for a new run."""
        self.balance = self.initial_balance
        self.positions = 0.0
        self.trades = []
        # Initialize equity curve with NaNs, then set the first value.
        self.equity_curve = pd.Series(np.nan, index=self.data.index, dtype='float64')
        if not self.data.empty:
            self.equity_curve.iloc[0] = self.initial_balance
        self._last_buy_price = 0.0

    def run(self) -> dict:
        """
        Runs the backtest simulation.
        """
        self._reset_state()

        if self.data.empty:
            print("No data to backtest.")
            # Return performance based on initial state if data is empty
            return {
                "performance": self.calculate_performance(), # Will use initial balance
                "trades": self.trades,
                "equity_curve": self.equity_curve # Will be empty or single value
            }


        print(f"Starting backtest for strategy: {self.strategy.strategy_name} on symbol: {self.strategy.symbol}")
        print(f"Initial balance: {self.initial_balance:.2f}. Data points: {len(self.data)}")

        for i, timestamp in enumerate(self.data.index):
            current_price = self.data['close'].iloc[i]

            current_data_window = self.data.iloc[:i+1]

            if current_data_window.empty:
                if i > 0: self.equity_curve.iloc[i] = self.equity_curve.iloc[i-1]
                else: self.equity_curve.iloc[i] = self.initial_balance # Should be set by _reset_state
                continue

            signals_output = self.strategy.generate_signals(current_data_window)
            signal = signals_output.get('signal', 'hold')

            if signal == 'buy' and self.positions == 0:
                cost_per_unit = current_price * (1 + self.commission_rate)
                units_to_buy = self.trade_size
                total_cost = cost_per_unit * units_to_buy

                if self.balance >= total_cost:
                    self.balance -= total_cost
                    self.positions += units_to_buy
                    self._last_buy_price = current_price
                    self.trades.append({
                        'timestamp': timestamp, 'type': 'buy', 'price': current_price,
                        'quantity': units_to_buy, 'commission': current_price * units_to_buy * self.commission_rate,
                        'cost': total_cost, 'pnl': 0
                    })
            elif signal == 'sell' and self.positions > 0:
                units_to_sell = self.positions
                proceeds_per_unit = current_price * (1 - self.commission_rate)
                total_proceeds = proceeds_per_unit * units_to_sell
                self.balance += total_proceeds

                commission_on_buy = self._last_buy_price * units_to_sell * self.commission_rate
                commission_on_sell = current_price * units_to_sell * self.commission_rate
                gross_pnl = (current_price - self._last_buy_price) * units_to_sell
                net_pnl = gross_pnl - commission_on_buy - commission_on_sell

                self.trades.append({
                    'timestamp': timestamp, 'type': 'sell', 'price': current_price,
                    'quantity': units_to_sell, 'commission': commission_on_sell,
                    'proceeds': total_proceeds, 'pnl': net_pnl
                })
                self.positions = 0; self._last_buy_price = 0

            self.equity_curve.loc[timestamp] = self.balance + (self.positions * current_price)

        # Fill forward any NaNs in equity curve that might occur if loop had issues or data was too short
        self.equity_curve.ffill(inplace=True)
        if self.equity_curve.isna().any() and not self.data.empty: # If still NaN after ffill (e.g. all data too short)
            self.equity_curve.fillna(self.initial_balance, inplace=True)


        final_equity_value = self.equity_curve.iloc[-1] if not self.equity_curve.empty else self.initial_balance
        print(f"Backtest finished. Final balance: {self.balance:.2f}, Final positions: {self.positions}, Final equity: {final_equity_value:.2f}")

        performance_metrics = self.calculate_performance()

        return {
            "performance": performance_metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }

    def calculate_performance(self) -> dict:
        if self.data.empty and not self.trades: # No data, no trades
             return { "initial_equity": self.initial_balance, "final_equity": self.initial_balance, "total_return_percentage": 0.0, "number_of_round_trip_trades":0 }


        final_equity = self.initial_balance
        if not self.equity_curve.empty:
            last_valid_equity = self.equity_curve.dropna().iloc[-1] if not self.equity_curve.dropna().empty else self.initial_balance
            final_equity = last_valid_equity
        else: # If equity curve is totally empty (e.g. data was empty from start)
            final_equity = self.balance + (self.positions * self.data['close'].iloc[-1] if self.positions > 0 and not self.data.empty else 0)


        total_return_percentage = ((final_equity - self.initial_balance) / self.initial_balance) * 100

        num_round_trips = 0; winning_trades = 0; losing_trades = 0
        total_pnl_from_trades = 0; profit_from_wins = 0; loss_from_losses = 0

        for trade in self.trades:
            if trade['type'] == 'sell':
                num_round_trips += 1
                total_pnl_from_trades += trade['pnl']
                if trade['pnl'] > 0: winning_trades += 1; profit_from_wins += trade['pnl']
                else: losing_trades += 1; loss_from_losses += trade['pnl']

        win_rate = (winning_trades / num_round_trips) * 100 if num_round_trips > 0 else 0
        avg_win_profit = profit_from_wins / winning_trades if winning_trades > 0 else 0
        avg_loss_amount = loss_from_losses / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(profit_from_wins / loss_from_losses) if loss_from_losses != 0 else np.inf

        sharpe_ratio_simplified = np.nan
        if not self.equity_curve.dropna().empty and len(self.equity_curve.dropna()) > 1:
            returns = self.equity_curve.dropna().pct_change().dropna()
            if len(returns) > 1 and returns.std() != 0:
                # Assuming daily-like periodicity for annualization factor 252. Adjust if data is different.
                # For intraday data, this annualization is often debated.
                # For now, use a placeholder annualization or simply report periodic Sharpe.
                annualization_factor = 1 # No annualization, effectively periodic Sharpe
                if len(self.data) > 0 :
                    time_delta_seconds = (self.data.index[1] - self.data.index[0]).total_seconds() if len(self.data.index) > 1 else (24*60*60)
                    periods_per_year = (252 * 24 * 60 * 60) / time_delta_seconds if time_delta_seconds > 0 else 252 # Default to daily if unknown
                    annualization_factor = np.sqrt(periods_per_year)

                sharpe_ratio_simplified = returns.mean() / returns.std() * annualization_factor

        return {
            "initial_equity": self.initial_balance, "final_equity": final_equity,
            "total_return_percentage": total_return_percentage, "total_pnl_from_trades": total_pnl_from_trades,
            "number_of_round_trip_trades": num_round_trips, "winning_trades": winning_trades, "losing_trades": losing_trades,
            "win_rate_percentage": win_rate, "average_profit_per_winning_trade": avg_win_profit,
            "average_loss_per_losing_trade": avg_loss_amount, "profit_factor": profit_factor,
            "sharpe_ratio_simplified": sharpe_ratio_simplified,
            "data_period_start": self.data.index[0] if not self.data.empty else None,
            "data_period_end": self.data.index[-1] if not self.data.empty else None,
        }

if __name__ == '__main__':
    print("--- Backtester Demonstration ---")

    # 1. Generate Sample Data
    num_points = 100 # Keep it shorter for faster demo, but enough for some EMA calc
    base_time = pd.to_datetime('2023-01-01 00:00:00')
    test_index = [base_time + pd.Timedelta(minutes=5*i) for i in range(num_points)]

    # Data that creates a crossover for EMACrossStrategy (e.g., fast=10, mid=20)
    prices = np.concatenate([
        np.linspace(100, 90, num_points//2), # Downtrend
        np.linspace(91, 110, num_points//2)  # Uptrend with crossover
    ])
    sample_data = pd.DataFrame({
        'open': prices - np.random.rand(num_points) * 0.2,
        'high': prices + np.random.rand(num_points) * 0.5 + 0.2,
        'low': prices - np.random.rand(num_points) * 0.5 - 0.2,
        'close': prices,
        'volume': np.random.randint(100, 1000, size=num_points)
    }, index=pd.DatetimeIndex(test_index))
    sample_data['low'] = sample_data[['low', 'open', 'close']].min(axis=1)
    sample_data['high'] = sample_data[['high', 'open', 'close']].max(axis=1)

    # 2. Instantiate Strategy (EMACrossStrategy for this demo)
    # Ensure strategy config matches intended behavior for sample data
    ema_config = {'fast_ema_period': 10, 'mid_ema_period': 20, 'slow_ema_period': 30} # slow_ema > mid_ema
    ema_strategy = EMACrossStrategy(symbol="TEST/USD", config=ema_config)

    # 3. Instantiate Backtester
    backtester = Backtester(strategy=ema_strategy, historical_data=sample_data,
                            initial_balance=10000.0, commission_rate=0.001, trade_size=1.0)

    # 4. Run Backtest
    results = backtester.run()

    # 5. Print Results
    print("\n--- Backtest Performance ---")
    if results and results.get("performance"):
        for key, value in results["performance"].items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').capitalize()}: {value:.2f}")
            else:
                print(f"  {key.replace('_', ' ').capitalize()}: {value}")

    print("\n--- Trades Log ---")
    if results and results.get("trades"):
        if not results["trades"]:
            print("  No trades were executed.")
        for i, trade in enumerate(results["trades"]):
            print(f"  Trade {i+1}:")
            for key, value in trade.items():
                if isinstance(value, float) and key not in ['quantity']:
                     print(f"    {key.capitalize()}: {value:.2f}")
                else:
                     print(f"    {key.capitalize()}: {value}")

    # print("\n--- Equity Curve (Last 5 points) ---")
    # if results and "equity_curve" in results and not results["equity_curve"].empty:
    #     print(results["equity_curve"].tail())
    # else:
    #     print("  Equity curve is empty or not available.")

    print("\n--- Advanced Strategy (brief example) ---")
    # This will be very slow due to HTF calculations on expanding window if not careful.
    # For demo, use fewer points or pre-calculated indicators for AdvancedStrategy
    # Or a config with very short periods.
    adv_config_demo = {
        'fast_ema': 5, 'mid_ema': 10, 'slow_ema': 15,
        'htf_period': '30min', 'htf_ema_period': 5,
        'volume_ma_period': 10, 'rsi_period': 7,
        'macd_fast': 6, 'macd_slow': 13, 'macd_signal': 4,
        'pivot_left': 3, 'pivot_right': 3, 'atr_period': 7,
        'require_htf_trend_for_buy': False, # To allow signals even if HTF is flat/short data
        'require_volume_spike_for_buy': False
    }
    # Use a shorter dataset for advanced strategy demo to speed it up
    adv_strategy_demo = AdvancedExampleStrategy(symbol="ADV/USD", config=adv_config_demo)
    backtester_adv = Backtester(strategy=adv_strategy_demo, historical_data=sample_data.head(60), # 60 points
                                initial_balance=10000.0, commission_rate=0.001, trade_size=0.5)
    results_adv = backtester_adv.run()
    print("\n--- Advanced Strategy Performance (on 60 data points) ---")
    if results_adv and results_adv.get("performance"):
         for key, value in results_adv["performance"].items():
            if isinstance(value, float): print(f"  {key.replace('_', ' ').capitalize()}: {value:.2f}")
            else: print(f"  {key.replace('_', ' ').capitalize()}: {value}")
    print("\nAdvanced Strategy Trades:")
    if results_adv and results_adv.get("trades"):
        if not results_adv["trades"]: print("  No trades for Advanced Strategy.")
        for i, trade in enumerate(results_adv["trades"]): print(f"  Trade {i+1}: {trade['type']} @ {trade['price']:.2f}, P&L: {trade['pnl']:.2f}")

    print("\nBacktester demonstration finished.")
