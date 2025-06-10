# trading_bot/strategies/advanced_example_strategy.py

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from trading_bot.utils.indicators import (
    get_ema_signals,
    get_higher_timeframe_trend,
    get_volume_indicators,
    get_oscillator_signals,
    get_price_volatility_signals,
    # calculate_atr is used by get_price_volatility_signals, no need to call separately usually
)

class AdvancedExampleStrategy(BaseStrategy):
    """
    An example strategy that combines multiple indicators for signal generation.
    """
    def __init__(self, symbol: str, config: dict = None):
        strategy_name = "AdvancedExampleStrategy"

        default_config = {
            # EMA Signals
            'fast_ema': 10, 'mid_ema': 20, 'slow_ema': 50,
            # Higher Timeframe Trend
            'htf_period': '1h', 'htf_ema_period': 50, # Ensure data has DatetimeIndex for this
            # Volume Indicators
            'volume_ma_period': 20, 'volume_spike_multiplier': 2.0, 'min_big_order_value': 10000,
            # Oscillator Signals
            'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30, 'rsi_buy_threshold': 65, # Example custom threshold
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            # Price/Volatility Signals
            'pivot_left': 5, 'pivot_right': 5,
            'atr_period': 14, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 4.0,
            # Strategy specific
            'require_volume_spike_for_buy': True,
            'require_htf_trend_for_buy': True,
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__(strategy_name, symbol, merged_config)
        print(f"{self.strategy_name} initialized for {self.symbol} with config: {self.config}")

    def generate_signals(self, data: pd.DataFrame) -> dict:
        """
        Generates trading signals based on a combination of indicators.
        """
        # Ensure necessary columns are present (at least 'open', 'high', 'low', 'close', 'volume')
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            return {'signal': 'hold', 'reason': f"Missing required data columns: {missing_cols}"}

        # Ensure data is not empty and has enough rows for slowest period
        # Max of: slow_ema, htf_ema_period (if data is resampled, this applies to original data length for resample to make sense)
        # volume_ma_period, rsi_period, macd_slow+macd_signal, atr_period, pivot_left+pivot_right+1
        # This is a rough estimate, each indicator function handles its own length checks.
        min_data_len = max(self.config['slow_ema'], self.config['volume_ma_period'],
                           self.config['rsi_period']+1, self.config['macd_slow'] + self.config['macd_signal'],
                           self.config['atr_period']+1, self.config['pivot_left'] + self.config['pivot_right'] + 1)
        if len(data) < min_data_len:
            return {'signal': 'hold', 'reason': f"Insufficient data. Need at least {min_data_len} bars."}

        # --- Calculate all indicators ---
        indicator_values = {}

        # 1. EMA Signals
        ema_signals = get_ema_signals(
            prices=data['close'],
            fast_period=self.config['fast_ema'],
            mid_period=self.config['mid_ema'],
            slow_period=self.config['slow_ema']
        )
        indicator_values['ema_signals'] = ema_signals

        # 2. Higher Timeframe Trend (requires DatetimeIndex)
        if isinstance(data.index, pd.DatetimeIndex):
            htf_signals = get_higher_timeframe_trend(
                prices_df=data, # expects ohlc
                htf_period=self.config['htf_period'],
                htf_ema_period=self.config['htf_ema_period']
            )
            indicator_values['htf_signals'] = htf_signals
        else:
            indicator_values['htf_signals'] = {'htfTrendUp': False, 'htfTrendDown': False, 'reason': 'Data has no DatetimeIndex'}
            if self.config.get('require_htf_trend_for_buy'): # If HTF is mandatory, cannot proceed for buy
                 indicator_values['htf_signals']['htfTrendUp'] = False # Ensure it fails buy condition

        # 3. Volume Indicators
        volume_signals = get_volume_indicators(
            df=data, # expects close, volume
            volume_ma_period=self.config['volume_ma_period'],
            spike_multiplier=self.config['volume_spike_multiplier'],
            min_big_order_value=self.config['min_big_order_value']
        )
        indicator_values['volume_signals'] = volume_signals

        # 4. Oscillator Signals (RSI, MACD)
        osc_signals = get_oscillator_signals(
            prices=data['close'],
            rsi_period=self.config['rsi_period'],
            rsi_overbought=self.config['rsi_overbought'],
            rsi_oversold=self.config['rsi_oversold'],
            macd_fast=self.config['macd_fast'],
            macd_slow=self.config['macd_slow'],
            macd_signal=self.config['macd_signal']
        )
        indicator_values['osc_signals'] = osc_signals

        # 5. Price/Volatility Signals (Pivots, ATR)
        pv_signals = get_price_volatility_signals(
            df=data, # expects high, low, close
            pivot_left=self.config['pivot_left'],
            pivot_right=self.config['pivot_right'],
            atr_period=self.config['atr_period']
        )
        indicator_values['pv_signals'] = pv_signals

        # --- Initialize signal dictionary ---
        final_signal = {'signal': 'hold', 'details': indicator_values}

        # --- Trading Logic ---
        latest_close = data['close'].iloc[-1]

        # Buy Condition
        is_buy_condition = True # Start with True, then AND with conditions

        is_buy_condition &= ema_signals.get('trendUp', False)
        is_buy_condition &= ema_signals.get('emaFastCrossAboveMid', False)

        if self.config.get('require_htf_trend_for_buy'):
            is_buy_condition &= indicator_values['htf_signals'].get('htfTrendUp', False)

        is_buy_condition &= not osc_signals.get('isRsiOverbought', True) # Not overbought
        # Example of custom RSI threshold for buying: RSI must be < a certain level but not necessarily oversold
        is_buy_condition &= (osc_signals.get('rsi_value', 100) < self.config.get('rsi_buy_threshold', 75))

        if self.config.get('require_volume_spike_for_buy', False):
            is_buy_condition &= volume_signals.get('isVolumeSpike', False)

        is_buy_condition &= osc_signals.get('isMacdBullish', False)

        if is_buy_condition:
            final_signal['signal'] = 'buy'
            final_signal['reason'] = "All buy conditions met (EMA trend, EMA cross, HTF trend, RSI not OB, MACD bullish)"
            if self.config.get('require_volume_spike_for_buy'):
                 final_signal['reason'] += " with volume spike."
            # Calculate SL/TP using ATR
            if pd.notna(pv_signals.get('latest_atr')):
                final_signal['stop_loss'] = latest_close - (pv_signals['latest_atr'] * self.config['atr_sl_multiplier'])
                final_signal['take_profit'] = latest_close + (pv_signals['latest_atr'] * self.config['atr_tp_multiplier'])
            return final_signal # Return immediately on buy signal

        # Sell Condition (example for closing a long or initiating short)
        is_sell_condition = False # Start with False, then OR with conditions

        # Condition 1: MACD Dead Cross
        if osc_signals.get('macdDeadCross', False):
            is_sell_condition = True
            final_signal['reason'] = "MACD Dead Cross"

        # Condition 2: Trend down AND price breaks below support
        if not is_sell_condition: # Only check if not already a sell signal
            if ema_signals.get('trendDown', False) and \
               pd.notna(pv_signals.get('currentSupport')) and \
               latest_close < pv_signals['currentSupport']:
                is_sell_condition = True
                final_signal['reason'] = f"Trend Down and broke support {pv_signals['currentSupport']:.2f}"

        if is_sell_condition:
            final_signal['signal'] = 'sell'
            # Calculate SL/TP using ATR (for shorting, SL is above, TP is below)
            if pd.notna(pv_signals.get('latest_atr')):
                final_signal['stop_loss'] = latest_close + (pv_signals['latest_atr'] * self.config['atr_sl_multiplier'])
                final_signal['take_profit'] = latest_close - (pv_signals['latest_atr'] * self.config['atr_tp_multiplier'])
            return final_signal # Return immediately on sell signal

        # Default to hold if no conditions met
        final_signal['reason'] = "No strong buy or sell signals."
        return final_signal

if __name__ == '__main__':
    print("--- AdvancedExampleStrategy Demonstration ---")

    # Generate sample data (long enough for all indicators)
    # Min length for default config: max(50, 50, 20, 14+1, 26+9, 14+1, 5+5+1) = max(50,50,20,15,35,15,11) = 50.
    # HTF resample also needs enough data. For 1h resample from 5min, 1 htf_ema_period = 12 * 5min candles.
    # So htf_ema_period 50 needs 50*12 = 600 5min candles. This is a lot.
    # Let's use smaller htf_ema_period for demo, or ensure enough data.
    # For demo, let's make htf_ema_period smaller in a custom config for testing.

    num_points = 200 # Decent length for most indicators with default periods
    base_time = pd.to_datetime('2023-01-01 00:00:00')
    test_index = [base_time + pd.Timedelta(minutes=5*i) for i in range(num_points)]

    # Create data that might trigger a buy signal towards the end
    # Start with a consolidation/slight downtrend, then a clear uptrend with volume spike
    prices = np.linspace(100, 95, num_points//2).tolist() + \
             np.linspace(96, 120, num_points//2).tolist()
    volumes = np.random.randint(100, 300, num_points).tolist()
    volumes[-num_points//4:] = [v * 3 for v in volumes[-num_points//4:]] # Volume spike in last quarter

    sample_data = pd.DataFrame({
        'open': np.array(prices) - np.random.rand(num_points) * 0.5,
        'high': np.array(prices) + np.random.rand(num_points) * 1.0,
        'low': np.array(prices) - np.random.rand(num_points) * 1.0,
        'close': prices,
        'volume': volumes
    }, index=pd.DatetimeIndex(test_index))
    # Ensure low is lowest and high is highest
    sample_data['low'] = sample_data[['low', 'open', 'close']].min(axis=1)
    sample_data['high'] = sample_data[['high', 'open', 'close']].max(axis=1)


    # Default config strategy
    print("\n--- Testing with Default Config ---")
    adv_strategy_default = AdvancedExampleStrategy(symbol="BTC/USD")
    # Default config has htf_ema_period=50. 50 * 12 = 600 points needed for 5min data.
    # The sample data (200pts) is too short for this specific HTF setting.
    # The htf_signals will likely indicate no trend or insufficient data.
    signals_default = adv_strategy_default.generate_signals(sample_data)
    print("Default Config Signals:")
    print(f"  Signal: {signals_default.get('signal')}")
    print(f"  Reason: {signals_default.get('reason', 'N/A')}")
    if 'stop_loss' in signals_default:
        print(f"  Stop Loss: {signals_default['stop_loss']:.2f}")
        print(f"  Take Profit: {signals_default['take_profit']:.2f}")
    # print("  Details:", signals_default.get('details')) # Can be very verbose


    # Custom config strategy for more reactive HTF
    print("\n--- Testing with Custom Config (Reactive HTF) ---")
    custom_config_reactive_htf = {
        'htf_period': '30min', # Shorter HTF
        'htf_ema_period': 10,  # Shorter HTF EMA period (10 * 6 = 60 5-min candles)
        'fast_ema': 5, 'mid_ema': 10, 'slow_ema': 20, # Faster main EMAs
        'rsi_buy_threshold': 70, # Allow buying up to RSI 70
        'require_volume_spike_for_buy': False # Don't strictly need volume spike
    }
    adv_strategy_custom = AdvancedExampleStrategy(symbol="ETH/USD", config=custom_config_reactive_htf)
    signals_custom = adv_strategy_custom.generate_signals(sample_data) # Use same 200pt data
    print("Custom Config Signals:")
    print(f"  Signal: {signals_custom.get('signal')}")
    print(f"  Reason: {signals_custom.get('reason', 'N/A')}")
    if 'stop_loss' in signals_custom:
        print(f"  Stop Loss: {signals_custom['stop_loss']:.2f}")
        print(f"  Take Profit: {signals_custom['take_profit']:.2f}")

    # Example: Print specific indicator values from details for the custom run
    print("\n  Selected Indicator Details (Custom Config Run):")
    if signals_custom.get('details'):
        details = signals_custom['details']
        print(f"    EMA Fast: {details.get('ema_signals', {}).get('fast_ema', 'N/A'):.2f}")
        print(f"    EMA Mid Cross: {details.get('ema_signals', {}).get('emaFastCrossAboveMid', 'N/A')}")
        print(f"    HTF Trend Up: {details.get('htf_signals', {}).get('htfTrendUp', 'N/A')}")
        print(f"    RSI Value: {details.get('osc_signals', {}).get('rsi_value', 'N/A'):.2f}")
        print(f"    Is RSI Overbought: {details.get('osc_signals', {}).get('isRsiOverbought', 'N/A')}")
        print(f"    MACD Bullish: {details.get('osc_signals', {}).get('isMacdBullish', 'N/A')}")
        print(f"    Volume Spike: {details.get('volume_signals', {}).get('isVolumeSpike', 'N/A')}")
        print(f"    Current Resistance: {details.get('pv_signals', {}).get('currentResistance', 'N/A')}")
        print(f"    ATR: {details.get('pv_signals', {}).get('latest_atr', 'N/A'):.3f}")

    print("\nAdvanced strategy demonstration finished.")
