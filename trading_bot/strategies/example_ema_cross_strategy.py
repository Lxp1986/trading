# trading_bot/strategies/example_ema_cross_strategy.py

import pandas as pd
import numpy as np # Added for test data generation
from .base_strategy import BaseStrategy
from trading_bot.utils.indicators import get_ema_signals

class EMACrossStrategy(BaseStrategy):
    """
    An example strategy based on EMA crossover signals.
    Uses self.config for parameters.
    """
    DEFAULT_CONFIG = {
        'fast_ema_period': 10,
        'mid_ema_period': 20,
        'slow_ema_period': 50, # Must be > mid_ema_period for get_ema_signals
        # Add other potential params like 'ema_sell_cross_enabled': False
    }

    def __init__(self,
                 symbol: str,
                 config: dict = None):
        """
        Initializes the EMACrossStrategy.

        Args:
            symbol (str): Trading symbol.
            config (dict, optional): Strategy-specific configuration parameters.
                                     These will override defaults.
        """
        # Merge provided config with defaults
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__("EMACrossStrategy", symbol, merged_config)

        # Validate that fast < mid < slow after merging
        # This is important because get_ema_signals expects this order.
        fp = self.get_parameter('fast_ema_period')
        mp = self.get_parameter('mid_ema_period')
        sp = self.get_parameter('slow_ema_period')

        if not (fp < mp < sp):
            print(f"Warning: EMA periods ({fp},{mp},{sp}) are not strictly ascending (fast < mid < slow). "
                  f"Adjusting to defaults ({self.DEFAULT_CONFIG['fast_ema_period']},{self.DEFAULT_CONFIG['mid_ema_period']},{self.DEFAULT_CONFIG['slow_ema_period']}) for internal calculations if logic demands it, "
                  f"or strategy might behave unexpectedly if get_ema_signals fails.", file=sys.stderr if 'sys' in globals() else None)
            # Consider raising an error or auto-adjusting if critical:
            # raise ValueError(f"EMA periods must be fast < mid < slow. Got: Fast={fp}, Mid={mp}, Slow={sp}")
            # Or, the internal logic in generate_signals will handle it with its min/max adjustments.

        print(f"{self.strategy_name} initialized for {self.symbol} with EMA periods: "
              f"Fast={fp}, "
              f"Mid={mp}, "
              f"Slow={sp}")

    def generate_signals(self, data: pd.DataFrame) -> dict:
        """
        Generates trading signals based on EMA crossover logic.

        Args:
            data (pd.DataFrame): DataFrame containing at least 'close' prices.

        Returns:
            dict: Signals dictionary (e.g., {'signal': 'buy'/'sell'/'hold'}).
        """
        if 'close' not in data.columns:
            return {'signal': 'hold', 'reason': "Data does not contain 'close' column."}

        close_prices = data['close']

        # Retrieve parameters from self.config using get_parameter
        fast_period = self.get_parameter('fast_ema_period')
        mid_period = self.get_parameter('mid_ema_period')
        slow_period = self.get_parameter('slow_ema_period')

        # Ensure periods are valid for get_ema_signals (fast < mid < slow)
        # This is a simplified handling; get_ema_signals itself might have more robust checks or fail.
        # The print in __init__ should serve as a warning if config is bad.
        # Forcing it here to prevent runtime errors in get_ema_signals if periods are not ordered.
        # This ensures that the call to get_ema_signals will not fail due to period ordering.
        # However, if the user intended specific values that are now overridden, the strategy's
        # behavior might not be as expected from their config.
        # A more robust solution would be strict validation in __init__ or set_parameters.

        _fp = fast_period
        _mp = mid_period
        _sp = slow_period

        if not (_fp < _mp < _sp):
            print(f"Warning: EMA periods in config ({_fp},{_mp},{_sp}) not valid for 3-EMA cross. Using defaults for calculation: 10,20,50", file=sys.stderr if 'sys' in globals() else None)
            _fp, _mp, _sp = 10, 20, 50 # Fallback to known good relative order for get_ema_signals
            if not (_fp < _mp < _sp): # Should not happen with these defaults
                 _fp = min(fast_period, mid_period-1, slow_period-2)
                 _mp = min(mid_period, slow_period-1)
                 _sp = slow_period
                 if not (_fp < _mp < _sp): # Final fallback
                     _fp, _mp, _sp = 10,20,50


        ema_signals = get_ema_signals(
            prices=close_prices,
            fast_period=_fp,
            mid_period=_mp,
            slow_period=_sp
        )

        signal_details = { # Store actual parameters used for signal calc and the results
            'params_used': {'fast': _fp, 'mid': _mp, 'slow': _sp},
            'ema_values': ema_signals
        }
        final_signal = {'signal': 'hold', 'details': signal_details}

        if ema_signals.get('emaFastCrossAboveMid'):
            final_signal['signal'] = 'buy'
            final_signal['reason'] = f"Fast EMA ({ema_signals.get('fast_ema', 'N/A'):.2f}) crossed above Mid EMA ({ema_signals.get('mid_ema', 'N/A'):.2f})"

        # Example for a sell signal (emaFastCrossBelowMid - assuming get_ema_signals provides it)
        # if ema_signals.get('emaFastCrossBelowMid', False): # Check if key exists and is True
        #     final_signal['signal'] = 'sell'
        #     final_signal['reason'] = f"Fast EMA ({ema_signals.get('fast_ema', 'N/A'):.2f}) crossed below Mid EMA ({ema_signals.get('mid_ema', 'N/A'):.2f})"

        return final_signal

if __name__ == '__main__':
    # Adjust sys.path for direct execution (if trading_bot is not in PYTHONPATH)
    import os
    import sys
    CLI_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CLI_DIR) # This should be 'trading_bot' directory
    TRADING_BOT_ROOT = os.path.dirname(PROJECT_ROOT) # This should be parent of 'trading_bot' dir
    if TRADING_BOT_ROOT not in sys.path:
        sys.path.insert(0, TRADING_BOT_ROOT)
    # Need to re-import after path adjustment if previous failed
    from trading_bot.utils.indicators import get_ema_signals


    print("--- Example EMACrossStrategy Demonstration (using self.config) ---")

    prices_gc = pd.Series(
        np.linspace(50, 40, 25).tolist() +
        np.linspace(39, 30, 20).tolist() +
        np.linspace(31, 55, 15).tolist()
    )
    sample_ohlcv_data = pd.DataFrame({'close': prices_gc})
    sample_ohlcv_data['open'] = sample_ohlcv_data['close'] - np.random.rand(len(sample_ohlcv_data)) * 0.5
    sample_ohlcv_data['high'] = sample_ohlcv_data['close'] + np.random.rand(len(sample_ohlcv_data)) * 0.5
    sample_ohlcv_data['low'] = sample_ohlcv_data['close'] - np.random.rand(len(sample_ohlcv_data)) * 0.5
    sample_ohlcv_data['volume'] = np.random.randint(100, 1000, size=len(sample_ohlcv_data))

    # Instantiate strategy with default config (from DEFAULT_CONFIG)
    strategy_default = EMACrossStrategy(symbol="BTC/USD")
    print(f"\nTesting with default config: {strategy_default.get_parameters()}")
    signals_default = strategy_default.generate_signals(sample_ohlcv_data)
    print("Generated Signals (Default Config):", signals_default)

    # Instantiate strategy with custom config via constructor
    custom_strategy_config = {'fast_ema_period': 5, 'mid_ema_period': 10, 'slow_ema_period': 15}
    strategy_custom = EMACrossStrategy(symbol="ETH/USD", config=custom_strategy_config)
    print(f"\nTesting with custom config: {strategy_custom.get_parameters()}")

    prices_short_ema_gc = pd.Series(
        np.linspace(50,45,10).tolist() + np.linspace(44,40,10).tolist() + np.linspace(41,50,10).tolist()
    )
    sample_short_ema_data = pd.DataFrame({'close': prices_short_ema_gc})
    signals_custom = strategy_custom.generate_signals(sample_short_ema_data)
    print("Generated Signals (Custom Config):", signals_custom)

    # Test set_parameters to update config
    print("\nTesting set_parameters...")
    strategy_default.set_parameters({'fast_ema_period': 8, 'mid_ema_period': 18, 'slow_ema_period': 28})
    print(f"Config after set_parameters: {strategy_default.get_parameters()}")
    signals_after_set_params = strategy_default.generate_signals(sample_ohlcv_data)
    print("Signals after set_parameters (8, 18, 28):", signals_after_set_params)

    # Test with config that would violate fast < mid < slow, to see fallback/warning
    invalid_config = {'fast_ema_period': 25, 'mid_ema_period': 15, 'slow_ema_period': 30}
    print(f"\nTesting with invalid config for period ordering: {invalid_config}")
    strategy_invalid = EMACrossStrategy(symbol="BAD/CFG", config=invalid_config) # Warning should print
    # The generate_signals should use fallback (10,20,50) for calculation
    signals_invalid = strategy_invalid.generate_signals(sample_ohlcv_data)
    print("Signals (Invalid Config - should use fallback for calc):", signals_invalid)
    assert signals_invalid['details']['params_used'] == {'fast': 10, 'mid': 20, 'slow': 50}

    # Test with insufficient data
    insufficient_data = sample_ohlcv_data.head(5)
    print(f"\nTesting with insufficient data (5 points) for current config of BTC/USD strategy")
    signals_insufficient = strategy_default.generate_signals(insufficient_data) # Uses (8,18,28)
    print("Signals (Insufficient Data):", signals_insufficient)
