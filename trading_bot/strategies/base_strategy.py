# trading_bot/strategies/base_strategy.py

from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    def __init__(self, strategy_name: str, symbol: str, config: dict = None):
        """
        Initializes the BaseStrategy.

        Args:
            strategy_name (str): Name of the strategy.
            symbol (str): Trading symbol (e.g., 'BTC/USD').
            config (dict, optional): Strategy-specific configuration parameters.
                                     These parameters will be used by the strategy's logic.
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        # Store the configuration. If None, initialize as empty dict.
        # Concrete strategies can define default configs and merge them with provided config.
        self.config = config if config is not None else {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> dict:
        """
        Generates trading signals based on the input data.

        This method must be implemented by concrete strategy classes.

        Args:
            data (pd.DataFrame): A DataFrame containing OHLCV data and potentially
                                 pre-calculated indicators. It's assumed that the
                                 DataFrame is indexed by timestamp and contains
                                 at least 'close' prices.

        Returns:
            dict: A dictionary containing trading signals.
                  Example: {'signal': 'buy'/'sell'/'hold',
                            'stop_loss': price,
                            'take_profit': price,
                            'confidence': 0.0-1.0,
                            'details': {<indicator_values>}}
        """
        pass

    def set_parameters(self, params_dict: dict):
        """
        Sets or updates strategy-specific parameters in the config.

        Args:
            params_dict (dict): A dictionary of parameters to set or update.
        """
        if not isinstance(params_dict, dict):
            raise ValueError("Parameters must be provided as a dictionary.")

        self.config.update(params_dict)
        print(f"Config updated for strategy {self.strategy_name} on {self.symbol}: {params_dict}")

    def get_parameters(self) -> dict:
        """
        Retrieves the current strategy configuration parameters.

        Returns:
            dict: The current config of the strategy.
        """
        return self.config.copy() # Return a copy to prevent direct modification

    def get_parameter(self, param_name: str, default_value=None):
        """
        Retrieves a specific parameter's value from the config.

        Args:
            param_name (str): The name of the parameter to retrieve.
            default_value (optional): The value to return if the parameter is not found in config.

        Returns:
            The parameter's value or the default_value.
        """
        return self.config.get(param_name, default_value)

if __name__ == '__main__':
    # This class is abstract and cannot be instantiated directly.
    # Example of how a concrete class might use it:

    class MyConfigurableStrategy(BaseStrategy):
        DEFAULT_CONFIG = {'lookback': 10, 'threshold_pct': 0.5, 'min_volume': 100}

        def __init__(self, symbol: str, config: dict = None):
            # Merge user-provided config with defaults
            merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
            super().__init__("ConfigurableStrategy", symbol, merged_config)
            print(f"{self.strategy_name} initialized for {self.symbol} with effective config: {self.config}")

        def generate_signals(self, data: pd.DataFrame) -> dict:
            print(f"Generating signals for {self.symbol} using {self.strategy_name} with config: {self.config}...")
            if data.empty:
                return {'signal': 'hold', 'reason': 'No data provided'}

            latest_close = data['close'].iloc[-1]
            lookback = self.get_parameter('lookback') # No default needed if using merged_config

            if len(data) < lookback:
                return {'signal': 'hold', 'reason': f'Not enough data for lookback {lookback}'}

            avg_price = data['close'].tail(lookback).mean()
            threshold_pct = self.get_parameter('threshold_pct')
            min_volume_check = self.get_parameter('min_volume', 0) # Example of a default if not in default_config

            if 'volume' in data and data['volume'].iloc[-1] < min_volume_check:
                 return {'signal': 'hold', 'reason': f'Volume {data["volume"].iloc[-1]} below min {min_volume_check}'}


            if latest_close > avg_price * (1 + threshold_pct / 100):
                return {'signal': 'buy', 'price': latest_close, 'avg_price': avg_price}
            elif latest_close < avg_price * (1 - threshold_pct / 100):
                return {'signal': 'sell', 'price': latest_close, 'avg_price': avg_price}
            else:
                return {'signal': 'hold', 'avg_price': avg_price}

    # Example usage:
    dummy_ohlcv = pd.DataFrame({
        'close': [100, 101, 102, 100, 99, 103, 105, 106, 104, 107, 110],
        'volume': [150, 120, 130, 160, 140, 100, 110, 180, 190, 105, 200]
    })

    # Initialize with default config
    strategy_default_cfg = MyConfigurableStrategy(symbol="BTC/USD")
    signals_default = strategy_default_cfg.generate_signals(dummy_ohlcv)
    print("Signals (Default Config):", signals_default)

    # Initialize with custom config overriding some defaults
    custom_user_config = {'lookback': 7, 'threshold_pct': 0.8, 'new_param': 'test'}
    strategy_custom_cfg = MyConfigurableStrategy(symbol="ETH/USD", config=custom_user_config)
    signals_custom = strategy_custom_cfg.generate_signals(dummy_ohlcv)
    print("Signals (Custom Config):", signals_custom)
    print(f"Strategy ETH/USD full config: {strategy_custom_cfg.get_parameters()}")


    # Update parameters post-initialization
    strategy_default_cfg.set_parameters({'lookback': 3, 'min_volume': 150})
    signals_updated = strategy_default_cfg.generate_signals(dummy_ohlcv)
    print("Signals (BTC/USD, after set_parameters):", signals_updated)
    print(f"BTC/USD 'min_volume' parameter: {strategy_default_cfg.get_parameter('min_volume')}")
    print(f"BTC/USD 'threshold_pct' (should be default): {strategy_default_cfg.get_parameter('threshold_pct')}")
    print(f"BTC/USD full config: {strategy_default_cfg.get_parameters()}")
