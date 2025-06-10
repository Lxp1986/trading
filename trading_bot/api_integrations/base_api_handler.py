# trading_bot/api_integrations/base_api_handler.py

from abc import ABC, abstractmethod
import pandas as pd

class BaseAPIHandler(ABC):
    """
    Abstract base class for API handlers to interact with cryptocurrency exchanges.
    """
    def __init__(self, api_key: str = None, api_secret: str = None, config: dict = None):
        """
        Initializes the BaseAPIHandler.

        Args:
            api_key (str, optional): API key for the exchange.
            api_secret (str, optional): API secret for the exchange.
            config (dict, optional): Additional configuration parameters for the API handler
                                     (e.g., base_url, specific exchange settings).
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.config = config if config is not None else {}

    @abstractmethod
    def get_balance(self) -> dict:
        """
        Retrieves the account balance.

        Returns:
            dict: A dictionary of asset balances (e.g., {'BTC': 1.5, 'USD': 10000}).
        """
        pass

    @abstractmethod
    def get_ticker(self, symbol: str) -> dict:
        """
        Retrieves the current market ticker for a specific symbol.

        Args:
            symbol (str): The trading symbol (e.g., 'BTC/USD').

        Returns:
            dict: Current market data (e.g., {'symbol': 'BTC/USD', 'bid': 40000, 'ask': 40005, 'last_price': 40002}).
        """
        pass

    @abstractmethod
    def get_historical_data(self,
                            symbol: str,
                            timeframe: str,
                            start_date_str: str = None, # Using string for flexibility with API formats
                            end_date_str: str = None,
                            limit: int = 100) -> pd.DataFrame:
        """
        Retrieves historical OHLCV data.

        Args:
            symbol (str): The trading symbol (e.g., 'BTC/USD').
            timeframe (str): The timeframe for the data (e.g., '1m', '5m', '1h', '1d').
            start_date_str (str, optional): Start date string (format depends on API).
            end_date_str (str, optional): End date string (format depends on API).
            limit (int, optional): Number of data points to retrieve.

        Returns:
            pd.DataFrame: OHLCV data with a DatetimeIndex.
                          Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                          Timestamp should be timezone-aware (UTC) if possible.
        """
        pass

    @abstractmethod
    def place_order(self,
                    symbol: str,
                    order_type: str, # e.g., 'market', 'limit'
                    side: str,       # 'buy' or 'sell'
                    amount: float,   # Amount of base asset to buy/sell
                    price: float = None, # Required for limit orders
                    params: dict = None  # Extra parameters for the exchange
                   ) -> dict:
        """
        Places a trading order.

        Args:
            symbol (str): The trading symbol.
            order_type (str): Type of order ('market', 'limit', etc.).
            side (str): 'buy' or 'sell'.
            amount (float): Quantity of the asset to trade.
            price (float, optional): Price for limit orders.
            params (dict, optional): Additional exchange-specific parameters.

        Returns:
            dict: Confirmation of the order (e.g., {'id': '123', 'status': 'open', 'symbol': symbol, ...}).
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str = None) -> dict:
        """
        Cancels an open order.

        Args:
            order_id (str): The ID of the order to cancel.
            symbol (str, optional): The trading symbol (required by some exchanges).

        Returns:
            dict: Confirmation of the cancellation.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str, symbol: str = None) -> dict:
        """
        Retrieves the status of a specific order.

        Args:
            order_id (str): The ID of the order.
            symbol (str, optional): The trading symbol.

        Returns:
            dict: Status of the order (e.g., {'id': '123', 'status': 'filled', 'filled_amount': 1.0, ...}).
        """
        pass

    @abstractmethod
    def get_open_orders(self, symbol: str = None) -> list:
        """
        Retrieves a list of all open orders.

        Args:
            symbol (str, optional): Filter by symbol. If None, return for all symbols if supported.

        Returns:
            list: A list of order dictionaries.
        """
        pass

if __name__ == '__main__':
    # This is an abstract class and cannot be instantiated.
    # Example of a concrete class (simplified):
    class MyDummyAPIHandler(BaseAPIHandler):
        def get_balance(self) -> dict: return {'USD': 1000}
        def get_ticker(self, symbol: str) -> dict: return {'symbol': symbol, 'last_price': 50000}
        def get_historical_data(self, symbol: str, timeframe: str, start_date_str=None, end_date_str=None, limit: int=100) -> pd.DataFrame: return pd.DataFrame()
        def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float=None, params: dict=None) -> dict: return {'id': 'dummy_order_1', 'status': 'open'}
        def cancel_order(self, order_id: str, symbol: str=None) -> dict: return {'id': order_id, 'status': 'cancelled'}
        def get_order_status(self, order_id: str, symbol: str=None) -> dict: return {'id': order_id, 'status': 'open'}
        def get_open_orders(self, symbol: str=None) -> list: return []

    # dummy_handler = MyDummyAPIHandler() # This would work if all methods are implemented
    # print(dummy_handler.get_balance())
    print("BaseAPIHandler defined. Concrete implementations are needed.")
