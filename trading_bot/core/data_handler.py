class DataHandler:
    def __init__(self, api_manager):
        self.api_manager = api_manager # Expects an instance of APIManager

    def get_realtime_price(self, exchange_name, symbol):
        """Fetches real-time price data for a symbol from an exchange."""
        if exchange_name not in self.api_manager.connections:
            return {"status": "error", "message": f"Not connected to {exchange_name} for real-time data."}

        # This would use the api_manager to get live ticker or market data
        print(f"Fetching real-time price for {symbol} from {exchange_name}...")
        ticker_data = self.api_manager.get_ticker(exchange_name, symbol)

        if ticker_data.get("status") == "error": # Propagate error if get_ticker failed
            return ticker_data

        # Assuming get_ticker returns a dict with 'last_price'
        return {
            "status": "success",
            "symbol": symbol,
            "exchange": exchange_name,
            "price": ticker_data.get("last_price"),
            "timestamp": "YYYY-MM-DD HH:MM:SS" # Placeholder for actual timestamp
        }

    def get_historical_data(self, exchange_name, symbol, timeframe, start_date, end_date):
        """Fetches historical OHLCV data for a symbol from an exchange."""
        if exchange_name not in self.api_manager.connections:
            return {"status": "error", "message": f"Not connected to {exchange_name} for historical data."}

        # Placeholder for fetching historical data
        print(f"Fetching historical data for {symbol} ({timeframe}) from {exchange_name} between {start_date} and {end_date}...")

        # In a real implementation, this would call something like:
        # data = self.api_manager.fetch_historical_ohlcv(exchange_name, symbol, timeframe, start_date, end_date)
        # return {"status": "success", "data": data}

        # Example mock data
        mock_data = [
            {"timestamp": "2023-01-01T00:00:00", "open": 30000, "high": 30100, "low": 29900, "close": 30050, "volume": 100},
            {"timestamp": "2023-01-01T00:05:00", "open": 30050, "high": 30200, "low": 30000, "close": 30150, "volume": 120},
        ]
        return {"status": "success",
                "exchange": exchange_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": mock_data}

if __name__ == '__main__':
    # Example Usage (optional, for testing)
    # This requires a mock or real APIManager instance
    class MockAPIManager:
        def __init__(self):
            self.connections = {}
        def connect(self, exchange_name, key, secret):
            self.connections[exchange_name] = {"status": "connected"}
            print(f"Mock connected to {exchange_name}")
            return {"status": "success", "exchange": exchange_name}

        def get_ticker(self, exchange_name, symbol):
            if exchange_name in self.connections:
                return {
                    "exchange": exchange_name,
                    "symbol": symbol,
                    "last_price": 40000.0, # Mock price
                }
            return {"status": "error", "message": "Not connected"}

    mock_api_manager = MockAPIManager()
    mock_api_manager.connect("Binance", "k", "s")

    data_handler = DataHandler(api_manager=mock_api_manager)

    realtime_price = data_handler.get_realtime_price("Binance", "BTC/USD")
    print(realtime_price)

    historical_data = data_handler.get_historical_data("Binance", "BTC/USD", "5m", "2023-01-01", "2023-01-02")
    print(historical_data)

    realtime_price_disconnected = data_handler.get_realtime_price("Kraken", "ETH/USD") # Not connected
    print(realtime_price_disconnected)
