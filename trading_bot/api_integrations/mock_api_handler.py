# trading_bot/api_integrations/mock_api_handler.py

import pandas as pd
import numpy as np
import datetime
import time
import uuid

from .base_api_handler import BaseAPIHandler

class MockAPIHandler(BaseAPIHandler):
    """
    Mock API Handler for testing and development purposes.
    Simulates interactions with a cryptocurrency exchange.
    """
    def __init__(self, api_key: str = "mock_key", api_secret: str = "mock_secret", config: dict = None):
        super().__init__(api_key, api_secret, config if config is not None else {})
        self._initialize_mock_state()
        # print("MockAPIHandler initialized.") # Keep it less verbose for library use

    def _initialize_mock_state(self):
        """Initializes or resets the mock state of the API handler."""
        self.mock_balances = self.config.get('initial_balances', {'USD': 100000.0, 'BTC': 5.0, 'ETH': 100.0})
        self.mock_open_orders = {} # Stores open orders by id
        self.mock_order_history = {} # Stores all orders by id
        # self.order_id_counter = 1 # Replaced by uuid
        self.current_sim_time = pd.to_datetime(self.config.get('start_sim_time', '2023-01-01T00:00:00Z'))

        self.mock_market_prices = {
            'BTC/USD': self.config.get('initial_btc_price', 40000.0),
            'ETH/USD': self.config.get('initial_eth_price', 2500.0),
            'ADA/USD': self.config.get('initial_ada_price', 0.5)
        }

    def _get_next_order_id(self) -> str:
        return str(uuid.uuid4())

    def _update_market_price(self, symbol: str):
        if symbol in self.mock_market_prices:
            change = np.random.uniform(-0.005, 0.005)
            self.mock_market_prices[symbol] *= (1 + change)
            self.mock_market_prices[symbol] = round(self.mock_market_prices[symbol], 4) # Keep prices reasonable
        else: # Auto-initialize new symbols encountered
             self.mock_market_prices[symbol] = round(np.random.uniform(1,1000), 4)


    def get_balance(self) -> dict:
        # print(f"MockAPI: Fetching balances. Current: {self.mock_balances}")
        return self.mock_balances.copy()

    def get_ticker(self, symbol: str) -> dict:
        self._update_market_price(symbol)

        last_price = self.mock_market_prices.get(symbol)
        if last_price is None: # Should be initialized by _update_market_price if new
            raise ValueError(f"Symbol {symbol} has no mock price defined and was not auto-initialized.")

        bid = round(last_price * 0.999, 4)
        ask = round(last_price * 1.001, 4)

        ticker_data = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
            'bid': bid, 'ask': ask, 'last_price': last_price,
            'volume': round(np.random.uniform(1, 1000), 4)
        }
        # print(f"MockAPI: Ticker for {symbol}: Last Price {last_price:.2f}")
        return ticker_data

    def get_historical_data(self,
                            symbol: str,
                            timeframe: str,
                            start_date_str: str = None,
                            end_date_str: str = None,
                            limit: int = 100) -> pd.DataFrame:
        # print(f"MockAPI: Generating historical data for {symbol}, timeframe {timeframe}, limit {limit}")
        end_dt = pd.to_datetime(end_date_str, utc=True) if end_date_str else pd.Timestamp.now(tz='UTC')

        # Updated freq_map to use modern pandas offset aliases
        freq_map = {
            '1m': 'min', '5m': '5min', '15m': '15min',
            '1h': 'H', '4h': '4H', '1d': 'D'
        }
        freq = freq_map.get(timeframe, 'D')

        if start_date_str:
            start_dt = pd.to_datetime(start_date_str, utc=True)
            index = pd.date_range(start=start_dt, end=end_dt, freq=freq, name='timestamp')
            if not index.empty and len(index) > limit: index = index[-limit:]
        else:
            index = pd.date_range(end=end_dt, periods=limit, freq=freq, name='timestamp')

        if index.empty: return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).set_index('timestamp')

        data_len = len(index)
        current_price_base = self.mock_market_prices.get(symbol, np.random.uniform(100,50000))

        price_path = np.random.randn(data_len).cumsum() * (current_price_base * 0.01) # Price steps related to current price
        prices = current_price_base + price_path - price_path[0] # Start around current_price_base
        prices = np.maximum(prices, 0.01)

        df = pd.DataFrame(index=index)
        df['open'] = np.round(prices - np.random.uniform(0, 0.005, data_len) * prices, 4)
        df['close'] = np.round(prices, 4)
        df['low'] = np.round(df[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.01, data_len) * prices, 4)
        df['high'] = np.round(df[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.01, data_len) * prices, 4)
        df['volume'] = np.round(np.random.uniform(1, 100, data_len) * np.random.randint(1, 10, data_len), 4)

        df.low = np.minimum(df.low, df.open)
        df.low = np.minimum(df.low, df.close)
        df.high = np.maximum(df.high, df.open)
        df.high = np.maximum(df.high, df.close)
        df.clip(lower=0.0001, inplace=True) # Ensure no negative prices/volumes for crypto

        return df

    def place_order(self,
                    symbol: str, order_type: str, side: str,
                    amount: float, price: float = None, params: dict = None) -> dict:
        if amount <= 0: raise ValueError("Order amount must be positive.")
        order_id = self._get_next_order_id()

        # Ensure symbol exists in mock prices, if not add it for simulation
        if symbol not in self.mock_market_prices: self._update_market_price(symbol)

        current_market_price = self.mock_market_prices.get(symbol)

        order_info = {
            'id': order_id, 'symbol': symbol, 'type': order_type, 'side': side,
            'amount': amount, 'price': price if order_type == 'limit' else current_market_price,
            'timestamp': pd.Timestamp.now(tz='UTC').isoformat(), 'status': 'open',
            'filled_amount': 0.0, 'average_fill_price': 0.0, 'remaining': amount
        }

        if order_type == 'market':
            fill_price = current_market_price
            order_info.update({'price': fill_price, 'status': 'filled', 'filled_amount': amount,
                               'average_fill_price': fill_price, 'remaining': 0.0})

            base_asset, quote_asset = symbol.split('/')
            cost_or_proceeds = fill_price * amount
            commission = cost_or_proceeds * self.config.get('commission_rate', 0.001)

            if side == 'buy':
                required_quote = cost_or_proceeds + commission
                if self.mock_balances.get(quote_asset, 0) >= required_quote:
                    self.mock_balances[quote_asset] -= required_quote
                    self.mock_balances[base_asset] = self.mock_balances.get(base_asset, 0) + amount
                    # print(f"MockAPI: BUY market order {order_id} for {amount} {base_asset} @ {fill_price:.2f} filled.")
                else: order_info.update({'status':'rejected', 'reason':'Insufficient balance'})
            elif side == 'sell':
                if self.mock_balances.get(base_asset, 0) >= amount:
                    self.mock_balances[base_asset] -= amount
                    self.mock_balances[quote_asset] = self.mock_balances.get(quote_asset, 0) + (cost_or_proceeds - commission)
                    # print(f"MockAPI: SELL market order {order_id} for {amount} {base_asset} @ {fill_price:.2f} filled.")
                else: order_info.update({'status':'rejected', 'reason':f'Insufficient {base_asset} balance'})

        elif order_type == 'limit':
            # print(f"MockAPI: {side.upper()} limit order {order_id} for {amount} {symbol} @ {price:.2f} placed.")
            # Limit orders are initially open and are added to mock_open_orders
            if order_info['status'] == 'open': # Default status for new limit orders
                 self.mock_open_orders[order_id] = order_info
        # else market order:
            # If market order was filled or rejected, it's terminal and not "open".
            # So, it should not be in mock_open_orders.
            # The initial if condition `order_info['status'] not in ['rejected']` was too broad.
            # It allowed filled market orders into mock_open_orders.

        # All orders, regardless of initial status, go to history
        self.mock_order_history[order_id] = order_info

        # Explicitly remove market orders from mock_open_orders if they were added and are terminal
        # This check is a bit redundant if logic above is correct (market orders not added to open_orders)
        # but acts as a safeguard.
        if order_type == 'market' and order_info['status'] in ['filled', 'rejected']:
            if order_id in self.mock_open_orders: # Should not happen if not added initially
                self.mock_open_orders.pop(order_id, None)

        return order_info

    def cancel_order(self, order_id: str, symbol: str = None) -> dict:
        if order_id in self.mock_open_orders:
            order = self.mock_open_orders.pop(order_id)
            order.update({'status': 'cancelled', 'remaining': 0.0})
            self.mock_order_history[order_id] = order
            # print(f"MockAPI: Order {order_id} cancelled.")
            return {'id': order_id, 'status': 'cancelled', 'message': 'Order cancelled successfully.'}
        elif order_id in self.mock_order_history:
             return {'id': order_id, 'status': self.mock_order_history[order_id]['status'], 'message': 'Order not open or already processed.'}
        return {'id': order_id, 'status': 'error', 'message': 'Order not found.'}

    def get_order_status(self, order_id: str, symbol: str = None) -> dict:
        # Simulate potential fill of limit orders over time if ticker moves favorably
        if order_id in self.mock_open_orders:
            order = self.mock_open_orders[order_id]
            if order['type'] == 'limit' and order['status'] == 'open':
                current_price = self.mock_market_prices.get(order['symbol'])
                if current_price:
                    if order['side'] == 'buy' and current_price <= order['price']: # Buy limit triggered
                        order.update({'status': 'filled', 'filled_amount': order['amount'],
                                      'average_fill_price': order['price'], 'remaining': 0.0})
                        # Simulate balance update for filled limit buy
                        base, quote = order['symbol'].split('/')
                        cost = order['price'] * order['amount']
                        comm = cost * self.config.get('commission_rate', 0.001)
                        self.mock_balances[quote] -= (cost + comm)
                        self.mock_balances[base] = self.mock_balances.get(base,0) + order['amount']
                        # print(f"MockAPI: Buy Limit order {order_id} filled.")
                        self.mock_open_orders.pop(order_id) # remove from open
                        self.mock_order_history[order_id] = order
                    elif order['side'] == 'sell' and current_price >= order['price']: # Sell limit triggered
                        order.update({'status': 'filled', 'filled_amount': order['amount'],
                                      'average_fill_price': order['price'], 'remaining': 0.0})
                        # Simulate balance update for filled limit sell
                        base, quote = order['symbol'].split('/')
                        proceeds = order['price'] * order['amount']
                        comm = proceeds * self.config.get('commission_rate', 0.001)
                        self.mock_balances[base] -= order['amount']
                        self.mock_balances[quote] = self.mock_balances.get(quote,0) + (proceeds - comm)
                        # print(f"MockAPI: Sell Limit order {order_id} filled.")
                        self.mock_open_orders.pop(order_id) # remove from open
                        self.mock_order_history[order_id] = order
            return order
        elif order_id in self.mock_order_history:
            return self.mock_order_history[order_id]
        return {'id': order_id, 'status': 'not_found'}

    def get_open_orders(self, symbol: str = None) -> list:
        # Also check if any open limit orders should now be "filled" based on current price
        # This is a simplified check; real exchanges have complex matching.
        orders_to_remove = []
        for order_id, order in list(self.mock_open_orders.items()): # Iterate on list copy
             if order['type'] == 'limit' and order['status'] == 'open':
                # Call get_order_status which has fill logic for limits
                updated_order = self.get_order_status(order_id, order['symbol'])
                # If it got filled by get_order_status, it would be removed from mock_open_orders there.

        open_orders_list = list(self.mock_open_orders.values())
        if symbol:
            open_orders_list = [o for o in open_orders_list if o['symbol'] == symbol]
        # print(f"MockAPI: Found {len(open_orders_list)} open orders for symbol {symbol if symbol else 'all'}.")
        return open_orders_list

if __name__ == '__main__':
    print("--- MockAPIHandler Demonstration ---")
    handler_config = {
        'initial_balances': {'USD': 50000.0, 'BTC': 2.0},
        'commission_rate': 0.001 # 0.1%
    }
    mock_api = MockAPIHandler(config=handler_config)

    print("\n1. Initial Balances:")
    print(mock_api.get_balance())

    print("\n2. Get Ticker BTC/USD:")
    print(mock_api.get_ticker('BTC/USD'))
    print("\n3. Get Ticker ETH/USD (new symbol):")
    print(mock_api.get_ticker('ETH/USD'))

    print("\n4. Get Historical Data for BTC/USD (5m, 10 periods):")
    hist_data = mock_api.get_historical_data('BTC/USD', '5m', limit=10)
    print(hist_data.head())

    print("\n5. Placing Orders:")
    # Market Buy BTC
    print("   Placing Market Buy BTC/USD (0.1 BTC)...")
    buy_market_order = mock_api.place_order('BTC/USD', 'market', 'buy', 0.1)
    print("   Order Response:", buy_market_order)
    print("   Balances after market buy:", mock_api.get_balance())

    # Limit Sell BTC
    btc_ticker = mock_api.get_ticker('BTC/USD') # Get current price
    limit_sell_price = round(btc_ticker['last_price'] * 1.05, 2) # 5% above current
    print(f"\n   Placing Limit Sell BTC/USD (0.05 BTC @ {limit_sell_price})...")
    sell_limit_order = mock_api.place_order('BTC/USD', 'limit', 'sell', 0.05, price=limit_sell_price)
    print("   Order Response:", sell_limit_order)

    # Limit Buy ETH
    eth_ticker = mock_api.get_ticker('ETH/USD')
    limit_buy_price = round(eth_ticker['last_price'] * 0.95, 2) # 5% below current
    print(f"\n   Placing Limit Buy ETH/USD (1 ETH @ {limit_buy_price})...")
    buy_limit_order_eth = mock_api.place_order('ETH/USD', 'limit', 'buy', 1.0, price=limit_buy_price)
    print("   Order Response:", buy_limit_order_eth)

    print("\n6. Open Orders (before any limit fills):")
    open_orders = mock_api.get_open_orders()
    for order in open_orders: print(f"   - {order['id']}: {order['symbol']} {order['type']} {order['side']} {order['amount']} @ {order['price']} (Status: {order['status']})")
    if not open_orders: print("   No open orders.")

    print("\n7. Simulate Price Movement & Check Order Status for ETH Buy Limit:")
    # Simulate ETH price dropping to fill the buy limit order
    mock_api.mock_market_prices['ETH/USD'] = limit_buy_price - 10 # Price drops below limit buy
    print(f"   Simulated ETH/USD price drop to: {mock_api.mock_market_prices['ETH/USD']:.2f}")
    status_eth_limit = mock_api.get_order_status(buy_limit_order_eth['id'])
    print("   ETH Buy Limit Order Status after price drop:", status_eth_limit)
    print("   Balances after ETH limit buy potentially filled:", mock_api.get_balance())

    print("\n8. Open Orders (after ETH limit fill attempt):")
    open_orders_after_eth = mock_api.get_open_orders()
    for order in open_orders_after_eth: print(f"   - {order['id']}: {order['symbol']} {order['type']} {order['side']} {order['amount']} @ {order['price']} (Status: {order['status']})")
    if not open_orders_after_eth: print("   No open orders.")

    print("\n9. Cancel BTC Sell Limit Order:")
    cancel_response = mock_api.cancel_order(sell_limit_order['id'])
    print("   Cancel Response:", cancel_response)
    status_after_cancel = mock_api.get_order_status(sell_limit_order['id'])
    print("   BTC Sell Limit Order Status after cancel attempt:", status_after_cancel)

    print("\n10. Final Balances:")
    print(mock_api.get_balance())

    print("\n11. Get Open Orders (final check):")
    final_open_orders = mock_api.get_open_orders()
    if not final_open_orders: print("   No open orders remaining.")
    for order in final_open_orders: print(f"   - {order['id']} status: {order['status']}")

    print("\n--- MockAPIHandler Demonstration Finished ---")
