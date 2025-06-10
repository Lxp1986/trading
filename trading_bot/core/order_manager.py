class OrderManager:
    def __init__(self, api_manager):
        self.api_manager = api_manager # Expects an instance of APIManager
        self.active_orders = {}
        self.order_id_counter = 1

    def place_order(self, exchange_name, symbol, type, side, amount, price=None):
        """Places an order on the specified exchange."""
        if exchange_name not in self.api_manager.connections:
            return {"status": "error", "message": f"Not connected to {exchange_name} for placing order."}

        # Placeholder for order placement logic
        print(f"Placing {side} {type} order for {amount} of {symbol} at {price if price else 'market'} on {exchange_name}...")

        order_id = f"order_{self.order_id_counter}"
        self.order_id_counter += 1

        order_details = {
            "order_id": order_id,
            "exchange": exchange_name,
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "open" # Initial status
        }
        self.active_orders[order_id] = order_details

        # In a real scenario, this would interact with the exchange via api_manager
        # success = self.api_manager.submit_order(exchange_name, order_details)
        # if success:
        #     return {"status": "success", "order": order_details}
        # else:
        #     del self.active_orders[order_id]
        #     return {"status": "error", "message": "Failed to place order with exchange."}

        return {"status": "success", "order": order_details}

    def cancel_order(self, order_id):
        """Cancels an active order."""
        if order_id not in self.active_orders:
            return {"status": "error", "message": f"Order {order_id} not found."}

        # Placeholder for order cancellation logic
        print(f"Cancelling order {order_id}...")

        # In a real scenario, this would interact with the exchange via api_manager
        # success = self.api_manager.cancel_exchange_order(self.active_orders[order_id]['exchange'], order_id)
        # if success:
        #     self.active_orders[order_id]["status"] = "cancelled"
        #     return {"status": "success", "message": f"Order {order_id} cancelled."}
        # else:
        #     return {"status": "error", "message": f"Failed to cancel order {order_id} with exchange."}

        self.active_orders[order_id]["status"] = "cancelled"
        return {"status": "success", "message": f"Order {order_id} cancelled."}

    def get_order_status(self, order_id):
        """Retrieves the status of a specific order."""
        if order_id in self.active_orders:
            return {"status": "success", "order": self.active_orders[order_id]}
        else:
            # Potentially check with the exchange if not in active_orders (e.g., filled, partially filled)
            # status_from_exchange = self.api_manager.fetch_order_status(some_exchange_ref, order_id)
            return {"status": "error", "message": f"Order {order_id} not found in active list."}

if __name__ == '__main__':
    # Example Usage (optional, for testing)
    # This requires a mock or real APIManager instance
    class MockAPIManager:
        def __init__(self):
            self.connections = {}
        def connect(self, exchange_name, key, secret):
            self.connections[exchange_name] = True
            print(f"Mock connected to {exchange_name}")

    mock_api_manager = MockAPIManager()
    mock_api_manager.connect("Binance", "k", "s")

    order_manager = OrderManager(api_manager=mock_api_manager)

    order1 = order_manager.place_order("Binance", "BTC/USD", "limit", "buy", 0.1, 39000)
    print(order1)
    if order1['status'] == 'success':
        order_id1 = order1['order']['order_id']
        print(order_manager.get_order_status(order_id1))
        print(order_manager.cancel_order(order_id1))
        print(order_manager.get_order_status(order_id1))

    order2 = order_manager.place_order("Kraken", "ETH/USD", "market", "sell", 2.0) # Not connected
    print(order2)
    print(order_manager.get_order_status("non_existent_order"))
