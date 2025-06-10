# trading_bot/cli.py

import argparse
import pandas as pd
import json
import os
import sys
import numpy as np # Added for np.nan

# Adjust Python path to include the project root 'trading_bot'
# This allows finding modules like core, strategies, etc. when cli.py is in trading_bot/
# and run as `python trading_bot/cli.py` or if `trading_bot` itself is not directly in PYTHONPATH.
# For `python -m trading_bot.cli`, this might be less critical but good for robustness.
# Assuming the script is in /app/trading_bot/cli.py and /app/ is the project root for imports.
# For the sandbox, /app is the root. So trading_bot.core should work.

try:
    from trading_bot.core.engine import TradingBotEngine # Placeholder for future 'trade' command
    from trading_bot.strategies import EMACrossStrategy, AdvancedExampleStrategy, BaseStrategy
    from trading_bot.backtesting.backtester import Backtester
    from trading_bot.api_integrations.mock_api_handler import MockAPIHandler # For 'trade --mode mock'
except ImportError as e:
    # This block can help if running `python cli.py` directly from within `trading_bot` dir
    # Or if the module structure is not perfectly picked up.
    # For `python -m trading_bot.cli` this should ideally not be needed if PYTHONPATH is setup by -m.
    print(f"Initial ImportError: {e}. Attempting to adjust sys.path for local execution.", file=sys.stderr)
    # Get the directory of the cli.py script
    CLI_DIR = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (which should be the project root, e.g., /app for trading_bot)
    PROJECT_ROOT = os.path.dirname(CLI_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        print(f"Added {PROJECT_ROOT} to sys.path", file=sys.stderr)

    # Retry imports
    from trading_bot.core.engine import TradingBotEngine
    from trading_bot.strategies import EMACrossStrategy, AdvancedExampleStrategy, BaseStrategy
    from trading_bot.backtesting.backtester import Backtester
    from trading_bot.api_integrations.mock_api_handler import MockAPIHandler


# --- Strategy Mapping ---
# Allows selecting strategy class by string name from CLI
STRATEGY_MAP = {
    'EMACrossStrategy': EMACrossStrategy,
    'AdvancedExampleStrategy': AdvancedExampleStrategy,
    # Add other strategies here as they are created
}


def handle_backtest(args):
    print("Handling 'backtest' command...")
    print(f"Strategy: {args.strategy}")
    print(f"Data file: {args.data}")
    print(f"Initial balance: {args.initial_balance}")
    print(f"Commission rate: {args.commission}")
    print(f"Trade size: {args.trade_size}")
    print(f"Symbol: {args.symbol}")
    if args.strategy_config:
        print(f"Strategy Config: {args.strategy_config}")

    # 1. Load Data
    try:
        # Assuming data path is relative to where CLI is run, or absolute.
        data_df = pd.read_csv(
            args.data,
            comment='#', # To ignore comment lines if any were accidentally left
            parse_dates=['timestamp'], # Try to parse 'timestamp' column
            index_col='timestamp'      # Set 'timestamp' as index
        )

        # Explicitly convert index to DatetimeIndex if not already, and handle errors
        if not isinstance(data_df.index, pd.DatetimeIndex):
            print("Warning: Index was not automatically parsed as DatetimeIndex. Attempting manual conversion.", file=sys.stderr)
            try:
                data_df.index = pd.to_datetime(data_df.index, errors='raise')
            except Exception as e_idx:
                print(f"Error: Could not convert index to DatetimeIndex: {e_idx}", file=sys.stderr)
                print("Please ensure the 'timestamp' column is in a recognizable date format.", file=sys.stderr)
                return

        # Ensure required OHLCV columns exist and attempt to convert to numeric
        required_data_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_data_cols if col not in data_df.columns]
        if missing_cols:
            print(f"Error: Data CSV missing required columns: {', '.join(missing_cols)}", file=sys.stderr)
            return

        for col in required_data_cols:
            if data_df[col].dtype == 'object':
                print(f"Warning: Column '{col}' is of object type. Attempting conversion to numeric.", file=sys.stderr)
            # errors='coerce' will turn unparseable values into NaT/NaN
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

        # Drop rows with NaN in essential columns that might have resulted from coercion
        data_df.dropna(subset=required_data_cols, inplace=True)

        if data_df.empty:
            print("Error: Data became empty after cleaning (NaN removal or parsing issues). Check CSV format and content.", file=sys.stderr)
            return

        print(f"Data loaded and processed: {len(data_df)} rows.")
        # ---- DEBUG PRINT ----
        print("--- DataFrame Info (after processing) ---", file=sys.stderr)
        data_df.info(buf=sys.stderr)
        print(f"Index type: {type(data_df.index)}", file=sys.stderr)
        print("--- End DataFrame Info ---", file=sys.stderr)
        # ---- END DEBUG PRINT ----
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return

    # 2. Parse Strategy Config (if any)
    strategy_custom_config = {}
    if args.strategy_config:
        try:
            # Try parsing as JSON string first
            strategy_custom_config = json.loads(args.strategy_config)
        except json.JSONDecodeError:
            # Try parsing as file path
            try:
                with open(args.strategy_config, 'r') as f:
                    strategy_custom_config = json.load(f)
            except FileNotFoundError:
                print(f"Error: Strategy config file not found at {args.strategy_config}", file=sys.stderr)
                return
            except json.JSONDecodeError as e_file:
                print(f"Error: Invalid JSON in strategy config file {args.strategy_config}: {e_file}", file=sys.stderr)
                return
            except Exception as e_file_other:
                print(f"Error reading strategy config file {args.strategy_config}: {e_file_other}", file=sys.stderr)
                return
        print(f"Using custom strategy config: {strategy_custom_config}")


    # 3. Instantiate Strategy
    StrategyClass = STRATEGY_MAP.get(args.strategy)
    if not StrategyClass:
        print(f"Error: Strategy '{args.strategy}' not found. Available: {list(STRATEGY_MAP.keys())}", file=sys.stderr)
        return

    try:
        # Pass symbol and custom config to the strategy constructor
        strategy_instance = StrategyClass(symbol=args.symbol, config=strategy_custom_config)
    except Exception as e:
        print(f"Error instantiating strategy '{args.strategy}': {e}", file=sys.stderr)
        return

    # 4. Instantiate Backtester
    try:
        backtester = Backtester(
            strategy=strategy_instance,
            historical_data=data_df,
            initial_balance=args.initial_balance,
            commission_rate=args.commission,
            trade_size=args.trade_size
        )
    except Exception as e:
        print(f"Error instantiating Backtester: {e}", file=sys.stderr)
        return

    # 5. Run Backtest
    try:
        results = backtester.run()
    except Exception as e:
        print(f"Error during backtest execution: {e}", file=sys.stderr)
        # Potentially print more debug info here if needed
        import traceback
        traceback.print_exc()
        return

    # 6. Print Results
    print("\n--- Backtest Performance ---")
    if results and results.get("performance"):
        for key, value in results["performance"].items():
            if isinstance(value, float) and value is not np.nan : # check for NaN too
                print(f"  {key.replace('_', ' ').capitalize()}: {value:.2f}")
            else:
                print(f"  {key.replace('_', ' ').capitalize()}: {value}")

    print("\n--- Trades Log ---")
    if results and results.get("trades"):
        if not results["trades"]:
            print("  No trades were executed.")
        for i, trade in enumerate(results["trades"]):
            print(f"  Trade {i+1}: Timestamp: {trade['timestamp']}, Type: {trade['type']}, Price: {trade['price']:.2f}, Qty: {trade['quantity']}, P&L: {trade.get('pnl', 0):.2f}, Comm: {trade['commission']:.2f}")

    # Optional: Save equity curve or plot if matplotlib was imported and enabled
    # equity_df = results.get("equity_curve")
    # if equity_df is not None:
    #     equity_df.to_csv("equity_curve.csv")
    #     print("\nEquity curve saved to equity_curve.csv")


def handle_trade(args):
    print("Handling 'trade' command...")
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}")
    print(f"Mode: {args.mode}")
    if args.strategy_config:
        print(f"Strategy Config: {args.strategy_config}")

    if args.mode == 'mock':
        print("Mock trading mode selected.")
        # Placeholder: Instantiate MockAPIHandler, strategy, and potentially a simplified TradingBotEngine
        # api_handler = MockAPIHandler()
        # StrategyClass = STRATEGY_MAP.get(args.strategy)
        # if not StrategyClass: ... error ...
        # strategy_instance = StrategyClass(symbol=args.symbol, config=parsed_config)
        # engine = TradingBotEngine(strategy=strategy_instance, api_handler=api_handler, ...)
        # engine.start() / engine.run_once() etc.
        print("Mock trading functionality not fully implemented yet.")
    elif args.mode == 'live':
        print("Live trading mode selected. THIS IS NOT IMPLEMENTED AND WOULD INTERACT WITH REAL EXCHANGES.")
        print("Ensure you have correctly configured API keys and understand the risks.")
        print("Live trading functionality not implemented yet.")
    else:
        print(f"Error: Unknown trade mode '{args.mode}'", file=sys.stderr)


def handle_configure(args):
    print("Handling 'configure' command...")
    # Placeholder for future configuration management (e.g., API keys, default params)
    print("Configuration management not fully implemented yet.")
    print(f"Possible config action: {args.action}, Key: {args.key}, Value: {args.value}")


def main():
    parser = argparse.ArgumentParser(description="Trading Bot CLI")
    subparsers = parser.add_subparsers(title='Commands', dest='command', required=True)

    # --- Backtest Command ---
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest of a strategy.')
    backtest_parser.add_argument('--strategy', type=str, required=True, choices=STRATEGY_MAP.keys(), help='Name of the strategy class to use.')
    backtest_parser.add_argument('--data', type=str, required=True, help="Path to CSV file with historical OHLCV data. Must include 'timestamp', 'open', 'high', 'low', 'close', 'volume'.")
    backtest_parser.add_argument('--symbol', type=str, default='BTC/USD', help="Trading symbol (e.g., 'BTC/USD'). Default: BTC/USD")
    backtest_parser.add_argument('--initial_balance', type=float, default=10000.0, help='Initial capital for backtesting.')
    backtest_parser.add_argument('--commission', type=float, default=0.001, help='Commission rate per trade (e.g., 0.001 for 0.1%%).')
    backtest_parser.add_argument('--trade_size', type=float, default=1.0, help='Fixed amount of asset to trade per signal.')
    backtest_parser.add_argument('--strategy_config', type=str, help='Strategy-specific configuration as a JSON string or path to a JSON file.')
    backtest_parser.set_defaults(func=handle_backtest)

    # --- Trade Command (Placeholder) ---
    trade_parser = subparsers.add_parser('trade', help='Run the trading bot in live or mock mode.')
    trade_parser.add_argument('--strategy', type=str, required=True, choices=STRATEGY_MAP.keys(), help='Name of the strategy class to use.')
    trade_parser.add_argument('--symbol', type=str, required=True, help="Trading symbol (e.g., 'BTC/USD').")
    trade_parser.add_argument('--mode', type=str, choices=['live', 'mock'], default='mock', help='Trading mode: live or mock.')
    trade_parser.add_argument('--strategy_config', type=str, help='Strategy-specific configuration as a JSON string or path to a JSON file.')
    trade_parser.set_defaults(func=handle_trade)

    # --- Configure Command (Placeholder) ---
    config_parser = subparsers.add_parser('configure', help='Manage bot configurations (placeholder).')
    config_parser.add_argument('action', choices=['set', 'get', 'list'], help="Configuration action.")
    config_parser.add_argument('--key', type=str, help="Configuration key.")
    config_parser.add_argument('--value', type=str, help="Configuration value (for 'set').")
    config_parser.set_defaults(func=handle_configure)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    # The initial imports might fail if trading_bot is not in PYTHONPATH,
    # e.g. when running `python cli.py` from `trading_bot/` directory.
    # The try-except block for imports at the top attempts to handle this.
    # For `python -m trading_bot.cli` run from parent of `trading_bot`, it should work.
    main()
