# trading_bot/strategies/__init__.py

"""
Trading strategies for the bot.

This package contains the base strategy class and specific strategy implementations.
"""

from .base_strategy import BaseStrategy
from .example_ema_cross_strategy import EMACrossStrategy
from .advanced_example_strategy import AdvancedExampleStrategy

__all__ = [
    'BaseStrategy',
    'EMACrossStrategy',
    'AdvancedExampleStrategy'
]

print("trading_bot.strategies package initialized with advanced strategy")
