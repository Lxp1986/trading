# trading_bot/api_integrations/__init__.py

"""
API Integration layer for connecting to various exchanges.

This package contains the base API handler abstract class and concrete
implementations for different exchanges (e.g., a MockAPIHandler for testing).
"""

from .base_api_handler import BaseAPIHandler
from .mock_api_handler import MockAPIHandler

__all__ = [
    'BaseAPIHandler',
    'MockAPIHandler'
]

print("trading_bot.api_integrations package initialized")
