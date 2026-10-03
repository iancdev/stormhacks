"""Wheel hardware backends; importing this package does not load Windows drivers."""

from .windows import HardwareError, WindowsAdapter

__all__ = ["HardwareError", "WindowsAdapter"]
