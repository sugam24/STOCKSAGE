"""
Root api package forwarding to src.api.main for convenience.
"""

from src.api.main import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
