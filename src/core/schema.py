"""
core/schema.py
--------------
Pydantic data models for StockSage.

Steps covered:
  18. StockData model with typed fields.
  19. Valid & invalid instantiation examples (run as __main__).
  20. .model_dump()  – Pydantic object  → plain dict.
  21. .model_validate() – plain dict → Pydantic object.
"""

from pydantic import BaseModel, ValidationError


class StockData(BaseModel):
    """Represents a single stock's key data points."""

    ticker: str
    price: float
    pe_ratio: float


# ---------------------------------------------------------------------------
# Demonstration (steps 19–21) – run with:  uv run python -m core.schema
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ── Step 19a: Valid instance ────────────────────────────────────────────
    valid = StockData(ticker="AAPL", price=189.30, pe_ratio=28.5)
    print("✅ Valid instance created:")
    print(f"   {valid}\n")

    # ── Step 20: model_dump() → plain dict ─────────────────────────────────
    data_dict = valid.model_dump()
    print("📦 model_dump() result:")
    print(f"   {data_dict}")
    print(f"   type: {type(data_dict)}\n")

    # ── Step 21: model_validate() ← plain dict → Pydantic object ───────────
    raw = {"ticker": "MSFT", "price": 415.75, "pe_ratio": 34.2}
    from_dict = StockData.model_validate(raw)
    print("🔄 model_validate() result:")
    print(f"   {from_dict}\n")

    # ── Step 19b: Invalid instance – price='abc' ───────────────────────────
    print("❌ Attempting invalid instance (price='abc') …")
    try:
        bad = StockData(ticker="TSLA", price="abc", pe_ratio=70.0)
    except ValidationError as exc:
        print(f"   ValidationError caught:\n{exc}\n")
