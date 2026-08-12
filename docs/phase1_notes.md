# Phase 1 — Complete Reference & Retrospective (Days 1–15)

> This document serves as both a **technical reference** for every module built during Phase 1 and a **learning journal** capturing decisions, gotchas, and insights. Anyone reading the codebase for the first time — or the future version of yourself preparing for an interview — should be able to use this as a map.

---

## What Phase 1 Covers

Phase 1 is the foundation layer. No agents, no RAG, no graphs — just the individual building blocks that every future component depends on:

- Talking to an LLM (Groq) and getting reliable, structured responses
- Validating data with Pydantic so bugs surface at boundaries, not downstream
- Fetching real market data (yfinance) and real news (Tavily)
- Computing technical indicators (RSI, moving averages) from scratch
- Wiring everything together into a single end-to-end pipeline

---

## File-by-File Reference

### `src/core/config.py` — Day 1

**Purpose:** Loads environment variables (`GROQ_API_KEY`, etc.) at startup using Pydantic `BaseSettings`. This means if a required key is missing, the app crashes immediately with a clear error instead of failing silently at the first API call ten minutes later.

**How to run:** This module is imported by other modules; it doesn't have a standalone entry point.

**Key design choice:** `extra = "allow"` in the Settings Config class lets the `.env` file contain keys that aren't declared in the model (like `TAVILY_API_KEY`) without raising validation errors. The `ANTHROPIC_API_KEY` field is a leftover from an earlier iteration before the project migrated to Groq — it still loads fine because the key exists in `.env`.

---

### `src/core/llm/client.py` — Days 2, 12, 13

**Purpose:** The lowest-level wrapper around Groq. Three capabilities in one file:

1. **Basic completion** (`get_llm_response`) — send a system + user message, get text back
2. **Streaming** (`_stream_completion`) — prints tokens as they arrive, accumulates the full text, and reports token usage at the end
3. **Retry with exponential backoff** (`_call_with_backoff`) — catches `APIConnectionError`, `APITimeoutError`, and `RateLimitError`; waits 2s → 4s → gives up after 3 attempts

**How to run:**
```bash
# Normal question with streaming
uv run python -m src.core.llm.client

# Demo: intentionally bad API key to see retry logic in action
uv run python -m src.core.llm.client --test-bad-key
```

**What I learned:** Streaming changes the response shape entirely — you iterate over chunks, each with a `.choices[0].delta.content` that may be `None`. Token usage only arrives on the final chunk. The retry logic uses a simple `for` loop rather than a library like `tenacity` — keeping it manual made the backoff behaviour completely transparent.

---

### `src/chat.py` — Day 3

**Purpose:** A multi-turn CLI chatbot. Maintains a `messages` list that grows with each user/assistant turn, giving Groq conversational memory within a session.

**How to run:**
```bash
uv run python -m src.chat
```

Type stock-related questions; type `quit` or `exit` to end.

**Key design choice:** The `SYSTEM_INSTRUCTION` is long and very specific. It enforces:
- Domain restriction (finance only)
- No first-person pronouns — the model refers to itself as "StockSage"
- A professional analyst tone
- An investment disclaimer when appropriate

**What I learned:** System prompts are surprisingly powerful. A well-crafted system prompt can keep the model in character across dozens of turns. The model occasionally slips on the pronoun rule ("I think..." instead of "StockSage notes..."), but it's rare. The conversation history list is the simplest form of "memory" — no databases, no embeddings, just a growing list of dicts.

---

### `src/core/schema.py` — Day 4

**Purpose:** Defines `StockData`, the first Pydantic model in the project. Three fields: `ticker` (str), `price` (float), `pe_ratio` (float).

**How to run:**
```bash
uv run python -m src.core.schema
```

Demonstrates valid/invalid instantiation, `.model_dump()` (object → dict), and `.model_validate()` (dict → object).

**What I learned:** Pydantic v2 does not silently coerce `"abc"` into a float — it raises a `ValidationError` with a clear message. This is the entire point: if bad data enters the system, you find out at the boundary, not three function calls later. The `.model_dump()` / `.model_validate()` round-trip is the foundation for serialisation everywhere else in the project.

---

### `src/core/structured.py` — Day 5

**Purpose:** The `extract(prompt, schema)` function — the single most reused utility in the project. It takes any natural-language text and a Pydantic model class, and forces Groq to return data matching that exact schema.

**How it works internally:**
1. `_make_function_declaration(schema)` converts a Pydantic class into an OpenAI/Groq tool declaration using `.model_json_schema()`
2. The API call uses `tool_choice` set to a specific function name, which **forces** Groq to always call that function (no "I'd prefer to answer in text" escape)
3. The raw JSON arguments from the function call are validated into a real Pydantic object via `schema.model_validate()`

**How to run:**
```bash
uv run python -m src.core.structured
```

Extracts `StockData` from 3 example paragraphs (Apple, Microsoft, NVIDIA).

**What I learned:** The difference between `tool_choice="auto"` and `tool_choice={"type": "function", "function": {"name": "..."}}` is critical. With `auto`, the model might decide to answer in plain text. With a forced choice, it always returns structured JSON. This is the backbone of reliable data extraction. The `_convert_property` helper simplifies nested JSON Schema properties — Pydantic outputs lowercase types (`"string"`, `"number"`) which is exactly what Groq expects, so no case conversion is needed.

---

### `src/core/tool_calling.py` — Day 6

**Purpose:** The first working tool call — a manual, step-by-step implementation showing exactly how function calling works before abstracting it away in Day 7.

**How it works:**
1. `get_current_price(ticker)` fetches a live price via `yfinance`
2. `PRICE_TOOL_SCHEMA` describes this function in OpenAI-compatible JSON
3. `ask_about_stock(question)` sends the question with the tool available, detects if Groq returns a function call, executes the real function, sends the result back, and gets a final natural-language answer

**How to run:**
```bash
uv run python -m src.core.tool_calling
```

**What I learned:** This is the "aha moment" of the entire project. The model doesn't just generate text — it generates a structured request to call a specific function with specific arguments, you run that function in your own Python process with real data, and then the model incorporates that real data into its response. This is the fundamental mechanism behind every AI agent.

---

### `src/core/tools.py` — Day 7

**Purpose:** Generalises Day 6 into a reusable system:
- `ToolDispatcher` — a registry mapping tool names to Python callables
- `run_agent()` — an agentic while-loop that keeps sending messages to Groq, dispatching tool calls, appending results, and looping until the model returns plain text or the safety cap is hit

**How to run:**
```bash
uv run python -m src.core.tools
```

Asks two questions that each require both `get_current_price` and `get_pe_ratio`.

**Key design choices:**
- `MAX_ITERATIONS = 5` — the safety cap prevents infinite loops if the model keeps requesting tools
- `_normalize_schema()` recursively lowercases the `"type"` values in schemas (the hand-written schemas use `"OBJECT"`, `"STRING"` but Groq expects lowercase)
- `build_default_dispatcher()` is a convenience factory that pre-registers both financial tools

**What I learned:** The agentic loop is shockingly simple — it's just a while loop. The magic is in the protocol: Groq returns a `tool_calls` list, you dispatch each one, append a `"tool"` role message with the `tool_call_id`, and send it all back. The model then synthesises the results. The safety cap is essential — without it, a confused model could loop forever.

---

### `src/ingestion/explore_yfinance.py` — Day 8 (exploration)

**Purpose:** Exploratory script to understand what data yfinance actually provides. Not used by any other module — purely a learning exercise.

**How to run:**
```bash
uv run python -m src.ingestion.explore_yfinance
```

Prints `.info` keys (fundamentals) and `.history()` DataFrame shape/columns for AAPL.

---

### `src/ingestion/yfinance_client.py` — Day 8

**Purpose:** The clean, production-quality data access layer for Yahoo Finance. Two public functions:

| Function | Returns | Description |
|---|---|---|
| `get_fundamentals(ticker)` | `CompanyFundamentals` | Market cap, P/E, sector, 52-week range, business summary |
| `get_price_history(ticker, period)` | `PriceHistory` | Chronological list of `PriceBar` objects (OHLCV) |

**How to run:**
```bash
uv run python -m src.ingestion.yfinance_client
```

**Pydantic models defined here:**
- `CompanyFundamentals` — 15 fields, all optional except `ticker`
- `PriceBar` — single day's OHLCV; uses field aliases (`trade_date` ↔ `date`, `open_price` ↔ `open`)
- `PriceHistory` — wrapper with a `.trading_days` property

**What I learned:** The `PriceBar` alias system (`Field(..., alias="date")` with `populate_by_name=True`) caused confusion. When you call `.model_dump()`, the output uses the Python field names (`trade_date`, `open_price`), not the aliases. But `.model_dump(by_alias=True)` uses the aliases. Both round-trip correctly because `populate_by_name=True` accepts either. The `_safe_float` helper guards against `NaN` values from yfinance (some fields return `NaN` instead of `None`).

---

### `src/ingestion/explore_pandas.py` — Day 9 (exploration)

**Purpose:** Practice script for pandas fundamentals — not imported anywhere else. Covers Steps 47–50:
- Loading a DataFrame from yfinance
- Selecting columns and filtering rows
- Handling NaN with `.dropna()` and `.ffill()`
- Computing a 20-day SMA with `.rolling(20).mean()`

**How to run:**
```bash
uv run python -m src.ingestion.explore_pandas
```

---

### `src/ingestion/technical.py` — Day 10

**Purpose:** Computes technical indicators from price data. Three main functions:

| Function | Returns | Description |
|---|---|---|
| `compute_rsi(prices, period=14)` | `pd.Series` | Wilder's RSI, NaN for the first `period` entries |
| `compute_moving_averages(prices)` | `(float\|None, float\|None)` | Latest MA-20 and MA-50 |
| `compute_technical_signals(prices, ticker)` | `TechnicalSignals` | Combines RSI + MAs into a validated model |
| `get_technical_signals(ticker, period)` | `TechnicalSignals` | Fetches price data from yfinance, then computes signals |

**How to run:**
```bash
uv run python -m src.ingestion.technical
```

**RSI implementation details (Wilder's method):**
1. Compute day-over-day price changes
2. Separate gains (positive changes) and losses (negative changes)
3. Seed the first average with a simple mean of the first N values
4. Smooth subsequent averages: `new_avg = (prev_avg × (N-1) + today) / N`
5. RS = avg_gain / avg_loss; RSI = 100 - 100/(1+RS)
6. Edge cases: both zero → RSI 50; all gains → RSI 100; all losses → RSI 0

**What I learned:** The RSI formula looks simple on paper, but the smoothed moving average (SMMA) is different from a simple moving average. The seed step matters — if you skip it and just use a rolling window, you get different numbers. The edge case handling (zero average loss, zero average gain) is also important; without it, you get division-by-zero errors or `inf` values.

---

### `src/ingestion/tavily_client.py` — Day 11

**Purpose:** Searches for recent stock news using the Tavily API. Three public functions:

| Function | Description |
|---|---|
| `search_news(ticker, max_results=10)` | Queries Tavily for `"latest news about {ticker} stock"` |
| `filter_recent(results, max_age_days=30)` | Drops articles older than 30 days |
| `print_results(results)` | Pretty-prints title, URL, snippet, date |

**How to run:**
```bash
uv run python -m src.ingestion.tavily_client
```

**What I learned:** Tavily's `published_date` field is inconsistent — sometimes ISO-8601, sometimes RFC-2822, sometimes `None`. The `_parse_published_date` helper tries ISO first, then falls back to `email.utils.parsedate_to_datetime`. Articles with unparseable dates are excluded by `filter_recent()` rather than silently kept.

---

### `src/pipeline.py` — Day 15

**Purpose:** The Phase 1 capstone. For each ticker, this script chains every module together:

1. `get_fundamentals(ticker)` → `CompanyFundamentals`
2. `get_price_history(ticker)` → `PriceHistory`
3. `compute_technical_signals([bar.close for bar in history.bars])` → `TechnicalSignals`
4. `search_news(ticker)` + `filter_recent()` → news blurbs
5. Build a context prompt → `extract(prompt, TickerSummary)` → validated summary

**How to run:**
```bash
# Default: AAPL, MSFT, NVDA
uv run python -m src.pipeline

# Custom tickers
uv run python -m src.pipeline TSLA GOOG AMZN
```

**Output model (`TickerSummary`):**
- `ticker: str` — the symbol
- `summary: str` — 3–5 sentence AI-generated paragraph
- `sentiment: str` — one of `"bullish"`, `"bearish"`, `"neutral"`

After running all tickers, the script performs automated sanity checks (non-empty fields, valid sentiment values).

---

## Test Suite Reference

All tests live in `tests/` and are runnable via:

```bash
# Unit tests only (no network)
uv run pytest -v -m "not integration"

# Everything
uv run pytest -v
```

### `tests/test_schema.py`
- Valid instantiation, `.model_dump()` round-trip, `.model_validate()` from dict
- Invalid data (`price="abc"`, missing required field) raises `ValidationError`

### `tests/test_technical.py`
- **Hand-calculated RSI** with period=3 on the price series `[10, 12, 11, 13, 14, 12, 15]` — every index is verified against a manually computed expected value
- MA-20 and MA-50 with insufficient data → `None`
- MA-20 and MA-50 with exact/sufficient data → verified against `mean()`
- `TechnicalSignals` round-trip validation
- Integration: `get_technical_signals("AAPL")` returns a valid model with RSI in [0, 100]

### `tests/test_structured.py`
- `_convert_property()` passes through all JSON Schema types correctly
- `_make_function_declaration()` produces the right OpenAI/Groq tool shape
- Mocked test: `extract()` raises `ValueError` when Groq returns no function call
- Live integration: extracts `StockData` from 4 different paragraphs (AAPL, MSFT, NVDA, TSLA) and verifies ticker, price, and P/E ratio

### `tests/test_yfinance_client.py`
- `CompanyFundamentals` model: minimal valid, full valid, round-trip
- `PriceBar` model: valid creation, missing field raises error
- `PriceHistory` model: empty bars, bars with data, `.trading_days` property
- Integration: `get_fundamentals("AAPL")` returns correct data, case-insensitive, invalid ticker raises `ValueError`
- Integration: `get_price_history("AAPL")` returns >100 trading days, correct types, round-trip

### `tests/test_tools.py`
- `ToolDispatcher`: register/dispatch happy path, duplicate name guard, unknown name guard, `.tools` format
- `run_agent()` mocked: plain text returns immediately, single tool call resolved, safety cap triggers `RuntimeError` after N iterations, tool result correctly appended to message history
- Integration: dual-tool question answered end-to-end, single-tool doesn't trigger safety cap

---

## Interface Notes & Gotchas

These are the friction points discovered while wiring everything together in `pipeline.py`:

### 1. PriceHistory → TechnicalSignals gap
`get_price_history()` returns `PriceHistory` with `.bars` (a list of `PriceBar` objects), but `compute_technical_signals()` expects a `pd.Series` or `Sequence[float]`. The pipeline bridges this with:
```python
closing_prices = [bar.close for bar in history.bars]
signals = compute_technical_signals(closing_prices, ticker=ticker)
```
A `.closes` property on `PriceHistory` would eliminate this friction.

### 2. PriceBar field aliases
`PriceBar` uses aliases: `trade_date` ↔ `"date"`, `open_price` ↔ `"open"`. The `populate_by_name=True` config makes round-trips work, but it's confusing when reading the code for the first time.

### 3. Tavily date parsing
`published_date` has no guaranteed format. The fallback parser chain (ISO → RFC-2822 → `None`) works but feels fragile. Worth wrapping in a dedicated utility if more date-bearing APIs are added later.

### 4. Schema naming
The roadmap says `schemas.py` but the actual file is `schema.py` (singular). Similarly `test_schemas.py` in the roadmap vs `test_schema.py` in practice. This is fine — just something to be aware of if comparing against the curriculum.

### 5. `core/config.py` still references `ANTHROPIC_API_KEY`
This is a remnant from before the migration to Groq. It doesn't break anything (because `extra = "allow"` and the `.env` doesn't have that key), but it's misleading. Worth cleaning up.

---

## What I Learned — The Big Takeaways

**Pydantic is the best decision in the project.** Every module boundary is a validated model. When `pipeline.py` wired everything together, there were zero data-shape bugs — every mismatch was caught at the Pydantic layer during development. The pattern of "function returns a validated model, caller trusts the types" eliminated an entire category of bugs.

**Groq's function calling is the backbone of reliable AI output.** The difference between hoping the model outputs valid JSON and *forcing* it via `tool_choice` is enormous. With forced function calling, `extract()` has never returned malformed data in any test run. This is the single technique that makes structured output actually production-viable.

**The agentic loop is embarrassingly simple.** Before building it, "AI agent" sounded like a complex concept. In reality, it's a while loop: send messages → check for tool calls → dispatch → append results → repeat. The complexity isn't in the loop; it's in designing good tools and handling edge cases (safety cap, error handling, message format).

**Technical indicators are more subtle than they look.** Wilder's RSI uses a smoothed moving average that's different from a simple rolling mean. The seed step (first N values use a simple average, then subsequent values use exponential smoothing) is easy to get wrong. Edge cases (all gains, all losses, no movement) each need explicit handling.

**Streaming changes the API contract.** A non-streaming response gives you a single object with `.choices[0].message.content`. A streaming response gives you an iterator of chunks where `.choices[0].delta.content` may be `None` on any given chunk, and `.usage` only appears on the final chunk. You can't just add `stream=True` — you need to restructure the response handling.

---

## What Confused Me

**The `PriceBar` alias system** was the most confusing part of Pydantic. I expected `.model_dump()` to use the alias names, but it uses the Python field names by default. You need `.model_dump(by_alias=True)` to get the aliases. Both directions work for `.model_validate()` because of `populate_by_name=True`, but it took trial and error to understand the asymmetry.

**Where does `src` end and the module begin?** Running `python -m core.llm.client` fails with `ModuleNotFoundError` because Python doesn't know `core` is inside `src`. The fix is `python -m src.core.llm.client` (treating `src` as the top-level package). The `pyproject.toml` sets `pythonpath = ["src"]` for pytest, which is why tests can import `from core.schema import StockData` directly — but scripts invoked from the command line need the `src.` prefix.

**Groq's `tool_choice` variants** were initially confusing. `"auto"` means "call a tool if you want to". `"none"` means "never call a tool". `{"type": "function", "function": {"name": "X"}}` means "you MUST call function X". For `extract()` (structured output), forced choice is essential. For `run_agent()` (agentic loop), `"auto"` is correct because the model needs the freedom to stop calling tools and give a final answer.

**Tavily sometimes returns results with no `published_date` at all**, and sometimes the date is in a completely different format than expected. The first time I ran `filter_recent()`, it silently dropped everything because all dates failed to parse. Adding the RFC-2822 fallback parser and the `None`-check fixed it, but it was a frustrating debugging session.

---

## Day-by-Day Log

| Day | Commit Message | What Was Built |
|---|---|---|
| 1 | Project scaffold | Folder structure, `.env`, `config.py`, `.gitignore`, `pyproject.toml` |
| 2 | First Groq API call | `core/llm/client.py` — single-shot question → answer |
| 3 | Multi-turn conversation loop | `chat.py` — CLI chatbot with system prompt persona |
| 4 | First Pydantic schema | `core/schema.py` — `StockData` model with validation |
| 5 | Structured output extraction | `core/structured.py` — `extract(prompt, schema)` via function calling |
| 6 | First working tool call | `core/tool_calling.py` — manual function-calling round-trip |
| 7 | General tool-calling loop | `core/tools.py` — `ToolDispatcher` + `run_agent()` with safety cap |
| 8 | yfinance data client | `ingestion/yfinance_client.py` + `explore_yfinance.py` |
| 9 | Pandas practice | `ingestion/explore_pandas.py` — filtering, NaN handling, SMA |
| 10 | RSI and moving averages | `ingestion/technical.py` — Wilder's RSI from scratch |
| 11 | Tavily news search | `ingestion/tavily_client.py` — search, filter, display |
| 12 | Streaming responses | Updated `core/llm/client.py` with `stream=True` |
| 13 | Retry logic | Added exponential backoff to `core/llm/client.py` |
| 14 | Test suite | 5 test files covering schemas, technicals, yfinance, structured, tools |
| 15 | Phase 1 complete | `pipeline.py` — end-to-end integration + this document |

---

## What's Next (Phase 2 Preview)

Phase 2 (Days 16–30) introduces **Retrieval-Augmented Generation (RAG)**:
- Embedding financial documents into vector space
- Storing and retrieving them from a vector database
- Grounding Groq's answers in real documents rather than just live data

The `rag/` directory is currently empty — it will be populated starting Day 16.
