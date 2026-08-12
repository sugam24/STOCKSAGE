# StockSage 📈

**An AI-powered stock research assistant** that combines real-time market data, technical analysis, news intelligence, and LLM-driven summarisation into a single pipeline.

Built as a 75-day hands-on learning project. This repository represents **Phase 1 (Days 1–15)**: the foundational building blocks that every future agent, RAG pipeline, and graph workflow will depend on.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [End-to-End Pipeline](#end-to-end-pipeline)
  - [Interactive Chat](#interactive-chat)
  - [Individual Modules](#individual-modules)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Features

| Capability | Description |
|---|---|
| **LLM Integration** | Groq API (`llama-3.3-70b-versatile`) with streaming, retry logic, and exponential backoff |
| **Structured Output** | Force Groq to return Pydantic-validated JSON via function calling — no regex parsing |
| **Tool Calling** | Generic `ToolDispatcher` + agentic while-loop that resolves multi-tool chains automatically |
| **Market Data** | Real-time fundamentals and 6-month OHLCV price history via `yfinance` |
| **Technical Analysis** | Wilder's RSI (14-period) and SMA-20/SMA-50 computed from scratch with pandas |
| **News Search** | Recent, filtered news articles via the Tavily search API |
| **Integration Pipeline** | Single script that chains all of the above into a one-paragraph AI summary per ticker |
| **Test Suite** | Unit + integration tests covering schemas, technical indicators, data clients, structured extraction, and tool dispatch |

---

## Project Structure

```
StockSage/
├── .env                          # API keys (never committed)
├── .env.example                  # Template showing required keys
├── .gitignore                    # Standard Python ignores
├── pyproject.toml                # Dependencies, pytest config, pyright config
├── README.md                     # ← You are here
│
├── src/
│   ├── __init__.py
│   ├── chat.py                   # Day 3  — Multi-turn CLI chatbot
│   ├── pipeline.py               # Day 15 — End-to-end integration script
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Day 1  — Pydantic BaseSettings for env vars
│   │   ├── schema.py             # Day 4  — StockData Pydantic model
│   │   ├── structured.py         # Day 5  — extract(prompt, schema) → validated object
│   │   ├── tool_calling.py       # Day 6  — Single-tool function-calling demo
│   │   ├── tools.py              # Day 7  — ToolDispatcher + agentic while-loop
│   │   └── llm/
│   │       ├── __init__.py
│   │       └── client.py         # Day 2, 12, 13 — Groq client, streaming, retries
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── explore_yfinance.py   # Day 8  — yfinance exploration script
│   │   ├── explore_pandas.py     # Day 9  — Pandas practice on AAPL data
│   │   ├── yfinance_client.py    # Day 8  — get_fundamentals() + get_price_history()
│   │   ├── technical.py          # Day 10 — RSI, MA-20, MA-50, TechnicalSignals
│   │   └── tavily_client.py      # Day 11 — search_news() + filter_recent()
│   │
│   ├── rag/                      # Phase 2 — empty placeholder
│   ├── agents/                   # Phase 3 — empty placeholder
│   ├── graph/                    # Phase 3 — empty placeholder
│   └── api/                      # Phase 4 — empty placeholder
│
├── tests/
│   ├── __init__.py
│   ├── test_schema.py            # Day 14 — StockData model validation
│   ├── test_structured.py        # Day 14 — Structured extraction (unit + live)
│   ├── test_technical.py         # Day 14 — RSI & MA hand-calculated verification
│   ├── test_tools.py             # Day 14 — ToolDispatcher + agentic loop
│   └── test_yfinance_client.py   # Day 14 — CompanyFundamentals, PriceBar, PriceHistory
│
└── docs/
    └── phase1_notes.md           # Day 15 — Phase 1 retrospective & reference guide
```

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (recommended) *or* plain `pip`
- **API Keys:**
  - [Groq](https://console.groq.com/) — free tier, used for all LLM calls
  - [Tavily](https://tavily.com/) — free tier (1,000 searches/month), used for news

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/StockSage.git
cd StockSage

# Create and activate a virtual environment (uv handles this automatically)
uv sync

# Or, with plain pip:
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
pip install -e ".[dev]"
```

---

## Configuration

Copy the example env file and fill in your real keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
TAVILY_API_KEY=tvly-your_tavily_api_key_here
```

> **Important:** `.env` is listed in `.gitignore` and must never be committed.

---

## Usage

### End-to-End Pipeline

The flagship script — fetches fundamentals, price history, computes RSI/MAs, searches news, and asks Groq to summarise everything into a structured one-paragraph report:

```bash
# Default tickers: AAPL, MSFT, NVDA
uv run python -m src.pipeline

# Custom tickers
uv run python -m src.pipeline TSLA GOOG AMZN
```

**What it does for each ticker:**
1. Fetches company fundamentals (market cap, P/E, sector, etc.) from Yahoo Finance
2. Fetches 6 months of daily OHLCV price history
3. Computes RSI-14, MA-20, and MA-50 from the closing prices
4. Searches Tavily for recent news articles (last 30 days)
5. Sends all collected data to Groq, forcing a structured `TickerSummary` response (ticker, summary paragraph, sentiment)
6. Runs automated sanity checks on the output

### Interactive Chat

A multi-turn CLI chatbot with a strict financial-research persona:

```bash
uv run python -m src.chat
```

Type stock-related questions; the assistant stays in character. Type `quit` or `exit` to end.

### Individual Modules

Every module is independently runnable for exploration and testing:

| Command | What It Does |
|---|---|
| `uv run python -m src.core.llm.client` | Send a one-shot question to Groq with streaming output |
| `uv run python -m src.core.llm.client --test-bad-key` | Demo retry logic with an intentionally invalid API key |
| `uv run python -m src.core.structured` | Extract structured `StockData` from 3 example paragraphs |
| `uv run python -m src.core.tool_calling` | Single-tool function-calling demo (live AAPL price) |
| `uv run python -m src.core.tools` | Multi-tool agentic loop (price + P/E ratio for AAPL & MSFT) |
| `uv run python -m src.ingestion.explore_yfinance` | Explore the yfinance `.info` and `.history()` APIs |
| `uv run python -m src.ingestion.explore_pandas` | Practice pandas on AAPL price data (filter, NaN, SMA) |
| `uv run python -m src.ingestion.yfinance_client` | Fetch and validate AAPL fundamentals + price history |
| `uv run python -m src.ingestion.technical` | Compute and display RSI/MA signals for AAPL |
| `uv run python -m src.ingestion.tavily_client` | Search Tavily for recent AAPL news |

---

## Testing

```bash
# Run all unit tests (no API calls)
uv run pytest -v -m "not integration"

# Run everything including live API tests
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_technical.py -v

# Run integration tests that need the STOCKSAGE_INTEGRATION flag
STOCKSAGE_INTEGRATION=1 uv run pytest -v -m integration
```

**Test breakdown:**

| File | Tests | Requires Network |
|---|---|---|
| `test_schema.py` | StockData valid/invalid instantiation, round-trip | No |
| `test_technical.py` | RSI hand-calculation, MA edge cases, TechnicalSignals model | No (unit), Yes (integration) |
| `test_yfinance_client.py` | CompanyFundamentals/PriceBar/PriceHistory models | No (unit), Yes (integration) |
| `test_structured.py` | Function declaration shape, extract() mock + live | No (unit), Yes (integration) |
| `test_tools.py` | ToolDispatcher CRUD, run_agent() mock + safety cap | No (unit), Yes (integration) |

---

## Roadmap

This project follows a 75-day structured curriculum across 5 phases:

| Phase | Days | Focus | Status |
|---|---|---|---|
| **Phase 1: Foundations** | 1–15 | Groq API, Pydantic, yfinance, technical indicators, Tavily, testing | ✅ Complete |
| **Phase 2: RAG** | 16–30 | Embeddings, vector stores, retrieval-augmented generation | 🔲 Upcoming |
| **Phase 3: Agents** | 31–45 | Multi-agent orchestration, LangGraph, specialised agents | 🔲 Upcoming |
| **Phase 4: API & UI** | 46–60 | FastAPI backend, Streamlit frontend, deployment | 🔲 Upcoming |
| **Phase 5: Production** | 61–75 | Evaluation, guardrails, monitoring, optimisation | 🔲 Upcoming |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| LLM Provider | [Groq](https://groq.com/) (`llama-3.3-70b-versatile`) |
| Data Validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Market Data | [yfinance](https://github.com/ranaroussi/yfinance) |
| Data Analysis | [pandas](https://pandas.pydata.org/) |
| News Search | [Tavily](https://tavily.com/) |
| Testing | [pytest](https://pytest.org/) |
| Type Checking | [pyright](https://github.com/microsoft/pyright) |

---

## License

This project is for educational purposes. No license has been applied yet.
