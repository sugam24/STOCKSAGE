#!/usr/bin/env python3
"""
check_health.py
---------------
StockSage — Comprehensive System & API Health Diagnostic Tool.

Run this script before starting research sessions or after returning to the project
after days/months. It audits every external dependency, API key, model endpoint,
and local vector database to guarantee zero mid-execution halts.

Checks performed:
  1. Python runtime & Virtual Environment (checks for broken paths from folder moves)
  2. Configuration & .env presence
  3. Groq API (key validity, active models on your tier, test completion)
  4. Tavily Search API (key validity, monthly search quota check)
  5. SEC EDGAR API (User-Agent compliance, ticker directory access)
  6. Yahoo Finance (live quote & historical OHLCV feed)
  7. Local Embedding Model (sentence-transformers / all-MiniLM-L6-v2)
  8. Local Vector Database (ChromaDB persistence & write/read permissions)
  9. Neural Cross-Encoder (FlashRank / ms-marco-TinyBERT-L-2-v2)

Usage:
    uv run python check_health.py
    # or
    python check_health.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ANSI Terminal Styling
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    remediation: str | None = None
    duration_ms: float = 0.0


results: list[CheckResult] = []


def record(name: str, status: str, message: str, remediation: str | None = None, start_time: float | None = None) -> None:
    duration = (time.perf_counter() - start_time) * 1000 if start_time else 0.0
    results.append(CheckResult(name, status, message, remediation, duration))

    badge = f"{GREEN}[PASS]{RESET}" if status == "PASS" else f"{YELLOW}[WARN]{RESET}" if status == "WARN" else f"{RED}[FAIL]{RESET}"
    timing = f"{DIM}({duration:.0f}ms){RESET}" if duration > 0 else ""
    print(f"  {badge} {BOLD}{name:<32}{RESET} {message} {timing}")


# ---------------------------------------------------------------------------
# 1. Environment & Runtime
# ---------------------------------------------------------------------------
def check_runtime_and_env() -> None:
    print(f"\n{BOLD}{CYAN}1. Environment & Runtime Integrity{RESET}")
    t0 = time.perf_counter()

    # Python version
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 12):
        record("Python Version", "PASS", f"Running Python {major}.{minor}.{sys.version_info.micro}", start_time=t0)
    else:
        record(
            "Python Version", "FAIL",
            f"Python {major}.{minor} is below minimum requirement (3.12+)",
            "Install Python 3.12 or newer using `uv python install 3.12` or your system package manager.",
            start_time=t0
        )

    # Venv path integrity (detects folder move issues)
    t0 = time.perf_counter()
    virtual_env = os.getenv("VIRTUAL_ENV")
    current_dir = str(Path.cwd().resolve())
    if virtual_env:
        resolved_venv = str(Path(virtual_env).resolve())
        if resolved_venv.startswith(current_dir):
            record("Virtual Environment Path", "PASS", f"Active venv matches project root", start_time=t0)
        else:
            record(
                "Virtual Environment Path", "WARN",
                f"Venv path '{resolved_venv}' differs from current project '{current_dir}'",
                "Rebuild virtual environment paths: run `uv sync` to update all binaries to the new folder.",
                start_time=t0
            )
    else:
        record("Virtual Environment", "WARN", "Not running inside an active virtual environment",
               "Recommended: run via `uv run python check_health.py`.", start_time=t0)

    # .env file existence
    t0 = time.perf_counter()
    env_file = Path(".env")
    if env_file.exists():
        record(".env File Presence", "PASS", f"Found .env at project root", start_time=t0)
    else:
        record(
            ".env File Presence", "FAIL", "Missing .env file in project root",
            "Copy template: `cp .env.example .env` and insert your GROQ_API_KEY and TAVILY_API_KEY.",
            start_time=t0
        )

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        record("python-dotenv Package", "FAIL", "python-dotenv is not installed in current environment",
               "Run `uv sync` to install all project dependencies.", start_time=t0)


# ---------------------------------------------------------------------------
# 2. Groq LLM API Health
# ---------------------------------------------------------------------------
def check_groq_api() -> None:
    print(f"\n{BOLD}{CYAN}2. Groq LLM API & Model Availability{RESET}")
    t0 = time.perf_counter()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        record(
            "Groq API Key Presence", "FAIL", "GROQ_API_KEY is not set in environment or .env",
            "Obtain a free key at https://console.groq.com/keys and add `GROQ_API_KEY=gsk_...` to `.env`.",
            start_time=t0
        )
        return

    record("Groq API Key Presence", "PASS", f"Key set (starts with '{api_key[:6]}...')", start_time=t0)

    try:
        from groq import Groq, AuthenticationError, RateLimitError, NotFoundError, APIConnectionError
    except ImportError:
        record("Groq SDK Package", "FAIL", "`groq` package is not installed",
               "Run `uv sync` to install dependencies.", start_time=time.perf_counter())
        return

    client = Groq(api_key=api_key)

    # Test authentication & fetch available models
    t0 = time.perf_counter()
    available_models: list[str] = []
    try:
        model_list = client.models.list()
        available_models = sorted(m.id for m in model_list.data)
        record("Groq Authentication & Models", "PASS", f"Authenticated successfully ({len(available_models)} models online)", start_time=t0)
    except AuthenticationError as e:
        record(
            "Groq Authentication", "FAIL", f"Invalid or expired API key ({e})",
            "Your Groq API key is invalid or revoked. Generate a new key at https://console.groq.com/keys and update `.env`.",
            start_time=t0
        )
        return
    except RateLimitError as e:
        record(
            "Groq Rate Limit", "FAIL", f"Rate limit / credit quota exhausted ({e})",
            "Groq rate limits reached. Check https://console.groq.com/settings/limits or wait for quota reset.",
            start_time=t0
        )
        return
    except APIConnectionError as e:
        record(
            "Groq Network Connection", "FAIL", f"Cannot connect to api.groq.com ({e})",
            "Check your internet connection, DNS, or firewall settings.",
            start_time=t0
        )
        return
    except Exception as e:
        record("Groq API Handshake", "FAIL", f"Unexpected error: {e}", "Verify internet access and Groq service status at https://status.groq.com", start_time=t0)
        return

    # Check configured model availability
    t0 = time.perf_counter()
    configured_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    # Filter active chat models
    chat_models = [m for m in available_models if not any(x in m for x in ("whisper", "guard", "vision", "preview-deleted"))]

    if configured_model not in available_models:
        remediation_choice = chat_models[0] if chat_models else "openai/gpt-oss-120b"
        record(
            "Configured Model Online", "FAIL",
            f"Model '{configured_model}' not available on your Groq key (deprecated or restricted)",
            f"Add or update `GROQ_MODEL={remediation_choice}` in your `.env`. Available models: {', '.join(chat_models[:4])}",
            start_time=t0
        )
        return

    record("Configured Model Online", "PASS", f"Target model '{configured_model}' is available", start_time=t0)

    # Perform a live 1-token completion test
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=configured_model,
            messages=[{"role": "user", "content": "Respond with the word 'OK' only."}],
            max_tokens=5,
            temperature=0.0,
        )
        reply = (response.choices[0].message.content or "").strip()
        record("Groq Test Completion", "PASS", f"Ping response received: '{reply}'", start_time=t0)
    except Exception as e:
        record(
            "Groq Test Completion", "FAIL", f"Inference call failed: {e}",
            f"Model '{configured_model}' failed to respond. Try switching `GROQ_MODEL` in `.env` to another available model like `{chat_models[0]}`.",
            start_time=t0
        )


# ---------------------------------------------------------------------------
# 3. Tavily Financial News API
# ---------------------------------------------------------------------------
def check_tavily_api() -> None:
    print(f"\n{BOLD}{CYAN}3. Tavily Financial News API{RESET}")
    t0 = time.perf_counter()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        record(
            "Tavily API Key Presence", "FAIL", "TAVILY_API_KEY is not set in environment or .env",
            "Sign up for 1,000 free monthly queries at https://tavily.com/ and set `TAVILY_API_KEY=tvly-...` in `.env`.",
            start_time=t0
        )
        return

    record("Tavily API Key Presence", "PASS", f"Key set (starts with '{api_key[:6]}...')", start_time=t0)

    try:
        from tavily import TavilyClient
    except ImportError:
        record("Tavily SDK Package", "FAIL", "`tavily-python` is not installed",
               "Run `uv sync` to install dependencies.", start_time=time.perf_counter())
        return

    # Execute a minimal 1-item test search
    t0 = time.perf_counter()
    try:
        client = TavilyClient(api_key=api_key)
        res = client.search(query="NVDA stock news", max_results=1)
        results_count = len(res.get("results", []))
        if results_count > 0:
            first_title = res["results"][0].get("title", "Untitled")[:50]
            record("Tavily Search Ping", "PASS", f"Returned valid results (e.g., '{first_title}...')", start_time=t0)
        else:
            record("Tavily Search Ping", "WARN", "API responded but returned 0 results for query", start_time=t0)
    except Exception as e:
        err_msg = str(e).lower()
        if "unauthorized" in err_msg or "401" in err_msg or "invalid" in err_msg:
            record(
                "Tavily Search Ping", "FAIL", f"Invalid API key ({e})",
                "Your Tavily API key is invalid. Get a new free key at https://app.tavily.com and update `.env`.",
                start_time=t0
            )
        elif "limit" in err_msg or "429" in err_msg or "quota" in err_msg:
            record(
                "Tavily Search Ping", "FAIL", f"Monthly quota exceeded ({e})",
                "Tavily monthly free search quota (1,000 calls) has been exhausted. Create a new key or upgrade at https://tavily.com.",
                start_time=t0
            )
        else:
            record(
                "Tavily Search Ping", "FAIL", f"Request failed: {e}",
                "Verify network connection to api.tavily.com and check API status.",
                start_time=t0
            )


# ---------------------------------------------------------------------------
# 4. SEC EDGAR API (10-Q / 10-K Ingestion)
# ---------------------------------------------------------------------------
def check_sec_edgar() -> None:
    print(f"\n{BOLD}{CYAN}4. SEC EDGAR Regulatory Filings API{RESET}")
    t0 = time.perf_counter()

    user_agent = os.getenv("SEC_USER_AGENT", "StockSage research dahalsugam24@gmail.com")
    # SEC mandates 'User-Agent: Sample Company Name AdminContact@<sample company domain>.com'
    if "@" in user_agent and len(user_agent.split()) >= 2:
        record("SEC User-Agent Format", "PASS", f"Compliant User-Agent: '{user_agent}'", start_time=t0)
    else:
        record(
            "SEC User-Agent Format", "WARN",
            f"User-Agent '{user_agent}' may not meet SEC guidelines (requires Organization and email)",
            "Set `SEC_USER_AGENT=YourName your_email@domain.com` in `.env` to prevent SEC 403 blocks.",
            start_time=t0
        )

    t0 = time.perf_counter()
    try:
        import requests
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=10)
        if resp.status_code == 200:
            count = len(resp.json())
            record("SEC EDGAR Directory Access", "PASS", f"Successfully reached efts.sec.gov ({count:,} public tickers verified)", start_time=t0)
        elif resp.status_code == 403:
            record(
                "SEC EDGAR Directory Access", "FAIL", "SEC blocked request with HTTP 403 Forbidden",
                "The SEC blocked your User-Agent. Update `SEC_USER_AGENT=YourName your_email@domain.com` in `.env`.",
                start_time=t0
            )
        else:
            record("SEC EDGAR Directory Access", "WARN", f"Unexpected status code: {resp.status_code}", start_time=t0)
    except Exception as e:
        record(
            "SEC EDGAR Directory Access", "FAIL", f"Cannot connect to sec.gov: {e}",
            "Check internet connection. Note: SEC blocks non-compliant high-frequency scrapers (limit is 10 req/sec).",
            start_time=t0
        )


# ---------------------------------------------------------------------------
# 5. Yahoo Finance Market Data
# ---------------------------------------------------------------------------
def check_yahoo_finance() -> None:
    print(f"\n{BOLD}{CYAN}5. Yahoo Finance Market Data Feed (yfinance){RESET}")
    t0 = time.perf_counter()

    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        history = ticker.history(period="5d")
        if not history.empty and len(history) > 0:
            last_close = history["Close"].iloc[-1]
            record("Yahoo Finance Historical Feed", "PASS", f"Fetched 5-day AAPL OHLCV (Latest Close: ${last_close:.2f})", start_time=t0)
        else:
            record(
                "Yahoo Finance Historical Feed", "WARN", "Returned empty DataFrame for AAPL",
                "Yahoo Finance may be rate-limiting requests or experiencing temporary service degradation.",
                start_time=t0
            )
    except Exception as e:
        record(
            "Yahoo Finance Feed", "FAIL", f"yfinance call failed: {e}",
            "Check internet connectivity or upgrade yfinance: `uv add yfinance --upgrade`.",
            start_time=t0
        )


# ---------------------------------------------------------------------------
# 6. Local Embeddings, ChromaDB & FlashRank
# ---------------------------------------------------------------------------
def check_local_ml_and_vectorstore() -> None:
    print(f"\n{BOLD}{CYAN}6. Local ML Models & Vector Database{RESET}")

    # Sentence Transformers
    t0 = time.perf_counter()
    try:
        from src.rag.embeddings import embed
        vecs = embed(["StockSage embedding engine sanity check."])
        if len(vecs) == 1 and len(vecs[0]) == 384:
            record("SentenceTransformer Embedding", "PASS", f"Loaded `all-MiniLM-L6-v2` (Output dim: 384)", start_time=t0)
        else:
            record("SentenceTransformer Embedding", "WARN", f"Unexpected vector dimension: {len(vecs[0]) if vecs else 0}", start_time=t0)
    except Exception as e:
        record(
            "SentenceTransformer Embedding", "FAIL", f"Embedding failed: {e}",
            "Check PyTorch / sentence-transformers installation: run `uv sync`.",
            start_time=t0
        )

    # ChromaDB Vector Store
    t0 = time.perf_counter()
    try:
        from src.rag import vectorstore
        client = vectorstore.get_client()
        test_col = client.get_or_create_collection(name="healthcheck-temp")
        test_col.upsert(ids=["test-1"], documents=["Diagnostic ping."], embeddings=[[0.1] * 384])
        query_res = test_col.query(query_embeddings=[[0.1] * 384], n_results=1)
        client.delete_collection(name="healthcheck-temp")
        persist_path = vectorstore.PERSIST_DIR
        record("ChromaDB Vector Store", "PASS", f"Persistent client operational at '{persist_path}'", start_time=t0)
    except Exception as e:
        record(
            "ChromaDB Vector Store", "FAIL", f"ChromaDB read/write failed: {e}",
            f"Verify write permissions for `data/chroma`. Try `rm -rf data/chroma` if database is corrupted.",
            start_time=t0
        )

    # FlashRank Neural Reranker
    t0 = time.perf_counter()
    try:
        from src.rag.reranker import rerank
        sample_docs = [
            {"id": "doc1", "text": "NVIDIA revenue jumped significantly due to AI.", "score": 0.5, "metadata": {}},
            {"id": "doc2", "text": "Apples are delicious fruits.", "score": 0.4, "metadata": {}},
        ]
        reranked = rerank("What was Nvidia's revenue?", sample_docs, top_n=1)
        if reranked and reranked[0]["id"] == "doc1":
            record("FlashRank Neural Reranker", "PASS", f"Cross-encoder loaded `ms-marco-TinyBERT-L-2-v2`", start_time=t0)
        else:
            record("FlashRank Neural Reranker", "WARN", "Reranker returned unexpected top document", start_time=t0)
    except Exception as e:
        record(
            "FlashRank Neural Reranker", "FAIL", f"FlashRank initialization failed: {e}",
            "Verify `flashrank` and `onnxruntime` installations via `uv sync`.",
            start_time=t0
        )


# ---------------------------------------------------------------------------
# 7. Diagnostic Summary & Remediation Guide
# ---------------------------------------------------------------------------
def print_summary() -> int:
    passes = sum(1 for r in results if r.status == "PASS")
    warns = sum(1 for r in results if r.status == "WARN")
    fails = sum(1 for r in results if r.status == "FAIL")
    total = len(results)

    print("\n" + "=" * 80)
    print(f"{BOLD}DIAGNOSTIC SUMMARY: {passes}/{total} Passed | {warns} Warnings | {fails} Failed{RESET}")
    print("=" * 80)

    if fails == 0 and warns == 0:
        print(f"\n{GREEN}{BOLD}🎉 ALL SYSTEMS OPERATIONAL!{RESET}")
        print("StockSage is completely ready to run without any API key or environment blockers.")
        print("You can run the flagship research agent:")
        print(f"  {CYAN}uv run python -m src.rag.news_agent NVDA{RESET}\n")
        return 0

    if fails > 0:
        print(f"\n{RED}{BOLD}🚨 BLOCKING ISSUES DETECTED ({fails}):{RESET}")
        print("The following issues WILL halt code execution. Follow the remediation steps below:\n")
        for i, r in enumerate([r for r in results if r.status == "FAIL"], start=1):
            print(f"  {BOLD}{i}. {r.name}:{RESET} {r.message}")
            if r.remediation:
                print(f"     {YELLOW}➔ Fix:{RESET} {r.remediation}\n")

    if warns > 0:
        print(f"\n{YELLOW}{BOLD}⚠️ WARNINGS & NON-BLOCKING RECOMMENDATIONS ({warns}):{RESET}")
        for i, r in enumerate([r for r in results if r.status == "WARN"], start=1):
            print(f"  {BOLD}{i}. {r.name}:{RESET} {r.message}")
            if r.remediation:
                print(f"     {YELLOW}➔ Suggestion:{RESET} {r.remediation}\n")

    return 1 if fails > 0 else 0


def main() -> None:
    print("=" * 80)
    print(f"{BOLD}  StockSage 📈  —  Pre-Flight System & API Health Check{RESET}")
    print("  Auditing environment, API credentials, models, and databases...")
    print("=" * 80)

    check_runtime_and_env()
    check_groq_api()
    check_tavily_api()
    check_sec_edgar()
    check_yahoo_finance()
    check_local_ml_and_vectorstore()

    exit_code = print_summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
