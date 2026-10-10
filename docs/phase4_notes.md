# Phase 4 — Deployment, Full-Stack Architecture & Retrospective (Days 51–65)

> **StockSage Phase 4 Milestone**: Transforming an autonomous multi-agent Python backend into a production-grade, containerized web platform with FastAPI, Streamlit, Docker Compose, GitHub Actions CI, and LangSmith distributed tracing.

---

## 1. Executive Summary & Full-Stack Architecture

Phase 4 bridges the gap between terminal research scripts and a **real, publicly deployable financial technology application**. The complete StockSage system now features:

```mermaid
flowchart TD
    User([End User / Web Browser]) -->|Port 8501| StreamlitUI[Streamlit Interactive Terminal<br/>demo/app.py]
    
    StreamlitUI -->|HTTP POST /v1/analysis<br/>X-API-Key: Header| FastAPIServer[FastAPI Backend Engine<br/>src/api/main.py :8000]
    
    FastAPIServer -->|BackgroundTasks| TaskQueue[Asynchronous Worker Pool]
    TaskQueue --> LangGraphEngine[Multi-Agent LangGraph Pipeline<br/>src/graph/stocksage_graph.py]
    
    LangGraphEngine -->|Real-Time Progress| WebSocket[WebSocket Broadcaster<br/>/v1/ws/{session_id}]
    WebSocket -.->|Live Streaming Events| StreamlitUI
    
    LangGraphEngine -->|State Snapshots| SQLiteStorage[(SQLite Checkpoints & Memory<br/>data/)]
    LangGraphEngine -->|Observability Traces| LangSmith[LangSmith Tracing Platform]
    
    StreamlitUI -->|Poll Status & Fetch Report| FastAPIServer
```

---

## 2. Day-by-Day Implementation Directory

### Day 51: Intro to FastAPI (`src/api/main.py`)
- Created root and source application modules (`src/api/main.py` and `api/main.py`).
- Implemented `GET /health` (`status: ok`) and `GET /ping/{ticker}` (`status: pong`).

### Day 52: Wiring FastAPI to LangGraph
- Implemented `POST /v1/analysis` executing the compiled multi-agent state graph directly.
- Returns synthesized research report and structured metrics in JSON.

### Day 53: Asynchronous Endpoints & Background Tasks
- Solved HTTP timeout issues on long-running multi-agent runs:
  - `POST /v1/analysis` immediately allocates a UUID `session_id` and kicks off `run_analysis_task` in FastAPI `BackgroundTasks`.
  - `GET /v1/analysis/{session_id}` allows non-blocking status polling (`processing`, `pending_approval`, `completed`, `failed`).

### Day 54: WebSocket Live Progress Streaming
- Implemented persistent WebSocket stream at `/v1/ws/{session_id}`.
- Broadcasts granular milestone updates as each agent completes its tasks (`supervisor_started`, `data_agent_completed`, `risk_agent_completed`, `news_agent_completed`, `complete`).

### Days 55–56: Streamlit UI Skeleton & API Client (`demo/app.py`)
- Built an institutional-grade financial dashboard using Streamlit.
- Wired input form to `httpx.Client`, polling `/v1/analysis/{session_id}` every 2 seconds with an interactive progress spinner.

### Day 57: Interactive Plotly Visualizations
- Integrated interactive Plotly charts directly into the Streamlit terminal:
  - **Price Trajectory Chart:** 6-month historical closing price line chart with gradient shading.
  - **RSI Indicator Dial:** Real-time gauge highlighting Oversold ($<30$), Neutral ($30–70$), and Overbought ($>70$) momentum zones.

### Day 58: Human-in-the-Loop Review UI
- Connected the Day 44 LangGraph `interrupt()` gate to the web interface.
- When `status == "pending_approval"`, Streamlit displays intermediate fundamental data and risk metrics, giving the user clickable **"Approve Synthesis"** and **"Request Revision"** controls that post to `/v1/analysis/{session_id}/approval`.

### Day 59: Basic API Key Authentication
- Secured all `/v1/analysis` endpoints with an `X-API-Key` header dependency (`verify_api_key`).
- Unauthenticated requests are rejected immediately with HTTP 401 Unauthorized.

### Day 60: Containerization (`Dockerfile`)
- Authored an optimized, multi-layer `Dockerfile` based on `python:3.12-slim` and `uv`.
- Configured native health checks (`curl -f http://localhost:8000/health`) and non-root execution support.

### Day 61: Docker Compose Full-Stack Deployment (`docker-compose.yml`)
- Multi-service compose specification:
  - `stocksage-api`: FastAPI backend on port 8000.
  - `stocksage-ui`: Streamlit frontend on port 8501.
  - Volume mounting for persistent vector databases and SQLite checkpoints (`./data:/app/data`).

### Day 62: Automated API Test Suite (`tests/test_api.py`)
- Verified all endpoints using FastAPI's `TestClient`:
  - Health check and ping response schema validation.
  - 401 rejection on missing/invalid API keys.
  - Background task session initialization and polling.
  - Human approval state transitions.

### Day 63: GitHub Actions CI Pipeline (`.github/workflows/ci.yml`)
- Continuous integration workflow running on every push and pull request to `main`.
- Automatically checks out code, installs Python 3.12 via `astral-sh/setup-uv`, synchronizes dependencies, and executes all 114 offline unit tests.

### Day 64: LangSmith Observability
- Integrated distributed tracing for every agent forward pass and tool execution.
- Enables monitoring latency, token expenditure, and error trajectories per session at `smith.langchain.com`.

---

## 3. Retrospective Insights: From Terminal to Cloud

1. **Decoupling Compute from Web Workers:** Multi-agent workflows that take 10–25 seconds cannot run inside synchronous HTTP request-response cycles without causing proxy timeouts. Adopting asynchronous background tasks with WebSocket push notifications transformed the user experience.
2. **Container Layer Caching with `uv`:** Traditional `pip install` inside Docker takes several minutes. By copying only `pyproject.toml` and `uv.lock` first and running `uv sync --frozen`, Docker builds take under 15 seconds on cached runs.
3. **Defense-in-Depth Authentication:** Enforcing API keys at the FastAPI dependency layer ensures the expensive LLM agents and web search APIs cannot be drained by unauthorized crawlers or bots.
