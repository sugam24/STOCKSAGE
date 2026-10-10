"""
api/main.py
------------
Days 51–54, 58–59: Production FastAPI Backend for StockSage.

Endpoints:
  - GET  /health                      (Public health status)
  - GET  /ping/{ticker}               (Public ticker echo check)
  - POST /v1/analysis                 (Async background analysis trigger, protected by API Key)
  - GET  /v1/analysis/{session_id}    (Poll analysis status and report, protected by API Key)
  - POST /v1/analysis/{session_id}/approval (Human-in-the-Loop review resume)
  - WS   /v1/ws/{session_id}          (Real-time agent progress streaming)

Usage:
    uv run uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Security,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from src.graph.state import create_initial_state
from src.graph.stocksage_graph import build_stocksage_graph

load_dotenv()

EXPECTED_API_KEY = os.getenv("STOCKSAGE_API_KEY", "stocksage-dev-key-12345")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="StockSage Multi-Agent Financial Research API",
    description="Autonomous institutional equity analysis powered by LangGraph and Groq LLMs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-Memory Session & WebSocket Registry
# ---------------------------------------------------------------------------
SESSIONS: dict[str, dict[str, Any]] = {}
ACTIVE_WEBSOCKETS: dict[str, list[WebSocket]] = {}


class ConnectionManager:
    """Manages active WebSockets and broadcasts live event streams."""

    @staticmethod
    async def connect(session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in ACTIVE_WEBSOCKETS:
            ACTIVE_WEBSOCKETS[session_id] = []
        ACTIVE_WEBSOCKETS[session_id].append(websocket)

    @staticmethod
    def disconnect(session_id: str, websocket: WebSocket):
        if session_id in ACTIVE_WEBSOCKETS:
            if websocket in ACTIVE_WEBSOCKETS[session_id]:
                ACTIVE_WEBSOCKETS[session_id].remove(websocket)
            if not ACTIVE_WEBSOCKETS[session_id]:
                del ACTIVE_WEBSOCKETS[session_id]

    @staticmethod
    async def broadcast(session_id: str, event: dict[str, Any]):
        sockets = ACTIVE_WEBSOCKETS.get(session_id, [])
        dead = []
        for ws in sockets:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ConnectionManager.disconnect(session_id, ws)


# ---------------------------------------------------------------------------
# Security: API Key Dependency (Day 59)
# ---------------------------------------------------------------------------

async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """Validates the X-API-Key header against the configured secret."""
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header. Provide a valid StockSage API key.",
        )
    return api_key


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    app: str = "StockSage API"
    timestamp: float


class PingResponse(BaseModel):
    ticker: str
    status: str = "pong"


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL' or 'NVDA'.")
    query: str | None = Field(None, description="Optional custom natural-language question.")
    enable_hitl: bool = Field(False, description="Whether to pause for human approval before final report.")
    sync: bool = Field(False, description="If True, blocks until analysis is complete (Day 52 mode).")


class AnalysisLaunchResponse(BaseModel):
    session_id: str
    ticker: str
    status: str
    created_at: float


class SessionDetailResponse(BaseModel):
    session_id: str
    ticker: str
    status: str  # processing | pending_approval | completed | failed
    final_report: str | None = None
    stock_data: dict[str, Any] | None = None
    risk_metrics: dict[str, Any] | None = None
    news_context: list[str] | None = None
    error_log: list[str] = []
    created_at: float
    completed_at: float | None = None


class ApprovalPayload(BaseModel):
    action: str = Field("approve", description="'approve' or 'revise: <reason>'.")
    feedback: str | None = Field(None, description="Optional revision instructions.")


# ---------------------------------------------------------------------------
# Background Analysis Worker (Days 53 & 54)
# ---------------------------------------------------------------------------

async def run_analysis_task(session_id: str, ticker: str, query: str | None, enable_hitl: bool):
    """Executes the multi-agent graph in the background and broadcasts WS progress events."""
    ticker_clean = ticker.strip().upper()
    user_query = query.strip() if query else f"Comprehensive institutional research report for {ticker_clean}"

    SESSIONS[session_id] = {
        "session_id": session_id,
        "ticker": ticker_clean,
        "status": "processing",
        "created_at": time.time(),
        "final_report": None,
        "stock_data": {},
        "risk_metrics": {},
        "news_context": [],
        "error_log": [],
        "completed_at": None,
    }

    await ConnectionManager.broadcast(session_id, {
        "event": "supervisor_started",
        "message": f"Supervisor analyzing query for {ticker_clean}...",
        "timestamp": time.time(),
    })

    graph = build_stocksage_graph(enable_hitl=enable_hitl)
    initial_state = create_initial_state(ticker=ticker_clean, user_query=user_query)

    try:
        # Step through graph stream to broadcast events per agent
        async for output in graph.astream(initial_state):
            for node_name, node_output in output.items():
                if node_name == "data_agent":
                    SESSIONS[session_id]["stock_data"] = node_output.get("stock_data", {})
                    await ConnectionManager.broadcast(session_id, {
                        "event": "data_agent_completed",
                        "message": "Market fundamentals and technical indicators calculated.",
                        "data": node_output.get("stock_data"),
                    })
                elif node_name == "risk_agent":
                    SESSIONS[session_id]["risk_metrics"] = node_output.get("risk_metrics", {})
                    await ConnectionManager.broadcast(session_id, {
                        "event": "risk_agent_completed",
                        "message": "Quantitative volatility, VaR, and drawdown metrics computed.",
                        "data": node_output.get("risk_metrics"),
                    })
                elif node_name == "news_agent":
                    SESSIONS[session_id]["news_context"] = node_output.get("news_context", [])
                    await ConnectionManager.broadcast(session_id, {
                        "event": "news_agent_completed",
                        "message": "SEC 10-Q regulatory filings and news intelligence retrieved.",
                        "count": len(node_output.get("news_context", [])),
                    })
                elif node_name == "human_review":
                    SESSIONS[session_id]["status"] = "pending_approval"
                    await ConnectionManager.broadcast(session_id, {
                        "event": "pending_approval",
                        "message": "Human approval required to synthesize report.",
                    })
                elif node_name == "analyst_agent":
                    SESSIONS[session_id]["final_report"] = node_output.get("final_report", "")
                    await ConnectionManager.broadcast(session_id, {
                        "event": "analyst_agent_completed",
                        "message": "Executive investment report generated.",
                    })
                elif node_name == "finalize_turn":
                    # Mark complete
                    pass

        SESSIONS[session_id]["status"] = "completed"
        SESSIONS[session_id]["completed_at"] = time.time()
        await ConnectionManager.broadcast(session_id, {
            "event": "complete",
            "message": "Analysis finished successfully.",
            "final_report": SESSIONS[session_id]["final_report"],
        })

    except Exception as exc:
        SESSIONS[session_id]["status"] = "failed"
        SESSIONS[session_id]["error_log"].append(str(exc))
        SESSIONS[session_id]["completed_at"] = time.time()
        await ConnectionManager.broadcast(session_id, {
            "event": "error",
            "message": f"Pipeline failed: {exc}",
        })


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Step 232: Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        app="StockSage API",
        timestamp=time.time(),
    )


@app.get("/ping/{ticker}", response_model=PingResponse, tags=["System"])
def ping_ticker(ticker: str):
    """Step 234: Echo ping endpoint."""
    return PingResponse(ticker=ticker.upper().strip(), status="pong")


@app.post(
    "/v1/analysis",
    response_model=AnalysisLaunchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
)
async def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Steps 236–240: Start a multi-agent analysis session.
    Returns session_id immediately and executes in the background.
    """
    session_id = str(uuid.uuid4())
    ticker = request.ticker.strip().upper()

    if request.sync:
        # Day 52 synchronous mode: execute and wait
        await run_analysis_task(session_id, ticker, request.query, request.enable_hitl)
        return AnalysisLaunchResponse(
            session_id=session_id,
            ticker=ticker,
            status=SESSIONS[session_id]["status"],
            created_at=SESSIONS[session_id]["created_at"],
        )

    # Day 53 async background mode
    background_tasks.add_task(
        run_analysis_task,
        session_id,
        ticker,
        request.query,
        request.enable_hitl,
    )

    return AnalysisLaunchResponse(
        session_id=session_id,
        ticker=ticker,
        status="processing",
        created_at=time.time(),
    )


@app.get(
    "/v1/analysis/{session_id}",
    response_model=SessionDetailResponse,
    tags=["Analysis"],
)
def get_analysis_status(
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Step 241: Poll status or retrieve completed report for session_id."""
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session ID '{session_id}' not found.",
        )
    return SessionDetailResponse(**SESSIONS[session_id])


@app.post(
    "/v1/analysis/{session_id}/approval",
    tags=["Analysis"],
)
async def submit_approval(
    session_id: str,
    payload: ApprovalPayload,
    api_key: str = Depends(verify_api_key),
):
    """Step 262: Human-in-the-Loop review response (Approve or Revise)."""
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session ID '{session_id}' not found.",
        )

    session = SESSIONS[session_id]
    if session["status"] != "pending_approval":
        return {"status": session["status"], "message": "Session is not awaiting approval."}

    # Mark approved and resume
    session["status"] = "processing"
    await ConnectionManager.broadcast(session_id, {
        "event": "resuming_after_approval",
        "action": payload.action,
    })

    return {"status": "resumed", "action": payload.action}


@app.websocket("/v1/ws/{session_id}")
async def websocket_status_stream(websocket: WebSocket, session_id: str):
    """Step 244–245: WebSocket stream pushing real-time agent progression."""
    await ConnectionManager.connect(session_id, websocket)
    try:
        # Send current status on connect
        current = SESSIONS.get(session_id, {"status": "connected"})
        await websocket.send_json({"event": "status_update", "current": current})
        while True:
            # Keep socket open and listen for heartbeat
            await websocket.receive_text()
    except WebSocketDisconnect:
        ConnectionManager.disconnect(session_id, websocket)


def main():
    """CLI runner helper with dynamic PORT support for Railway / Render."""
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
