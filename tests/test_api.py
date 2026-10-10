"""
tests/test_api.py
------------------
Day 62: API Test Suite for StockSage FastAPI backend.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import SESSIONS, app

client = TestClient(app)
VALID_API_KEY = "stocksage-dev-key-12345"


def test_get_health_endpoint():
    """Step 232 & 280: Verify /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "StockSage API"
    assert "timestamp" in data


def test_get_ping_endpoint():
    """Step 234: Verify /ping/{ticker} echoes ticker back."""
    response = client.get("/ping/aapl")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["status"] == "pong"


def test_create_analysis_missing_api_key_rejected():
    """Step 268 & 280: Requests without X-API-Key are rejected with 401 Unauthorized."""
    response = client.post("/v1/analysis", json={"ticker": "AAPL"})
    assert response.status_code == 401
    assert "Invalid or missing X-API-Key" in response.json()["detail"]


def test_create_analysis_invalid_api_key_rejected():
    """Requests with wrong X-API-Key are rejected with 401."""
    response = client.post(
        "/v1/analysis",
        json={"ticker": "AAPL"},
        headers={"X-API-Key": "wrong-key-xyz"},
    )
    assert response.status_code == 401


def test_create_analysis_with_valid_api_key():
    """Step 280: POST /v1/analysis with valid key launches background task and returns session_id."""
    with patch("src.api.main.run_analysis_task") as mock_task:
        response = client.post(
            "/v1/analysis",
            json={"ticker": "NVDA", "enable_hitl": False},
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["ticker"] == "NVDA"
        assert data["status"] == "processing"


def test_get_analysis_status_not_found():
    """Polling a non-existent session_id returns 404."""
    response = client.get(
        "/v1/analysis/nonexistent-session-123",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert response.status_code == 404


def test_get_analysis_status_success():
    """Step 241: Polling an existing session returns status and details."""
    session_id = "test-session-abc"
    SESSIONS[session_id] = {
        "session_id": session_id,
        "ticker": "AAPL",
        "status": "completed",
        "created_at": 1000.0,
        "completed_at": 1010.0,
        "final_report": "# AAPL Report\nEverything looks good.",
        "stock_data": {"latest_close": 230.0},
        "risk_metrics": {"volatility_annualized": 0.22},
        "news_context": ["[1] News headline"],
        "error_log": [],
    }

    response = client.get(
        f"/v1/analysis/{session_id}",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["ticker"] == "AAPL"
    assert data["status"] == "completed"
    assert "# AAPL Report" in data["final_report"]


def test_submit_approval_endpoint():
    """Step 262: Resume session awaiting human approval."""
    session_id = "test-hitl-session"
    SESSIONS[session_id] = {
        "session_id": session_id,
        "ticker": "MSFT",
        "status": "pending_approval",
        "created_at": 1000.0,
    }

    response = client.post(
        f"/v1/analysis/{session_id}/approval",
        json={"action": "approve", "feedback": "Approved by user"},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "resumed"
    assert data["action"] == "approve"
    assert SESSIONS[session_id]["status"] == "processing"
