from unittest.mock import patch
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "RAG Assistant API"}


@patch("app.api.main.run_agent_chat")
def test_agent_chat_endpoint(mock_run_chat):
    mock_run_chat.return_value = {
        "answer": "Тестова відповідь мок-агента",
        "used_tools": ["search_pdf_documents"]
    }

    payload = {
        "session_id": "test_session_123",
        "message": "Що описує документ?"
    }
    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_session_123"
    assert data["answer"] == "Тестова відповідь мок-агента"
    assert "search_pdf_documents" in data["used_tools"]
    mock_run_chat.assert_called_once()


@patch("app.api.main.ingest_pdf")
def test_upload_pdf_endpoint(mock_ingest):
    mock_ingest.return_value = {
        "status": "success",
        "filename": "sample.pdf",
        "chunks_created": 4,
        "collection_name": "docs"
    }

    files = {"file": ("sample.pdf", b"%PDF-1.4 test body", "application/pdf")}
    response = client.post("/api/v1/upload", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_created"] == 4