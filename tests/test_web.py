"""Unit tests for FastAPI endpoints and Web dashboard."""

import pytest
from starlette.testclient import TestClient
from draftcraft.web.app import app

client = TestClient(app)


def test_dashboard_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "DraftCraft" in response.text
    assert "EAGLE & Speculative Engine" in response.text


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_api_info():
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert "pytorch_version" in data
    assert "cuda_available" in data


def test_api_verify_tokenizer():
    payload = {
        "target_vocab_size": 128256,
        "draft_vocab_size": 128256,
        "target_hidden_dim": 4096,
        "head_hidden_dim": 1024,
        "is_eagle": True,
    }
    response = client.post("/api/verify-tokenizer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "fully_aligned"
    assert data["overlap_ratio"] == 1.0


def test_api_profile():
    payload = {
        "alpha": 0.75,
        "lookahead_depth": 3,
        "draft_latency_ms": 3.0,
        "target_latency_ms": 25.0,
        "domain": "general",
        "num_steps": 20,
    }
    response = client.post("/api/profile", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["theoretical_speedup"] > 1.0
    assert len(data["acceptance_by_position"]) == 3


def test_api_matrix_eval():
    payload = {
        "domain": "code",
        "draft_latency_ms": 3.0,
        "target_latency_ms": 25.0,
        "min_lookahead": 1,
        "max_lookahead": 4,
    }
    response = client.post("/api/matrix-eval", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["points"]) > 0
    assert data["peak_speedup"] > 1.2
    assert data["recommended_depth"] >= 1


def test_api_train_start_and_status():
    payload = {
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.001,
        "hidden_dim": 32,
        "target_dim": 64,
        "vocab_size": 100,
        "num_samples": 8,
    }
    response = client.post("/api/train/start", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert job_id is not None

    status_resp = client.get(f"/api/train/status/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] in ["running", "completed"]
