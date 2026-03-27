import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from graph_rag.api import app

client = TestClient(app)

@patch("graph_rag.api.GraphRAGPipeline")
def test_query_endpoint(mock_pipeline_class):
    # Setup mock pipeline instance
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = {
        "query": "What is aspirin?",
        "answer": "Aspirin is a drug.",
        "context": "Context about aspirin.",
        "entry_nodes": ["Aspirin"],
        "intent": "drug_info",
        "strategy": "targeted",
        "timing": {"total_ms": 100.0}
    }
    mock_pipeline_class.return_value = mock_pipeline
    
    # Test request
    response = client.post(
        "/query",
        json={"query": "What is aspirin?", "domain": "medical"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is aspirin?"
    assert data["answer"] == "Aspirin is a drug."
    assert "Aspirin" in data["entities"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}

@patch("graph_rag.api.GraphRAGPipeline")
def test_invalid_domain(mock_pipeline_class):
    # Setup mock to raise FileNotFoundError (simulates missing config)
    mock_pipeline_class.side_effect = FileNotFoundError("Config for domain 'non_existent' not found")
    
    response = client.post(
        "/query",
        json={"query": "test", "domain": "non_existent"}
    )
    assert response.status_code == 404
    assert "non_existent" in response.json()["detail"]

