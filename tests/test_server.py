from fastapi.testclient import TestClient

from critic_agent.server import app


def test_health_papers_and_static():
    client = TestClient(app)
    assert client.get("/api/health").json()["ok"] is True
    assert isinstance(client.get("/api/papers").json(), list)
    assert "Critic-Agent Explorer" in client.get("/").text
    assert client.get("/api/run", params={"paper": "../secret.pdf"}).status_code == 404
