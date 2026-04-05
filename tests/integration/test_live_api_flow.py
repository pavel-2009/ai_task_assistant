import pytest


@pytest.mark.integration

def test_live_seed_and_ping(integration_httpx_client_a, seeded_tasks_and_index):
    ping = integration_httpx_client_a.get("/ping")
    ping_json = ping.json()

    assert ping.status_code in (200, 503)
    models = ping_json["models"]
    required_services = {
        "embedding",
        "vector_db",
        "semantic_search",
        "ner",
        "rag",
        "data_drift",
    }
    assert required_services.issubset(models.keys())
    assert all(models[service]["ready"] for service in required_services)
    assert seeded_tasks_and_index["count"] == 20
