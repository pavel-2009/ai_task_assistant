import pytest


@pytest.mark.integration

def test_live_seed_and_ping(integration_httpx_client_a, seeded_tasks_and_index):
    ping = integration_httpx_client_a.get("/ping")

    assert ping.status_code in (200, 503)
    assert seeded_tasks_and_index["count"] == 20
