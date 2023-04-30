# tests/test_fastapi_service.py

import pytest
from httpx import AsyncClient
from parser.service import app

API_KEY = "foobar"
API_URL = "/process"


@pytest.mark.asyncio
async def test_process_logs_success():
    test_data = {
        "log": "INFO: This is a test log message\nDEBUG: This is another log message",
        "window_size": 10,
        "size": 512,
        "model": "bert-base-uncased",
        "show_boundaries": False,
        "tokenize": False,
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(API_URL, json=test_data, headers={"Authorization": f"Bearer {API_KEY}"})

    print("Response body:", response.content)

    assert response.status_code == 200
    assert "results" in response.json()


@pytest.mark.asyncio
async def test_process_logs_invalid_api_key():
    test_data = {
        "log": "INFO: This is a test log message\nDEBUG: This is another log message",
        "window_size": 10,
        "size": 512,
        "model": "bert-base-uncased",
        "show_boundaries": False,
        "tokenize": False,
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(API_URL, json=test_data, headers={"Authorization": f"Bearer invalid-key"})

    assert response.status_code == 401
