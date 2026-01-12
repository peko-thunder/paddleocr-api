from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ocr_ready" in data
    assert data["version"] == "1.0.0"


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ocr_ready"] is True


def test_ocr_with_invalid_file_type(client):
    response = client.post(
        "/ocr", files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_ocr_with_image(client):
    test_image_path = Path(__file__).parent / "test_images" / "sample_japanese.png"

    if not test_image_path.exists():
        pytest.skip("Test image not found")

    with open(test_image_path, "rb") as f:
        response = client.post(
            "/ocr", files={"file": ("sample_japanese.png", f, "image/png")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "results" in data
    assert "full_text" in data
