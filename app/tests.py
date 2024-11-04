from fastapi.testclient import TestClient
from .main import app


client = TestClient(app)


def test_replicate_image_generation():
    response = client.post(
        "/generate/", json={"prompt": "A man Driving a car in patchy roads in india"}
    )
    assert response.status_code == 200, response.text


def test_replicate_fine_tune_():
    response = client.post(
        "/generate/", json={"prompt": "A man Driving a car in patchy roads in india"}
    )
    assert response.status_code == 200, response.text
