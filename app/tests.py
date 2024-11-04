from fastapi.testclient import TestClient
from .main import app


client = TestClient(app)


def test_replicate_image_generation():
    response = client.post(
        "/generate/",
        json={
            "apply_watermark": True,
            "guidance_scale": 7.5,
            "height": 1024,
            "high_noise_frac": 0.8,
            "lora_scale": 0.6,
            "negative_prompt": "",
            "num_inference_steps": 50,
            "num_outputs": 1,
            "prompt": "An astronaut riding a rainbow unicorn",
            "prompt_strength": 0.8,
            "width": 1024,
        },
    )
    assert response.status_code == 200, response.text


def test_replicate_fine_tune_():
    response = client.post(
        "/fine-tune/",
        json={
            "caption_prefix": "a photo of TOK",
            "file_link": "https://tfm-storage.blr1.cdn.digitaloceanspaces.com/base-training.zip",
            "is_lora": True,
            "max_train_steps": 1000,
            "token_string": "TOK",
            "use_face_detection_instead": False,
        },
    )
    assert response.status_code == 200, response.text
