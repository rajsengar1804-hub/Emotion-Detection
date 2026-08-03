from huggingface_hub import hf_hub_download
import os

MODEL_NAME = "emotion_final_model.keras"

if not os.path.exists(MODEL_NAME):
    print("Downloading model from Hugging Face...")

    hf_hub_download(
        repo_id="rajsengar9340/emotion_detection",
        filename=MODEL_NAME,
        local_dir="."
    )

    print("Download completed!")