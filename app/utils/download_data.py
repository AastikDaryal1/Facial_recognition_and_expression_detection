import os
import subprocess

BUCKET_NAME = "face-emotion-dataset"

def download_dataset():
    if not os.path.exists("data/team_faces"):
        print("Downloading dataset from GCS...")
        subprocess.run([
            "gsutil", "-m", "cp", "-r",
            f"gs://{BUCKET_NAME}", "./data"
        ])
        print("Download complete.")
    else:
        print("Dataset already exists.")