import os
import zipfile
from huggingface_hub import snapshot_download

REPO_ID = "huanngzh/DeepFashion-MultiModal-Parts2Whole"
LOCAL_DIR = "data/DeepFashion-MultiModal-Parts2Whole"

os.makedirs(os.path.dirname(LOCAL_DIR), exist_ok=True)
snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR)
print(f"Downloaded {REPO_ID} to {LOCAL_DIR}")

# Unzip the downloaded zip files in the LOCAL_DIR
print(f"Unzipping zip files in {LOCAL_DIR}...")
for file in os.listdir(LOCAL_DIR):
    if file.endswith(".zip"):
        zip_path = os.path.join(LOCAL_DIR, file)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(LOCAL_DIR)
        print(f"Unzipped {zip_path} to {os.path.splitext(zip_path)[0]}")
