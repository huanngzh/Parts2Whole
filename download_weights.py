from huggingface_hub import snapshot_download

REPO_ID = "huanngzh/Parts2Whole"
snapshot_download(repo_id=REPO_ID, local_dir="pretrained_weights/parts2whole")
