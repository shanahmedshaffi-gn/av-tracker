import os
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir) # Removed symlinks arg as deprecated warning suggested
        print("Success.")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")

if __name__ == "__main__":
    os.makedirs("pretrained_models", exist_ok=True)
    
    # Download SepFormer for 2 speakers
    download_model("speechbrain/sepformer-libri2mix", "pretrained_models/sepformer-libri2mix")
    
    # Download SepFormer for 3 speakers
    download_model("speechbrain/sepformer-libri3mix", "pretrained_models/sepformer-libri3mix")
    
    # Download SepFormer for 4 speakers (community model)
    download_model("hahmadraz/sepformer-libri4mix", "pretrained_models/sepformer-libri4mix")