from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Ministral-8B-Instruct-2410",
    local_dir="ministral_8b",
)
