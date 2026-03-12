import os
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

# -------- Hugging Face download behavior --------
# 官方支持的环境变量：
# - HF_HUB_DOWNLOAD_TIMEOUT
# - HF_HUB_ETAG_TIMEOUT
# - HF_HUB_DISABLE_XET
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

repo_id = "Guowei-Zou/DMPO-datasets"
repo_type = "dataset"

base_dir = Path("dmpo_datasets")
base_dir.mkdir(parents=True, exist_ok=True)

datasets = {
    "gym": [
        "hopper-medium-v2",
        "walker2d-medium-v2",
        "ant-medium-expert-v2",
        "Humanoid-medium-v3",
        "kitchen-complete-v0",
        "kitchen-mixed-v0",
        "kitchen-partial-v0",
    ],
    "robomimic": [
        "lift-img",
        "lift",
        "can-img",
        "can",
        "square-img",
        "square",
        "transport-img",
        "transport",
    ],
}

files = [
    "train.npz",
    "normalization.npz",
]


def already_exists(local_root: Path, relative_path: str) -> bool:
    target = local_root / relative_path
    return target.exists() and target.is_file()


def download_with_retry(
    repo_id: str,
    repo_type: str,
    filename: str,
    local_dir: str,
    max_retries: int = 5,
) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=local_dir,
            )
        except Exception as e:
            last_err = e
            print(f"[WARN] download failed ({attempt}/{max_retries}) for {filename}: {e}")
            if attempt < max_retries:
                sleep_s = 5 * attempt
                print(f"[INFO] retrying in {sleep_s}s ...")
                time.sleep(sleep_s)
    raise last_err


failed = []

for domain, tasks in datasets.items():
    for task in tasks:
        for file in files:
            relative_path = f"{domain}/{task}/{file}"
            local_path = base_dir / relative_path

            if already_exists(base_dir, relative_path):
                print(f"[SKIP] exists: {relative_path}")
                continue

            print(f"[DOWNLOADING] {relative_path}")
            try:
                saved_path = download_with_retry(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    filename=relative_path,
                    local_dir=str(base_dir),
                    max_retries=5,
                )
                print(f"[OK] saved to: {saved_path}")
            except Exception as e:
                print(f"[ERROR] failed: {relative_path}")
                print(f"        reason: {e}")
                failed.append(relative_path)

print(f"\nAll done. Base dir: {base_dir.resolve()}")
if failed:
    print("\nFailed files:")
    for x in failed:
        print(f" - {x}")
else:
    print("No failed files.")