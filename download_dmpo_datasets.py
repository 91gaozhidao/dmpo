from huggingface_hub import hf_hub_download
import os

repo_id = "Guowei-Zou/DMPO-datasets"
repo_type = "dataset"

# 保存目录
base_dir = "dmpo_datasets"
os.makedirs(base_dir, exist_ok=True)

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
        "can-img",
        "square-img",
        "transport-img",
    ]
}

files = [
    "train.npz",
    "normalization.npz"
]

for domain, tasks in datasets.items():
    for task in tasks:
        for file in files:
            path = f"{domain}/{task}/{file}"

            print("Downloading:", path)

            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=path,
                local_dir=base_dir,
                local_dir_use_symlinks=False
            )

print("All datasets downloaded to:", base_dir)