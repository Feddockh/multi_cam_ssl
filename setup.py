import os
import subprocess
import sys

DINOV2_REPO_URL = "https://github.com/facebookresearch/dinov2.git"
DINOV2_DIR = "dinov2"

def clone_dinov2():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base_dir, DINOV2_DIR)
    if os.path.exists(target_path):
        print(f"Directory '{DINOV2_DIR}' already exists. Skipping clone.")
    else:
        print(f"Cloning DINOv2 repository into '{target_path}'...")
        try:
            subprocess.check_call(["git", "clone", DINOV2_REPO_URL, target_path])
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while cloning the repository: {e}")
            sys.exit(1)

if __name__ == "__main__":

    # Clone the DINOv2 repository
    clone_dinov2()