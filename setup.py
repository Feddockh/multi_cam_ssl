import os
import subprocess
import sys
import zipfile
import gdown


DINOV2_REPO_URL = "https://github.com/facebookresearch/dinov2.git"
DINOV2_DIR = "dinov2"

GOOGLE_DRIVE_ZIP_URL = "https://drive.google.com/file/d/1QGvp8OITsTZmngmOifnyMlVrkXU74AiO/view?usp=drive_link"
ZIP_FILENAME = "image_data.zip"
DATA_DIR = ZIP_FILENAME.replace(".zip", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clone_dinov2():
    """
    This function clones the DINOv2 repository from GitHub into the current directory and
    checks if the directory already exists to avoid re-cloning.
    """
    target_path = os.path.join(BASE_DIR, DINOV2_DIR)
    if os.path.exists(target_path):
        print(f"Directory '{DINOV2_DIR}' already exists so skipping clone")
        return
    
    # Try to clone the repository
    print(f"Cloning DINOv2 repository into '{target_path}'")
    try:
        subprocess.check_call(["git", "clone", DINOV2_REPO_URL, target_path])
        print("Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while cloning the repository: {e}")
        sys.exit(1)

def download_and_extract_zip():
    """
    This function downloads the image data zip file from Google Drive and extracts it.
    The zip file is then deleted after extraction.
    The extracted files are placed in the 'image_data' directory.
    If the directory already exists, it skips the download and extraction process.
    """
    zip_path = os.path.join(BASE_DIR, ZIP_FILENAME)
    extract_dir = os.path.join(BASE_DIR, DATA_DIR)
    if os.path.exists(extract_dir):
        print(f"Data directory '{DATA_DIR}' already exists so skipping download and extraction")
        return
    
    # Download the zip file if it doesn't exist
    if not os.path.exists(zip_path):
        print("Downloading zipped data file from Google Drive")
        gdown.download(GOOGLE_DRIVE_ZIP_URL, zip_path, quiet=False, fuzzy=True)
    else:
        print(f"Zip file '{ZIP_FILENAME}' already exists so skipping download")
    
    # Extract the zip file into DATA_DIR
    print(f"Extracting '{ZIP_FILENAME}' to '{DATA_DIR}'")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print("Extraction complete.")

    # Delete the zip file after extraction
    os.remove(zip_path)
    print(f"Deleted zip file '{ZIP_FILENAME}' after extraction.")

def main():
    clone_dinov2()
    download_and_extract_zip()
    print("Setup complete. You can now use the DINOv2 repository and the image data.")

if __name__ == "__main__":
    main()