# download_data.py
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

def main():
    # Replace this link with your actual Zenodo direct download link once uploaded
    ZENODO_URL = "https://zenodo.org/records/21978244/files/AlloGNN_Data.zip?download=1"
    
    zip_path = Path("AlloGNN_Data.zip")
    data_dir = Path("data")
    checkpoints_dir = Path("output/checkpoints")
    
    print("==========================================================")
    print("      AlloGNN Data and Checkpoint Acquisition Pipeline    ")
    print("==========================================================")
    
    # 1. Download ZIP file from Zenodo
    if not zip_path.exists():
        print(f"Downloading precomputed dataset and pre-trained checkpoint from Zenodo...")
        print(f"Source URL: {ZENODO_URL}")
        try:
            urllib.request.urlretrieve(ZENODO_URL, zip_path)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            return
    else:
        print("Data ZIP archive already exists locally. Skipping download.")

    # 2. Extract ZIP
    print("\nExtracting precomputed graphs, ESM2 embeddings, and split records...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction completed successfully.")
    except Exception as e:
        print(f"Error extracting data: {e}")
        return

    # 3. Clean up ZIP file
    try:
        os.remove(zip_path)
        print("Cleaned up temporary ZIP archive.")
    except Exception as e:
         print(f"Warning: Failed to clean up ZIP file: {e}")

    print("\n==========================================================")
    print("  Setup successful! Directory structure initialized.")
    print("  You can now run 'python evaluation.py' to verify results.")
    print("==========================================================")

if __name__ == "__main__":
    main()