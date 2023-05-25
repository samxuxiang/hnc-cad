#!/usr/bin/env python3
import gdown
import zipfile
import hashlib
import requests
from pathlib import Path


def download_file(url, target_file_path):
    print(f"Downloading file from [{url}] as [{target_file_path}]")
    req = requests.get(url, allow_redirects=True)
    with open(target_file_path, 'wb') as target_file:
        target_file.write(req.content)


def download():
    download_path = Path("./data")
    if not download_path.is_dir():
        download_path.mkdir(parents=True, exist_ok=True)
    
    # download CAD data
    md5 = "8eb6aa7cf6d4717acc1f7f3ec94a1cb9"
    data_url = "https://drive.google.com/uc?id=1DWlunuHbYT6r1LrPry2vUFODffa5rZUl"
    data_zip_file_name = "cad_raw.zip"
    target_zip_file_path = download_path.joinpath(data_zip_file_name)

    # download requested dataset zip file from Google Drive
    if not target_zip_file_path.is_file():
        gdown.download(data_url, str(target_zip_file_path), quiet=False)
    else:
        print(f"Skipping downloading dataset from Google Drive, file [{target_zip_file_path}] already exists.")
    
    # verify md5
    print("Verifying MD5 hash...")
    assert hashlib.md5(open(target_zip_file_path, 'rb').read()).hexdigest() == md5
   
    # unzip
    print("Unzipping dataset...")
    with zipfile.ZipFile(target_zip_file_path, 'r') as target_ds_zip_file:
        target_ds_zip_file.extractall(download_path)

    print("All Complete...")


def main():
    download()
   

if __name__ == "__main__":
    main()