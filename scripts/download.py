#!/usr/bin/env python3

import gdown
import zipfile
import hashlib
import requests
import traceback
from pathlib import Path
import pdb 

def download_file(url, target_file_path):
    print(f"Downloading file from [{url}] as [{target_file_path}]")
    req = requests.get(url, allow_redirects=True)
    with open(target_file_path, 'wb') as target_file:
        target_file.write(req.content)


def download():
    download_path = Path("./data")
    if not download_path.is_dir():
        download_path.mkdir(parents=True, exist_ok=True)
    
    # Download CAD data
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
    pdb.set_trace()
    # unzip
    print("Unzipping dataset...")
    with zipfile.ZipFile(target_zip_file_path, 'r') as target_ds_zip_file:
        target_ds_zip_file.extractall(download_path)
  
  
    # # Download pretrained models
    # best_epoch = 585
    # release_url = "https://github.com/threedle/GeoCode/releases/latest/download"
    # if args.model_path:
    #     best_ckpt_file_name = f"procedural_{args.domain}_last_ckpt.zip"
    #     latest_ckpt_file_name = f"procedural_{args.domain}_epoch{best_epoch:03d}_ckpt.zip"
    #     exp_target_dir = model_path.joinpath(f"exp_geocode_{args.domain}")
    #     exp_target_dir.mkdir(exist_ok=True)

    #     best_ckpt_url = f"{release_url}/{best_ckpt_file_name}"
    #     best_ckpt_file_path = exp_target_dir.joinpath(best_ckpt_file_name)
    #     download_file(best_ckpt_url, best_ckpt_file_path)

    #     print(f"Unzipping checkpoint file [{best_ckpt_file_path}]...")
    #     with zipfile.ZipFile(best_ckpt_file_path, 'r') as best_ckpt_file:
    #         best_ckpt_file.extractall(exp_target_dir)

    #     latest_ckpt_url = f"{release_url}/{latest_ckpt_file_name}"
    #     latest_ckpt_file_path = exp_target_dir.joinpath(latest_ckpt_file_name)
    #     download_file(latest_ckpt_url, latest_ckpt_file_path)

    #     print(f"Unzipping checkpoint file [{latest_ckpt_file_path}]...")
    #     with zipfile.ZipFile(latest_ckpt_file_path, 'r') as latest_ckpt_file:
    #         latest_ckpt_file.extractall(exp_target_dir)

    print("Done")


def main():
    download()
   

if __name__ == "__main__":
    main()