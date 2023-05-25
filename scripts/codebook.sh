#!/bin/bash

# SOLID CODEBOOK
python codebook/train.py --output proj_log/solid --batchsize 256 --format solid --device 0
python codebook/extract_code.py --checkpoint proj_log/solid --format solid --epoch 250 --device 0 

# PROFILE CODEBOOK
python codebook/train.py --output proj_log/profile --batchsize 256 --format profile --device 0
python codebook/extract_code.py --checkpoint proj_log/profile --format profile --epoch 250 --device 0

# LOOP CODEBOOK
python codebook/train.py --output proj_log/loop --batchsize 256 --format loop --device 0
python codebook/extract_code.py --checkpoint proj_log/loop --format loop --epoch 250 --device 0