#!/bin/bash

# SOLID CODEBOOK
python codebook/train.py --output proj_log/solid --batchsize 256 --format solid --device 0

# PROFILE CODEBOOK
python codebook/train.py --output proj_log/profile --batchsize 256 --format profile --device 0

# LOOP CODEBOOK
python codebook/train.py --output proj_log/loop --batchsize 256 --format loop --device 0
