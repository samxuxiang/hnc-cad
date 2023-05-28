#!/bin/bash

# SOLID CODEBOOK
python codebook/extract_code.py --checkpoint proj_log/solid --format solid --epoch 250 --device 0

# PROFILE CODEBOOK
python codebook/extract_code.py --checkpoint proj_log/profile --format profile --epoch 250 --device 0

# LOOP CODEBOOK
python codebook/extract_code.py --checkpoint proj_log/loop --format loop --epoch 250 --device 0