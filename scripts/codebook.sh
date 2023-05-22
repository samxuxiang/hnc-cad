#!/bin/bash

# SOLID CODEBOOK
# python codebook/train.py --output ./proj_log/solid/10000_freq1_rand_threshold7 --batchsize 256 --format solid --device 2
# python codebook/cluster.py --checkpoint ./proj_log/solid/nocmd_10000_freq1_rand_threshold7_short --device 0
# python codebook/extract_code.py --checkpoint ./proj_log/solid/nocmd_10000_freq1_rand_threshold7_short --format solid --device 0

# PROFILE CODEBOOK
# python codebook/train.py --output ./proj_log/profile/5000_freq1_rand_threshold7 --batchsize 256 --format profile --device 0
# python codebook/cluster2.py --checkpoint ./proj_log/profile/5000_freq1_rand_threshold7_short_short --device 0
# python codebook/extract_code.py --checkpoint ./proj_log/profile/5000_freq1_rand_threshold7_short --format profile --device 0

# LOOP CODEBOOK
# python codebook/train.py --output ./proj_log/loop/5000_freq1_rand_threshold7_short_short --batchsize 256 --format loop --device 3
# python codebook/cluster3.py --checkpoint ./proj_log/loop/5000_freq1_rand_threshold7_short --device 2 
# python codebook/extract_code.py --checkpoint ./proj_log/loop/5000_freq1_rand_threshold7_short --format loop --device 0