#!/bin/bash\

# train full model for conditional generation
python gen/train_full.py --output proj_log/gen_full --batchsize 256 \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --mode cond --device 0