#!/bin/bash\

# train code-tree generator 
python gen/train_code.py --output proj_log/code --batchsize 512 \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --device 0

# train CAD model generator
python gen/train_cad.py --output ./proj_log/cad --batchsize 256 \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --device 0

