#!/bin/bash\

# train code-tree generator 
python gen/train_code.py --output proj_log/gen_code --batchsize 512 \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --mode uncond --device 0,1

# train CAD model generator
python gen/train_cad.py --output ./proj_log/gen_cad --batchsize 512 \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --mode uncond --device 0,1
