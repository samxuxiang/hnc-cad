#!/bin/bash\

# train neural code-tree generator 
python gen/train_code.py --output proj_log/code --batchsize 512 \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --device 0

# train CAD model generator
python gen/train.py --output ./proj_log/cad --batchsize 256 \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --device 0

# 1) sample code-tree, 2) sample CAD model (sketch-and-extrude sequence)
python gen/rand_gen.py --code_weight proj_log/code  --cad_weight proj_log/cad --output proj_log/rand \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --device 0

# convert obj format to stl & step
python gen/convert.py --data_folder proj_log/rand

# visualize CAD 
python gen/cad_img.py --input_dir proj_log/rand --output_dir  proj_log/rand_visual

######## EVALUATION ##########
python gen/sample_points.py --in_dir proj_log/rand --out_dir pcd 
python gen/eval_cad.py --fake proj_log/rand --real data/test_set


