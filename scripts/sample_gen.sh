#!/bin/bash\

# 1) sample code-tree, 2) sample CAD model (sketch-and-extrude sequence)
python gen/rand_gen.py --code_weight proj_log/code  --cad_weight proj_log/cad --output proj_log/rand \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --device 0

# convert obj format to stl & step
python gen/convert.py --data_folder proj_log/rand

# visualize CAD 
python gen/cad_img.py --input_dir proj_log/rand --output_dir  proj_log/rand_visual

