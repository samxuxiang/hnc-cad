#!/bin/bash\

# sample code-tree & CAD model (visualization)
python gen/rand_gen.py --code_weight proj_log/code  --cad_weight proj_log/cad --output result/sample \
    --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --mode sample --device 0

# sample code-tree & CAD model (evaluation, slow)
# python gen/rand_gen.py --code_weight proj_log/code  --cad_weight proj_log/cad --output result/eval \
#     --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --mode eval --device 0

# convert obj format to stl & step
python gen/convert.py --data_folder result/sample

# visualize CAD 
python gen/cad_img.py --input_dir result/sample --output_dir  result/sample_visual

