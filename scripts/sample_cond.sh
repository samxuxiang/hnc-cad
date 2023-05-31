#!/bin/bash\

python gen/ac_gen.py --output result/ac --weight proj_log/gen_full \
                --solid_code solid.pkl --profile_code  profile.pkl --loop_code loop.pkl --mode cond --device 0

# convert obj format to stl & step
python gen/convert.py --data_folder result/ac

# visualize CAD 
python gen/cad_img.py --input_dir result/ac --output_dir  result/ac_visual