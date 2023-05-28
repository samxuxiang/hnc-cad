#!/bin/bash\

# sample surface point cloud
python gen/sample_points.py --in_dir result/eval --out_dir pcd 

# run evaluation script
python gen/eval_cad.py --fake result/eval --real data/testset


