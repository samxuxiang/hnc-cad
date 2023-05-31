#!/bin/bash\

# sample surface point cloud (please convert obj format to stl & step first)
python gen/sample_points.py --in_dir result/random_eval --out_dir pcd 

# run evaluation script
CUDA_VISIBLE_DEVICES=0 python gen/eval_cad.py --fake result/random_eval --real data/testset


