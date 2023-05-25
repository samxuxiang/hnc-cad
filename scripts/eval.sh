#!/bin/bash\

# sample surface point cloud
python gen/sample_points.py --in_dir proj_log/rand --out_dir pcd 

# run evaluation script
python gen/eval_cad.py --fake proj_log/rand --real data/test_set


