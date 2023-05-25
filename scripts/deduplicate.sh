#!/bin/bash

# PROCESS SOLID
python data_process/deduplicate.py --data_path data/solid --format solid

# PROCESS PROFILE
python data_process/deduplicate.py --data_path data/profile --format profile

# PROCESS LOOP
python data_process/deduplicate.py --data_path data/loop --format loop

# PROCESS FULL CAD MODEL
python data_process/deduplicate.py --data_path data/model --format model