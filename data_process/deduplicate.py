import os
import argparse
from tqdm import tqdm
import multiprocessing
import numpy as np
from hashlib import sha256
import pickle


def hash_model(data):
    cmd = np.hstack(data['cad_cmd'])
    param =  np.vstack(data['cad_param'])
    ext = np.hstack(data['cad_ext'])
    hash_str = sha256(np.ascontiguousarray(ext).flatten()).hexdigest() +'_'+\
        sha256(np.ascontiguousarray(cmd).flatten()).hexdigest() +'_'+\
        sha256(np.ascontiguousarray(param).flatten()).hexdigest()
    uid = data['tmp_uid']
    return hash_str, uid


def hash_loop(data):
    param =  np.vstack(data['param']) 
    cmd = np.hstack(data['cmd'])
    hash_str = sha256(np.ascontiguousarray(param).flatten()).hexdigest() +'_'+\
        sha256(np.ascontiguousarray(cmd).flatten()).hexdigest()
    uid = data['tmp_uid']
    return hash_str, uid


def hash_profile(data):
    bboxes =  np.vstack(data['profile']) 
    hash_str = sha256(np.ascontiguousarray(bboxes).flatten()).hexdigest() 
    uid = data['tmp_uid']
    return hash_str, uid


def hash_solid(data):
    hash_str = sha256(np.ascontiguousarray(data['solid']).flatten()).hexdigest() 
    uid = data['tmp_uid']
    return hash_str, uid


def parallel_hash(data, format):
    """ Parallel hash generated data """
    duplicate_groups = {}
    process_func = {
        "solid": hash_solid,
        "profile": hash_profile,
        "loop": hash_loop,
        "model": hash_model
    }
    num_cpus = multiprocessing.cpu_count()
    objs_iter = multiprocessing.Pool(num_cpus).imap(process_func[format], data)
    for h, uid in tqdm(objs_iter, total=len(data)):
        if len(h)>0:
            if not h in duplicate_groups:
                duplicate_groups[h] = []
            duplicate_groups[h].append([uid])
    return duplicate_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--format", type=str, required=True)
    args = parser.parse_args()

    # Load faces
    with open(os.path.join(args.data_path, 'train.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    # Assign UID
    for idx, data in enumerate(dataset):
        data['tmp_uid'] = idx
 
    # Hash 
    print('Removing Duplicate...')
    gen_len = len(dataset)
    gen_groups = parallel_hash(dataset, args.format)
    
    # Find unique data
    num_files_in_groups = []
    for g in tqdm(gen_groups.values()):
        num_files_in_group = len(g)
        num_files_in_groups.append(num_files_in_group)
    unique_count = np.sum(np.array(num_files_in_groups)==1)
    unique_percent = (unique_count / gen_len) * 100.0

    # Create deduplicate dataste
    unique_uid = {}
    for g in tqdm(gen_groups.values()):
        uid = g[0][0] # only choose one 
        unique_uid[uid] = True
   
    save_dataset = []
    for data in tqdm(dataset):
        uid = data['tmp_uid']
        if uid in unique_uid.keys():
            save_dataset.append(data)
        else:
            pass
    
    with open(os.path.join(args.data_path,"train_deduplicate.pkl"), "wb") as tf:
        pickle.dump(save_dataset, tf)

    print("Duplicate Stats:")
    print(f"\tUnique Percentage: {unique_percent:.2f}%")
    print(f"\tUnique Dataset Length: {len(save_dataset)}")
