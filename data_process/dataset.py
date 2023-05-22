from utils import process_solid, process_profile, process_loop, process_model
from tqdm import tqdm
import multiprocessing
import json 
from pathlib import Path
from glob import glob
import itertools


class Loader():
    """ Process dataset """
    def __init__(self, datapath, bit, format):
        self.datapath = datapath
        self.bit = bit
        self.format = format

    def load_all_obj(self):
        print(f"Processing {self.format} data...")
        with open('data_process/train_val_test_split.json') as f:
            data_split = json.load(f)
       
        project_folders = []
        for i in range(0, 100):
            cur_dir =  Path(self.datapath) / str(i).zfill(4)
            project_folders += sorted(glob(str(cur_dir)+'/*/'))

        # Parallel process
        iter_data = zip(
            project_folders,
            itertools.repeat(self.bit),
        )

        process_func = {
            "solid": process_solid,
            "profile": process_profile,
            "loop": process_loop,
            "model": process_model
        }

        samples = []
        num_cpus = multiprocessing.cpu_count()
        load_iter = multiprocessing.Pool(num_cpus).imap(process_func[self.format], iter_data)
        for data_sample in tqdm(load_iter, total=len(project_folders)):
            samples += data_sample
        
        print('Splitting data...')
        train_samples = []
        test_samples = []
        val_samples = []
        for data in tqdm(samples):
            if data['name'] in data_split['train']:
                train_samples.append(data)
            elif data['name'] in data_split['test']:
                test_samples.append(data)
            elif data['name'] in data_split['validation']:
                val_samples.append(data)
            else:
                train_samples.append(data) # put into training if no match

        print(f"Data Summary")
        print(f"\tTraining data: {len(train_samples)}")
        print(f"\tValidation data: {len(val_samples)}")
        print(f"\tTest data: {len(test_samples)}")
        return train_samples, test_samples, val_samples

