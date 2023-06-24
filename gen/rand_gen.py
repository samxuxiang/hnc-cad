import os
import torch
import argparse
from tqdm import tqdm
from config import *
import multiprocessing
from hashlib import sha256
import numpy as np 
from dataset import CADData
from model.decoder import SketchDecoder, ExtDecoder, CodeDecoder
from utils import CADparser, write_obj_sample


def raster_cad(data): 
    (coord, ext), uid = data
    parser = CADparser(CAD_BIT)
    try:
        parsed_data = parser.perform(coord, ext)
        return [parsed_data], [uid]
    except Exception as error_msg:  
        return [], []


def pad_code(total_code):
    keys = np.ones(len(total_code))
    padding = np.zeros(MAX_CODE-len(total_code)).astype(int)  
    total_code = np.concatenate([total_code, padding], axis=0)
    seq_mask = 1-np.concatenate([keys, padding]) == 1   
    return total_code, seq_mask


def hash_sketch(sketch, ext):
    hash_str = sha256(np.ascontiguousarray(sketch).flatten()).hexdigest() +'_'+\
        sha256(np.ascontiguousarray(ext).flatten()).hexdigest()
    return hash_str


def parse_aug(args):
    total_size = {
        "sample": RANDOM_SAMPLE_TOTAL,
        "eval": RANDOM_EVAL_TOTAL,
    }
    bsz_size = {
        "sample": RANDOM_SAMPLE_BS,
        "eval": RANDOM_EVAL_BS,
    }
    code_top_p = {
        "sample": code_top_p_sample,
        "eval": code_top_p_eval,
    }
    cad_top_p = {
        "sample": cad_top_p_sample,
        "eval": cad_top_p_eval,
    }
    return total_size[args.mode], bsz_size[args.mode], code_top_p[args.mode], cad_top_p[args.mode]


@torch.inference_mode()
def sample(args):
    total_size, bsz, code_top_p, cad_top_p = parse_aug(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    dataset = CADData(CAD_TRAIN_PATH, args.solid_code, args.profile_code, args.loop_code, mode='uncond')
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=False, 
                                             batch_size=32,
                                             num_workers=4)
    code_size = dataset.solid_unique_num + dataset.profile_unique_num + dataset.loop_unique_num
    
    # Load model weights
    sketch_dec = SketchDecoder(mode='uncond', num_code=code_size) 
    sketch_dec.load_state_dict(torch.load(os.path.join(args.cad_weight, 'sketch_dec_epoch_350.pt')))
    sketch_dec = sketch_dec.cuda().eval()

    ext_dec = ExtDecoder(mode='uncond', num_code=code_size) 
    ext_dec.load_state_dict(torch.load(os.path.join(args.cad_weight, 'ext_dec_epoch_350.pt')))
    ext_dec = ext_dec.cuda().eval()

    code_dec = CodeDecoder(mode='uncond', num_code=code_size)
    code_dec.load_state_dict(torch.load(os.path.join(args.code_weight, 'code_epoch_350.pt')))
    code_dec = code_dec.cuda().eval()

    # Random sampling 
    sample_cad = []
    while len(sample_cad) < total_size:
        # Sample code
        codes = code_dec.sample(n_samples=bsz, latent_z=None, latent_mask=None, top_k=0, top_p=code_top_p).detach().cpu().numpy() 
        codes_pad = []
        codes_pad_mask = []
        for code in codes:
            if len(np.where(code==0)[0])==0:
                continue
            code = code[:np.where(code==0)[0][0]+1]
            code, code_mask = pad_code(code)
            codes_pad.append(code)
            codes_pad_mask.append(code_mask)
        codes_pad = torch.LongTensor(np.vstack(codes_pad)).cuda()
        codes_pad_mask = torch.BoolTensor(np.vstack(codes_pad_mask)).cuda()

        # Sample CAD
        xy_samples, _code_, _codes_pad_mask_, _, _ = sketch_dec.sample(codes_pad, codes_pad_mask, 
                                                                              latent=None, latent_mask=None,top_k=0, top_p=cad_top_p)
        cad_samples = ext_dec.sample(xy_samples, _code_, _codes_pad_mask_, 
                                        latent=None, latent_mask=None, top_k=0, top_p=cad_top_p)

        sample_cad+=cad_samples
        print(len(sample_cad))

    # Raster CAD
    iter_data = zip(
        sample_cad,
        np.arange(len(sample_cad)),
    )
    processed_cads = []
    num_cpus = multiprocessing.cpu_count()
    load_iter = multiprocessing.Pool(num_cpus).imap(raster_cad, iter_data) 
    for data_sample, data_uid in load_iter:
        processed_cads += data_sample

    # Save as obj format
    for cad_name, cad_obj in enumerate(processed_cads):
        save_folder = os.path.join(result_folder, str(cad_name).zfill(4))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        write_obj_sample(save_folder, cad_obj)
    
    # Novel / Unique
    if args.mode == 'eval':
        duplicate_groups = {}
        with tqdm(dataloader, unit="batch") as batch_data:
            for _, coord, sketch_mask,_,_, ext, ext_mask,_,_ in batch_data:
                for idx in range(len(coord)):
                    xy_sample = coord[idx][~sketch_mask[idx]].detach().cpu().numpy()
                    ext_sample = ext[idx][~ext_mask[idx]].detach().cpu().numpy()
                    uid = hash_sketch(xy_sample, ext_sample)
                    if uid not in duplicate_groups:
                        duplicate_groups[uid] = [1]
                    else:
                        pass

        for sample in sample_cad: 
            xy_sample = sample[0]
            ext_sample = sample[1]
            uid = hash_sketch(xy_sample, ext_sample)
            if uid not in duplicate_groups:
                duplicate_groups[uid] = [2]
            else:
                duplicate_groups[uid].append(2)

        # Compute Novel and Unique scores
        non_novel_count = 0 
        gen_count = 0
        non_unique_count = 0
        for _, value in duplicate_groups.items():
            gen_count += (np.array(value)==2).sum()
            if 2 in value and 1 in value:
                non_novel_count += (np.array(value)==2).sum()
            if 2 in value:
                non_unique_count += (np.array(value)==2).sum()-1
        novel_pec = (gen_count-non_novel_count)/gen_count
        unique_pec = (gen_count-non_unique_count)/gen_count
        print(f'Novel:{novel_pec}')
        print(f'Unique:{unique_pec}')

     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_weight", type=str, help="Pretrained code-tree model", required=True)
    parser.add_argument("--cad_weight", type=str, help="Pretrained CAD model", required=True)
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--device", type=str, help="CUDA Device Index", required=True)
    parser.add_argument("--mode", type=str, required=True, help="eval | sample")
    parser.add_argument("--solid_code", type=str, required=True)
    parser.add_argument("--profile_code", type=str, required=True)
    parser.add_argument("--loop_code", type=str, required=True)
    args = parser.parse_args()

    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    sample(args)
