import pickle
import torch 
import os 
import argparse
import numpy as np 
from config import *
from hashlib import sha256
from dataset import SolidData, ProfileData, LoopData
from model.encoder import SolidEncoder, ProfileEncoder, LoopEncoder


def get_unique_code(codes, is_numpy=True):
    if is_numpy:
        unique_codes = np.unique(codes, axis=0)  
        unique_codes_dict = {}
        for code_idx, code in enumerate(unique_codes):
            key = sha256(np.ascontiguousarray(code).flatten()).hexdigest()
            unique_codes_dict[key] = code_idx
    else:
        code_idx = 0
        unique_codes_dict = {}
        for code in codes:
            code_uid = sha256(np.ascontiguousarray(code).flatten()).hexdigest()
            if code_uid not in unique_codes_dict:
                unique_codes_dict[code_uid] = code_idx
                code_idx += 1
    # Re-assign
    code_unique = [] 
    for code in codes:
        uid = sha256(np.ascontiguousarray(code).flatten()).hexdigest() 
        code_unique.append(unique_codes_dict[uid])
    return code_unique


@torch.no_grad()
def extract_code(args):
    """
    Extract assigned code index per data
    """
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device 

    # Initialize dataset loader
    data_func = {
        "solid": SolidData,
        "profile": ProfileData,
        "loop": LoopData
    }

    data_path = {
        "solid": SOLID_FULL_PATH,
        "profile": PROFILE_FULL_PATH,
        "loop": LOOP_FULL_PATH
    }

    dataset = data_func[args.format](data_path[args.format])
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=False, 
                                             batch_size=1024,
                                             num_workers=6)
    
    # Load model weights
    enc_func = {
        "solid": SolidEncoder,
        "profile": ProfileEncoder,
        "loop": LoopEncoder
    }

    encoder = enc_func[args.format]()
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint, f'enc_epoch_{args.epoch}.pt')))
    encoder = encoder.cuda().eval()

    # Extract codes
    print('Extracting Codes...')
    codes = []
    uid = []
    for param, seq_mask, _, cad_uid in dataloader:
        param = param.cuda()
        seq_mask = seq_mask.cuda()
        _, _, _, code = encoder(param, seq_mask)
        code_index = torch.argmax(code.transpose(0,1), 2).detach().cpu().numpy()
        codes.append(code_index)
        uid.append(cad_uid)
    uid = np.hstack(uid)
    codes = np.vstack(codes)  
    codes_unique = get_unique_code(codes, is_numpy=True) 

    code_dict = {}
    code_dict['unique_num'] = np.max(codes_unique)+1
    code_dict['content'] = {}
    for uid, code_unq in zip(uid, codes_unique):
        code_dict['content'][uid] = code_unq
    
    print(f'[Done] {np.max(codes_unique)+1} Codes Extracted')
    with open(args.checkpoint.split('/')[-1]+'.pkl', "wb") as tf:
        pickle.dump(code_dict, tf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to pretrained model", required=True)
    parser.add_argument("--device", type=str, help="CUDA device", required=True)
    parser.add_argument("--format", type=str, help="Data type", required=True)
    parser.add_argument("--epoch", type=int, help="Pretrained epoch", required=True)
    args = parser.parse_args()
        
    # Start training 
    extract_code(args)