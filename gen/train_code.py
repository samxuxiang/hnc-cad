import os
import torch
import argparse
from config import *
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F 
from model.decoder import CodeDecoder
from model.network import schedule_with_warmup
from dataset import CodeData
from torch.utils.tensorboard import SummaryWriter


def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    
    # Initialize dataset loader
    dataset = CodeData(CAD_TRAIN_PATH, args.solid_code, args.profile_code, args.loop_code)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=6)
    code_size = dataset.solid_unique_num + dataset.profile_unique_num + dataset.loop_unique_num
    model = CodeDecoder(args.mode, code_size)
    model = nn.DataParallel(model)
    model = model.to(device).train()
    
    # Initialize optimizer
    network_parameters = list(model.parameters()) 
    optimizer = torch.optim.AdamW(network_parameters, lr=1e-3)
    scheduler = schedule_with_warmup(optimizer, 2000)
    
    # logging 
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print('Start training...')
    for epoch in range(UNCOND_TRAIN_EPOCH):  
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        for code, code_mask in dataloader:
            code = code.cuda()
            code_mask = code_mask.cuda()
            
            logits = model(code[:, :-1])

            valid_mask = (~code_mask).reshape(-1)
            c_pred = logits.reshape(-1, logits.shape[-1]) 
            c_target = code.reshape(-1)
            code_loss = F.cross_entropy(c_pred[valid_mask], c_target[valid_mask])
        
            total_loss = code_loss

            # logging
            if iters % 10 == 0:
                writer.add_scalar("Loss/Total", total_loss, iters)

            # Backprop 
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(network_parameters, max_norm=1.0)  # clip gradient
            optimizer.step()
            scheduler.step()  # linear warm up to 1e-3
            iters += 1
            progress_bar.update(1)

        progress_bar.close()
        writer.flush()

        # save model after n epoch
        if (epoch+1) % 50 == 0:
            torch.save(model.module.state_dict(), os.path.join(args.output,'code_epoch_'+str(epoch+1)+'.pt'))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help='Path to save weights and logging')
    parser.add_argument("--batchsize", type=int, required=True, help='Training batchsize')
    parser.add_argument("--device", type=str, required=True, help='Cuda device')
    parser.add_argument("--solid_code", type=str, required=True, help='Extracted solid codes (.pkl)')
    parser.add_argument("--profile_code", type=str, required=True, help='Extracted profile codes (.pkl)')
    parser.add_argument("--loop_code", type=str, required=True, help='Extracted loop codes (.pkl)')
    parser.add_argument("--mode", type=str, required=True, help='uncond | cond')
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)



