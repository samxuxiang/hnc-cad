import os
import torch
import argparse
from tqdm import tqdm
from config import *
from dataset import CADData
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
from model.decoder import SketchDecoder, ExtDecoder
from torch.optim.lr_scheduler import LambdaLR


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")

    # Initialize dataset loader
    dataset = CADData(CAD_TRAIN_PATH, args.solid_code, args.profile_code, args.loop_code)
    code_size = dataset.solid_unique_num + dataset.profile_unique_num + dataset.loop_unique_num
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=8)
    
    # Initialize models    
    sketch_dec = SketchDecoder(code_size) 
    sketch_dec = nn.DataParallel(sketch_dec)
    sketch_dec = sketch_dec.to(device).train()

    ext_dec = ExtDecoder(code_size) 
    ext_dec = nn.DataParallel(ext_dec)
    ext_dec = ext_dec.to(device).train()

    params = list(sketch_dec.parameters()) + list(ext_dec.parameters()) 
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, 2000)
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print('Start training...')
    for epoch in range(TOTAL_TRAIN_EPOCH):  
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
  
        for pixel, coord, sketch_mask, pixel_aug, coord_aug, ext, ext_mask, code, code_mask in dataloader:
            pixel = pixel.to(device)
            coord = coord.to(device)
            sketch_mask = sketch_mask.to(device)
            ext = ext.to(device)
            ext_mask = ext_mask.to(device)
            code = code.to(device)
            code_mask = code_mask.to(device)
            pixel_aug = pixel_aug.to(device)
            coord_aug = coord_aug.to(device)

            # Pass through sketch decoder
            sketch_logits = sketch_dec(pixel_aug[:, :-1], coord_aug[:, :-1, :], code, code_mask)
            
            # Pass through extrude decoder
            ext_logits = ext_dec(ext[:, :-1], code, code_mask)

            # Compute loss
            valid_mask =  (~sketch_mask).reshape(-1) 
            sketch_pred = sketch_logits.reshape(-1, sketch_logits.shape[-1]) 
            sketch_gt = pixel.reshape(-1)
            sketch_loss = F.cross_entropy(sketch_pred[valid_mask], sketch_gt[valid_mask])     

            valid_mask =  (~ext_mask).reshape(-1) 
            ext_pred = ext_logits.reshape(-1, ext_logits.shape[-1]) 
            ext_gt = ext.reshape(-1)
            ext_loss = F.cross_entropy(ext_pred[valid_mask], ext_gt[valid_mask])     

            total_loss = sketch_loss + ext_loss 
            
            # logging
            if iters % 20 == 0:
                writer.add_scalar("Loss/Total", total_loss, iters) 
                writer.add_scalar("Loss/sketch", sketch_loss, iters)
                writer.add_scalar("Loss/ext", ext_loss, iters)
                
            # Update model
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)  
            optimizer.step()
            scheduler.step()  # linear warm up to 1e-3
            iters += 1
            progress_bar.update(1)

        progress_bar.close()
        writer.flush()

        # # save model after n epoch
        if (epoch+1) % 50 == 0:
            torch.save(sketch_dec.module.state_dict(), os.path.join(args.output,'sketch_dec_epoch_'+str(epoch+1)+'.pt'))
            torch.save(ext_dec.module.state_dict(), os.path.join(args.output,'ext_dec_epoch_'+str(epoch+1)+'.pt'))
            
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--batchsize", type=int, help="Training batchsize", required=True)
    parser.add_argument("--device", type=str, help="CUDA device", required=True)
    parser.add_argument("--solid_code", type=str, required=True, help='Extracted solid codes')
    parser.add_argument("--profile_code", type=str, required=True, help='Extracted profile codes')
    parser.add_argument("--loop_code", type=str, required=True, help='Extracted loop codes')
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)