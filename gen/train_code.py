import os
import torch
import argparse
from config import *
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F 
from model.code import CodeModel
from dataset import CodeData
from torch.utils.tensorboard import SummaryWriter
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
    dataset = CodeData(CAD_TRAIN_PATH, args.solid_code, args.profile_code, args.loop_code)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=6)
    code_size = dataset.solid_unique_num + dataset.profile_unique_num + dataset.loop_unique_num

    model = CodeModel(code_size)
    model = model.to(device).train()
    
    # Initialize optimizer
    network_parameters = list(model.parameters()) 
    optimizer = torch.optim.AdamW(network_parameters, lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, 2000)
    
    # logging 
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print('Start training...')
    for epoch in range(TOTAL_TRAIN_EPOCH):  
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
            torch.save(model.state_dict(), os.path.join(args.output,'code_epoch_'+str(epoch+1)+'.pt'))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help='Path to save weights and logging')
    parser.add_argument("--batchsize", type=int, required=True, help='Training batchsize')
    parser.add_argument("--device", type=str, required=True, help='Cuda device')
    parser.add_argument("--solid_code", type=str, required=True, help='Extracted solid codes (.pkl)')
    parser.add_argument("--profile_code", type=str, required=True, help='Extracted profile codes (.pkl)')
    parser.add_argument("--loop_code", type=str, required=True, help='Extracted loop codes (.pkl)')
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)



