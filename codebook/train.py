import os
import torch
import argparse
from tqdm import tqdm
from config import * 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import SolidData, ProfileData, LoopData
from model.encoder import SolidEncoder, ProfileEncoder, LoopEncoder
from model.decoder import SolidDecoder, ProfileDecoder, LoopDecoder
from model.network import get_constant_schedule_with_warmup, squared_emd_loss


def parse_aug(format):
    """
    Find the corresponding function to run
    """
    data_func = {
        "solid": SolidData,
        "profile": ProfileData,
        "loop": LoopData
    }[format]

    data_path = {
        "solid": SOLID_TRAIN_PATH,
        "profile": PROFILE_TRAIN_PATH,
        "loop": LOOP_TRAIN_PATH
    }[format]

    enc_func = {
        "solid": SolidEncoder,
        "profile": ProfileEncoder,
        "loop": LoopEncoder
    }[format]

    dec_func = {
        "solid": SolidDecoder,
        "profile": ProfileDecoder,
        "loop": LoopDecoder
    }[format]

    return data_func, data_path, enc_func, dec_func



def train(args):
    # gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    data_func, data_path, enc_func, dec_func = parse_aug(args.format)
    
    # Initialize dataset loader
    dataset = data_func(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=6)
    
    # Initialize models
    encoder = enc_func()
    encoder = encoder.cuda().train()

    decoder = dec_func()
    decoder = decoder.cuda().train()

    params = list(decoder.parameters()) + list(encoder.parameters()) 
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, 2000)
    writer = SummaryWriter(log_dir=args.output)
    
    # Main training loop
    iters = 0
    print('Start training...')
    for epoch in range(TOTAL_TRAIN_EPOCH):  
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for param, seq_mask, ignore_mask,_ in dataloader:
            param = param.cuda()
            seq_mask = seq_mask.cuda()
            ignore_mask = ignore_mask.cuda()
           
            # Pass through encoder 
            latent_code, vq_loss, selection, _ = encoder(param, seq_mask)
            
            # Pass through decoder
            param_logits = decoder(param, seq_mask, ignore_mask, latent_code)
            
            # Compute loss                                    
            param_loss = squared_emd_loss(logits=param_logits, 
                                            labels=param, 
                                            num_classes=param_logits.shape[-1], 
                                            mask=ignore_mask)

            total_loss = param_loss + vq_loss

            # logging
            if iters % 10 == 0:
                writer.add_scalar("Loss/Total", total_loss, iters) 
                writer.add_scalar("Loss/param", param_loss, iters) 
                writer.add_scalar("Loss/vq", vq_loss, iters) 

            if iters % 20 == 0 and selection is not None:
                writer.add_histogram('selection', selection, iters)
        
            # Update model
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)  # clip gradient
            optimizer.step()
            scheduler.step()  # linear warm up to 1e-3
            iters += 1
            progress_bar.update(1)

        progress_bar.close()
        writer.flush()

        # Re-init codebook 
        if epoch<REINIT_TRAIN_EPOCH:
            # Compute cluster data count & data to cluster distance
            code_encoded = []
            for param, seq_mask, _,_ in dataloader:
                param = param.cuda()
                seq_mask = seq_mask.cuda()
                with torch.no_grad():
                    _, _code_encoded_ = encoder.count_code(param, seq_mask)
                    code_encoded.append(_code_encoded_.reshape(-1,256).detach().cpu())
          
            code_encoded = torch.vstack(code_encoded)
            code_encoded = code_encoded[torch.randperm(code_encoded.size()[0])] # random shuffle
            reinit_count = encoder.codebook.reinit(code_encoded)
            # print(f'{reinit_count} Codes Reinitialied')

        # Save model after n epoch
        if (epoch+1) % 50 == 0:
            torch.save(encoder.state_dict(), os.path.join(args.output,'enc_epoch_'+str(epoch+1)+'.pt'))
            torch.save(decoder.state_dict(), os.path.join(args.output,'dec_epoch_'+str(epoch+1)+'.pt'))
       
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output folder to save the data", required=True)
    parser.add_argument("--batchsize", type=int, help="Batch size", required=True)
    parser.add_argument("--device", type=str, help="CUDA device", required=True)
    parser.add_argument("--format", type=str, help="Data type", required=True)
    args = parser.parse_args()

    # Create training folder
    result_folder = args.output
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    # Start training 
    train(args)