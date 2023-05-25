from .layers.transformer import *
from .layers.improved_transformer import *
import torch.nn as nn
import torch
from config import *
from model.network import *


class SolidDecoder(nn.Module):
  """
  Transformer-based decoder for solid
  """

  def __init__(self):
    """
    Initializes model.
    """
    super(SolidDecoder, self).__init__()
    self.embed_dim = DECODER_CONFIG['embed_dim']
    self.param_embed = Embedder(2**BIT, 32)
    self.param_fc = nn.Sequential(
        nn.Linear(32*SOLID_PARAM_SEQ, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.LeakyReLU(),
    )

    self.pos_embed = PositionalEncoding(max_len=MAX_SOLID+1, d_model=self.embed_dim)
    self.mask_token = nn.Parameter(torch.zeros(32))

    layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, nhead=DECODER_CONFIG['num_heads'], 
        dim_feedforward=DECODER_CONFIG['hidden_dim'], dropout=DECODER_CONFIG['dropout_rate'])
    self.network = TransformerDecoder(layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

    self.param_logit1 = nn.Linear(self.embed_dim, 32*SOLID_PARAM_SEQ)
    self.param_logit2 = nn.Linear(32, 2**BIT)

  
  def forward(self, param, seq_mask, ignore_mask, latent_code):
    """ forward pass """
    bs = len(param)
    p_embeds = self.param_embed(param)
    p_embeds[ignore_mask] = self.mask_token # replaced with masked token
    p_embeds = p_embeds.flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))    
    box_embeds = p_embeds
    
    input_embeds = self.pos_embed(torch.cat([latent_code, box_embeds], axis=1).transpose(0,1))
    
    # Pass through decoder 
    seq_mask = torch.cat([(torch.zeros([bs, 1])==1).cuda(), seq_mask], axis=1)
    decoder_out = self.network(tgt=input_embeds, tgt_key_padding_mask=seq_mask, memory=None)
    decoder_out = decoder_out[1:].transpose(1,0) 
    
    param_logits1 = self.param_logit1(decoder_out) 
    param_logits2 = self.param_logit2(param_logits1.view(param_logits1.shape[0], param_logits1.shape[1], SOLID_PARAM_SEQ, -1))
    
    return param_logits2
  

class ProfileDecoder(nn.Module):
  """
  Transformer-based decoder for profile 
  """

  def __init__(self):
    super(ProfileDecoder, self).__init__()
    self.embed_dim = DECODER_CONFIG['embed_dim']
    self.bbox_embed = Embedder(2**BIT,  32)
    self.bbox_fc = nn.Sequential(
        nn.Linear(32*PROFILE_PARAM_SEQ, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.LeakyReLU(),
    )

    self.pos_embed = PositionalEncoding(max_len=MAX_PROFILE+1, d_model=self.embed_dim)
    self.mask_token = nn.Parameter(torch.zeros(32))

    layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, nhead=DECODER_CONFIG['num_heads'], 
        dim_feedforward=DECODER_CONFIG['hidden_dim'], dropout=DECODER_CONFIG['dropout_rate'])
    self.network = TransformerDecoder(layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

    self.param_logit1 = nn.Linear(self.embed_dim, 32*PROFILE_PARAM_SEQ)
    self.param_logit2 = nn.Linear(32, 2**BIT+PROFILE_PARAM_SEQ)
    
                       
  def forward(self, coord, seq_mask, ignore_mask, latent_code):
    """ forward pass """
    bs = len(coord)
    b_embeds = self.bbox_embed(coord)
    b_embeds[ignore_mask] = self.mask_token
    b_embeds = b_embeds.flatten(start_dim=2, end_dim=3)
    bbox_embeds = self.bbox_fc(b_embeds.flatten(0,1)).unflatten(0,(b_embeds.shape[0], b_embeds.shape[1]))       
   
    input_embeds = self.pos_embed(torch.cat([latent_code, bbox_embeds], axis=1).transpose(0,1))

    # Pass through decoder
    seq_mask = torch.cat([(torch.zeros([bs, 1])==1).cuda(), seq_mask], axis=1)
    decoder_out = self.network(tgt=input_embeds, tgt_key_padding_mask=seq_mask, memory=None)
    decoder_out = decoder_out[1:]
    decoder_out = decoder_out.transpose(0,1)
    
    # Logits fc
    param_logits1 = self.param_logit1(decoder_out) 
    param_logits2 = self.param_logit2(param_logits1.view(param_logits1.shape[0], param_logits1.shape[1], PROFILE_PARAM_SEQ, -1))
    return param_logits2
    

class LoopDecoder(nn.Module):
  """
  Transformer-based decoder for loop
  """

  def __init__(self):
    """
    Initializes.
    """
    super(LoopDecoder, self).__init__()
    self.embed_dim = DECODER_CONFIG['embed_dim']
    self.param_embed = Embedder(2**BIT+LOOP_PARAM_PAD, 32)
    self.param_fc = nn.Sequential(
        nn.Linear(32*LOOP_PARAM_SEQ, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.LeakyReLU(),
    )

    self.pos_embed = PositionalEncoding(max_len=MAX_LOOP+1, d_model=self.embed_dim)
    self.mask_token = nn.Parameter(torch.zeros(32))

    layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, nhead=DECODER_CONFIG['num_heads'], 
        dim_feedforward=DECODER_CONFIG['hidden_dim'], dropout=DECODER_CONFIG['dropout_rate'])
    self.network = TransformerDecoder(layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

    self.param_logit1 = nn.Linear(self.embed_dim, 32*LOOP_PARAM_SEQ)
    self.param_logit2 = nn.Linear(32, 2**BIT+LOOP_PARAM_PAD)

  
  def forward(self, coord, seq_mask, ignore_mask, latent_code):
    """ forward pass """
    bs = len(coord)
    p_embeds = self.param_embed(coord)
    p_embeds[ignore_mask] = self.mask_token
    p_embeds = p_embeds.flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))    

    input_embeds = self.pos_embed(torch.cat([latent_code, p_embeds], axis=1).transpose(0,1))
    seq_mask = torch.cat([(torch.zeros([bs, 1])==1).cuda(), seq_mask], axis=1)
  
    decoder_out = self.network(tgt=input_embeds, tgt_key_padding_mask=seq_mask, memory=None)
    decoder_out = decoder_out[1:].transpose(0,1)

    param_logits1 = self.param_logit1(decoder_out) 
    param_logits2 = self.param_logit2(param_logits1.view(param_logits1.shape[0], param_logits1.shape[1], LOOP_PARAM_SEQ, -1))

    return param_logits2