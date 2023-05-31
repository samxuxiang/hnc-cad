import torch.nn as nn
import torch
import torch.nn.functional as F
from .layers.transformer import *
from .layers.improved_transformer import *
from config import *
from model.network import *


class SketchEncoder(nn.Module):
  """
  Transformer Encoder 
  """
  def __init__(self):
    super(SketchEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.coord_embed_x = Embedder(2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.coord_embed_y = Embedder(2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.pixel_embeds = Embedder(2**CAD_BIT * 2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=MAX_CAD, d_model=self.embed_dim)
    layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'], 
        dim_feedforward=ENCODER_CONFIG['hidden_dim'], dropout=ENCODER_CONFIG['dropout_rate'])
    self.encoder = TransformerEncoder(layers, ENCODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))
   
  
  def forward(self, pixel, coord, mask):
    """ forward pass """
    coord_embed = self.coord_embed_x(coord[...,0]) + self.coord_embed_y(coord[...,1]) # [bs, vlen, dim]
    pixel_embed = self.pixel_embeds(pixel)
    embed_inputs = pixel_embed + coord_embed 
    input_embeds = self.pos_embed(embed_inputs.transpose(0,1))
    outputs = self.encoder(src=input_embeds, src_key_padding_mask=mask)  # [seq_len, bs, dim]    
    return outputs.transpose(0,1)



class ExtEncoder(nn.Module):
  """
  Transformer Encoder 
  """
  def __init__(self):
    super(ExtEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.ext_embed = Embedder(2**CAD_BIT+EXT_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=MAX_EXT, d_model=self.embed_dim)
    layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'], 
        dim_feedforward=ENCODER_CONFIG['hidden_dim'], dropout=ENCODER_CONFIG['dropout_rate'])
    self.encoder = TransformerEncoder(layers, ENCODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))
   

  def forward(self, extrude, mask):
    """ forward pass """
    embed_inputs = self.ext_embed(extrude)
    input_embeds = self.pos_embed(embed_inputs.transpose(0,1))
    outputs = self.encoder(src=input_embeds, src_key_padding_mask=mask)  # [seq_len, bs, dim]    
    return outputs.transpose(0,1)
