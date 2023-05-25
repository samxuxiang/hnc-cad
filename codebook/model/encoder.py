import torch.nn as nn
from .layers.transformer import *
from .layers.improved_transformer import *
from config import *
from model.network import *


class SolidEncoder(nn.Module):

  def __init__(self):
    """
    Initializes Solid Model.
    """
    super(SolidEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.param_embed = Embedder(2**BIT, 32)
    self.param_fc = nn.Sequential(
        nn.Linear(32*SOLID_PARAM_SEQ, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.LeakyReLU(),
    )

    self.pos_embed = PositionalEncoding(max_len=MAX_SOLID, d_model=self.embed_dim)
    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'], 
                                             dim_feedforward=ENCODER_CONFIG['hidden_dim'], dropout=ENCODER_CONFIG['dropout_rate'])
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, ENCODER_CONFIG['num_layers'], encoder_norm)

    commitment_cost = 0.25
    decay = 0.99
    self.code_dim = 256
    self.codebook = VectorQuantizerEMA(SOLID_CODEBOOK_DIM, self.code_dim, commitment_cost, decay)

    self.bottleneck = nn.Sequential(
        nn.Linear(self.embed_dim, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.Tanh()
    )
    

  def forward(self, param, seq_mask):
    """ forward pass """
    p_embeds = self.param_embed(param).flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))    
    box_embeds = p_embeds
    encoder_input = self.pos_embed(box_embeds.transpose(0,1))
    
    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)  

    # Avg pool latent code 
    z_mask = (~seq_mask).transpose(0,1).unsqueeze(2).repeat(1,1,self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    vq_loss, quantized, encodings_flat, selection = self.codebook(code_encoded)
    latent_code = quantized.transpose(0,1)
    
    return latent_code, vq_loss, selection, encodings_flat


  def count_code(self, param, seq_mask):
    """ Codebook usage """
    p_embeds = self.param_embed(param).flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))    
    box_embeds = p_embeds
    encoder_input = self.pos_embed(box_embeds.transpose(0,1))
    
    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)  

    # Avg pool latent code 
    z_mask = (~seq_mask).transpose(0,1).unsqueeze(2).repeat(1,1,self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    code_dist = self.codebook.count_code(code_encoded)
    
    return code_dist, code_encoded
  

class ProfileEncoder(nn.Module):

  def __init__(self):
    """
    Initializes Profile Model.
    """
    super(ProfileEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.bbox_embed = Embedder(2**BIT,  32)
    self.bbox_fc = nn.Sequential(
        nn.Linear(32*PROFILE_PARAM_SEQ, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.LeakyReLU(),
    )
   
    self.pos_embed = PositionalEncoding(max_len=MAX_PROFILE, d_model=self.embed_dim)

    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'], 
                                             dim_feedforward=ENCODER_CONFIG['hidden_dim'], dropout=ENCODER_CONFIG['dropout_rate'])
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, ENCODER_CONFIG['num_layers'], encoder_norm)

    commitment_cost = 0.25
    decay = 0.99
    self.code_dim = self.embed_dim
    self.codebook = VectorQuantizerEMA(PROFILE_CODEBOOK_DIM, self.code_dim, commitment_cost, decay)

    self.bottleneck = nn.Sequential(
        nn.Linear(self.embed_dim, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.Tanh()
    )
    

  def forward(self, coord, seq_mask):
    """ forward pass """
    p_embed = self.bbox_embed(coord).flatten(start_dim=2, end_dim=3)
    coord_embed = self.bbox_fc(p_embed.flatten(0,1)).unflatten(0,(p_embed.shape[0], p_embed.shape[1]))    
    encoder_input = self.pos_embed(coord_embed.transpose(0,1))

    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)  
    
    # Avg pool latent code 
    z_mask = (~seq_mask).transpose(0,1).unsqueeze(2).repeat(1,1,self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    vq_loss, quantized, encodings_flat, selection = self.codebook(code_encoded)
    latent_code = quantized.transpose(0,1)

    return latent_code, vq_loss, selection, encodings_flat


  def count_code(self, coord, seq_mask):
    """ Codebook usage """
    p_embed = self.bbox_embed(coord).flatten(start_dim=2, end_dim=3)
    coord_embed = self.bbox_fc(p_embed.flatten(0,1)).unflatten(0,(p_embed.shape[0], p_embed.shape[1]))    
    encoder_input = self.pos_embed(coord_embed.transpose(0,1))
   
    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)  
    
    # Avg pool latent code 
    z_mask = (~seq_mask).transpose(0,1).unsqueeze(2).repeat(1,1,self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    code_dist = self.codebook.count_code(code_encoded)

    return code_dist, code_encoded
  

class LoopEncoder(nn.Module):

  def __init__(self):
    """
    Initializes Loop Model.
    """
    super(LoopEncoder, self).__init__()
    self.embed_dim = ENCODER_CONFIG['embed_dim']
    self.param_embed = Embedder(2**BIT+LOOP_PARAM_PAD, 32)
    self.param_fc = nn.Sequential(
        nn.Linear(32*LOOP_PARAM_SEQ, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.LeakyReLU(),
    )

    self.pos_embed = PositionalEncoding(max_len=MAX_LOOP, d_model=self.embed_dim)

    encoder_layers = TransformerEncoderLayerImproved(d_model=self.embed_dim, nhead=ENCODER_CONFIG['num_heads'], 
                                             dim_feedforward=ENCODER_CONFIG['hidden_dim'], dropout=ENCODER_CONFIG['dropout_rate'])
    encoder_norm = LayerNorm(self.embed_dim)
    self.encoder = TransformerEncoder(encoder_layers, ENCODER_CONFIG['num_layers'], encoder_norm)

    commitment_cost = 0.25
    decay = 0.99
    self.code_dim = 256
    self.codebook = VectorQuantizerEMA(LOOP_CODEBOOK_DIM, self.code_dim, commitment_cost, decay)

    self.bottleneck = nn.Sequential(
        nn.Linear(self.embed_dim, self.embed_dim),
        nn.BatchNorm1d(self.embed_dim),
        nn.Tanh()
    )
    

  def forward(self, coord, seq_mask):
    """ forward pass """
    p_embeds = self.param_embed(coord).flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))    
    encoder_input = self.pos_embed(p_embeds.transpose(0,1))
    
    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask) 
   
    # Avg pool latent code 
    z_mask = (~seq_mask).transpose(0,1).unsqueeze(2).repeat(1,1,self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)
  
    vq_loss, quantized, encodings_flat, selection = self.codebook(code_encoded)
    latent_code = quantized.transpose(0,1)
    return latent_code, vq_loss, selection, encodings_flat


  def count_code(self, coord, seq_mask):
    """ Codebook usage """
    p_embeds = self.param_embed(coord).flatten(start_dim=2, end_dim=3)
    p_embeds = self.param_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))    
    encoder_input = self.pos_embed(p_embeds.transpose(0,1))
    
    # Pass through encoder
    outputs = self.encoder(src=encoder_input, src_key_padding_mask=seq_mask)  

    # Avg pool latent code 
    z_mask = (~seq_mask).transpose(0,1).unsqueeze(2).repeat(1,1,self.embed_dim)
    avg_z = (outputs * z_mask).sum(dim=0, keepdim=False) / z_mask.sum(dim=0, keepdim=False)
    code_encoded = self.bottleneck(avg_z).unsqueeze(0)

    code_dist = self.codebook.count_code(code_encoded)
    
    return code_dist, code_encoded