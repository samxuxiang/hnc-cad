from .layers.transformer import *
from .layers.improved_transformer import *
import torch.nn as nn
import torch
import numpy as np
from config import *
from model.network import *
import torch.nn as nn

class SketchDecoder(nn.Module):
  """
  Autoregressive generative model 
  """

  def __init__(self, mode, num_code=None):
    super(SketchDecoder, self).__init__()
    self.embed_dim = DECODER_CONFIG['embed_dim']
    self.coord_embed_x = Embedder(2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.coord_embed_y = Embedder(2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.pixel_embeds = Embedder(2**CAD_BIT * 2**CAD_BIT+SKETCH_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=MAX_CAD, d_model=self.embed_dim)
    self.mode = mode 

    if self.mode == 'uncond': 
      self.code_embed = Embedder(num_code+CODE_PAD, self.embed_dim)
      self.mempos_embed = PositionalEncoding(max_len=MAX_CODE, d_model=self.embed_dim)
    else:
      self.code_embed = Embedder(num_code+CODE_PAD, self.embed_dim)
      self.mempos_embed = PositionalEncoding(max_len=MAX_CODE+MAX_EXT+MAX_CAD, d_model=self.embed_dim)

    layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, nhead=DECODER_CONFIG['num_heads'], 
        dim_feedforward=DECODER_CONFIG['hidden_dim'], dropout=DECODER_CONFIG['dropout_rate'])
    self.network = TransformerDecoder(layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))
 
    self.pixel_logit = nn.Linear(self.embed_dim, 2**CAD_BIT * 2**CAD_BIT+SKETCH_PAD)
    
  
  def forward(self, pixel, coord, code, code_mask, latent_z=None, latent_mask=None):
    """ forward pass """
    if pixel[0] is None:
        bs = len(pixel)
        seqlen = 0
    else:
        bs, seqlen = pixel.shape[0], pixel.shape[1]  
    
    # Context
    context_embeds = torch.zeros((bs, 1, self.embed_dim)).cuda()

    # Token embedding
    if seqlen > 0:   
        coord_embed = self.coord_embed_x(coord[...,0]) + self.coord_embed_y(coord[...,1]) # [bs, vlen, dim]
        pixel_embed = self.pixel_embeds(pixel)
        embed_inputs = pixel_embed + coord_embed 
        decoder_input = self.pos_embed(torch.cat([context_embeds, embed_inputs], dim=1).transpose(0,1))
    else:
        decoder_input = self.pos_embed(context_embeds.transpose(0,1))
  
    # Memory embedding
    if self.mode == 'cond':
      latent_z = torch.cat([latent_z, self.code_embed(code) ], 1)
      memory_embeds = self.mempos_embed(latent_z.transpose(0,1))
      memory_key_padding_mask = torch.cat([latent_mask, code_mask], 1)
    else:
      latent_z = self.code_embed(code) 
      memory_embeds = self.mempos_embed(latent_z.transpose(0,1))
      memory_key_padding_mask = code_mask
   
    # Decoder
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(seqlen+1).cuda()  # masked with -inf  
    decoder_out = self.network(tgt=decoder_input, memory=memory_embeds, memory_key_padding_mask=memory_key_padding_mask, \
                                tgt_mask=nopeak_mask)
    decoder_out = decoder_out.transpose(0,1)
    
    pixel_logits = self.pixel_logit(decoder_out)

    return pixel_logits


  def sample(self, code, code_mask, latent=None, latent_mask=None, top_k=0, top_p=0.95):    
    # Mapping from pixel index to xy coordiante
    pixel2xy = {}
    x=np.linspace(0, 2**CAD_BIT-1, 2**CAD_BIT)
    y=np.linspace(0, 2**CAD_BIT-1, 2**CAD_BIT)
    xx,yy=np.meshgrid(x,y)
    xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
    for pixel, xy in enumerate(xy_grid):
      pixel2xy[pixel] = xy+SKETCH_PAD
    
    pix_samples = []
    xy_samples = []
    latent_samples = []
    code_samples = []
    latent_mask_samples = []
    code_mask_samples = []
    n_samples = len(code)

    for k in range(MAX_CAD):
      if k == 0:
        pixel_seq = [None] * n_samples
        xy_seq = [None] * n_samples
      
      with torch.no_grad():
        p_pred = self.forward(pixel_seq, xy_seq, code, code_mask, latent, latent_mask)
        p_logits = p_pred[:, -1, :]

      next_pixels = []
      for logit in p_logits: 
        filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
        next_pixel = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
        next_pixels.append(next_pixel.item())

      # Convert pixel index to xy coordinate
      next_xys = []
      for pixel in next_pixels:
        if pixel >= SKETCH_PAD:
          xy = pixel2xy[pixel-SKETCH_PAD]
        else:
          xy = np.array([pixel, pixel]).astype(int)
        next_xys.append(xy)
      next_xys = np.vstack(next_xys)  # [BS, 2]
      next_pixels = np.vstack(next_pixels)  # [BS, 1]

      # Add next tokens
      nextp_seq = torch.LongTensor(next_pixels).view(len(next_pixels), 1).cuda()
      nextxy_seq = torch.LongTensor(next_xys).unsqueeze(1).cuda()
      
      if xy_seq[0] is None:
        pixel_seq = nextp_seq
        xy_seq = nextxy_seq
      else:
        pixel_seq = torch.cat([pixel_seq, nextp_seq], 1)
        xy_seq = torch.cat([xy_seq, nextxy_seq], 1)
      
      # Early stopping
      done_idx = np.where(next_pixels==0)[0]

      if len(done_idx) > 0:
        done_pixs = pixel_seq[done_idx] 
        done_xys = xy_seq[done_idx]
        done_code = code[done_idx]
        done_code_mask = code_mask[done_idx]
        
        for pix, xy, _code_, _code_mask_ in zip(done_pixs, done_xys, done_code, done_code_mask):
          pix = pix.detach().cpu().numpy()
          xy = xy.detach().cpu().numpy()
          pix_samples.append(pix)
          xy_samples.append(xy)
          code_samples.append(_code_)
          code_mask_samples.append(_code_mask_)

        if self.mode == 'cond':
          done_latent = latent[done_idx]
          done_latent_mask = latent_mask[done_idx]
          for _latent_, _latent_mask_ in zip(done_latent, done_latent_mask):
            latent_samples.append(_latent_)
            latent_mask_samples.append(_latent_mask_)
        
      left_idx = np.where(next_pixels!=0)[0]
      if len(left_idx) == 0:
        break # no more jobs to do
      else:
        pixel_seq = pixel_seq[left_idx]
        xy_seq = xy_seq[left_idx]
        if self.mode == 'cond':
          latent = latent[left_idx]
          latent_mask = latent_mask[left_idx]
        code = code[left_idx]
        code_mask = code_mask[left_idx]

    if self.mode == 'cond':
      latent_samples = torch.stack(latent_samples)
      latent_mask_samples = torch.stack(latent_mask_samples)
    code_samples = torch.stack(code_samples)
    code_mask_samples = torch.stack(code_mask_samples)
    return xy_samples, code_samples, code_mask_samples,latent_samples, latent_mask_samples



class ExtDecoder(nn.Module):
  """
  Autoregressive generative model 
  """

  def __init__(self, mode, num_code=None):
    super(ExtDecoder, self).__init__()
    self.embed_dim = DECODER_CONFIG['embed_dim']
    self.ext_embed = Embedder(2**CAD_BIT+EXT_PAD, self.embed_dim)
    self.pos_embed = PositionalEncoding(max_len=MAX_EXT, d_model=self.embed_dim)
    self.mode = mode

    layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, nhead=DECODER_CONFIG['num_heads'], 
        dim_feedforward=DECODER_CONFIG['hidden_dim'], dropout=DECODER_CONFIG['dropout_rate'])
    self.network = TransformerDecoder(layers, DECODER_CONFIG['num_layers'], LayerNorm(self.embed_dim))

    if self.mode == 'uncond':
      self.code_embed = Embedder(num_code+CODE_PAD, self.embed_dim)
      self.mempos_embed = PositionalEncoding(max_len=MAX_CODE, d_model=self.embed_dim)
    else:
      self.code_embed = Embedder(num_code+CODE_PAD, self.embed_dim)
      self.mempos_embed = PositionalEncoding(max_len=MAX_CODE+MAX_EXT+MAX_CAD, d_model=self.embed_dim)

    self.logit = nn.Linear(self.embed_dim, 2**CAD_BIT+EXT_PAD)
   
  
  def forward(self, extrude, code, code_mask, latent_z=None, latent_mask=None):
    """ forward pass """
    if extrude[0] is None:
        bs = len(extrude)
        seqlen = 0
    else:
        bs, seqlen = extrude.shape[0], extrude.shape[1]  
    
    # Context
    context_embeds = torch.zeros((bs, 1, self.embed_dim)).cuda()
  
    # Token embedding
    if seqlen > 0:   
        embed_inputs = self.ext_embed(extrude)
        decoder_input = self.pos_embed(torch.cat([context_embeds, embed_inputs], dim=1).transpose(0,1))
    else:
        decoder_input = self.pos_embed(context_embeds.transpose(0,1))
  
    # Memory embedding
    if self.mode == 'cond':
      latent_z = torch.cat([latent_z, self.code_embed(code) ], 1)
      memory_embeds = self.mempos_embed(latent_z.transpose(0,1))
      memory_key_padding_mask = torch.cat([latent_mask, code_mask], 1)
    else:
      latent_z = self.code_embed(code) 
      memory_embeds = self.mempos_embed(latent_z.transpose(0,1))
      memory_key_padding_mask = code_mask

    # Decoder
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(seqlen+1).cuda()  # masked with -inf  
    decoder_out = self.network(tgt=decoder_input, memory=memory_embeds, memory_key_padding_mask=memory_key_padding_mask, tgt_mask=nopeak_mask)
    decoder_out = decoder_out.transpose(0,1)
  
    logits = self.logit(decoder_out)

    return logits


  def sample(self, xy_samples, code, code_mask, latent=None, latent_mask=None, top_k=0, top_p=0.95):    
    # Mapping from pixel index to xy coordiante
    cad_samples = []
    n_samples = len(code)

    for k in range(MAX_EXT):
      if k == 0:
        ext_seq = [None] * n_samples
      
      with torch.no_grad():
        p_pred = self.forward(ext_seq, code, code_mask, latent, latent_mask)
        p_logits = p_pred[:, -1, :]
  
      next_exts = []
      for logit in p_logits: 
        filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
        next_ext = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
        next_exts.append(next_ext.item())
      next_exts = np.vstack(next_exts)  # [BS, 1]

      # Add next tokens
      nextp_seq = torch.LongTensor(next_exts).view(len(next_exts), 1).cuda()
      
      if ext_seq[0] is None:
        ext_seq = nextp_seq
      else:
        ext_seq = torch.cat([ext_seq, nextp_seq], 1)
      
      # Early stopping
      done_idx = np.where(next_exts==0)[0]
      if len(done_idx) > 0:
        done_exts = ext_seq[done_idx] 
        done_xys = [xy_samples[idx] for idx in done_idx]
        for ext, xy in zip(done_exts, done_xys):
          ext = ext.detach().cpu().numpy()
          cad_samples.append([xy, ext])

      left_idx = np.where(next_exts!=0)[0]
      if len(left_idx) == 0:
        break # no more jobs to do
      else:
        ext_seq = ext_seq[left_idx]
        xy_samples = [xy_samples[idx] for idx in left_idx]
        if self.mode == 'cond':
          latent = latent[left_idx]
          latent_mask = latent_mask[left_idx]
        code = code[left_idx]
        code_mask = code_mask[left_idx]
    
    return cad_samples



class CodeDecoder(nn.Module):

  def __init__(self, mode, num_code):
    super(CodeDecoder, self).__init__()
    self.embed_dim = CODE_CONFIG['embed_dim']
    self.dropout = CODE_CONFIG['dropout_rate']
    self.mode = mode

    # Position embeddings
    self.pos_embed = PositionalEncoding(max_len=MAX_CODE, d_model=self.embed_dim)

    # Discrete vertex value embeddings
    self.code_embed = Embedder(num_code+CODE_PAD, self.embed_dim)
  
    # Transformer decoder
    if self.mode == 'uncond':
      decoder_layers = TransformerDecoderLayerImproved_v2(d_model=self.embed_dim,   # no cross attention, uncond AR
                          dim_feedforward= CODE_CONFIG['hidden_dim'],
                          nhead=CODE_CONFIG['num_heads'], dropout=self.dropout)
    else:
      self.mempos_embed = PositionalEncoding(max_len=MAX_EXT+MAX_CAD, d_model=self.embed_dim)
      decoder_layers = TransformerDecoderLayerImproved(d_model=self.embed_dim,   # with cross attention, cond AR
                          dim_feedforward= CODE_CONFIG['hidden_dim'],
                          nhead=CODE_CONFIG['num_heads'], dropout=self.dropout)
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, CODE_CONFIG['num_layers'], decoder_norm)
    self.fc = nn.Linear(self.embed_dim, num_code+CODE_PAD)
    

  def forward(self, code, latent_z=None, latent_mask=None):
    """ forward pass """
    if code[0] is None:
      bs = len(code)
      seq_len = 0
    else:
      bs, seq_len = code.shape[0], code.shape[1]

    # Context embedding values
    context_embedding = torch.zeros((bs, 1, self.embed_dim)).cuda() 
      
    if seq_len > 0:
      embeddings = self.code_embed(code)
      decoder_inputs = torch.cat([context_embedding, embeddings], axis=1) 
      decoder_inputs = self.pos_embed(decoder_inputs.transpose(0,1))   
    
    else:
      decoder_inputs = self.pos_embed(context_embedding.transpose(0,1))  
    
    if self.mode == 'uncond':
      memory_embeds = None
    else:
      memory_embeds = self.mempos_embed(latent_z.transpose(0,1))
    
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_inputs.shape[0]).cuda()  # masked with -inf
    decoder_out = self.decoder(tgt=decoder_inputs, memory=memory_embeds, memory_key_padding_mask=latent_mask, tgt_mask=nopeak_mask)
    
    # Get logits
    logits = self.fc(decoder_out) 
    return logits.transpose(0,1)
    

  def sample(self, n_samples=10, latent_z=None, latent_mask=None, top_k=0, top_p=0.95):
    """
    sample from distribution (top-k, top-p)
    """
    for k in range(MAX_CODE):
        if k == 0:
          v_seq = [None] * n_samples
       
        # pass through decoder
        with torch.no_grad():
          logits = self.forward(code=v_seq, latent_z=latent_z, latent_mask=latent_mask)
          logits = logits[:, -1, :] 
        
        # Top-p sampling 
        next_vs = []
        for logit in logits:   
            filtered_logits = top_k_top_p_filtering(logit.clone(), top_k=top_k, top_p=top_p)
            next_v = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
            next_vs.append(next_v.item())

        # Add next tokens
        next_seq = torch.LongTensor(next_vs).view(len(next_vs), 1).cuda()
        if v_seq[0] is None:
            v_seq = next_seq
        else:
            v_seq = torch.cat([v_seq, next_seq], 1)
    
    return v_seq


