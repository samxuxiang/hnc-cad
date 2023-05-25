from .layers.transformer import *
from .layers.improved_transformer import *
from config import *
from model.network import *
import torch.nn as nn

class CodeModel(nn.Module):

  def __init__(self, num_code):
    super(CodeModel, self).__init__()
    self.embed_dim = CODE_CONFIG['embed_dim']
    self.dropout = CODE_CONFIG['dropout_rate']

    # Position embeddings
    self.pos_embed = PositionalEncoding(max_len=MAX_CODE, d_model=self.embed_dim)

    # Discrete vertex value embeddings
    self.code_embed = Embedder(num_code+CODE_PAD, self.embed_dim)
  
    # Transformer decoder
    decoder_layers = TransformerDecoderLayerImproved_v2(d_model=self.embed_dim,   # no cross attention, uncond AR
                        dim_feedforward= CODE_CONFIG['hidden_dim'],
                        nhead=CODE_CONFIG['num_heads'], dropout=self.dropout)
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, CODE_CONFIG['num_layers'], decoder_norm)
    self.fc = nn.Linear(self.embed_dim, num_code+CODE_PAD)
    

  def forward(self, code):
    """ forward pass """
    if code[0] is None:
      bs = len(code)
      seq_len = 0
    else:
      bs, seq_len = code.shape[0], code.shape[1]

    # Context embedding values
    context_embedding = torch.zeros((bs, 1, self.embed_dim)).cuda() # [bs, 1, dim]
      
    if seq_len > 0:
      embeddings = self.code_embed(code)
      decoder_inputs = torch.cat([context_embedding, embeddings], axis=1) # [bs, seqlen+1, dim]
      decoder_inputs = self.pos_embed(decoder_inputs.transpose(0,1))   # [seqlen+1, bs, dim]
    
    else:
      decoder_inputs = self.pos_embed(context_embedding.transpose(0,1))   # [1, bs, dim]
    
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_inputs.shape[0]).cuda()  # masked with -inf
    decoder_out = self.decoder(tgt=decoder_inputs, memory=None, tgt_mask=nopeak_mask)
    
    # Get logits
    logits = self.fc(decoder_out) 
    return logits.transpose(0,1)
    

  def sample(self, n_samples=10):
    """
    sample from distribution (top-k, top-p)
    """
    top_k = code_top_k
    top_p = code_top_p

    for k in range(MAX_CODE):
        if k == 0:
          v_seq = [None] * n_samples
         
        # pass through decoder
        with torch.no_grad():
          logits = self.forward(code=v_seq)
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


