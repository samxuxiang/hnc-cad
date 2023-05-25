import torch.nn as nn
import torch
import torch.nn.functional as F
from config import *
import numpy as np 
from torch.optim.lr_scheduler import LambdaLR


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) 
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

        self.count = np.zeros(self._num_embeddings).astype(int)  # code activated count


    def count_code(self, inputs):
        seqlen, bs = inputs.shape[0], inputs.shape[1]
        
        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        code_dist = distances.min(1).values.reshape(seqlen,bs) # distance to closest code
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        for idx in encoding_indices.detach().cpu().numpy():
            self.count[idx]+=1

        return code_dist


    def reinit(self, code_encoded):
        reinit_idx = np.where(self.count<REINIT_THRESHOLD)[0]
        with torch.no_grad():
            self._embedding.weight[reinit_idx] = code_encoded[:len(reinit_idx)].cuda()
        self.count = np.zeros(self._num_embeddings).astype(int) # reset count to zero
        return len(reinit_idx)


    def forward(self, inputs):
        seqlen, bs = inputs.shape[0], inputs.shape[1]
        
        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
       
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).reshape(seqlen, bs, self._embedding_dim)

        encodings_flat = encodings.reshape(inputs.shape[0], inputs.shape[1], -1)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach() 
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), encodings_flat, encoding_indices


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x):
        return self.embed(x)
    

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



def squared_emd_loss_one_hot_labels(y_pred, y_true, mask=None):
    """
    Squared EMD loss that considers the distance between classes as opposed to the cross-entropy
    loss which only considers if a prediction is correct/wrong.

    Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks.
    Le Hou, Chen-Ping Yu, Dimitris Samaras
    https://arxiv.org/abs/1611.05916

    Args:
        y_pred (torch.FloatTensor): Predicted probabilities of shape (batch_size x ... x num_classes)
        y_true (torch.FloatTensor): Ground truth one-hot labels of shape (batch_size x ... x num_classes)
        mask (torch.FloatTensor): Binary mask of shape (batch_size x ...) to ignore elements (e.g. padded values)
                                  from the loss
    
    Returns:
        torch.tensor: Squared EMD loss
    """
    tmp = torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)
    if mask is not None:
        tmp = tmp * mask
    return torch.sum(tmp) / tmp.shape[0]


def squared_emd_loss(logits, labels, num_classes=-1, mask=None):
    """
    Squared EMD loss that considers the distance between classes as opposed to the cross-entropy
    loss which only considers if a prediction is correct/wrong.

    Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks.
    Le Hou, Chen-Ping Yu, Dimitris Samaras
    https://arxiv.org/abs/1611.05916

    Args:
        logits (torch.FloatTensor): Predicted logits of shape (batch_size x ... x num_classes)
        labels (torch.LongTensor): Ground truth class labels of shape (batch_size x ...)
        mask (torch.FloatTensor): Binary mask of shape (batch_size x ...) to ignore elements (e.g. padded values)
                                  from the loss
    
    Returns:
        torch.tensor: Squared EMD loss
    """
    y_pred = torch.softmax(logits, dim=-1)
    y_true = F.one_hot(labels, num_classes=num_classes).float()
    return squared_emd_loss_one_hot_labels(y_pred, y_true, mask=mask)
