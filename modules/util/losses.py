import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(x, x_decode, mu, logvar):
    x = x.contiguous().view(x.size(0), -1)
    x_decode = x_decode.contiguous().view(x_decode.size(0), -1)
    BCE = F.binary_cross_entropy(x_decode, x, reduction='sum')
    # BCE = F.binary_cross_entropy_with_logits(x_decode, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD
