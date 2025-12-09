import torch
from torch import nn
import torch.nn.functional as F    
from typing import Tuple, Optional

from utils import fft2, ifft2


class fft_pooling(nn.Module):
    def __init__(self, bandwidth) -> None:
        super().__init__()
        self.bw = bandwidth

    def forward(self, X_f: torch.Tensor) -> torch.Tensor:
        batch_dim, depth, height, width = X_f.shape
        
        mask = torch.zeros_like(X_f, dtype=torch.bool)
        cutoff_h = int(self.bw * (height // 2))
        cutoff_w = int(self.bw * (width // 2))
        mask[..., :cutoff_h, :cutoff_w] = True  
        mask[..., :cutoff_h, -cutoff_w:] = True  
        mask[..., -cutoff_h:, :cutoff_w] = True  
        mask[..., -cutoff_h:, -cutoff_w:] = True    
        X_f = X_f[mask].reshape(
            batch_dim, 
            depth, 
            cutoff_h * 2, 
            cutoff_w * 2,
        )
        return X_f



class fft_relu(nn.Module):
    def __init__(self, units: Optional[int] = None) -> None:
        super().__init__()
        if units is not None:
            self.batchnorm = nn.BatchNorm2d(units)
        else:
            self.batchnorm = None
            
    def forward(self, X_f: torch.Tensor) -> torch.Tensor:
        X_t = ifft2(X_f)
        X_t = F.relu(X_t)
        if self.batchnorm is not None:
            X_t = self.batchnorm(X_t)
        X_f = fft2(X_t)
        return X_f


class fft_conv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size,
        sampling_frequencies: Tuple,
    ) -> None:
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, dtype=torch.complex64)
        self.params = nn.Parameter(conv.weight)  # shape = out_channels, 1, k, k 
        self.sampling_frequencies = sampling_frequencies
        
    def forward(self, X_f: torch.Tensor) -> torch.Tensor:
        filters = torch.fft.fft2(self.params, s=self.sampling_frequencies) 
        X_f = (filters*X_f.unsqueeze(1)).sum(dim=2)
        return X_f