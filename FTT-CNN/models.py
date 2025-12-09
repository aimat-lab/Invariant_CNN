import torch
import torch.nn as nn

from typing import List, Tuple, Union

from fft_layers import fft_pooling, fft_relu, fft_conv
from utils import fft2
       

class FFT_CNN(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        input_shape: Tuple[int, int],
        conv_units: List[int] = [64, 64, 64], 
        dense_units: List[int] = [64, 32, 2],
        kernel_sizes: Union[List[int],int] = 4,
        bandwidths: List[float] = .5,
        **kwargs,     
    ):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.input_shape = input_shape
        self.conv_units = conv_units
        self.dense_units = dense_units
        
        if isinstance(bandwidths, list):
            self.bandwidths = bandwidths
        else:
            self.bandwidths = [bandwidths**(i+1) for i in range(len(conv_units))]
            
        if isinstance(kernel_sizes, list):
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes]*len(conv_units)

        layers = []
        in_channels = self.input_dim
        sampling_freqs = self.input_shape
        print(f'input: {self.input_shape}')
        for units, kernel, bw in zip(self.conv_units, self.kernel_sizes, self.bandwidths):
            layers.extend(
                [
                    fft_conv(
                        in_channels=in_channels,
                        out_channels=units,
                        kernel_size=kernel,
                        sampling_frequencies=sampling_freqs,
                    ),
                    fft_relu(units=units),
                    fft_pooling(bandwidth=bw),
                ]
            )
            in_channels = units
            sampling_freqs = self._reduce(*sampling_freqs, bw)
            print(f'pooled h,w: {sampling_freqs}')
        self.backbone = nn.Sequential(*layers)
       
        layers = []
        prev_units = sampling_freqs[0]*sampling_freqs[1]*units
        print(f'embedding: {prev_units}')
        for units in dense_units[:-1]:
            layers.extend(
                [
                    nn.Linear(prev_units, units),
                    nn.ReLU(),
                    # nn.Dropout(p=0.2),
                ]
            )
            prev_units = units
        layers.append(nn.Linear(prev_units, dense_units[-1]))
        self.mlp = nn.Sequential(*layers)
    
    @staticmethod
    def _reduce(h, w, bw):
        h = int(bw * (h // 2)) * 2
        w = int(bw * (w // 2)) * 2
        return h, w
    
    def forward(self, X_t: torch.Tensor) -> torch.Tensor:
        batch_dim = X_t.shape[0]
        X_f = self.backbone(fft2(X_t))
        embedding = torch.abs(X_f).reshape(batch_dim, -1)
        out = self.mlp(embedding)
        return out
        



class flip_invariant_FFT_CNN(FFT_CNN):
    def __init__(
        self, 
        input_dim: int, 
        input_shape: Tuple[int, int],
        conv_units: List[int] = [64, 64, 64], 
        dense_units: List[int] = [64, 32, 2],
        kernel_sizes: Union[List[int],int] = 4,
        bandwidths: List[float] = .5,
        **kwargs,
    ):
        super().__init__(
            input_dim, 
            input_shape, 
            conv_units, 
            dense_units, 
            kernel_sizes, 
            bandwidths, 
            **kwargs,
        )
        
    def forward(self, X_t: torch.Tensor) -> torch.Tensor:
        batch_dim = X_t.shape[0]
        
        X_f = self.backbone(fft2(X_t))
        embedding = torch.abs(X_f).reshape(batch_dim, -1)

        X_f_flipped = self.backbone(fft2(torch.flip(X_t, dims=[2])))
        embedding_flipped = torch.abs(X_f_flipped).reshape(batch_dim, -1)
        
        out = self.mlp(embedding+embedding_flipped)
        return out
