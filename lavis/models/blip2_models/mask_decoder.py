import torch
from torch import nn
from torch.nn import functional as F
from typing import Type

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 4, kernel_size=3, padding=1),  # 增加的卷积层
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 8),
            activation(),
            nn.Conv2d(transformer_dim // 8, transformer_dim // 8, kernel_size=3, padding=1),  # 增加的卷积层
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 8, 1, kernel_size=2, stride=2
            ),
        )
        self.linear_projection = nn.Linear(257, 256)
        

    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        projected_embeddings = self.linear_projection(image_embeddings.transpose(1, 2)).transpose(1, 2)
        
        batch_size, num_tokens, embedding_dim = projected_embeddings.shape
        src = projected_embeddings.transpose(1, 2).view(batch_size, embedding_dim, int(num_tokens ** 0.5), int(num_tokens ** 0.5))
        
        upscaled_embedding = self.output_upscaling(src)
        
        masks = F.sigmoid(upscaled_embedding)  
        return masks

