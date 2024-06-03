import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type



def check_tensor(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")
    # print(f"{name} - min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}, std: {tensor.std().item()}")

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
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.linear_projection = nn.Linear(257, 256)
        for param in self.linear_projection.parameters():
            param.requires_grad = False
        


    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = self.predict_masks(image_embeddings)
        mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(
            image_embeddings.size(0), -1, -1
        )

        projected_embeddings = self.linear_projection(image_embeddings.transpose(1, 2)).transpose(1, 2)
        check_tensor(projected_embeddings, "Projected Embeddings")
        batch_size, num_tokens, embedding_dim = projected_embeddings.shape

        src = projected_embeddings.transpose(1, 2).view(batch_size, embedding_dim, int(num_tokens ** 0.5), int(num_tokens ** 0.5))
        check_tensor(src, "Source after reshaping")

        upscaled_embedding = self.output_upscaling(src)
        check_tensor(upscaled_embedding, "Upscaled Embedding")

        hyper_in_list: List[torch.Tensor] = [
            self.output_hypernetworks_mlps[i](output_tokens[:, i, :])
            for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        masks = F.relu(masks)
        check_tensor(masks, "Masks")
        return masks

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

