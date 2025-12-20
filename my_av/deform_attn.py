import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math


class MSDeformableAttention(nn.Module):
    """
    Multi-scale Deformable Attention (practical API).
    value_list: list of feature maps per level: [(B, C, H_l, W_l), ...]
    query: (B, Len_q, C)
    reference_points: (B, Len_q, 2) in normalized coords [0,1] (x, y)
    outputs: (B, Len_q, C)
    """

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int):
        super(MSDeformableAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # Input projection for values
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # Sampling offsets: each query predicts (num_heads * num_levels * num_points * 2) offsets
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        # Attention weights: (num_heads * num_levels * num_points)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Standard Xavier + zero inits
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

        # ---- Initialize sampling_offsets.bias as a small grid around zero ----
        H, L, P = self.num_heads, self.num_levels, self.num_points
        num = H * L * P

        # Build a (num, 2) grid of sinusoidal offsets in Python
        # No tensors yet → no device issues
        
        grid_list = []
        for p in range(P):
            theta = 2 * math.pi * (p / P)
            dx = 0.02 * math.cos(theta)
            dy = 0.02 * math.sin(theta)
            grid_list.append([dx, dy])

        # Repeat for all heads and levels
        grid_list = grid_list * (H * L)  # length = num

        # Convert to tensor on the correct device using new_tensor()
        bias = self.sampling_offsets.bias
        grid = bias.new_tensor(grid_list).view(-1)  # (num*2,)

        if grid.numel() == bias.numel():
            bias.data.copy_(grid)
        else:
            nn.init.constant_(bias, 0.)



    def forward(self, query: torch.Tensor,
                reference_points: torch.Tensor,
                value_list: List[torch.Tensor]) -> torch.Tensor:
        """
        query: (B, Len_q, C)
        reference_points: (B, Len_q, 2) normalized (x, y) in [0,1]
        value_list: list of feature maps per level, each (B, C, H_l, W_l)
        returns: (B, Len_q, C)
        """
        B, Len_q, C = query.shape
        assert C == self.embed_dim
        assert len(value_list) == self.num_levels

        # flatten and project values per level later as needed
        # project query to get sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(query)  # (B, Len_q, num_heads*num_levels*num_points*2)
        attention_weights = self.attention_weights(query)  # (B, Len_q, num_heads*num_levels*num_points)
        attention_weights = attention_weights.view(B, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(B, Len_q, self.num_heads, self.num_levels, self.num_points)

        sampling_offsets = sampling_offsets.view(B, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        # sampling_offsets are predicted in normalized coordinate offsets (we'll treat them as normalized displacement)
        # value projection
        # We'll project each value map's channels into embed and split heads later.
        # Prepare the sampled values accumulator
        outputs = query.new_zeros(B, Len_q, self.embed_dim)

        # project values per level and reshape for grid sampling
        proj_values = []
        for lvl in range(self.num_levels):
            value = value_list[lvl]  # (B, C, H, W)
            Bv, Cv, Hl, Wl = value.shape
            assert Bv == B and Cv == C, f"got {Bv} needed {B}, got {Cv} needed {C} "
            # flatten spatial and project channels in-linear using conv or linear across channels
            # We use a linear projection implemented as 1x1 conv alternative: here use linear applied to channel dim via view
            # Simpler: reshape to (B, H, W, C) and apply value_proj as linear on last dim, then permute back
            value_ = value.permute(0, 2, 3, 1)  # (B, H, W, C)
            value_ = self.value_proj(value_)  # (B, H, W, C)
            value_ = value_.permute(0, 3, 1, 2)  # (B, C, H, W)

            # Reshape value tensor to separate heads: (B, num_heads, head_dim, H, W)
            value_per_head = value_.reshape(B, self.num_heads, C//self.num_heads, Hl, Wl)

            proj_values.append(value_per_head)

        # For each level, use grid_sample to sample points
        # We'll compute for each level the sampling grid in normalized coords [-1,1] for grid_sample.
        # reference_points: (B, Len_q, 2) normalized [0,1] -> convert to normalized coords for grid_sample later
        # Expand reference_points to be compatible with heads/points
        for lvl in range(self.num_levels):
            value_l = proj_values[lvl]  # (B, num_heads, head_dim, H, W)
            B, num_heads, head_dim, H, W = value_l.shape
            # Reshape value tensor for batched sampling: 
            value_for_sampling = value_l.reshape(B * num_heads, head_dim, H, W) # (B * num_heads, head_dim, H, W)

            # prepare grid of sampling locations
            # reference_points: (B, Len_q, 2) -> (B, Len_q, 1, 1, 2)  -> (B, Len_q, num_heads, 1, 2)
            ref = reference_points.view(B, Len_q, 1, 1, 2).expand(-1, -1, num_heads, -1, -1)
            # offsets for this level: sampling_offsets[..., lvl, p, 2]
            offsets_lvl = sampling_offsets[..., lvl, :, :]  # shape (B, Len_q, num_heads, num_points, 2) 
            
            # Create sampling locations:
            sampling_locations = ref + offsets_lvl  # (B, Len_q, num_heads, num_points, 2)

            # Reshape sampling locations for each head
            # (B, Len_q, num_heads, num_points, 2) -> (B, num_heads, Len_q, num_points, 2) 
            sampling_grid = sampling_locations.permute(0, 2, 1, 3, 4)  
            # (B, num_heads, Len_q, num_points, 2) -> (B * num_heads, Len_q, num_points, 2)
            sampling_grid = sampling_grid.reshape(B * num_heads, Len_q, self.num_points, 2)

            # convert [0,1] to [-1,1]
            sampling_grid = 2.0 * sampling_grid - 1.0

            # grid_sample
            sampled = F.grid_sample(
                value_for_sampling,      # (B * num_heads, head_dim, H, W)
                sampling_grid,            # (B * num_heads, Len_q, num_points, 2)
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            # Output: (B * num_heads, head_dim, Len_q, num_points)
            # Reshape back to (B, num_heads, head_dim, Len_q, num_points)
            sampled = sampled.reshape(B, num_heads, head_dim, Len_q, self.num_points)

            # (B, num_heads, head_dim, Len_q, num_points) -> (B, Len_q, num_heads, num_points, head_dim)
            sampled = sampled.permute(0, 3, 1, 4, 2)

            # attention weights for this level
            attn = attention_weights[:, :, :, lvl, :]  # (B, Len_q, num_heads, num_points)
            attn = attn.unsqueeze(-1)  # (B, Len_q, num_heads, num_points, 1)
            # weighted sum across points
            weighted_sampled = (sampled * attn).sum(dim=3)  # (B, Len_q, num_heads, head_dim)
            # concat heads back
            weighted_sampled = weighted_sampled.view(B, Len_q, num_heads * head_dim)  # (B, Len_q, embed_dim)
            outputs += weighted_sampled

        outputs = self.output_proj(outputs)
        return outputs


# Example minimal test / usage
if __name__ == "__main__":
    B = 2
    Len_q = 100
    C = 256
    num_heads = 8
    num_levels = 3
    num_points = 4

    model = MSDeformableAttention(embed_dim=C,
                                  num_heads=num_heads,
                                  num_levels=num_levels,
                                  num_points=num_points)

    # build random multi-scale features
    feat1 = torch.randn(B, C, 64, 64)  # level 0
    feat2 = torch.randn(B, C, 32, 32)  # level 1
    feat3 = torch.randn(B, C, 16, 16)  # level 2
    value_list = [feat1, feat2, feat3]

    query = torch.randn(B, Len_q, C)
    # reference points normalized in [0,1]
    reference_points = torch.rand(B, Len_q, 2)

    out = model(query, reference_points, value_list)
    print("out.shape:", out.shape)  # expect (B, Len_q, C)
