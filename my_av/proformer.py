import torch
import torch.nn as nn

from .temporal_self_attn import ProposalDeformableSelfAttention
from .spatial_cross_attn import ProposalSpatialCrossAttention
from .pos_encoding import LearnedPositionalEncoding
from .mlp import MLP, FFN
from .utils.utils import proposal_to_anchor

class ProFormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_proposals: int,
                 num_forecast_steps: int,
                 embed_dim: int,
                 num_heads: int,
                 num_levels: int,
                 num_points: int,
                 ffn_dim: int,
                 mlp_dim: int,
                 state_dim: int = 3,
                 shared: bool = True):
        super(ProFormer, self).__init__()

        self.num_proposals = num_proposals
        self.num_forecast_steps = num_forecast_steps

        self.learned_pos_encoding = LearnedPositionalEncoding(num_feats=embed_dim // 2, 
                                                              row_num_embed=num_proposals,
                                                              col_num_embed=num_forecast_steps)
        
        if shared:
            layer = ProFormerLayer(embed_dim=embed_dim,
                                   num_heads=num_heads,
                                   num_levels=num_levels,
                                   num_points=num_points,
                                   ffn_dim=ffn_dim,
                                   mlp_dim=mlp_dim,
                                   state_dim=state_dim)
            
            self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([
                ProFormerLayer(embed_dim=embed_dim,
                               num_heads=num_heads,
                               num_levels=num_levels,
                               num_points=num_points,
                               ffn_dim=ffn_dim,
                               mlp_dim=mlp_dim,
                               state_dim=state_dim)
                for _ in range(num_layers)
            ])

    def forward(self,
                ego_features: torch.tensor,
                image_feats: list,
                **kwargs):

        
        B, C = ego_features.shape # ego_features: (B, C)
        mask = ego_features.new_zeros(size=(B, self.num_proposals, self.num_forecast_steps), dtype=torch.bool) # (B, N, T)

        bev_pos = self.learned_pos_encoding(mask).permute(0, 2, 3, 1) # (B, C, N, T) -> (B, N, T, C)
        queries = bev_pos + ego_features.view(B, 1, 1, C) # (B, N, T, C)

        _, N, T, _ = queries.shape
        K = len(self.layers)
        props = torch.empty(size=(B, K, N, T, 3), device=queries.device)
        for k, layer in enumerate(self.layers):
            queries, proposals = layer(queries=queries,
                                       image_feats=image_feats,
                                       bev_pos=bev_pos,
                                       **kwargs) # (B, N, T, C) , (B, N, T, 3)

            props[:, k, ...] = proposals


        return queries, props # (B, N, T, C) , (B, K, N, T, 3)
        

class ProFormerLayer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_levels: int,
                 num_points: int,
                 ffn_dim: int,
                 mlp_dim: int,
                 state_dim: int = 3):
        super(ProFormerLayer, self).__init__()
        

        self.layers = nn.ModuleList([
            ProposalDeformableSelfAttention(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            num_points=num_points),
            nn.LayerNorm(embed_dim),
            ProposalSpatialCrossAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          num_levels=num_levels,
                                          num_points=num_points),
            nn.LayerNorm(embed_dim),
            FFN(input_dim=embed_dim,
                hidden_dim=ffn_dim,
                output_dim=embed_dim),
            nn.LayerNorm(embed_dim)
        ])

        self.mlp = MLP(input_dim=embed_dim,
                       hidden_dim=mlp_dim,
                       output_dim=state_dim) # predict (x,y,heading) offsets


    def forward(self,
                queries: torch.tensor,
                image_feats: list,
                bev_pos: torch.tensor = None,
                **kwargs):

        proposals = self.mlp(queries) # (B, N, T, 3)
        # Temporal Self-Attention
        queries = self.layers[0](queries, proposals[..., :2], query_pos=bev_pos) # (B, N, T, C)

        # norm
        queries = self.layers[1](queries)

        # Spatial Cross-Attention
        identity = queries # residual connection
        img_coords, mask_view = proposal_to_anchor(proposals, **kwargs) # (B, N, T, 4, num_points_in_pillar, D, 2), (B, N, T, D)
        queries = self.layers[2](queries,
                                 img_coords,
                                 image_feats, 
                                 mask_view=mask_view, 
                                 identity=identity) # (B, N, T, C)

        # norm
        queries = self.layers[3](queries)

        # FFN
        identity = queries # residual connection
        queries = self.layers[4](queries, identity=identity) # (B, N, T, C)

        # norm
        queries = self.layers[5](queries)

        return queries, proposals