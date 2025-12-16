import torch
from .deform_attn import MSDeformableAttention


class ProposalDeformableSelfAttention(MSDeformableAttention):
    """
    Proposal-anchored deformable self-attention from the planning stage.
    Inputs:
        queries: (B, N, T, C)      - query for (n,t)
        proposals: (B, N, T, 2)     - predicted (x,y) anchor in normalized coords [0,1]

    Output:
        attended_query: (B, N, T, C)
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_points: int):
        """
        Args:
            embed_dim: model embedding dim
            num_heads: number of attention heads
            num_points: number of offset points to sample 
        """
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, num_levels=1, num_points=num_points)

    def forward(self,
                queries,
                proposals,
                query_pos=None,
                identity = None):
        """
        queries : (B, N, T, C)
        proposals: (B, N, T, 2), normalized [0,1]
        query_pos: (B, N, T, C) positional encoding for queries (optional)
        identity : (B, N, T, C) residual connection (optional).
        """

        if query_pos is not None:
            queries = queries + query_pos

        B, N, T, C = queries.shape

        query = queries.flatten(1, 2) # (B, NT, C)
        reference_points = proposals.flatten(1, 2) # (B, NT, 2)
        value_list = [queries.permute(0,3,1,2)] # (B, N, T, C) - > (B, C, N, T)

        out = super().forward(query=query,
                              reference_points=reference_points,
                              value_list=value_list) # (B, N*T, C)
        
        out = out.view(B, N, T, C)

        if identity is not None:
            out += identity

        return out


if __name__ == "__main__":
    B = 2
    N = 4
    T = 6
    C = 256
    num_heads = 8
    num_points = 4

    model = ProposalDeformableSelfAttention(embed_dim=C, num_heads=num_heads, num_points=num_points)

    proposals = torch.rand(B, N, T, 2)
    queries = torch.randn(B, N, T, C)

    out = model(queries, proposals)


    print(out.shape) # expect (B, N, T, C)
