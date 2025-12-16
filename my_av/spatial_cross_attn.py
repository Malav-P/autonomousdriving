import torch

from .deform_attn import MSDeformableAttention


class ProposalSpatialCrossAttention(MSDeformableAttention):
    """
    Proposal-anchored deformable self-attention from the planning stage.
    Inputs:
        queries: (B, N, T, C)      - query for (n,t)
        anchor_xy: (B, N, T, 4, num_points_in_pillar, D, 2)     - predicted (x,y) anchor in normalized coords [0,1]
        I : image features
    Output:
        attended_query: (B, N*T, C)
    """

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int):
        """
        Args:
            embed_dim: model embedding dim
            num_heads: number of attention heads
            num_levels: number of levels in feature pyramid
            num_points: number of offset points to sample 
        """
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points)

    def forward(self,
                queries,
                anchor_xy,
                I,
                query_pos=None,
                mask_view=None,
                identity = None):
        """
        queries: (B, N, T, C)      - query for (n,t)
        anchor_xy: (B, N, T, 4, num_points_in_pillar, D, 2)     - predicted (x,y) anchor in normalized coords [0,1]]
        I : image features. I is a list. Each item in I is a list of length self.num_levels
        query_pos: (B, N, T, C) positional encoding for queries (optional)
        mask_view: (B, N, T, D)   - visibility mask for each camera view
        identity : (B, N, T, C) residual connection (optional). If None, use queries as identity
        """


        B, N, T, C = queries.shape
        _, _, _, _, num_points_in_pillar, D, _ = anchor_xy.shape

        if query_pos is not None:
            queries = queries + query_pos


        if mask_view is not None:
            # Expand mask_view to match anchor_xy shape for broadcasting
            mask_expanded = mask_view[:, :, :, None, None, :, None]  # (B, N, T, 1, 1, D, 1)

            # In-place assignment
            anchor_xy.masked_fill_(~mask_expanded, -2.0)  # Set to -2.0 where mask is False this falls outside [-1,1] range so grid sample will give zeros

            Vhit = mask_view.sum(dim=-1)  # (B, N, T)
            inv_Vhit = torch.where(Vhit != 0, 1.0 / Vhit, torch.zeros_like(Vhit))


        query = queries.flatten(1, 2) # (B, NT, C)

        out = query.new_zeros(query.shape)

        for z in range(num_points_in_pillar):
            for j in range(4):
                for i in range(D):
                    reference_points = anchor_xy[..., j, z, i, :].flatten(1, 2) # (B, N, T, 2) - > (B, NT, 2)
                    value_list = I[i]
                    out += super().forward(query=query,
                                        reference_points=reference_points,
                                        value_list=value_list) # (B, N*T, C)
                    

        out = out.view(B, N, T, C)

        if mask_view is not None:
            out *= inv_Vhit[..., None]  # inv_Vhit is (B, N, T, 1)

        if identity is not None:
            out += identity

        return out 


if __name__ == "__main__":
    torch.manual_seed(0)

    # -------------------------------
    # Hyperparameters for the test
    # -------------------------------
    B = 2
    N = 3
    T = 2
    C = 64
    num_heads = 8
    num_levels = 2
    num_points = 4
    num_points_in_pillar = 5
    D = 3

    # -------------------------------
    # Instantiate module
    # -------------------------------
    attn = ProposalSpatialCrossAttention(
        embed_dim=C,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points
    )

    # -------------------------------
    # Fake inputs
    # -------------------------------
    queries = torch.randn(B, N, T, C)
    anchor_xy = torch.rand(B, N, T, 4, num_points_in_pillar, D, 2)

    # Image features
    H1, W1 = 32, 32
    H2, W2 = 16, 16
    I = []
    for cam in range(D):
        per_level = [torch.randn(B, C, H1, W1), torch.randn(B, C, H2, W2)]
        I.append(per_level)

    # -------------------------------
    # Test 1: Normal case (no mask)
    # -------------------------------
    out = attn(queries, anchor_xy, I)
    print("Output shape (normal):", out.shape)

    # -------------------------------
    # Test 2: mask_view all False
    # -------------------------------
    mask_view = torch.randint(0, 1, size=(B, N, T, D), dtype=torch.bool)  # all False
    anchor_xy_test = anchor_xy.clone()  # avoid modifying original anchors

    out_masked = attn(queries, anchor_xy_test, I, mask_view=mask_view)

    # print("Output with mask_view all False:", out_masked)
    print("Max value should be 0:", out_masked.max().item())
    print("Min value should be 0:", out_masked.min().item())

    # Verify
    assert torch.all(out_masked == 0), "Output is not zero when mask_view is all False!"
    print("Test passed: output is zero when mask_view is all False")
