import torch
import pytest

from my_av.spatial_cross_attn import ProposalSpatialCrossAttention

@pytest.fixture
def fake_inputs():
    # torch.manual_seed(0)

    B, N, T, C = 2, 3, 2, 64
    num_heads = 8
    num_levels = 2
    num_points = 4
    num_points_in_pillar = 5
    D = 3

    # Queries
    queries = torch.randn(B, N, T, C)

    # Anchors
    anchor_xy = torch.rand(B, N, T, 4, num_points_in_pillar, D, 2)

    # Image features
    H1, W1 = 32, 32
    H2, W2 = 16, 16
    I = []
    for cam in range(D):
        per_level = [torch.randn(B, C, H1, W1), torch.randn(B, C, H2, W2)]
        I.append(per_level)

    return queries, anchor_xy, I, B, N, T, C, D, num_points_in_pillar

@pytest.fixture
def attn_module(fake_inputs):
    queries, *_ = fake_inputs
    C = queries.shape[-1]
    attn = ProposalSpatialCrossAttention(
        embed_dim=C,
        num_heads=8,
        num_levels=2,
        num_points=4
    )
    return attn

def test_forward_normal(attn_module, fake_inputs):
    queries, anchor_xy, I, B, N, T, C, D, num_points_in_pillar = fake_inputs
    out = attn_module(queries, anchor_xy.clone(), I)
    assert out.shape == (B, N, T, C), "Output shape is incorrect for normal forward pass"

def test_forward_mask_all_false(attn_module, fake_inputs):
    queries, anchor_xy, I, B, N, T, C, D, num_points_in_pillar = fake_inputs
    # mask_view all False
    mask_view = torch.zeros(B, N, T, D, dtype=torch.bool)
    out_masked = attn_module(queries, anchor_xy.clone(), I, mask_view=mask_view)
    # Output should be all zeros
    assert torch.all(out_masked == 0), "Output is not zero when mask_view is all False"
