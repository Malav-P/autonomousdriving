import torch
import pytest
from my_av.deform_attn import MSDeformableAttention  # adjust import if needed

@pytest.fixture
def ms_deformable_attention_inputs():
    B = 2
    Len_q = 100
    C = 256
    num_heads = 8
    num_levels = 3
    num_points = 4

    model = MSDeformableAttention(embed_dim=C, num_heads=num_heads, num_levels=num_levels, num_points=num_points)

    # Multi-scale features
    feat1 = torch.randn(B, C, 64, 64)  # level 0
    feat2 = torch.randn(B, C, 32, 32)  # level 1
    feat3 = torch.randn(B, C, 16, 16)  # level 2
    value_list = [feat1, feat2, feat3]

    # Query
    query = torch.randn(B, Len_q, C)

    # Reference points normalized in [0,1]
    reference_points = torch.rand(B, Len_q, 2)

    return model, query, reference_points, value_list, B, Len_q, C

def test_ms_deformable_attention_forward(ms_deformable_attention_inputs):
    model, query, reference_points, value_list, B, Len_q, C = ms_deformable_attention_inputs

    out = model(query, reference_points, value_list)

    # Check output shape
    assert out.shape == (B, Len_q, C), f"Expected shape {(B, Len_q, C)}, got {out.shape}"

    # Optional: check dtype
    assert out.dtype == query.dtype, "Output dtype does not match query dtype"

    # Optional: check finite values
    assert torch.isfinite(out).all(), "Output contains NaNs or infs"
