import torch
import pytest
import numpy as np

from my_av.proformer import ProFormer 

@pytest.fixture
def proformer_inputs():

    B = 2
    N = 6      # num proposals
    T = 4      # forecast steps
    C = 32     # embed dim
    L = 3      # num layers
    D = 3      # num cameras

    # Create model
    model = ProFormer(
        num_layers=L,
        num_proposals=N,
        num_forecast_steps=T,
        embed_dim=C,
        num_heads=4,
        num_levels=3,
        num_points=4,
        ffn_dim=64,
        mlp_dim=64,
        state_dim=3,
        shared=True
    )

    # Ego features
    ego_features = torch.randn(B, C)

    # Fake multi-scale image features
    # Image features
    H1, W1 = 32, 32
    H2, W2 = 16, 16
    H3, W3 = 8, 8
    img_feats = []
    for cam in range(D):
        per_level = [torch.randn(B, C, H1, W1), torch.randn(B, C, H2, W2), torch.randn(B, C, H3, W3)]
        img_feats.append(per_level)


    ############################################ GENERATE CAMERA INTRINSICS and EXTRINSICS
    # Random translations
    translations = (torch.rand(D, 3) * 2 - 1) 

    # Random quaternions
    q = torch.randn(D, 4)
    q = q / q.norm(dim=1, keepdim=True)

    # Stack as (x, y, z, qw, qx, qy, qz)
    extrinsics = torch.cat([translations, q[:, :1], q[:, 1:]], dim=1).unsqueeze(0) # (D, 9) - > (1, D, 9) batch dim

    # Intrinsics
    img_shapes = torch.tensor([(900, 1600)] * D)
    cx = torch.empty(D, 1).uniform_(300, 500)
    cy = torch.empty(D, 1).uniform_(300, 500)
    distortion = torch.randn(D, 5) * 0.01

    intrinsics = torch.cat([img_shapes, cx, cy, distortion], dim=1).unsqueeze(0) # (D, 7) - > (1, D, 7) batch dim
    print(extrinsics.shape)
    print(intrinsics.shape)
    ######################################################################


    # kwargs for proposal_to_anchor
    kwargs = dict(
        half_width=torch.tensor([0.5]),
        half_length=torch.tensor([1.0]),
        rear_axle_to_center=torch.tensor([0.5]),
        zmin=0.0,
        zmax=1.0,
        num_points_in_pillar=4,
        pc_range=np.array([-50, -50, 0., 50., 50., 5.], dtype=np.float32),
        extrinsics=extrinsics,
        intrinsics=intrinsics
    )

    return model, ego_features, img_feats, kwargs, B, N, T, C


def test_proformer_forward(proformer_inputs):
    (
        model,
        ego_features,
        img_feats,
        kwargs,
        B, N, T, C
    ) = proformer_inputs

    out, props = model(ego_features, img_feats, **kwargs)

    # Shape check
    assert out.shape == (B, N, T, C), \
        f"Expected shape {(B, N, T, C)}, got {out.shape}"

    # Dtype check
    assert out.dtype == ego_features.dtype, "Output dtype mismatch"

    # Finite check
    assert torch.isfinite(out).all(), "Output contains NaNs or Infs"

    print("output max value:", out.max().item())
    print("output min value:", out.min().item())
