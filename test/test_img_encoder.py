import torch
import pytest
from my_av.encoders import ImageEncoder

@pytest.fixture
def img_encoder_inputs():
    B = 2
    C = 3
    H = 224
    W = 224
    backbone_name = 'resnet18'
    out_indices = [-1, -2]
    fpn_out_channels = 64

    model = ImageEncoder(backbone_name=backbone_name,
                         out_indices=out_indices,
                         fpn_out_channels=fpn_out_channels,
                         pretrained=False)
    
    inputs = torch.randn(B, C, H, W)

    return B, inputs, out_indices, fpn_out_channels, model

def test_img_encoder_forward(img_encoder_inputs):
    B, inputs, out_indices, fpn_out_channels, model = img_encoder_inputs

    out = model(inputs)

    # Check output len
    assert len(out) == len(out_indices), f"Expected output len {len(out_indices)}, got {len(out)}"

    # Check output shapes
    for i in range(len(out)):
        expected_first_two_dims = (B, fpn_out_channels)
        assert out[i].shape[:2] == expected_first_two_dims, f"Expected shape {expected_first_two_dims}, got {out[i].shape[:2]}"

    # Optional: check finite values
    for output in out:
        assert torch.isfinite(output).all(), "Output contains NaNs or infs"
