import torch
import pytest
from my_av.fpn import FPN 

@pytest.fixture
def fpn_inputs():
    B = 2
    in_channels_list = [64, 128, 256]
    out_channels = 256
    H, W = 256, 256
    model = FPN(in_channels_list=in_channels_list, out_channels=out_channels)

    inputs = [torch.randn(B, c, H // (2 ** i), W // (2 ** i)) for i, c in enumerate(in_channels_list)]


    return B, inputs, out_channels, model

def test_fpn_forward(fpn_inputs):
    B, inputs, out_channels, model = fpn_inputs

    out = model(inputs)

    # Check output len
    assert len(out) == len(inputs), f"Expected output len {len(inputs)}, got {len(out)}"

    # Check output shapes
    for i in range(len(out)):
        expected_shape = (B, out_channels, inputs[i].shape[2], inputs[i].shape[3])
        assert out[i].shape == expected_shape, f"Expected shape {expected_shape}, got {out[i].shape}"

    # Optional: check finite values
    for output in out:
        assert torch.isfinite(output).all(), "Output contains NaNs or infs"
