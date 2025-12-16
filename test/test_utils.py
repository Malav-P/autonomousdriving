import torch
import numpy as np
from my_av.utils.utils import world_to_image_ftheta, rotation_to_heading, bev_to_world, _get_corners  # adjust import


def test_bev_to_world_basic():
    # Simple test inputs
    bev_coords = torch.tensor([
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    pc_range = torch.tensor([0.0, 10.0, 20.0, 10.0, 20.0, 30.0], dtype=torch.float32)

    world_coords = bev_to_world(bev_coords, pc_range)

    # Check shape
    assert world_coords.shape == bev_coords.shape

    # Check that minimum bev_coords maps to pc_range min
    assert torch.allclose(world_coords[0], pc_range[:3], atol=1e-6)

    # Check that maximum bev_coords maps to pc_range max
    assert torch.allclose(world_coords[2], pc_range[3:], atol=1e-6)

    # Check that center bev_coords maps to midpoint of range
    midpoint = (pc_range[:3] + pc_range[3:]) / 2
    assert torch.allclose(world_coords[1], midpoint, atol=1e-6)



def test_heading_identity():
    R = np.eye(3)  # no rotation
    heading = rotation_to_heading(R)
    assert np.isclose(heading, 0.0, atol=1e-8)

def test_heading_90_deg():
    # 90 degree rotation around Z-axis
    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    heading = rotation_to_heading(R)
    assert np.isclose(heading, theta, atol=1e-8)

def test_heading_negative_90_deg():
    # -90 degree rotation around Z-axis
    theta = -np.pi / 2
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    heading = rotation_to_heading(R)
    assert np.isclose(heading, theta, atol=1e-8)

def test_heading_arbitrary():
    # small rotation
    theta = 0.123
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    heading = rotation_to_heading(R)
    assert np.isclose(heading, theta, atol=1e-8)


def test_world_to_image_ftheta_practical_ftheta_coefs():

    # Practical f-theta coefficients from a real camera
    k0 = 0.0
    k1 = 924.317142
    k2 = -4.383656
    k3 = -15.204016
    k4 = 1.238666

    w, h = 1920, 1080
    cx, cy = 960.483185, 541.324927

    # Camera at origin, no rotation
    tx, ty, tz = 0.0, 0.0, 0.0
    qw, qx, qy, qz = 0.5, -0.5, 0.5, -0.5 # rig/anchor frame to opencv camera frame

    world_coords = torch.tensor([[1.0, 0.0, 0.0], # point directly in front of camera
                                 [1000.0, 0.0, 0.0], # point far along optical axis
                                 [1.0, 10000.0, 0.0],  # point way to the left   
                                 [-1.0, 0.0, 0.0],  # point behind camera
                                 [0.0, 0.0, 0.0]                            
                                 ], dtype=torch.float32) 

    intrinsics = torch.tensor([[[w, h, cx, cy, k0, k1, k2, k3, k4]]], dtype=torch.float32)
    extrinsics = torch.tensor([[[ qx, qy, qz, qw,  tx, ty, tz]]], dtype=torch.float32)

    img_pts, masks = world_to_image_ftheta(world_coords, intrinsics, extrinsics, normalize=False)


    assert torch.allclose(img_pts[0,0], torch.tensor([cx, cy]), atol=1e-4)
    assert masks[0,0,0] == True, "Point directly in front of camera should be valid"

    assert torch.allclose(img_pts[1,0], torch.tensor([cx, cy]), atol=1e-4)
    assert masks[1,0,0] == True, "Point far along optical axis should still be valid"

    x, y = img_pts[2,0]
    assert x < 0.0, "Point far to left should project outside image"
    assert masks[2,0,0] == False

    assert torch.allclose(img_pts[3,0], torch.tensor([cx, cy]), atol=1e-4)
    assert masks[3,0,0] == False, "Point at camera center should be invalid"

    assert torch.allclose(img_pts[4,0], torch.tensor([cx, cy]), atol=1e-4)
    assert masks[4,0,0] == False, "Point behind camera should be invalid"


def test_get_corners_zero_heading():
    """
    Case: single proposal at origin, heading=0.
    No rear offset, so corners are axis aligned and easy to verify.
    """
    # ---- Inputs ----
    proposals = torch.tensor([
        [0.0, 0.0, 0.0]   # x, y, heading
    ])  # shape (B*num_frames=1, 3)

    half_width = torch.tensor([1.0])     # B=1
    half_length = torch.tensor([2.0])    # B=1
    rear_axle_to_center = torch.tensor([0.0])  # B=1

    # ---- Expected ----
    expected = torch.tensor([
        [ 2.0,  1.0],
        [ 2.0, -1.0],
        [-2.0, -1.0],
        [-2.0,  1.0],
    ])  # shape (4,2)

    # ---- Run ----
    out = _get_corners(
        proposals=proposals,
        half_width=half_width,
        half_length=half_length,
        rear_axle_to_center=rear_axle_to_center,
    )

    # out has shape (B*num_frames, 4, 2) == (1,4,2)
    assert out.shape == (1, 4, 2)

    # ---- Compare ----
    torch.testing.assert_close(out[0], expected)

