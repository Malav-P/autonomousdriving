import torch
import numpy as np

def proposal_to_anchor(proposals: torch.tensor,
                       half_width: float,
                       half_length: float,
                       rear_axle_to_center: float,
                       zmin: float,
                       zmax: float,
                       num_points_in_pillar: float,
                       pc_range: np.ndarray,
                       intrinsics: torch.tensor,
                       extrinsics: torch.tensor):
    
    corners = _get_corners(proposals=proposals,
                           half_width=half_width,
                           half_length=half_length,
                           rear_axle_to_center=rear_axle_to_center) # (B, N, T, 4, 2)

    reference_points = _lift_corners(corners=corners,
                                     zmin=zmin,
                                     zmax=zmax,
                                     num_points_in_pillar=num_points_in_pillar) # (B, N, T, 4, num_points_in_pillar, 3)

    pc_range = torch.tensor(pc_range)
    world_coords = bev_to_world(bev_coords=reference_points,
                                pc_range=pc_range) # (B, N, T, 4, num_points_in_pillar, 3)

    img_coords, mask = world_to_image_ftheta(world_coords=world_coords,
                                             intrinsics=intrinsics,
                                             extrinsics=extrinsics,
                                             normalize=True)
    
    mask_view = mask[..., 0].any(dim=4).any(dim=3) # (B, N, T, D)

    return img_coords, mask_view


def _get_corners(proposals: torch.tensor,
                 half_width,
                 half_length,
                 rear_axle_to_center) -> torch.tensor:
    """
    Given proposal locations (x, y, heading), compute the four corners of the box of the vehicle.
    I THINK `proposals` SHOULD BE IN BEV (NORMALIZED) COORDINATES

    Args:
        proposals : tensor of shape (B*num_frames, ..., 3)
        half_width: tensor of shape (B,)
        half_length: tensor of shape (B,)
        rear_axle_to_center: tensor of shape (B,)

    Returns:
        corners :  tensor of shape (B*num_frames, ..., 4, 2)
    """
    B = half_length.shape[0]

    proposals = proposals.view(B, -1, *proposals.shape[1:]) # (B, num_frames, ..., 3)

    _, num_frames, *batch_shape, _ = proposals.shape

    corners = torch.empty(size=(B, num_frames, *batch_shape, 4, 2))

    for b in range(B):
        x, y, headings = proposals[b, ..., 0], proposals[b, ..., 1], proposals[b, ..., 2]
        h_width =torch.zeros_like(x)+half_width[b]
        h_length = torch.zeros_like(x)+half_length[b]

        cos_yaw = torch.cos(headings)[...,None]
        sin_yaw = torch.sin(headings)[...,None]

        x=x[...,None]+rear_axle_to_center * cos_yaw
        y=y[...,None]+rear_axle_to_center * sin_yaw

        # Compute the four corners
        corners_x = torch.stack([h_length, h_length, -h_length, -h_length],dim=-1)
        corners_y = torch.stack([h_width, -h_width, -h_width, h_width],dim=-1)

        # Rotate corners by yaw
        rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
        rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

        # Translate corners to the center of the bounding box
        corners[b] = torch.stack((rot_corners_x + x, rot_corners_y + y), dim=-1) # (num_frames, ..., 4, 2)

    corners = corners.flatten(0, 1)
    return corners

def _lift_corners(corners: torch.tensor,
                  zmin: float,
                  zmax: float,
                  num_points_in_pillar) -> torch.tensor:
    """
    Lift 2d corner points into 3d to allow for camera projection onto image views.
    I THINK `corners` and `zmin`  and `zmax` SHOULD BE IN BEV COORDINATES   


    Args:
        corners: tensor of shape (..., 4, 2)
        zmin: minimum z pos of pillar
        zmax: maximum z pos of pillar
        num_points_in_pillar: number of points to sample in pillar    

    Returns:
        reference_points: tensor of shape (..., 4, num_points_in_pillar, 3)
    """
    # corners: (..., 4, 2)
    *batch_dims, _, _ = corners.shape

    zs = torch.linspace(
        zmin,
        zmax,
        steps=num_points_in_pillar
    )

    # Expand z to match corners
    # -> (..., 4, num_points_in_pillar)
    z = zs.view(1, 1, num_points_in_pillar).expand(*batch_dims, 4, num_points_in_pillar)

    # Expand x,y across the pillar dimension
    x = corners[..., :, 0].unsqueeze(-1).expand(*batch_dims, 4, num_points_in_pillar)
    y = corners[..., :, 1].unsqueeze(-1).expand(*batch_dims, 4, num_points_in_pillar)

    # Stack into (x, y, z)
    reference_points = torch.stack((x, y, z), dim=-1)

    return reference_points


def bev_to_world(bev_coords: torch.tensor,
                 pc_range: torch.tensor):
    """
    Convert bev coordinates (i.e. coordinates that the transformer processes) to 3d world coordinates    
    he scaling takes [-1, 1] to [xmin, xmax], [ymin, ymax], [zmin, zmax] defined by pc_range

    Args:
        bev_coords: (..., 3) tensor of coordinates to transform
        pc_range: (xmin, ymin, zmin, xmax, ymax, zmax) point cloud range

    Returns:
        world_coords: (..., 3) tensor of world coordinates
    """

    world_coords = ((bev_coords + 1) / 2 ) * (pc_range[3:] - pc_range[:3]) + pc_range[:3]


    return world_coords

def world_to_image_ftheta(world_coords: torch.tensor,
                           intrinsics: torch.tensor,
                           extrinsics: torch.tensor,
                           normalize=False,
                           eps = 1e-8):
    """
    Project world coords into image coords using ftheta model. These world coordinates should be in the rig frame. I.e. origin at rear axle projected onto ground
    with x axis forward, y-axis left and z axis up.

    Args:
        world_coords: (B*num_frames, ..., 3) tensor of world coords
        extrinsics: (B, D, 7) tensor of D camera quaternions + translations (qx, qy, qz, qw, x, y, z) defining camera pose in rig frame
        intrinsics: (B, D, 9) tensor of D camera intrinsics in fisheye model (w, h, cx, cy, k0, k1, k2, k3, k4)
        normalize: whether to normalize image coords to [0, 1]
        eps: small value to avoid divide by zero

    Returns:
        img_pts: (..., D, 2) tensor of image points
        bev_masks: (..., D, 1) tensor of valid masks
    """

    B, D, _ = extrinsics.shape
    world_coords = world_coords.view(B, -1, *world_coords.shape[1:]) # (B, num_frames, ..., 3)
    _, num_frames, *batch_shape, _ = world_coords.shape

    bev_masks = torch.empty(size=(B, num_frames, *batch_shape, D, 1), dtype=torch.bool)
    img_pts = torch.empty(size=(B, num_frames, *batch_shape, D, 2))

    for b in range(B):
        w_coords = world_coords[b]
        for cam in range(D):
            qx, qy, qz, qw, x, y, z = extrinsics[b, cam]
            w, h, cx, cy, k0, k1, k2, k3, k4 = intrinsics[b, cam]

            R = torch.tensor(quat_to_rotmat([qw, qx, qy, qz])) # (3, 3)
            t = torch.tensor([x, y, z])

            # Transform world coords to camera frame
            cam_coords = (w_coords - t) @ R  # (..., 3)

            X = cam_coords[..., 0]
            Y = cam_coords[..., 1]
            Z = cam_coords[..., 2]  

            # points must be infront of camera (i.e. z > eps)
            bev_mask  = Z > eps # (...,)

            norm = torch.sqrt(X**2 + Y**2 + Z**2)
            norm_safe = torch.where(norm < eps, torch.tensor(1.0), norm)
            cos_theta = torch.clamp(Z / norm_safe, -1.0, 1.0)  # valid acos domain
            theta = torch.acos(cos_theta)

            ftheta = k0 + k1 * theta + k2 * theta**2 + k3 * theta**3 + k4 * theta**4 # (...,)
            rp = torch.sqrt(X**2 + Y**2)

            # add small epsilon to avoid division by zero
            rp_safe = torch.where(rp < eps, torch.tensor(1.0), rp)

            img_x = cx + (ftheta * X / rp_safe)
            img_y = cy + (ftheta * Y / rp_safe)


            # handle exact zero case: set img_x = cx, img_y = cy when rp == 0
            img_x = torch.where(rp < eps, cx, img_x)
            img_y = torch.where(rp < eps, cy, img_y)

            bev_mask = (bev_mask& (img_x > 0.0)
                                & (img_x < w)
                                & (img_y < h)
                                & (img_y > 0.0))
            
            if normalize:
                img_x /= w
                img_y /= h


            img_pts[b, ..., cam, 0] = img_x
            img_pts[b, ..., cam, 1] = img_y
            bev_masks[b, ..., cam, 0] = bev_mask

    img_pts = img_pts.flatten(0, 1)
    bev_masks = bev_masks.flatten(0, 1)

    return img_pts, bev_masks


def world_to_image_pinhole(world_coords: torch.tensor,
                           cameras: torch.tensor,
                           img_shapes: torch.tensor,
                           normalize=False,
                           eps = 1e-8):
    
    """
    Project world coords into image coords using pinhole model. These world coordinates should be in the rig frame. I.e. origin at rear axle projected onto ground
    with x axis forward, y-axis left and z axis up.

    Args:
        world_coords: (..., 3) tensor of world coords
        cameras: (D, 3, 4) D camera projection matrices. This assumes camera projection matrices are pytorch / opencv style so that top-left of image is origin
        img_shapes: (D, 2) list of D image shapes (h, w)
        normalize: whether to normalize image coords to [0, 1]
        eps: small value to avoid divide by zero

    Returns:
        img_pts: (..., D, 2) tensor of image points
        bev_masks: (..., D, 1) tensor of valid masks
    """
    D, _, _ = cameras.shape
    *batch_shape, _ = world_coords.shape

    ones = torch.ones_like(world_coords[..., :1]) # (..., 1)
    homogenous_world_coords = torch.cat([world_coords, ones], dim=-1) # (..., 4)

    image_coords = torch.empty(size=(*batch_shape, D, 2))
    bev_masks = torch.empty(size=(*batch_shape, D, 1))

    for i, camera in enumerate(cameras):
        homo_img_coords = homogenous_world_coords @ camera.T # (..., 3) each 3-vector is of the form z (x, y, 1)
        h, w = img_shapes[i]

        z = torch.clamp(homo_img_coords[..., 2:3], min=eps)
        image_coords_i = homo_img_coords / z # coordinates here have (0, 0) at the top-left of image. shape of this tensor (..., 3)

        # points must be infront of camera (i.e. z > eps)
        bev_mask  = homo_img_coords[..., 2:3] > eps # (..., 1)

        bev_mask = (bev_mask& (image_coords_i[..., 1:2] > 0.0)
                            & (image_coords_i[..., 1:2] < h)
                            & (image_coords_i[..., 0:1] < w)
                            & (image_coords_i[..., 0:1] > 0.0))
        
        # normalize coords to [0, 1]
        if normalize:
            image_coords_i[..., 0] /= w 
            image_coords_i[..., 1] /= h
        
        image_coords[..., i, :] = image_coords_i[..., :2] # (..., 2)
        bev_masks[..., i, :] = bev_mask

    return image_coords, bev_masks
      


def quat_to_rotmat(q):
    """Convert quaternion [qw, qx, qy, qz] to rotation matrix."""
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float32)
    return R

def rotation_to_heading(R):
    """
    Extract the heading (yaw) angle from a 3x3 rotation matrix.
    
    Heading is defined as the rotation around the Z-axis,
    i.e., the angle between the vehicle's x-axis projected onto
    the horizontal plane and the anchor frame x-axis.
    
    Args:
        R (np.ndarray): 3x3 rotation matrix (vehicle in anchor frame)
    
    Returns:
        float: heading angle in radians, positive counterclockwise
    """
    # Vehicle x-axis in anchor frame
    x_vehicle = R[:, 0]
    
    # Project onto horizontal plane (ignore z)
    x_proj = x_vehicle[:2]
    
    # Compute heading angle
    heading = np.arctan2(x_proj[1], x_proj[0])
    
    return heading




if __name__ == "__main__":

    ############################# _get_corners ##########################################
    # Simple sanity tests

    # Parameters
    half_width = 1.0
    half_length = 2.0
    rear_axle_to_center = 0.5

    # Test proposals: (x, y, heading)
    proposals = torch.tensor([
        [0.0, 0.0, 0.0],            # heading = 0°
        [0.0, 0.0, torch.pi / 2],   # heading = 90°
    ])

    corners = _get_corners(
        proposals,
        half_width=half_width,
        half_length=half_length,
        rear_axle_to_center=rear_axle_to_center,
    )

    print("Corners output shape:", corners.shape)
    print("Corners:\n", corners)

    # --- Expected checks ---
    # For heading = 0°:
    #   Center should be shifted to (0.5, 0)
    #   Corners should be:
    #       (+2,+1), (+2,-1), (-2,-1), (-2,+1) offset by (0.5, 0)
    #
    # For heading = 90°:
    #   Center shifts to (0, 0.5)
    #   Corners should be:
    #       (+1,+2.5), (+1,-1.5), (-1,+2.5), (-1,-1.5) 


    ############################# _lift_corners ##########################################
    pillar_height = 1
    num_points_in_pillar = 2
    reference_points = _lift_corners(corners=corners, pillar_height=pillar_height, num_points_in_pillar=num_points_in_pillar)

    print(reference_points)
    print("ref points shape: ", reference_points.shape)

