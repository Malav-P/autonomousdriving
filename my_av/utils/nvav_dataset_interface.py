import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from physical_ai_av.dataset import PhysicalAIAVDatasetInterface



def nvav_collator(batch):
    """
    batch: list of dataset items
    Each item is:
        {
            "ego_features": (vel_ego, acc_ego, nav_goal),
            "image_features": frames_list_of_tensors,
            "gt_proposals": gt
        }
    """
    D = len(batch[0]["image_features"])

    # Extract individual fields
    vel_list   = []
    acc_list   = []
    goal_list  = []
    img_list   = [[] for _ in range(D)]
    gt_list    = []
    ext_list   = []
    int_list   = []
    vdims_list = []

    for item in batch:
        vel, acc, goal = item["ego_features"]
        vel_list.append(torch.from_numpy(vel))   # (N, 3)
        acc_list.append(torch.from_numpy(acc))   # (N, 3)
        goal_list.append(torch.from_numpy(goal)) # (N,)
        for cam in range(D):
            img_list[cam].append(torch.from_numpy(item["image_features"][cam]))
        gt_list.append(torch.from_numpy(item["gt_proposals"]))
        ext_list.append(torch.from_numpy(item["extrinsics"]))
        int_list.append(torch.from_numpy(item["intrinsics"]))
        vdims_list.append(torch.from_numpy(item["vehicle_dims"]))

    # Stack ego features into (B, N, ·)
    vel = torch.stack(vel_list, dim=0)
    acc = torch.stack(acc_list, dim=0)
    goal = torch.stack(goal_list, dim=0)

    # Flatten (B, N, d) → (B*N, d)
    vel = vel.flatten(0, 1)
    acc = acc.flatten(0, 1)
    goal = goal.flatten(0, 1)

    for cam in range(D):
        img_list[cam] = torch.stack(img_list[cam], dim=0).flatten(0, 1)

    gt_props  = torch.stack(gt_list, dim=0).flatten(0,1)
    exts      = torch.stack(ext_list, dim=0)
    ints      = torch.stack(int_list, dim=0)
    vdims     = torch.stack(vdims_list, dim=0)

    return {
        "ego_features": (vel, acc, goal),
        "image_features": img_list,
        "gt_proposals": gt_props,
        "extrinsics": exts,
        "intrinsics": ints,
        "vehicle_dims": vdims
    }




class NVAVDataset(Dataset):
    def __init__(self,
                 camera_names: list,
                 dt: float, T: int,
                 kappa_epsilon: float = 1e-3,
                 pc_range: np.ndarray = None
                 ):
        """
        Dataset Class for interfacing with the Nvidia Autonomous vehicles dataset

        Args:
            camera_names: list of camera names that will be used.
            dt: time interval between sampled future timestamps in microseconds
            T: int, number of future time stamps to sample
            kappa_epsilon : threshold to determine whether a turn is happening
            pc_range (np.ndarray) array xmin, ymin, zmin, xmax, ymax, zmax containing point cloud ranges. If passed, velocity and ego will be linearly scaled
                              to [-1, 1] using these scales

        """
        self.ds = PhysicalAIAVDatasetInterface(token=True)

        self.camera_names = camera_names
        self.kappa_epsilon = kappa_epsilon
        self.pc_range = pc_range
        self.dt = dt
        self.T = T

        self.fps = 30
        self.downsample = 0.4
        self.num_frames_to_choose = 16

    def __len__(self):
        # number of clips
        return len(self.ds.clip_index.index)

    def __getitem__(self, idx):
        """
        Docstring for __getitem__
        
        Returns:
            vel_ego (np.ndarray) (1, 3)
            acc_ego (np.ndarray) (1, 3)
            nav_goal (np.ndarray) (1,)

            image_features ( list(torch.tensor) ), each item in list is shape (B, 3, H_i, W_i). This tensor is normalized according to imagenet mean and std

            gt_proposals (np.ndarray) (1, T, 3) array of ground truth proposal trajectories
        """
        # print("A")
        clip_id = self.ds.clip_index.index[idx]
        chunk_id = self.ds.get_clip_chunk(clip_id)

        all_available = self.ds.chunk_sensor_presence.loc[chunk_id, self.camera_names].all()

        if not all_available:
            raise RuntimeError(f"Chunk ID {chunk_id} does not have all requested camera views present")

        try:
            # test to see if data is downloaded
            self.ds.get_clip_feature(clip_id, "vehicle_dimensions", maybe_stream=False)
        except FileNotFoundError:

            # Path to the dataset cache
            cache_path = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--nvidia--PhysicalAI-Autonomous-Vehicles"

            # Remove the directory if it exists
            if cache_path.exists() and cache_path.is_dir():
                shutil.rmtree(cache_path)
                print(f"Removed cache directory: {cache_path}")
            else:
                print(f"Cache directory does not exist: {cache_path}")

            self.ds.download_clip_features(clip_id, self.camera_names + ["egomotion", "sensor_extrinsics", "camera_intrinsics", "vehicle_dimensions"])
        
        # print("B")
        camera_readers = []
        for camera_name in self.camera_names:
            camera_readers.append(self.ds.get_clip_feature(clip_id, camera_name, maybe_stream=False, use_torch_codec=False))

        camera_readers = tuple(camera_readers)
        ego_reader = self.ds.get_clip_feature(clip_id, "egomotion", maybe_stream=False)

        n_frames = len(camera_readers[0].timestamps) - (self.dt / 1e6) * self.T * self.fps # ensure that we dont pick a frame that is too close to the end where we cant forecast T steps into the future

        # ### OPTION 1, RANDOMLY SAMPLE 16 frames frames
        # frame_idx = np.random.randint(0, n_frames, size=self.num_frames_to_choose)

        ### OPTION 2, RANDOMLY SAMPLE 16 consecutive frames somewhere in the clip, FASTER
        start_idx = np.random.randint(0, n_frames-self.num_frames_to_choose)
        frame_idx = np.arange(start_idx, start_idx + self.num_frames_to_choose, dtype=int)


        timestamps = _get_mean_timestamp(frame_idx=frame_idx, camera_readers=camera_readers)

        # ego features
        vel_ego, acc_ego, nav_goal = _get_ego_features(timestamps=timestamps,
                                                       reader=ego_reader,
                                                       kappa_epsilon=self.kappa_epsilon,
                                                       pc_range=self.pc_range)

        # convert nav_goal to one-hot
        indices = nav_goal + 1
        one_hot_nav_goal = np.eye(3, dtype=np.float32)[indices]
        
        # print("C")
        # image features
        frames = []
        for reader in camera_readers:
            frame = reader.decode_images_from_frame_indices(frame_idx) # (N, H, W, C)

            frame = frame.transpose(0, 3, 1, 2) # (N, C, H, W)
            
            frames.append(frame) 

            reader.close()


        # print("D")
        # camera intrinsics, extrinsics, and vehicle dimensions
        reader = self.ds.get_clip_feature(clip_id, "sensor_extrinsics", maybe_stream=False)
        extrinsics = reader.loc[self.camera_names].to_numpy().astype(np.float32) # each row qx, qy, qz, qw, x, y, z

        reader = self.ds.get_clip_feature(clip_id, "camera_intrinsics", maybe_stream=False)
        intrinsics = reader.loc[self.camera_names, ["width", "height", "cx", "cy", "fw_poly_0","fw_poly_1",  "fw_poly_2",  "fw_poly_3",  "fw_poly_4"]].to_numpy()
        
        reader = self.ds.get_clip_feature(clip_id, "vehicle_dimensions", maybe_stream=False)
        vehicle_dims = reader.loc[["length", "width", "rear_axle_to_bbox_center"]].to_numpy()
        
        if self.pc_range is not None:
            # scale to [-1, 1]
            xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
            vehicle_dims[[0, 2]] = 2* (vehicle_dims[[0, 2]] - xmin) / (xmax - xmin) - 1.0
            vehicle_dims[1] = 2* (vehicle_dims[1] - ymin) / (ymax - ymin) - 1.0

        
        # ground truth trajectories
        gt_proposals = _get_gt_proposals(timestamps=timestamps,
                                         egomotion_reader=ego_reader,
                                         dt=self.dt,
                                         T=self.T,
                                         pc_range=self.pc_range)

        return {"ego_features": (vel_ego, acc_ego, one_hot_nav_goal),
                "image_features": frames,
                "gt_proposals": gt_proposals,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "vehicle_dims": vehicle_dims,
                }

def _add_turn_lead(nav_goal: np.ndarray, min_len: int=30, lead: int=30):
    """
    Add a lead signal to the navigation goal. This ensures that vehicle is aware of upcoming turn.
    
    Args:
        nav_goal: 1d array to add lead signal
        min_len: minimum number of consecutive frames needed to classify as a turn
        lead: number of frames of lead signal to add

    Returns:
        nav_goal_lead: 1d array of same length as nav_goal with the lead signal added
    """

    N = len(nav_goal)
    result = np.zeros(N, dtype=int)

    for sign in (1, -1):
        mask = (nav_goal == sign).astype(int)

        # --- find runs ---
        x = np.concatenate(([0], mask, [0]))
        dx = np.diff(x)

        run_starts = np.where(dx == 1)[0]
        run_ends   = np.where(dx == -1)[0]
        run_lengths = run_ends - run_starts

        # --- keep only valid runs ---
        valid = run_lengths >= min_len
        run_starts = run_starts[valid]
        run_ends   = run_ends[valid]

        # --- difference array prepend ---
        diff = np.zeros(N + 1, dtype=int)
        prepend_starts = np.maximum(0, run_starts - lead)

        diff[prepend_starts] += sign
        diff[run_ends]       -= sign

        result += (np.cumsum(diff[:-1]) != 0).astype(int) * sign

    # resolve rare overlap conflicts (e.g. +1 and -1 overlap)
    result[result > 0] = 1
    result[result < 0] = -1

    return result



def _get_ego_features(timestamps: np.ndarray,
                     reader,
                     kappa_epsilon: float,
                     pc_range=None,
                     add_lead_signal=True):
    
    """
    Get velocity, acceleration, navigation goal features from current ego_motion reader

    Args:
        timestamps (np.ndarray) array of timestamps where features are being requested. shape (N,)
        reader : ego motion reader object
        kappa_epsilon : threshold to determine whether a turn is happening
        pc_range (np.ndarray) array xmin, ymin, zmin, xmax, ymax, zmax containing point cloud ranges. If passed, velocity and ego will be linearly scaled
                              to [-1, 1] using these scales
        add_lead_signal (bool) : whether to add lead signal to the navigation goal. Default True

    Returns:
        velocity_ego (np.ndarray) describing ego velocity in ego frame. Potentially normalized (N, 3)
        acceleration_ego (np.ndarray) describing acceleration in ego frame. Potentially normalized (N, 3)
        nav_goal (np.ndarray) of int describing whether vehicle is moving left (-1) straight (0) or right (1) shape (N,)

    """
    
    ego_features = reader(timestamps)

    velocity_anchor = ego_features.velocity  # (N, 3)
    acceleration_anchor = ego_features.acceleration  # (N, 3)
    curvature = ego_features.curvature.squeeze()  # (N,)
    transforms = ego_features.pose  # N poses

    N, _ = velocity_anchor.shape

    # Convert velocity and acceleration to vehicle frame
    _, R = transforms.inv().as_components()
    velocity_ego = np.einsum('nij,nj->ni', R.as_matrix(), velocity_anchor)  # (N, 4)
    acceleration_ego = np.einsum('nij, nj->ni', R.as_matrix(), acceleration_anchor)

    if pc_range is not None:
        # Linear scale to [-1, 1]
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range

        velocity_ego = 2.0 * (velocity_ego - pc_range[:3]) / (pc_range[3:] - pc_range[:3]) - 1.0
        acceleration_ego = 2.0 * (acceleration_ego - pc_range[:3]) / (pc_range[3:] - pc_range[:3]) - 1.0


    # Get navigation goal based on curvature
    nav_goal = np.zeros(shape=(N), dtype=int)
    nav_goal[curvature >  kappa_epsilon] = -1      # greater than +kappa_epsilon → column 0
    nav_goal[curvature < -kappa_epsilon] = 1      # less than -kappa_epsilon → column 2

    if add_lead_signal:
        nav_goal = _add_turn_lead(nav_goal, min_len=30, lead=30)


    return velocity_ego.astype(np.float32), acceleration_ego.astype(np.float32), nav_goal


def _get_mean_timestamp(frame_idx: np.ndarray,
                        camera_readers: tuple):
    """
    Given a frame index and multiple camera views, get the timestamp associated with the frame. Since different camera views may have slightly
    different timestamps for a frame, they are average across cameras.
    
    Args:
        frame_idx: np.ndarray of shape (B,) representing frames we are querying
        camera_readers: tuple of cameras from which to get timestamps

    Returns:
        np.ndarray (float) of shape (B,) containing timestamps of the frames
    """

    n_readers = len(camera_readers)
    B = frame_idx.shape[0]

    # average timestamps across all cameras
    timestamps = np.empty(shape=(B, n_readers))
    for i, reader in enumerate(camera_readers):
        timestamps[:, i] = reader.timestamps[frame_idx]
    timestamps = timestamps.mean(axis=-1) # (B,)

    return timestamps

def _get_gt_proposals(timestamps: np.ndarray,
                     egomotion_reader,
                     dt: float,
                     T: int,
                     pc_range: np.ndarray = None) -> np.ndarray:
    """
    
    Args:
        timestamps: np.ndarray of timestamps
        egomotion_reader: egomotion reader
        dt: float, time interval between sampled future timestamps in microseconds
        T: int, number of future time stamps to sample
        pc_range: (6,) array defining point cloud range. If not None, proposals will be normalized to this range


    Returns:
        proposals: (B, T, 3) array of future proposals in (x, y, heading) format

    """
    B = timestamps.shape[0]

    transforms = egomotion_reader(timestamps).pose  # B poses
    future_timestamps = timestamps[:, None] + (dt * np.arange(1, T + 1))[None, :] # (B, T)

    proposals = np.empty(shape=(B, T, 3))  # (B, T, 3)

    for i in range(B):
        future_poses = egomotion_reader(future_timestamps[i]).pose  # T poses
        rel_poses = transforms[i].inv() * (future_poses)  # T relative poses

        t, R = rel_poses.as_components()

        R = R.as_matrix()  # (T, 3, 3)

        x_vehicle = R[..., 0]  # (T, 3)
        x_proj = x_vehicle[..., :2] # (T, 2)

        heading = np.arctan2(x_proj[..., 1], x_proj[..., 0])  # (T,)
        x, y, _ = t.T  # each (T,)

        proposals[i, :, 0] = x  # (T,)
        proposals[i, :, 1] = y  # (T,)
        proposals[i, :, 2] = heading  # (T,)

    if pc_range is not None:
        # Linear scale to [-1, 1]
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        proposals[..., 0] = 2.0 * (proposals[..., 0] - x_min) / (x_max - x_min) - 1.0
        proposals[..., 1] = 2.0 * (proposals[..., 1] - y_min) / (y_max - y_min) - 1.0
        

    return proposals  # (B, T, 3)



def plot_bev_trajectory(clip_id, ds, num_points=100):
    reader = ds.get_clip_feature(clip_id, "egomotion", maybe_stream=False)

    time = np.linspace(reader.time_range[0], reader.time_range[1], num_points)
    t, R = reader(time).pose.as_components()

    curvature = reader(time).curvature.squeeze()  # (num_points,)
    cmax = np.abs(curvature).max()


    # Extract x and y (anchor frame)
    x = t[:, 0]
    y = t[:, 1]

    plt.figure(figsize=(8, 8))

    # Plot line (optional but keeps shape visible)
    plt.plot(-y, x, linewidth=1, alpha=0.3, color='black')

    # Scatter with curvature color mapping
    sc = plt.scatter(-y, x, c=curvature, cmap='coolwarm', s=25, vmin=-cmax, vmax=cmax)

    # Add colorbar to interpret curvature values
    cbar = plt.colorbar(sc)
    cbar.set_label("Curvature (1/m)")

    plt.xlabel("Y position (meters)")
    plt.ylabel("X position (meters)")
    plt.title("BEV Trajectory Colored by Curvature")
    plt.axis('equal')
    plt.grid(True)

    # Invert sign of xtick labels (but keep spacing)
    xticks = plt.xticks()[0]
    plt.xticks(xticks, [-int(tick) for tick in xticks])

    out_path = f"{clip_id}_bev_trajectory.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"BEV trajectory plot saved as {out_path}")