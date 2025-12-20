import pytest
import numpy as np

from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
from my_av.utils.nvav_dataset_interface import _get_gt_proposals, _get_ego_features, _get_mean_timestamp, _add_turn_lead, NVAVDataset

# --- Fixtures -------------------------------------------------------------
@pytest.fixture
def readers():
    ds = PhysicalAIAVDatasetInterface(token=True)
    clip_id = ds.clip_index.index[0] # First clip in the dataset

    reader1 = ds.get_clip_feature(clip_id, "camera_cross_left_120fov", maybe_stream=True)
    reader2 = ds.get_clip_feature(clip_id, "camera_cross_right_120fov", maybe_stream=True)
    reader3 = ds.get_clip_feature(clip_id, "camera_front_wide_120fov", maybe_stream=True)

    egomotion_reader = ds.get_clip_feature(clip_id, "egomotion", maybe_stream=True)
    camera_readers = (reader1, reader2, reader3)


    return camera_readers, egomotion_reader


# --- Tests ----------------------------------------------------------------

def test_get_ego_features_shape(readers):
    camera_readers, egomotion_reader = readers
    N = camera_readers[2].timestamps.size

    velocity_ego, acceleration_ego, nav_goal = _get_ego_features(camera_readers[2].timestamps, egomotion_reader, kappa_epsilon=1e-2)

    assert velocity_ego.shape == (N, 3)  # (N, 3)
    assert acceleration_ego.shape == (N, 3)  # (N, 3)
    assert nav_goal.shape == (N,)


def test_get_gt_proposals_shape(readers):
    camera_readers, egomotion_reader = readers
    frame_idx = np.array([0, 2, 4])  # 3 frames
    dt = 0.5 * 1e6  # 0.5 seconds in microseconds
    T = 5 # 5 future timestamps

    timestamps = _get_mean_timestamp(frame_idx, camera_readers)

    proposals = _get_gt_proposals(timestamps, egomotion_reader, dt, T)

    assert proposals.shape == (3, 5, 3)  # (N, T, 3)


def test_get_gt_proposals_progresses_forward(readers):
    camera_readers, egomotion_reader = readers
    frame_idx = np.array([1])  # single frame
    dt = 1 * 1e6  # 1 seconds in microseconds
    T = 3 # 3 future timestampss

    timestamps = _get_mean_timestamp(frame_idx, camera_readers)

    proposals = _get_gt_proposals(timestamps, egomotion_reader, dt, T)

    x = proposals[0, :, 0]

    # x should be strictly increasing (forward motion)
    assert np.all(np.diff(x) > 0)


def test_get_gt_proposals_normalized(readers):
    camera_readers, egomotion_reader = readers
    frame_idx = np.array([0])  # single frame
    dt = 5 * 1e6  # 5 seconds in microseconds
    T = 3 # 3 future timestamps

    xmin, xmax = -50.0, 50.0
    ymin, ymax = -50.0, 50.0
    zmin, zmax = -5.0, 3.0

    pc_range = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

    timestamps = _get_mean_timestamp(frame_idx, camera_readers)

    proposals = _get_gt_proposals(timestamps, egomotion_reader, dt, T)
    proposals_norm = _get_gt_proposals(timestamps, egomotion_reader, dt, T, pc_range=pc_range)


    # Check that proposals are scaled to -1 and 1
    assert np.all(np.abs(proposals_norm) <= np.abs(proposals))

def test_nvav_dataset_basic():

    camera_names = ["camera_cross_left_120fov", "camera_cross_right_120fov", "camera_front_wide_120fov"]
    dt = 0.5 * 1e6  # 0.5 seconds in microseconds
    T = 5 # 5 future timestamps

    dataset = NVAVDataset(camera_names=camera_names, dt=dt, T=T)

    item = dataset[0]

    assert item["image_features"][0].shape[:2] == (dataset.num_frames_to_choose, 3) # should be (N, 3, H, W)

def test_add_turn_lead():
    np.random.seed(42)
    N = 700

    # Start with zeros
    nav_goal = np.zeros(N, dtype=int)

    # Add some left turns (+1)
    nav_goal[100:150] = 1    # 50 samples → valid
    nav_goal[300:320] = 1    # 20 samples → too short, should be ignored
    nav_goal[500:540] = 1    # 40 samples → valid

    # Add some right turns (-1)
    nav_goal[200:250] = -1   # 50 samples → valid
    nav_goal[600:615] = -1   # 15 samples → too short, ignored

    # Apply lead function
    nav_goal_lead = _add_turn_lead(nav_goal, min_len=30, lead=30)

    assert (nav_goal_lead[70:150] == 1).all()
    assert (nav_goal_lead[270:300] == 0).all()
    assert (nav_goal_lead[470:540] == 1).all()
    assert (nav_goal_lead[170:250] == -1).all()
    assert (nav_goal_lead[570:600] == 0).all()