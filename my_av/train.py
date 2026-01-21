import torch
import numpy as np
from torch.utils.data import DataLoader
from my_av.loss import mon_loss

from torchvision.transforms import v2

transform = v2.Compose(
    [
        v2.Resize((432, 768)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
)

def train(model, ego_encoder, img_encoder, dataset, optimizer, loss_fn, epochs=10, batch_size=1, device="cuda", accum_iter=1):
    # Move model to the device
    model.to(device)
    model.train()

    ego_encoder.to(device)
    img_encoder.to(device)
    ego_encoder.train()
    img_encoder.train()

    # Data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=nvav_collator, num_workers=0)


    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            # Unpack batch (common case: (inputs, targets))
            vel_ego, acc_ego, nav_goal = batch["ego_features"] # (B*N, 3) (B*N, 3) (B*N,)
            img_feats = batch["image_features"]
            gt_proposals = batch["gt_proposals"] # (B*num_frames, T, 3)
            extrinsics = batch["extrinsics"] # (B, D, 7)
            intrinsics = batch["intrinsics"] # (B, D, 9)
            vdims = batch["vehicle_dims"] # (B, 3), each row (length, width, rear_axle_to_bbox_center)
            
            kwargs = dict(
                        half_length=vdims[:, 0]/2,
                        half_width=vdims[:, 1]/2,
                        rear_axle_to_center=vdims[:, 2],
                        zmin=0.0,
                        zmax=1.0,
                        num_points_in_pillar=4,
                        pc_range=dataset.pc_range,
                        extrinsics=extrinsics, 
                        intrinsics=intrinsics, 
                    )

            ego = torch.cat([vel_ego[:, :2], acc_ego[:, :2], nav_goal], dim=-1).to(device) # shape -> (B, 7)

            ego_feats = ego_encoder(ego)

            for cam, img_feat in enumerate(img_feats):
                img_feat = transform(img_feat.to(device))
                img_feats[cam] = img_encoder(img_feat)
            
            # Forward pass
            # with torch.autograd.set_detect_anomaly(True):

            outputs, proposals = model(ego_feats, img_feats, **kwargs)

            loss = loss_fn(proposals, gt_proposals, lambda_val=0.1)
            loss = loss / accum_iter

            # with torch.autograd.detect_anomaly():
            loss.backward()

            print(loss.item())


            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(loader)):
                optimizer.step()
                optimizer.zero_grad()
                print("\n")

            

            # total_loss += loss.item()

        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
    from my_av.utils.nvav_dataset_interface import NVAVDataset, FrameDecoder, nvav_collator
    from my_av.proformer import ProFormer
    from my_av.encoders import EgoMotionEncoder, ImageEncoder

    np.random.seed(42)
    torch.manual_seed(42)

    camera_names = ["camera_cross_left_120fov", "camera_cross_right_120fov", "camera_front_wide_120fov"]

    # Initialize the frame decoder
    ds_interface = PhysicalAIAVDatasetInterface(token=True)
    decoder = FrameDecoder(ds_interface, camera_names=camera_names)
    decoder.start()

    N = 64      # num proposals
    T = 6      # forecast steps
    C = 256     # embed dim
    L = 4      # num layers
    D = 3      # num cameras

    accum_iter = 2  # gradient accumulation steps

    dt = 0.5 * 1e6  # 0.5 seconds in microseconds

    backbone_name = 'resnet34'
    out_indices = [-1, -2]
    fpn_out_channels = C

    # point cloud range
    pc_range = np.array([-50, -50, 0., 50., 50., 5.], dtype=np.float32)


    # Create model
    model = ProFormer(
        num_layers=L,
        num_proposals=N,
        num_forecast_steps=T,
        embed_dim=C,
        num_heads=4,
        num_levels=2,
        num_points=4,
        ffn_dim=1024,
        mlp_dim=256,
        state_dim=3,
        shared=True
    )


    img_encoder = ImageEncoder(backbone_name=backbone_name,
                         out_indices=out_indices,
                         fpn_out_channels=fpn_out_channels,
                         pretrained=True)
    ego_encoder = EgoMotionEncoder(num_input_features=7, num_output_features=C)
    dataset = NVAVDataset(ds_interface=ds_interface, camera_names=camera_names, dt=dt, T=T, pc_range=pc_range, frame_decoder=decoder, prefetch_size=8)
    loss_fn = mon_loss
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": ego_encoder.parameters()},
        {"params": img_encoder.parameters()}
    ], lr=1e-4)


    train(model=model,
          ego_encoder=ego_encoder,
          img_encoder=img_encoder,
          dataset=dataset,
          optimizer=optimizer,
          loss_fn=loss_fn,
          accum_iter=accum_iter)
    

    decoder.stop()
    