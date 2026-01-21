import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from my_av.loss import mon_loss
from my_av.utils.nvav_dataset_interface import nvav_collator


class AVMotionPredictionModule(pl.LightningModule):
    def __init__(
        self,
        model,
        ego_encoder,
        img_encoder,
        pc_range,
        learning_rate=1e-4,
        lambda_val=0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'ego_encoder', 'img_encoder'])
        
        self.model = model
        self.ego_encoder = ego_encoder
        self.img_encoder = img_encoder
        self.pc_range = pc_range
        self.learning_rate = learning_rate
        self.lambda_val = lambda_val
        
        # Image transforms
        self.transform = v2.Compose([
            v2.Resize((432, 768)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, ego_feats, img_feats, **kwargs):
        return self.model(ego_feats, img_feats, **kwargs)

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     batch["ego_features"] = (batch["ego_features"][0].to(device), batch["ego_features"][1].to(device), batch["ego_features"][2].to(device))
    #     batch["image_features"] = [img_feat.to(device) for img_feat in batch["image_features"]]
    #     return batch
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        vel_ego, acc_ego, nav_goal = batch["ego_features"]
        img_feats = batch["image_features"]
        gt_proposals = batch["gt_proposals"]
        extrinsics = batch["extrinsics"]
        intrinsics = batch["intrinsics"]
        vdims = batch["vehicle_dims"]
        
        # Prepare kwargs for model
        kwargs = dict(
            half_length=vdims[:, 0] / 2,
            half_width=vdims[:, 1] / 2,
            rear_axle_to_center=vdims[:, 2],
            zmin=0.0,
            zmax=1.0,
            num_points_in_pillar=4,
            pc_range=self.pc_range,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        
        # Prepare ego features
        ego = torch.cat([vel_ego[:, :2], acc_ego[:, :2], nav_goal], dim=-1)
        ego_feats = self.ego_encoder(ego)
        
        # Process image features
        processed_img_feats = []
        for img_feat in img_feats:
            img_feat = self.transform(img_feat)
            processed_img_feats.append(self.img_encoder(img_feat))
        
        # Forward pass
        outputs, proposals = self(ego_feats, processed_img_feats, **kwargs)
        
        # Compute loss
        loss = mon_loss(proposals, gt_proposals, lambda_val=self.lambda_val)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=32)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
            {"params": self.ego_encoder.parameters()},
            {"params": self.img_encoder.parameters()}
        ], lr=self.learning_rate)
        
        return optimizer


if __name__ == "__main__":
    from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
    from my_av.utils.nvav_dataset_interface import NVAVDataset, FrameDecoder
    from my_av.proformer import ProFormer
    from my_av.encoders import EgoMotionEncoder, ImageEncoder
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    pl.seed_everything(42)
    
    # Configuration
    camera_names = ["camera_cross_left_120fov", "camera_cross_right_120fov", "camera_front_wide_120fov"]
    N = 64      # num proposals
    T = 6       # forecast steps
    C = 256     # embed dim
    L = 4       # num layers
    D = 3       # num cameras
    dt = 0.5 * 1e6  # 0.5 seconds in microseconds
    num_frames_to_choose = 32
    pc_range = np.array([-50, -50, 0., 50., 50., 5.], dtype=np.float32)
    
    # Initialize dataset
    ds_interface = PhysicalAIAVDatasetInterface(token=True)
    decoder = FrameDecoder(ds_interface, camera_names=camera_names, num_frames_to_choose=num_frames_to_choose)
    decoder.start()
    
    dataset = NVAVDataset(
        ds_interface=ds_interface,
        camera_names=camera_names,
        dt=dt,
        T=T,
        pc_range=pc_range,
        frame_decoder=decoder,
        prefetch_size=8
    )
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=nvav_collator,
        num_workers=0
    )
    
    # Create models
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
    
    img_encoder = ImageEncoder(
        backbone_name='resnet34',
        out_indices=[-1, -2],
        fpn_out_channels=C,
        pretrained=True
    )
    
    ego_encoder = EgoMotionEncoder(
        num_input_features=7,
        num_output_features=C
    )
    
    # Create Lightning module
    lightning_module = AVMotionPredictionModule(
        model=model,
        ego_encoder=ego_encoder,
        img_encoder=img_encoder,
        pc_range=pc_range,
        learning_rate=1e-4,
        lambda_val=0.1
    )
    
    # Create trainer with gradient accumulation
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=1,
        accumulate_grad_batches=2,
        log_every_n_steps=1,
        enable_checkpointing=True,
        default_root_dir='./lightning_logs'
    )
    
    # Train
    trainer.fit(lightning_module, train_loader)
    
    # Cleanup
    decoder.stop()