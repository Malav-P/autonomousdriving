import torch.nn as nn
import timm

from .fpn import FPN

class ImageEncoder(nn.Module):
    def __init__(self,
                 backbone_name: str = 'resnet34',
                 out_indices: list = None,
                 fpn_out_channels: int = 256,
                 pretrained: bool = True):
        """
        Args:
            backbone_name (str): name of the backbone model from timm
            out_indices (list[int], optional): indices of backbone feature maps to use.
                If None, all feature maps are used. Defaults to None.
            fpn_out_channels (int): number of output channels for FPN layers
            pretrained (bool): whether to use a pretrained backbone
        """
        super(ImageEncoder, self).__init__()

        # Load backbone model from timm
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=out_indices)
        in_channels_list = self.backbone.feature_info.channels()

        if out_indices is not None:
            in_channels_list = [in_channels_list[i] for i in out_indices]

        # Create FPN
        self.fpn = FPN(in_channels_list=in_channels_list,
                       out_channels=fpn_out_channels)
        


    def forward(self, x):
        """
        Args:
            x (Tensor): input image tensor (B, C, H, W)
        Returns:
            list[Tensor]: FPN feature maps (B, fpn_out_channels, H_i, W_i)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass features through FPN
        fpn_features = self.fpn(features)

        return fpn_features

class EgoMotionEncoder(nn.Module):
    def __init__(self,
                 num_input_features: int,
                 num_output_features: int):
        
        """
        Args:
            num_input_features (int): size of the input ego-motion vector
            num_output_features (int): size of the output feature vector
        """

        super(EgoMotionEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_input_features, num_output_features)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): input ego-motion tensor (B, num_input_features)
        Returns:
            Tensor: encoded ego-motion features (B, num_output_features)
        """
        out = self.encoder(x)
        return out


