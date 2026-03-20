import segmentation_models_pytorch as smp
import torch.nn as nn

class EyeSegmentationModel(nn.Module):
    def __init__(self, n_classes=4): # ТЕПЕРЬ 4 КЛАССА
        super(EyeSegmentationModel, self).__init__()
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)
