import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ManusEyeModel(nn.Module):
    """
    Профессиональная модель сегментации глаза на 4 класса.
    Использует EfficientNet-B0 как более точный и современный энкодер.
    """
    def __init__(self, n_classes=4):
        super(ManusEyeModel, self).__init__()
        self.model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
            activation=None # Используем Logits для стабильности обучения
        )

    def forward(self, x):
        return self.model(x)

# Константы для всего проекта
class EyeConfig:
    CLASSES = {
        0: "background",
        1: "sclera",
        2: "iris",
        3: "pupil"
    }
    # Контрастные цвета BGR
    COLORS = {
        "sclera": (200, 200, 200), # Светло-серый
        "iris": (0, 255, 0),       # Ярко-зеленый
        "pupil": (0, 0, 255),      # Чистый красный
        "glint": (255, 255, 0)     # Желтый (блики)
    }
    INPUT_SIZE = (256, 256)
