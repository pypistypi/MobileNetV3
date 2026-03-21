import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from eye_core import ManusEyeModel, EyeConfig
import torchvision.transforms.functional as TF
import random

# --- НАСТРОЙКИ ---
DATA_DIR = 'datasets/big_dataset'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-4


class EyeDataset(Dataset):
    def __init__(self, base_dir, augment=True):
        self.base_dir = base_dir
        self.augment = augment
        self.images = [f for f in os.listdir(os.path.join(base_dir, 'images')) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.base_dir, 'images', img_name)

        mask_i_name = img_name.replace('.jpg', '_i.png')
        mask_p_name = img_name.replace('.jpg', '_p.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_i = cv2.imread(os.path.join(self.base_dir, 'masks_i', mask_i_name), 0)
        mask_p = cv2.imread(os.path.join(self.base_dir, 'masks_p', mask_p_name), 0)

        final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        final_mask[mask_i >= 200] = 1
        final_mask[(mask_i > 50) & (mask_i < 200)] = 2
        final_mask[mask_p > 50] = 3

        # Стандартное изменение размера
        image = cv2.resize(image, (256, 256))
        final_mask = cv2.resize(final_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Простая аугментация без лишних библиотек
        if self.augment:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                final_mask = cv2.flip(final_mask, 1)

        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.tensor(image), torch.tensor(final_mask, dtype=torch.long)


def train():
    dataset = EyeDataset(DATA_DIR, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ManusEyeModel(n_classes=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Начинаем обучение Manus-Eye-v3 на {DEVICE} (без Albumentations)...")
    # ... (далее цикл обучения без изменений)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Эпоха [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss / len(loader):.4f}")

    # Сохраняем "мозги" новой модели
    torch.save(model.state_dict(), "manus_eye_v3.pth")
    print("Обучение завершено! Модель сохранена как manus_eye_v3.pth")


if __name__ == "__main__":
    train()
