import cv2
import torch
import numpy as np
import os
from s2_model_unet import EyeSegmentationModel

# --- НАСТРОЙКИ ---
VIDEO_PATH = 'input_video.mp4'  # Путь к вашему видео (или 0 для веб-камеры)
MODEL_PATH = 'eye_segmentation_final.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Цвета для маски (BGR)
COLOR_SCLERA = (240, 240, 240)  # Белый
COLOR_IRIS = (0, 255, 0)  # Зеленый
COLOR_PUPIL = (0, 0, 255)  # Красный


# -----------------

def process_frame(frame, model):
    h, w = frame.shape[:2]
    # Подготовка кадра для нейросети
    img_input = cv2.resize(frame, (256, 256))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_input).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.argmax(output[0], dim=0).cpu().numpy().astype(np.uint8)

    # Создаем чистую маску на черном фоне
    mask_color = np.zeros((256, 256, 3), dtype=np.uint8)
    mask_color[mask == 1] = COLOR_SCLERA
    mask_color[mask == 2] = COLOR_IRIS
    mask_color[mask == 3] = COLOR_PUPIL

    # Растягиваем маску до исходного размера видео с антиалиасингом
    mask_full = cv2.resize(mask_color, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask_full


def start_tracking():
    # 1. Загрузка модели
    print(f"Загрузка модели на {DEVICE}...")
    model = EyeSegmentationModel(n_classes=4).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"ОШИБКА: Файл {MODEL_PATH} не найден!")
        return
    model.eval()

    # 2. Открытие видеопотока
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Ошибка открытия видео!")
        return

    print("Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret: break  # Конец видео

        # Обработка кадра
        mask_only = process_frame(frame, model)

        # Объединяем оригинал и маску для отображения (горизонтально)
        # Слева - оригинал, Справа - только маска на черном фоне
        combined_view = np.hstack((frame, mask_only))

        # Показываем результат
        cv2.imshow('Eye Tracker: Original (Left) | Mask (Right)', combined_view)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_tracking()
