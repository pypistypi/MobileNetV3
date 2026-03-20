import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from s2_model_unet import EyeSegmentationModel

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')
SEGMENTOR_PATH = os.path.join(BASE_DIR, 'pretrained_eye_model.pth')
INPUT_DIR = os.path.join(BASE_DIR, 'input_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_results')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Цвета (BGR)
COLOR_SCLERA = (220, 220, 220)  # Белок
COLOR_IRIS = (0, 255, 0)  # Радужка
COLOR_PUPIL = (0, 0, 255)  # Зрачок
COLOR_GLINT = (255, 255, 0)  # Блики


# -----------------

def get_pure_segmentation(model, img_crop, h_orig, w_orig):
    img_resized = cv2.resize(img_crop, (256, 256))
    input_tensor = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        # Получаем маску всех 4 классов через argmax
        mask_256 = torch.argmax(output[0], dim=0).cpu().numpy().astype(np.uint8)

    # Вывод для отладки: какие классы вообще есть в маске?
    unique_classes = np.unique(mask_256)
    print(f"Найдено классов в маске: {unique_classes}") # Раскомментируйте для проверки в консоли

    mask_full = cv2.resize(mask_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return mask_full


def process_pipeline():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    detector = YOLO(DETECTOR_PATH)
    segmentor = EyeSegmentationModel(n_classes=4).to(DEVICE)
    segmentor.load_state_dict(torch.load(SEGMENTOR_PATH, map_location=DEVICE))
    segmentor.eval()

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.bmp'))]

    for img_name in image_files:
        img = cv2.imread(os.path.join(INPUT_DIR, img_name))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_orig, w_orig = img.shape[:2]

        final_overlay = img.copy()
        final_mask_only = np.zeros_like(img)

        results = detector.predict(img, conf=0.01, verbose=False)
        targets = [map(int, b.xyxy[0]) for b in results[0].boxes] if results[0].boxes else [(0, 0, w_orig, h_orig)]

        for (x1, y1, x2, y2) in targets:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            ch, cw = crop.shape[:2]

            mask = get_pure_segmentation(segmentor, crop, ch, cw)

            # --- ИСПРАВЛЕННАЯ ЛОГИКА ОТРИСОВКИ ---
            color_crop = np.zeros_like(crop)

            # 1. Склера (Класс 1)
            color_crop[mask == 1] = COLOR_SCLERA

            # 2. РАДУЖКА (Исправление: если класс 2 пуст, берем часть класса 1 или 3)
            # Временно закрашиваем класс 2, если он есть
            color_crop[mask == 2] = COLOR_IRIS

            # 3. ЗРАЧОК (Класс 3)
            color_crop[mask == 3] = COLOR_PUPIL

            # --- ХИТРОСТЬ ДЛЯ ВОЗВРАТА РАДУЖКИ ---
            # Если нейросеть "слила" радужку со склерой (класс 1),
            # мы выделим её вокруг зрачка (класс 3)
            if 2 not in np.unique(mask) and 3 in np.unique(mask):
                # Создаем зону вокруг зрачка (радужка обычно окружает зрачок)
                pupil_mask = (mask == 3).astype(np.uint8)
                kernel = np.ones((15, 15), np.uint8)  # Размер "кольца" радужки
                iris_zone = cv2.dilate(pupil_mask, kernel, iterations=2)
                # Убираем сам зрачок из этой зоны
                iris_zone = cv2.subtract(iris_zone, pupil_mask)
                # Ограничиваем зоной, которую сеть считает "глазом" (класс 1)
                iris_final = cv2.bitwise_and(iris_zone, iris_zone, mask=(mask == 1).astype(np.uint8))
                color_crop[iris_final > 0] = COLOR_IRIS

            # 4. БЛИКИ
            _, glints = cv2.threshold(gray[y1:y2, x1:x2], 230, 255, cv2.THRESH_BINARY)
            eye_zone = (mask > 0).astype(np.uint8)
            glints = cv2.bitwise_and(glints, glints, mask=eye_zone)
            color_crop[glints > 0] = COLOR_GLINT

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"result_{img_name}"), final_overlay)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_only_{img_name}"), final_mask_only)
        print(f"Обработано: {img_name}")


if __name__ == "__main__":
    process_pipeline()
