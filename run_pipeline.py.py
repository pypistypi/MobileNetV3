import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from s2_model_unet import EyeSegmentationModel  # Убедитесь, что переименовали файл!

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')
SEGMENTOR_PATH = os.path.join(BASE_DIR, 'eye_segmentation_final.pth')
INPUT_DIR = os.path.join(BASE_DIR, 'input_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_results')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Цвета для ИДЕАЛЬНОЙ визуализации (BGR)
COLOR_SCLERA = (240, 240, 240)  # Светло-серый (белок)
COLOR_IRIS = (0, 255, 0)  # Зеленый (радужка)
COLOR_PUPIL = (0, 0, 255)  # Красный (зрачок)
COLOR_GLINT = (255, 255, 0)  # Ярко-голубой/желтый (блики)


# -----------------

def get_segmentation_refined(model, img_crop, h_orig, w_orig):
    """Получение плавной маски высокого качества"""
    img_resized = cv2.resize(img_crop, (256, 256))
    input_tensor = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        # Используем Softmax для получения вероятностей (нужно для плавных краев)
        probs = torch.softmax(output[0], dim=0).cpu().numpy()
        mask = np.argmax(probs, axis=0).astype(np.uint8)

    # Создаем цветную маску 256x256
    color_mask_256 = np.zeros((256, 256, 3), dtype=np.uint8)
    color_mask_256[mask == 1] = COLOR_SCLERA
    color_mask_256[mask == 2] = COLOR_IRIS
    color_mask_256[mask == 3] = COLOR_PUPIL

    # Плавное растягивание до оригинального размера (Anti-aliasing)
    color_mask_full = cv2.resize(color_mask_256, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    # Легкое размытие для устранения "лесенок" на краях
    color_mask_full = cv2.GaussianBlur(color_mask_full, (3, 3), 0)

    return color_mask_full, mask


def detect_glints_v3(image_gray, pupil_mask_full):
    """Поиск бликов внутри зоны зрачка"""
    # Порог для самых ярких точек (бликов)
    _, glints = cv2.threshold(image_gray, 235, 255, cv2.THRESH_BINARY)
    # Ограничиваем поиск зоной зрачка (с небольшим расширением)
    kernel = np.ones((5, 5), np.uint8)
    dilated_pupil = cv2.dilate(pupil_mask_full, kernel, iterations=1)
    glints = cv2.bitwise_and(glints, glints, mask=dilated_pupil)
    return glints


def process_pipeline():
    if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR); return
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(f"Загрузка моделей на {DEVICE}...")
    detector = YOLO(DETECTOR_PATH)
    segmentor = EyeSegmentationModel(n_classes=4).to(DEVICE)  # Теперь 4 класса!
    segmentor.load_state_dict(torch.load(SEGMENTOR_PATH, map_location=DEVICE))
    segmentor.eval()

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Найдено изображений: {len(image_files)}")

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        full_img = cv2.imread(img_path)
        if full_img is None: continue

        gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        h_orig, w_orig = full_img.shape[:2]

        # Итоговые изображения
        final_overlay = full_img.copy()
        final_mask_only = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)

        # 1. Детекция глаз (YOLO)
        results = detector.predict(full_img, conf=0.01, iou=0.3, agnostic_nms=True, verbose=False)
        boxes = results[0].boxes

        # Список областей для обработки (если YOLO не нашла - берем всё фото)
        targets = []
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                targets.append((x1, y1, x2, y2))
        else:
            targets.append((0, 0, w_orig, h_orig))

        for (x1, y1, x2, y2) in targets:
            # Обрезаем область глаза
            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w <= 0 or crop_h <= 0: continue
            eye_crop = full_img[y1:y2, x1:x2]

            # Сегментация (Склера, Радужка, Зрачок)
            color_mask_crop, raw_mask_256 = get_segmentation_refined(segmentor, eye_crop, crop_h, crop_w)

            # Детекция Бликов (Glints)
            # Растягиваем маску зрачка для детектора бликов
            pupil_mask_256 = (raw_mask_256 == 3).astype(np.uint8)
            pupil_mask_full = cv2.resize(pupil_mask_256, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            glints = detect_glints_v3(gray[y1:y2, x1:x2], pupil_mask_full)

            # Добавляем блики на цветную маску
            color_mask_crop[glints > 0] = COLOR_GLINT

            # Накладываем результат на итоговые фото
            # 0.5 - коэффициент прозрачности (можно менять от 0.1 до 1.0)
            final_overlay[y1:y2, x1:x2] = cv2.addWeighted(final_overlay[y1:y2, x1:x2], 1.0, color_mask_crop, 0.5, 0)
            final_mask_only[y1:y2, x1:x2] = color_mask_crop

            # Рисуем тонкую рамку вокруг найденного глаза
            if len(boxes) > 0:
                cv2.rectangle(final_overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Сохранение
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"result_{img_name}"), final_overlay)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_only_{img_name}"), final_mask_only)
        print(f"Обработано: {img_name}")


if __name__ == "__main__":
    process_pipeline()
