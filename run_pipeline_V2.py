import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from s2_model_unet import EyeSegmentationModel

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')
SEGMENTOR_PATH = os.path.join(BASE_DIR, 'eye_segmentation_final.pth')
INPUT_DIR = os.path.join(BASE_DIR, 'input_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_results')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Цвета (BGR)
COLOR_SCLERA = (230, 230, 230)  # Белок
COLOR_IRIS = (0, 255, 0)  # Радужка
COLOR_PUPIL = (0, 0, 255)  # Зрачок
COLOR_GLINT = (255, 255, 0)  # Блики


# -----------------

def starburst_refine_ellipse(image_gray, initial_mask, class_id, num_rays=48):
    """Уточнение границы зрачка/радужки с помощью Starburst и вписывания эллипса"""
    pixels = np.where(initial_mask == class_id)
    if len(pixels[0]) < 10: return None

    cy, cx = np.mean(pixels[0]), np.mean(pixels[1])
    points = []
    h, w = image_gray.shape

    for angle in np.linspace(0, 2 * np.pi, num_rays):
        dx, dy = np.cos(angle), np.sin(angle)
        for r in range(3, 80):
            px, py = int(cx + r * dx), int(cy + r * dy)
            if px < 1 or px >= w - 1 or py < 1 or py >= h - 1: break
            # Ищем максимальный градиент вдоль луча
            grad = abs(int(image_gray[py, px + 1]) - int(image_gray[py, px - 1])) + \
                   abs(int(image_gray[py + 1, px]) - int(image_gray[py - 1, px]))
            if grad > 25:  # Порог границы
                points.append((px, py))
                break

    if len(points) >= 5:
        return cv2.fitEllipse(np.array(points))
    return None


def get_refined_prediction(model, img_crop):
    """Нейросетевая сегментация с постобработкой"""
    h, w = img_crop.shape[:2]
    img_resized = cv2.resize(img_crop, (256, 256))
    img_tensor = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out[0], dim=0).cpu().numpy()
        mask_256 = np.argmax(probs, axis=0).astype(np.uint8)

    mask_full = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_full, img_resized


def process_pipeline():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    detector = YOLO(DETECTOR_PATH)
    segmentor = EyeSegmentationModel(n_classes=4).to(DEVICE)
    segmentor.load_state_dict(torch.load(SEGMENTOR_PATH, map_location=DEVICE))
    segmentor.eval()

    for img_name in [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.bmp'))]:
        img = cv2.imread(os.path.join(INPUT_DIR, img_name))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        res_img = img.copy()
        mask_img = np.zeros_like(img)

        # 1. Детекция (YOLO)
        results = detector.predict(img, conf=0.01, verbose=False)
        targets = [map(int, b.xyxy[0]) for b in results[0].boxes] if results[0].boxes else [(0, 0, w, h)]

        for (x1, y1, x2, y2) in targets:
            # Небольшое расширение зоны для надежности
            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue

            # 2. Сегментация
            mask, _ = get_refined_prediction(segmentor, crop)

            # 3. Starburst уточнение для зрачка (класс 3)
            pupil_ellipse = starburst_refine_ellipse(gray[y1:y2, x1:x2], mask, 3)

            # Рисуем элементы
            crop_mask = np.zeros_like(crop)
            crop_mask[mask == 1] = COLOR_SCLERA
            crop_mask[mask == 2] = COLOR_IRIS
            crop_mask[mask == 3] = COLOR_PUPIL

            # Если Starburst нашел эллипс - перекрываем им маску для идеальной формы
            if pupil_ellipse:
                cv2.ellipse(crop_mask, pupil_ellipse, COLOR_PUPIL, -1)
                cv2.ellipse(res_img[y1:y2, x1:x2], pupil_ellipse, (255, 255, 255), 1)  # Белый контур

            # 4. Блики
            _, glints = cv2.threshold(gray[y1:y2, x1:x2], 240, 255, cv2.THRESH_BINARY)
            crop_mask[glints > 0] = COLOR_GLINT

            # Сглаживание краев маски
            crop_mask = cv2.GaussianBlur(crop_mask, (3, 3), 0)

            # Наложение
            res_img[y1:y2, x1:x2] = cv2.addWeighted(res_img[y1:y2, x1:x2], 1.0, crop_mask, 0.4, 0)
            mask_img[y1:y2, x1:x2] = crop_mask

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"result_{img_name}"), res_img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_only_{img_name}"), mask_img)
        print(f"Идеально обработано: {img_name}")


if __name__ == "__main__":
    process_pipeline()
