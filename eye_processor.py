import cv2
import numpy as np


class EyeRefiner:
    @staticmethod
    def get_adaptive_thresholds(image_gray, mask_area):
        """Вычисляет статистическую норму цвета для конкретной области"""
        pixels = image_gray[mask_area > 0]
        if len(pixels) == 0: return 0, 255
        mean = np.mean(pixels)
        std = np.std(pixels)
        return mean - 2 * std, mean + 2 * std

    @staticmethod
    def starburst_contour(image_gray, center, initial_radius, num_rays=64):
        """
        Находит точки максимального градиента (границы) вдоль лучей.
        Возвращает набор точек для построения реального контура.
        """
        points = []
        h, w = image_gray.shape
        cx, cy = center

        for angle in np.linspace(0, 2 * np.pi, num_rays):
            dx, dy = np.cos(angle), np.sin(angle)
            # Ищем границу в диапазоне +/- 30% от начального радиуса
            for r in range(int(initial_radius * 0.7), int(initial_radius * 1.3)):
                px, py = int(cx + r * dx), int(cy + r * dy)
                if 1 < px < w - 1 and 1 < py < h - 1:
                    # Вычисляем локальный градиент
                    grad = abs(int(image_gray[py, px + 1]) - int(image_gray[py, px - 1]))
                    if grad > 30:  # Адаптивный порог контраста
                        points.append([px, py])
                        break
        return np.array(points, dtype=np.int32)

    @staticmethod
    def smooth_contour(points):
        """Сглаживание контура без потери геометрии (аппроксимация)"""
        if len(points) < 5: return points
        # Используем полигональную аппроксимацию для естественности линий
        epsilon = 0.005 * cv2.arcLength(points, True)
        return cv2.approxPolyDP(points, epsilon, True)
