"""
trail_extraction_test.py - Упрощенная версия скрипта для выделения лыжней из карт.
Фокус только на получении бинарной маски и скелетизации с точным выделением перекрестков.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import os

# ==== НАСТРОЙКИ (РЕДАКТИРОВАТЬ ЗДЕСЬ) ====

# Путь к изображению карты
IMAGE_PATH = "3FHqyr7_w7w.jpg"

# Базовая директория для сохранения результатов
BASE_SAVE_DIR = "test_results"

# Оптимизированные параметры HSV для light_green
HSV_PARAMS = {
    "h_min": 35,
    "h_max": 85,
    "s_min": 50,
    "s_max": 255,
    "v_min": 50,
    "v_max": 235
}

# Параметры определения перекрестков
MIN_JUNCTION_DISTANCE = 10     # Минимальное расстояние между перекрестками
MIN_BRANCH_LENGTH = 5          # Минимальная длина ветви для сохранения

# Параметры визуализации
JUNCTION_CIRCLE_RADIUS = 3    # Радиус кругов для отображения перекрестков (уменьшен)

# Показывать ли промежуточные шаги обработки
SHOW_STEPS = False

# ==== КОНЕЦ НАСТРОЕК ====


def get_save_dir(image_path, base_dir="test_results"):
    """Создает директорию для сохранения результатов обработки."""
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    save_dir = os.path.join(base_dir, base_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def load_image(image_path):
    """Загрузка изображения."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    return img  # Возвращаем в BGR формате, так как OpenCV использует BGR


def extract_mask(image, lower_hsv, upper_hsv):
    """
    Выделение лыжней с помощью цветовой фильтрации.

    Параметры:
    - image: Исходное изображение (BGR формат из OpenCV)
    - lower_hsv: Нижняя граница диапазона HSV [h, s, v]
    - upper_hsv: Верхняя граница диапазона HSV [h, s, v]

    Возвращает:
    - mask: Бинарная маска с выделенными лыжнями
    """
    # Конвертация в HSV для лучшего выделения цвета
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Создание маски цвета
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    # Применение морфологических операций для улучшения результата
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def skeletonize_mask(mask):
    """
    Скелетизация бинарной маски для получения центральных линий лыжней.

    Параметры:
    - mask: Бинарная маска с выделенными лыжнями

    Возвращает:
    - skeleton: Скелетизированное изображение
    """
    # Конвертация маски в формат для scikit-image
    binary = mask > 0

    # Применение скелетизации
    skeleton = morphology.skeletonize(binary)
    skeleton_uint8 = skeleton.astype(np.uint8) * 255

    return skeleton_uint8


def cleanup_short_branches(skeleton, min_length=5):
    """
    Удаляет короткие ветви из скелета, которые часто являются шумом.

    Параметры:
    - skeleton: Скелетизированное изображение
    - min_length: Минимальная длина ветви, которую нужно сохранить

    Возвращает:
    - cleaned_skeleton: Очищенный скелет
    """
    skel_uint8 = skeleton.astype(np.uint8) if skeleton.dtype != np.uint8 else skeleton

    # Поиск конечных точек (имеют только одного соседа)
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    # Находим количество соседей
    neighbor_count = cv2.filter2D(skel_uint8, -1, kernel)

    # Находим конечные точки
    endpoints = np.logical_and(skel_uint8 > 0, neighbor_count == 1)
    endpoints_coords = np.where(endpoints)

    # Создаем копию скелета для работы
    cleaned_skeleton = skel_uint8.copy()

    # Обрабатываем каждую конечную точку
    for y, x in zip(endpoints_coords[0], endpoints_coords[1]):
        # Начинаем с конечной точки и двигаемся по ветви
        current_y, current_x = y, x
        branch_points = [(current_y, current_x)]
        branch_length = 0

        while True:
            branch_length += 1

            # Получаем окрестность 3x3
            if (current_y <= 0 or current_y >= skel_uint8.shape[0]-1 or
                current_x <= 0 or current_x >= skel_uint8.shape[1]-1):
                break

            neighborhood = skel_uint8[current_y-1:current_y+2, current_x-1:current_x+2].copy()
            neighborhood[1, 1] = 0  # Исключаем текущую точку

            # Находим соседей
            next_points = list(zip(*np.where(neighborhood > 0)))

            if len(next_points) != 1:
                # Если у точки нет соседей или больше одного соседа, то достигли
                # конца ветви или перекрестка
                break

            # Определяем новую точку относительно текущего положения
            dy, dx = next_points[0]
            next_y, next_x = current_y + (dy - 1), current_x + (dx - 1)

            # Добавляем новую точку к ветви
            branch_points.append((next_y, next_x))
            current_y, current_x = next_y, next_x

        # Если ветвь короткая, удаляем её
        if branch_length < min_length:
            for point_y, point_x in branch_points:
                cleaned_skeleton[point_y, point_x] = 0

    return cleaned_skeleton


def find_junction_points_improved(skeleton, min_distance=10, cleanup_branches=True):
    """
    Улучшенный алгоритм для нахождения перекрестков лыжней.

    Параметры:
    - skeleton: Скелетизированное изображение
    - min_distance: Минимальное расстояние между перекрестками для кластеризации
    - cleanup_branches: Удалять ли короткие ветви перед поиском перекрестков

    Возвращает:
    - junction_coords: Список координат перекрестков
    """
    # Предварительная очистка от коротких ветвей, если требуется
    if cleanup_branches:
        skeleton = cleanup_short_branches(skeleton, min_length=MIN_BRANCH_LENGTH)

    # Подготавливаем скелет
    skel_uint8 = skeleton.astype(np.uint8) if skeleton.dtype != np.uint8 else skeleton

    # Создаем ядро для определения количества соседей
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    # Находим количество соседей для каждой точки скелета
    neighbors = cv2.filter2D(skel_uint8, -1, kernel)

    # Находим точки с 3 или более соседями (потенциальные перекрестки)
    junction_mask = np.logical_and(skel_uint8 > 0, neighbors >= 3)
    junction_coords = np.column_stack(np.where(junction_mask))

    # Проверяем каждую потенциальную точку перекрестка более точно
    verified_junctions = []

    for y, x in junction_coords:
        # Проверяем, что точка находится в границах изображения
        if y <= 0 or y >= skel_uint8.shape[0]-1 or x <= 0 or x >= skel_uint8.shape[1]-1:
            continue

        # Получаем окрестность 3x3
        patch = skel_uint8[y-1:y+2, x-1:x+2].copy()
        patch[1, 1] = 0  # Удаляем центральный пиксель

        if np.sum(patch) == 0:
            continue

        # Используем компонентную связность для определения количества ветвей
        # Важно: для 8-связности нужно использовать 4-связность при определении компонент
        _, labeled = cv2.connectedComponents(patch.astype(np.uint8), connectivity=4)
        num_branches = len(np.unique(labeled)) - 1  # Вычитаем фоновую компоненту

        if num_branches >= 3:
            verified_junctions.append((y, x))

    # Конвертируем в массив numpy
    if verified_junctions:
        junction_coords = np.array(verified_junctions)
    else:
        return np.array([]).reshape(0, 2)

    # Группируем близкие точки с помощью DBSCAN
    if min_distance > 1 and len(junction_coords) > 0:
        from sklearn.cluster import DBSCAN

        # Используем DBSCAN для группировки близких точек
        clustering = DBSCAN(eps=min_distance, min_samples=1).fit(junction_coords)
        labels = clustering.labels_

        # Находим центроиды каждого кластера
        unique_labels = np.unique(labels)
        junction_coords_clustered = []

        for label in unique_labels:
            mask = labels == label
            points = junction_coords[mask]
            centroid = np.mean(points, axis=0).astype(int)
            junction_coords_clustered.append(centroid)

        junction_coords = np.array(junction_coords_clustered)

    return junction_coords


def process_image(image_path, hsv_params, base_dir="test_results", show_steps=False):
    """
    Обработка изображения для выделения лыжней и нахождения перекрестков.

    Параметры:
    - image_path: Путь к изображению карты
    - hsv_params: Словарь с параметрами HSV
    - base_dir: Базовая директория для сохранения результатов
    - show_steps: Показывать ли промежуточные шаги обработки
    """
    # Получаем директорию для сохранения результатов
    save_dir = get_save_dir(image_path, base_dir)

    # Загрузка изображения
    image = load_image(image_path)

    # Параметры цветового диапазона
    lower_hsv = [hsv_params["h_min"], hsv_params["s_min"], hsv_params["v_min"]]
    upper_hsv = [hsv_params["h_max"], hsv_params["s_max"], hsv_params["v_max"]]

    print(f"Обработка изображения: {os.path.basename(image_path)}")
    print("Использование параметров light_green для HSV:")
    print(f"Lower HSV: {lower_hsv}")
    print(f"Upper HSV: {upper_hsv}")

    # Шаг 1: Выделение лыжней по цвету (получаем только маску)
    mask = extract_mask(image, lower_hsv, upper_hsv)

    # Шаг 2: Скелетизация маски
    skeleton = skeletonize_mask(mask)

    # Шаг 3: Находим перекрестки (используем улучшенную функцию)
    junctions = find_junction_points_improved(skeleton, min_distance=MIN_JUNCTION_DISTANCE, cleanup_branches=True)

    print(f"Найдено {len(junctions)} перекрестков")

    # Создаем изображение с наложенным скелетом и перекрестками для визуализации
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертируем для matplotlib
    overlay = rgb_image.copy()

    # Накладываем скелет на исходное изображение
    overlay[skeleton > 0] = [255, 0, 0]  # Красный цвет для лыжней

    # Накладываем перекрестки с уменьшенным радиусом
    for y, x in junctions:
        cv2.circle(overlay, (x, y), JUNCTION_CIRCLE_RADIUS, [0, 0, 255], -1)  # Синий цвет для перекрестков

    # Отображаем результаты, если требуется
    if show_steps:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.title('Исходное изображение')
        plt.imshow(rgb_image)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title('Маска лыжней')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title('Скелет лыжней')
        plt.imshow(skeleton, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title(f'Итоговый результат ({len(junctions)} перекрестков)')
        plt.imshow(overlay)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Сохраняем результаты
    cv2.imwrite(os.path.join(save_dir, "mask.png"), mask)
    cv2.imwrite(os.path.join(save_dir, "skeleton.png"), skeleton)

    # Сохраняем наложение для визуализации (в BGR для OpenCV)
    cv2.imwrite(
        os.path.join(save_dir, "result.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    print(f"Маски сохранены в: {save_dir}")
    return mask, skeleton, junctions


if __name__ == "__main__":
    # Проверяем, существует ли файл
    if not os.path.exists(IMAGE_PATH):
        print(f"Файл не найден: {IMAGE_PATH}")
        exit(1)

    # Создаем базовую директорию для результатов, если она не существует
    if not os.path.exists(BASE_SAVE_DIR):
        os.makedirs(BASE_SAVE_DIR)

    # Обработка изображения с оптимальными параметрами light_green
    mask, skeleton, junctions = process_image(IMAGE_PATH, HSV_PARAMS, BASE_SAVE_DIR, SHOW_STEPS)