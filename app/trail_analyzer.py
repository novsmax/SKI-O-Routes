"""
trail_extraction_test.py - Скрипт для выделения лыжней из карт.
Реализует выделение лыжных трасс с последующим совмещением результатов.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import os
from sklearn.cluster import DBSCAN


HSV_PARAMS = {
    "h_min": 55,
    "h_max": 70,
    "s_min": 50,
    "s_max": 255,
    "v_min": 50,
    "v_max": 215
}


MIN_JUNCTION_DISTANCE = 10
MIN_BRANCH_LENGTH = 5
JUNCTION_CIRCLE_RADIUS = 3


SHOW_STEPS = True


OVERLAY_ALPHA = 0.7


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
    return img


def extract_mask(image, lower_hsv, upper_hsv, max_area_ratio=0.1):
    """
    Выделение лыжней с помощью цветовой фильтрации с игнорированием больших областей.

    Параметры:
    - image: Исходное изображение (BGR формат из OpenCV)
    - lower_hsv: Нижняя граница диапазона HSV [h, s, v]
    - upper_hsv: Верхняя граница диапазона HSV [h, s, v]
    - max_area_ratio: Максимальное отношение площади связной области к общей площади карты

    Возвращает:
    - mask: Бинарная маска с выделенными лыжнями
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)


    total_area = image.shape[0] * image.shape[1]


    filtered_mask = np.zeros_like(mask)


    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]


        if area / total_area < max_area_ratio:
            filtered_mask[labels == i] = 255


    kernel = np.ones((3, 3), np.uint8)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return filtered_mask


def skeletonize_mask(mask):
    """
    Скелетизация бинарной маски для получения центральных линий лыжней.

    Параметры:
    - mask: Бинарная маска с выделенными лыжнями

    Возвращает:
    - skeleton: Скелетизированное изображение
    """

    binary = mask > 0


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


    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)


    neighbor_count = cv2.filter2D(skel_uint8, -1, kernel)


    endpoints = np.logical_and(skel_uint8 > 0, neighbor_count == 1)
    endpoints_coords = np.where(endpoints)


    cleaned_skeleton = skel_uint8.copy()


    for y, x in zip(endpoints_coords[0], endpoints_coords[1]):

        current_y, current_x = y, x
        branch_points = [(current_y, current_x)]
        branch_length = 0

        while True:
            branch_length += 1

            if (current_y <= 0 or current_y >= skel_uint8.shape[0]-1 or
                current_x <= 0 or current_x >= skel_uint8.shape[1]-1):
                break

            neighborhood = skel_uint8[current_y-1:current_y+2, current_x-1:current_x+2].copy()
            neighborhood[1, 1] = 0  

            next_points = list(zip(*np.where(neighborhood > 0)))

            if len(next_points) != 1:


                break

            dy, dx = next_points[0]
            next_y, next_x = current_y + (dy - 1), current_x + (dx - 1)

            branch_points.append((next_y, next_x))
            current_y, current_x = next_y, next_x

        if branch_length < min_length:
            for point_y, point_x in branch_points:
                cleaned_skeleton[point_y, point_x] = 0

    return cleaned_skeleton


def find_junction_points_improved(skeleton, min_distance=10, cleanup_branches=True):
    """
    Улучшенный алгоритм для нахождения перекрестков лыжней.
    """
    if cleanup_branches:
        skeleton = cleanup_short_branches(skeleton, min_length=MIN_BRANCH_LENGTH)

    skel_uint8 = skeleton.astype(np.uint8) if skeleton.dtype != np.uint8 else skeleton

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    neighbors = cv2.filter2D(skel_uint8, -1, kernel)

    junction_mask = np.logical_and(skel_uint8 > 0, neighbors >= 3)
    junction_coords = np.column_stack(np.where(junction_mask))

    verified_junctions = []

    for y, x in junction_coords:
        if y <= 0 or y >= skel_uint8.shape[0]-1 or x <= 0 or x >= skel_uint8.shape[1]-1:
            continue

        patch = skel_uint8[y-1:y+2, x-1:x+2].copy()
        patch[1, 1] = 0  

        if np.sum(patch) == 0:
            continue

        
        _, labeled = cv2.connectedComponents(patch.astype(np.uint8), connectivity=4)
        num_branches = len(np.unique(labeled)) - 1  

        if num_branches >= 3:
            verified_junctions.append((y, x))



    if verified_junctions:
        junction_coords = np.array(verified_junctions)
    else:
        return np.array([]).reshape(0, 2)

    if min_distance > 1 and len(junction_coords) > 0:
        clustering = DBSCAN(eps=min_distance, min_samples=1).fit(junction_coords)
        labels = clustering.labels_

        unique_labels = np.unique(labels)
        junction_coords_clustered = []

        for label in unique_labels:
            mask = labels == label
            points = junction_coords[mask]
            centroid = np.mean(points, axis=0).astype(int)
            junction_coords_clustered.append(centroid)

        junction_coords = np.array(junction_coords_clustered)


    return junction_coords


def process_map_with_dashed_lines(image_path, hsv_params, base_dir="test_results", show_steps=False, alpha=0.7):
    """
    Обработка карты с выделением всех лыжней одинаковыми тонкими линиями,
    включая достраивание пунктирных линий.
    """
    save_dir = get_save_dir(image_path, base_dir)

    image = load_image(image_path)



    lower_hsv = [hsv_params["h_min"], hsv_params["s_min"], hsv_params["v_min"]]
    upper_hsv = [hsv_params["h_max"], hsv_params["s_max"], hsv_params["v_max"]]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    initial_mask = green_mask.copy()
    initial_mask[blue_mask > 0] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(initial_mask, connectivity=8)

    total_area = image.shape[0] * image.shape[1]

    mask = np.zeros_like(initial_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area / total_area < 0.05:
            mask[labels == i] = 255

    solid_lines_mask = mask.copy()

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)

    solid_lines_mask = cv2.morphologyEx(solid_lines_mask, cv2.MORPH_OPEN, kernel_small)
    solid_lines = cv2.morphologyEx(solid_lines_mask, cv2.MORPH_OPEN, kernel_large)
    dashed_lines = cv2.subtract(mask, solid_lines)

    dashed_connected = cv2.dilate(dashed_lines, kernel_medium, iterations=2)
    dashed_connected = cv2.erode(dashed_connected, kernel_small, iterations=1)

    dashed_skeleton = skeletonize_mask(dashed_connected)

    all_lines = cv2.bitwise_or(solid_lines, dashed_skeleton)
    final_skeleton = skeletonize_mask(all_lines)
    junctions = find_junction_points_improved(final_skeleton, min_distance=MIN_JUNCTION_DISTANCE, cleanup_branches=True)

    print(f"Найдено {len(junctions)} перекрестков")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = rgb_image.copy()

    lines_overlay = np.zeros_like(rgb_image)

    lines_overlay[final_skeleton > 0] = [0, 255, 0]

    overlay = cv2.addWeighted(overlay, 1 - alpha, lines_overlay, alpha, 0)

    for y, x in junctions:
        cv2.circle(overlay, (x, y), JUNCTION_CIRCLE_RADIUS, [255, 255, 0], -1)

    if show_steps:
        plt.figure(figsize=(15, 15))

        plt.subplot(2, 3, 1)
        plt.title('Исходное изображение')
        plt.imshow(rgb_image)
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('Исходная маска лыжней')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title('Сплошные линии')
        plt.imshow(solid_lines, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.title('Пунктирные линии (достроенные)')
        plt.imshow(dashed_skeleton, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('Общий скелет линий')
        plt.imshow(final_skeleton, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        temp_img = rgb_image.copy()
        for y, x in junctions:
            cv2.circle(temp_img, (x, y), JUNCTION_CIRCLE_RADIUS, [255, 255, 0], -1)
        plt.title(f'Перекрестки ({len(junctions)})')
        plt.imshow(temp_img)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    cv2.imwrite(os.path.join(save_dir, "all_lines_skeleton.png"), final_skeleton)
    cv2.imwrite(
        os.path.join(save_dir, "result_uniform_lines.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    np.save(os.path.join(save_dir, "junctions.npy"), junctions)

    print(f"Результаты сохранены в: {save_dir}")
    return mask, final_skeleton, junctions

