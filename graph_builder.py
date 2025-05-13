"""
graph_builder.py - Модуль для построения графовой модели лыжней и поиска оптимальных маршрутов.

Этот модуль отвечает за преобразование скелетизированных лыжней и обнаруженных перекрестков
в графовую модель, расчет весов рёбер, поиск оптимальных маршрутов между контрольными пунктами
и визуализацию результатов.
"""

import os
import numpy as np
import networkx as nx
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from trail_analyzer import process_map_with_dashed_lines


def build_graph_from_skeleton(
        skeleton_image: np.ndarray,
        junctions: np.ndarray,
        scale_factor: float = 1.0,
        hsv_image: Optional[np.ndarray] = None
) -> nx.Graph:
    """
    Построение графа из скелетизированного изображения лыжней и координат перекрестков
    Оптимизированная версия с сохранением изначальной логики

    Args:
        skeleton_image: Бинарное изображение скелетизированных лыжней (значения 255 для лыжней)
        junctions: Массив координат перекрестков [[y1, x1], [y2, x2], ...]
        scale_factor: Коэффициент масштаба (метров на пиксель)
        hsv_image: Не используется, оставлен для совместимости

    Returns:
        G: Граф, представляющий сеть лыжней
    """
    import time
    start_time = time.time()
    import networkx as nx

    print(f"Построение графа из {len(junctions)} перекрестков...")

    # 1. Создаем граф
    G = nx.Graph()

    # 2. Добавляем вершины (перекрестки)
    for i, (y, x) in enumerate(junctions):
        G.add_node(i, pos=(x, y), coordinates=(y, x), type="junction")

    # 3. Создаем более эффективную структуру для проверки близости к перекресткам
    junction_proximity_radius = 7  # Увеличиваем радиус для надежности
    junction_dict = {}

    for j_idx, (jy, jx) in enumerate(junctions):
        junction_dict[j_idx] = (jy, jx)

    # 4. Получаем координаты всех пикселей скелета
    y_coords, x_coords = np.where(skeleton_image > 0)
    skeleton_coords = list(zip(y_coords, x_coords))

    # 5. Создаем изображение с метками сегментов
    labeled_image = np.zeros_like(skeleton_image, dtype=int)
    current_label = 1

    # 6. Для каждого пикселя скелета
    segments_processed = 0

    for y, x in skeleton_coords:
        # Пропускаем, если пиксель уже помечен
        if labeled_image[y, x] != 0:
            continue

        # Запускаем трассировку сегмента от текущего пикселя
        current_segment = [(y, x)]
        labeled_image[y, x] = current_label

        # Начальные перекрестки для этого сегмента
        segment_junctions = set()

        # Проверяем все перекрестки для стартовой точки сегмента
        for j_idx, (jy, jx) in enumerate(junctions):
            if (y - jy) ** 2 + (x - jx) ** 2 <= junction_proximity_radius ** 2:
                segment_junctions.add(j_idx)

        # Очередь для обхода в ширину
        queue = [(y, x)]
        visited = set([(y, x)])

        # Обход сегмента
        while queue:
            cy, cx = queue.pop(0)

            # Проверяем 8-соседство
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = cy + dy, cx + dx

                    # Пропускаем, если вышли за границы изображения
                    if not (0 <= ny < skeleton_image.shape[0] and 0 <= nx < skeleton_image.shape[1]):
                        continue

                    # Проверяем, что это пиксель скелета и не посещен
                    if skeleton_image[ny, nx] > 0 and (ny, nx) not in visited:
                        junction_nearby = False

                        # Проверяем близость к перекресткам
                        for j_idx, (jy, jx) in enumerate(junctions):
                            if (ny - jy) ** 2 + (nx - jx) ** 2 <= junction_proximity_radius ** 2:
                                segment_junctions.add(j_idx)
                                junction_nearby = True

                        # Добавляем точку в сегмент независимо от близости к перекрестку
                        labeled_image[ny, nx] = current_label
                        current_segment.append((ny, nx))
                        visited.add((ny, nx))

                        # Если точка не рядом с перекрестком, продолжаем трассировку
                        if not junction_nearby:
                            queue.append((ny, nx))

        # Если сегмент соединяет ровно 2 разных перекрестка
        if len(segment_junctions) == 2:
            node1, node2 = list(segment_junctions)

            # Измеряем длину сегмента (количество пикселей)
            segment_length_pixels = len(current_segment)
            segment_length_meters = segment_length_pixels * scale_factor

            # Рассчитываем итоговый вес
            weight = calculate_final_weight(segment_length_meters)

            # Добавляем ребро
            G.add_edge(
                node1,
                node2,
                weight=weight,
                length_meters=segment_length_meters,
                length_pixels=segment_length_pixels,
                trail_type='normal',
                speed_factor=1.0,
                coords_y=np.array([p[0] for p in current_segment]),
                coords_x=np.array([p[1] for p in current_segment])
            )

            segments_processed += 1

        current_label += 1

    end_time = time.time()
    print(f"Обработано {segments_processed} сегментов")
    print(f"Граф построен: {G.number_of_nodes()} вершин, {G.number_of_edges()} рёбер")
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")

    return G


def get_trail_type(
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    hsv_image: Optional[np.ndarray] = None
) -> Tuple[str, float]:
    """
    Упрощенная версия функции, без определения типа лыжни

    Args:
        y_coords: массив Y-координат пикселей сегмента
        x_coords: массив X-координат пикселей сегмента
        hsv_image: HSV-изображение карты (не используется)

    Returns:
        тип лыжни ('normal') и коэффициент скорости (1.0)
    """
    # Всегда возвращаем стандартный тип и коэффициент
    return 'normal', 1.0


def calculate_final_weight(
    length_meters: float,
    trail_type_factor: float = 1.0
) -> float:
    """
    Упрощенная версия расчета веса ребра, без учета типа лыжни и рельефа

    Args:
        length_meters: длина сегмента в метрах
        trail_type_factor: не используется, оставлен для совместимости

    Returns:
        вес ребра (время в секундах)
    """
    # Базовое время в секундах (скорость 5 м/с)
    return length_meters / 5.0


def remove_disconnected_vertices(G: nx.Graph) -> nx.Graph:
    """
    Удаляет все вершины, которые не связаны с самой большой компонентой связности графа

    Args:
        G: Исходный граф

    Returns:
        Граф, содержащий только самую большую компоненту связности
    """
    # Находим все компоненты связности
    components = list(nx.connected_components(G))

    if not components:
        print("Граф не содержит компонент связности")
        return G

    # Находим самую большую компоненту
    largest_component = max(components, key=len)

    # Создаем подграф только с вершинами из самой большой компоненты
    largest_subgraph = G.subgraph(largest_component).copy()

    print(f"Исходный граф: {G.number_of_nodes()} вершин, {G.number_of_edges()} рёбер")
    print(f"Самая большая компонента: {largest_subgraph.number_of_nodes()} вершин, {largest_subgraph.number_of_edges()} рёбер")
    print(f"Удалено {G.number_of_nodes() - largest_subgraph.number_of_nodes()} изолированных вершин")

    return largest_subgraph


def find_optimal_route(
    G: nx.Graph,
    start_node: int,
    end_node: int
) -> Tuple[Optional[List[int]], float]:
    """
    Находит оптимальный маршрут между двумя перекрестками

    Args:
        G: граф лыжной сети
        start_node: индекс начального перекрестка
        end_node: индекс конечного перекрестка

    Returns:
        path: список индексов перекрестков, образующих оптимальный маршрут
        length: общая длина маршрута (в секундах)
    """
    try:
        # Используем алгоритм Дейкстры для поиска кратчайшего пути
        path = nx.dijkstra_path(G, start_node, end_node, weight='weight')

        # Вычисляем общую длину пути
        length = nx.dijkstra_path_length(G, start_node, end_node, weight='weight')

        return path, length
    except nx.NetworkXNoPath:
        print(f"Не удалось найти путь между перекрестками {start_node} и {end_node}")
        return None, float('inf')


def find_optimal_route_multiple_kps(
    G: nx.Graph,
    control_points: List[int]
) -> Tuple[Optional[List[int]], float]:
    """
    Находит оптимальный маршрут через несколько контрольных пунктов

    Args:
        G: граф лыжной сети
        control_points: список индексов контрольных пунктов, которые нужно посетить

    Returns:
        path: список индексов перекрестков, образующих оптимальный маршрут
        total_length: общая длина маршрута (в секундах)
    """
    if len(control_points) < 2:
        return [], 0

    full_path = []
    total_length = 0

    # Проходим по контрольным пунктам последовательно
    for i in range(len(control_points) - 1):
        start_cp = control_points[i]
        end_cp = control_points[i + 1]

        # Ищем путь между парой контрольных пунктов
        path, length = find_optimal_route(G, start_cp, end_cp)

        if path is None:
            print(f"Предупреждение: не удалось найти путь от КП {start_cp} до КП {end_cp}")
            return None, float('inf')

        # Добавляем только уникальные перекрестки (чтобы избежать дублирования)
        if i > 0:
            # Пропускаем первый перекресток, так как он уже должен быть в full_path
            path = path[1:]

        full_path.extend(path)
        total_length += length

    return full_path, total_length


def visualize_route(
        original_image: np.ndarray,
        G: nx.Graph,
        path: List[int],
        junctions: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True
) -> np.ndarray:
    """
    Визуализирует найденный маршрут на карте.
    Отображает только перекрестки и участки пути оптимального маршрута.

    Args:
        original_image: исходное изображение карты
        G: граф лыжной сети
        path: оптимальный маршрут (список индексов перекрестков)
        junctions: координаты перекрестков
        save_path: путь для сохранения результата (опционально)
        show: показывать ли результат

    Returns:
        result: визуализация маршрута на карте
    """
    result = original_image.copy()

    # Создаем словарь для быстрого доступа к координатам перекрестков
    junction_coords = {}
    for i in range(len(junctions)):
        junction_coords[i] = (junctions[i][0], junctions[i][1])

    # Рисуем только перекрестки, которые входят в путь
    if path:
        for i, node in enumerate(path):
            if node in junction_coords:
                y, x = junction_coords[node]

                # Перекрестки на пути - красные кружки
                color = (0, 0, 255)  # BGR - красный
                radius = 7

                # Начальная точка - зеленая
                if i == 0:
                    color = (0, 255, 0)  # BGR - зеленый
                    radius = 12

                    # Добавляем метку "Start" большим шрифтом
                    cv2.putText(
                        result,
                        "Start",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,  # Увеличенный размер шрифта
                        color,
                        3  # Более толстые линии
                    )

                # Конечная точка - синяя
                elif i == len(path) - 1:
                    color = (255, 0, 0)  # BGR - синий
                    radius = 12

                    # Добавляем метку "Finish" большим шрифтом
                    cv2.putText(
                        result,
                        "Finish",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,  # Увеличенный размер шрифта
                        color,
                        3  # Более толстые линии
                    )

                cv2.circle(result, (x, y), radius, color, -1)

        # Рисуем маршрут
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]

            if node1 in junction_coords and node2 in junction_coords:
                # Проверяем, есть ли ребро между этими вершинами
                if G.has_edge(node1, node2):
                    # Если есть точные координаты сегмента, используем их
                    if 'coords_y' in G[node1][node2] and 'coords_x' in G[node1][node2]:
                        coords_y = G[node1][node2]['coords_y']
                        coords_x = G[node1][node2]['coords_x']

                        # Рисуем линию маршрута по точным координатам
                        for j in range(len(coords_y) - 1):
                            pt1 = (coords_x[j], coords_y[j])
                            pt2 = (coords_x[j + 1], coords_y[j + 1])
                            cv2.line(result, pt1, pt2, (0, 0, 255), 3)
                    else:
                        # Если точных координат нет, рисуем прямую линию
                        y1, x1 = junction_coords[node1]
                        y2, x2 = junction_coords[node2]
                        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Сохраняем результат, если указан путь
    if save_path:
        cv2.imwrite(save_path, result)

    # Показываем результат
    if show:
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Оптимальный маршрут')
        plt.tight_layout()
        plt.show()

    return result


def visualize_graph(
    G: nx.Graph,
    junctions: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Визуализирует построенный граф с использованием networkx

    Args:
        G: граф лыжной сети
        junctions: координаты перекрестков
        save_path: путь для сохранения результата (опционально)
        show: показывать ли результат
    """
    plt.figure(figsize=(12, 10))

    # Создаем словарь позиций вершин
    pos = {}
    for i in G.nodes():
        if i < len(junctions):
            y, x = junctions[i]
            pos[i] = (x, -y)  # Инвертируем y для правильной ориентации

    # Рисуем граф
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='blue')

    # Рисуем ребра
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='blue')

    # Добавляем метки вершин
    if len(G) <= 50:  # Показываем метки только если вершин не слишком много
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title('Графовая модель лыжней')
    plt.axis('off')
    plt.tight_layout()

    # Сохраняем, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=300)

    # Показываем результат
    if show:
        plt.show()


def process_map_and_build_graph(
    skeleton_path: str,
    junctions_path: str,
    original_image_path: str,
    scale_factor: float = 1.0,
    output_dir: Optional[str] = None
) -> nx.Graph:
    """
    Загружает данные, строит графовую модель и визуализирует результаты

    Args:
        skeleton_path: путь к изображению скелетизированных лыжней
        junctions_path: путь к файлу с координатами перекрестков (.npy)
        original_image_path: путь к исходному изображению карты
        scale_factor: коэффициент масштаба (метров на пиксель)
        output_dir: директория для сохранения результатов

    Returns:
        G: построенный граф
    """
    # Создаем директорию для результатов, если она не существует
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Загружаем данные
    skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    junctions = np.load(junctions_path)
    original_image = cv2.imread(original_image_path)

    if skeleton is None or original_image is None:
        raise FileNotFoundError("Не удалось загрузить файлы")

    # Строим граф
    G = build_graph_from_skeleton(skeleton, junctions, scale_factor)

    # Удаляем несвязанные вершины
    G = remove_disconnected_vertices(G)

    # Визуализируем граф
    if output_dir:
        visualize_graph(
            G,
            junctions,
            save_path=os.path.join(output_dir, "graph_visualization.png"),
            show=False
        )

    return G


def process_original_map(
    original_image_path: str,
    scale_factor: float = 1.0,
    hsv_params: Optional[Dict] = None,
    base_dir: str = "results",
    show_steps: bool = False
) -> Tuple[nx.Graph, np.ndarray]:
    """
    Полный процесс обработки карты: от исходного изображения до графовой модели

    Args:
        original_image_path: путь к исходному изображению карты
        scale_factor: коэффициент масштаба (метров на пиксель)
        hsv_params: параметры HSV для выделения лыжней
        base_dir: базовая директория для сохранения результатов
        show_steps: показывать ли промежуточные шаги обработки

    Returns:
        G: построенный граф
        junctions: координаты перекрестков
    """
    # Параметры HSV по умолчанию для выделения лыжней
    if hsv_params is None:
        hsv_params = {
            "h_min": 55,
            "h_max": 70,
            "s_min": 50,
            "s_max": 255,
            "v_min": 50,
            "v_max": 215
        }

    # 1. Выделение лыжней и перекрестков с помощью trail_analyzer
    print("Шаг 1: Выделение лыжней и перекрестков...")

    mask, skeleton, junctions = process_map_with_dashed_lines(
        original_image_path, hsv_params, base_dir, show_steps, 0.7
    )

    # Получаем директорию с результатами обработки
    filename = os.path.basename(original_image_path)
    base_name = os.path.splitext(filename)[0]
    save_dir = os.path.join(base_dir, base_name)

    # 2. Построение графовой модели
    print("Шаг 2: Построение графовой модели...")
    G = build_graph_from_skeleton(skeleton, junctions, scale_factor)

    # 3. Удаление несвязанных вершин
    print("Шаг 3: Удаление несвязанных вершин...")
    G = remove_disconnected_vertices(G)

    # 4. Визуализация графа
    print("Шаг 4: Визуализация графа...")
    visualize_graph(
        G,
        junctions,
        save_path=os.path.join(save_dir, "graph_visualization.png"),
        show=False
    )

    return G, junctions


if __name__ == "__main__":
    # Пример использования
    import argparse

    parser = argparse.ArgumentParser(description='Построение графовой модели лыжней')
    parser.add_argument('--image', required=True, help='Путь к исходному изображению карты')
    parser.add_argument('--scale', type=float, default=1.0, help='Коэффициент масштаба (метров на пиксель)')
    parser.add_argument('--output', default='results', help='Директория для сохранения результатов')
    parser.add_argument('--start', type=int, default=0, help='Индекс начального перекрестка')
    parser.add_argument('--end', type=int, default=1, help='Индекс конечного перекрестка')
    parser.add_argument('--show_steps', action='store_true', help='Показывать промежуточные шаги обработки')

    args = parser.parse_args()

    # Полный процесс обработки карты
    print(f"Обработка карты: {args.image}")
    print(f"Масштаб: {args.scale} м/пиксель")

    # 1. Обрабатываем исходное изображение и строим графовую модель
    G, junctions = process_original_map(
        args.image,
        args.scale,
        None,  # Используем параметры HSV по умолчанию
        args.output,
        args.show_steps
    )

    # 2. Ищем оптимальный маршрут
    path, length = find_optimal_route(G, args.start, args.end)

    if path:
        print(f"Найден оптимальный маршрут длиной {length:.2f} секунд")

        # 3. Визуализируем маршрут
        original_image = cv2.imread(args.image)

        # Получаем директорию для сохранения результатов
        filename = os.path.basename(args.image)
        base_name = os.path.splitext(filename)[0]
        save_dir = os.path.join(args.output, base_name)

        result = visualize_route(
            original_image,
            G,
            path,
            junctions,
            save_path=os.path.join(save_dir, "route_visualization.png"),
            show=True
        )
    else:
        print("Не удалось найти маршрут")