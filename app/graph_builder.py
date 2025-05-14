
import os
import numpy as np
import networkx as nx
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from app.trail_analyzer import process_map_with_dashed_lines


def build_graph_from_skeleton(
        skeleton_image: np.ndarray,
        junctions: np.ndarray,
        scale_factor: float = 1.0,
        hsv_image: Optional[np.ndarray] = None
) -> nx.Graph:
    """
    Построение графа из скелетизированного изображения лыжней и координат перекрестков
    """
    import time
    start_time = time.time()
    import networkx as nx

    print(f"Построение графа из {len(junctions)} перекрестков...")

    G = nx.Graph()

    for i, (y, x) in enumerate(junctions):
        G.add_node(i, pos=(x, y), coordinates=(y, x), type="junction")

    junction_proximity_radius = 7
    junction_dict = {}

    for j_idx, (jy, jx) in enumerate(junctions):
        junction_dict[j_idx] = (jy, jx)

    y_coords, x_coords = np.where(skeleton_image > 0)
    skeleton_coords = list(zip(y_coords, x_coords))

    labeled_image = np.zeros_like(skeleton_image, dtype=int)
    current_label = 1

    segments_processed = 0

    for y, x in skeleton_coords:
        if labeled_image[y, x] != 0:
            continue

        current_segment = [(y, x)]
        labeled_image[y, x] = current_label

        segment_junctions = set()

        for j_idx, (jy, jx) in enumerate(junctions):
            if (y - jy) ** 2 + (x - jx) ** 2 <= junction_proximity_radius ** 2:
                segment_junctions.add(j_idx)

        queue = [(y, x)]
        visited = set([(y, x)])


        while queue:
            cy, cx = queue.pop(0)


            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = cy + dy, cx + dx


                    if not (0 <= ny < skeleton_image.shape[0] and 0 <= nx < skeleton_image.shape[1]):
                        continue


                    if skeleton_image[ny, nx] > 0 and (ny, nx) not in visited:
                        junction_nearby = False


                        for j_idx, (jy, jx) in enumerate(junctions):
                            if (ny - jy) ** 2 + (nx - jx) ** 2 <= junction_proximity_radius ** 2:
                                segment_junctions.add(j_idx)
                                junction_nearby = True


                        labeled_image[ny, nx] = current_label
                        current_segment.append((ny, nx))
                        visited.add((ny, nx))


                        if not junction_nearby:
                            queue.append((ny, nx))


        if len(segment_junctions) == 2:
            node1, node2 = list(segment_junctions)


            segment_length_pixels = len(current_segment)
            segment_length_meters = segment_length_pixels * scale_factor


            weight = calculate_final_weight(segment_length_meters)


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

    return length_meters / 5.0


def remove_disconnected_vertices(G: nx.Graph) -> nx.Graph:
    """
    Удаляет все вершины, которые не связаны с самой большой компонентой связности графа

    Args:
        G: Исходный граф

    Returns:
        Граф, содержащий только самую большую компоненту связности
    """

    components = list(nx.connected_components(G))

    if not components:
        print("Граф не содержит компонент связности")
        return G


    largest_component = max(components, key=len)


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

        path = nx.dijkstra_path(G, start_node, end_node, weight='weight')


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


    for i in range(len(control_points) - 1):
        start_cp = control_points[i]
        end_cp = control_points[i + 1]


        path, length = find_optimal_route(G, start_cp, end_cp)

        if path is None:
            print(f"Предупреждение: не удалось найти путь от КП {start_cp} до КП {end_cp}")
            return None, float('inf')


        if i > 0:

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


    junction_coords = {}
    for i in range(len(junctions)):
        junction_coords[i] = (junctions[i][0], junctions[i][1])


    if path:
        for i, node in enumerate(path):
            if node in junction_coords:
                y, x = junction_coords[node]


                color = (0, 0, 255)
                radius = 7


                if i == 0:
                    color = (0, 255, 0)
                    radius = 12


                    cv2.putText(
                        result,
                        "Start",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        3
                    )


                elif i == len(path) - 1:
                    color = (255, 0, 0)
                    radius = 12


                    cv2.putText(
                        result,
                        "Finish",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        3
                    )

                cv2.circle(result, (x, y), radius, color, -1)


        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]

            if node1 in junction_coords and node2 in junction_coords:

                if G.has_edge(node1, node2):

                    if 'coords_y' in G[node1][node2] and 'coords_x' in G[node1][node2]:
                        coords_y = G[node1][node2]['coords_y']
                        coords_x = G[node1][node2]['coords_x']


                        for j in range(len(coords_y) - 1):
                            pt1 = (coords_x[j], coords_y[j])
                            pt2 = (coords_x[j + 1], coords_y[j + 1])
                            cv2.line(result, pt1, pt2, (0, 0, 255), 3)
                    else:

                        y1, x1 = junction_coords[node1]
                        y2, x2 = junction_coords[node2]
                        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)


    if save_path:
        cv2.imwrite(save_path, result)


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


    pos = {}
    for i in G.nodes():
        if i < len(junctions):
            y, x = junctions[i]
            pos[i] = (x, -y)


    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='blue')


    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='blue')


    if len(G) <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title('Графовая модель лыжней')
    plt.axis('off')
    plt.tight_layout()


    if save_path:
        plt.savefig(save_path, dpi=300)


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

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    junctions = np.load(junctions_path)
    original_image = cv2.imread(original_image_path)

    if skeleton is None or original_image is None:
        raise FileNotFoundError("Не удалось загрузить файлы")


    G = build_graph_from_skeleton(skeleton, junctions, scale_factor)


    G = remove_disconnected_vertices(G)


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

    if hsv_params is None:
        hsv_params = {
            "h_min": 55,
            "h_max": 70,
            "s_min": 50,
            "s_max": 255,
            "v_min": 50,
            "v_max": 215
        }


    print("Шаг 1: Выделение лыжней и перекрестков...")

    mask, skeleton, junctions = process_map_with_dashed_lines(
        original_image_path, hsv_params, base_dir, show_steps, 0.7
    )


    filename = os.path.basename(original_image_path)
    base_name = os.path.splitext(filename)[0]
    save_dir = os.path.join(base_dir, base_name)


    print("Шаг 2: Построение графовой модели...")
    G = build_graph_from_skeleton(skeleton, junctions, scale_factor)


    print("Шаг 3: Удаление несвязанных вершин...")
    G = remove_disconnected_vertices(G)


    print("Шаг 4: Визуализация графа...")
    visualize_graph(
        G,
        junctions,
        save_path=os.path.join(save_dir, "graph_visualization.png"),
        show=False
    )

    return G, junctions


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Построение графовой модели лыжней')
    parser.add_argument('--image', required=True, help='Путь к исходному изображению карты')
    parser.add_argument('--scale', type=float, default=1.0, help='Коэффициент масштаба (метров на пиксель)')
    parser.add_argument('--output', default='results', help='Директория для сохранения результатов')
    parser.add_argument('--start', type=int, default=0, help='Индекс начального перекрестка')
    parser.add_argument('--end', type=int, default=1, help='Индекс конечного перекрестка')
    parser.add_argument('--show_steps', action='store_true', help='Показывать промежуточные шаги обработки')

    args = parser.parse_args()


    print(f"Обработка карты: {args.image}")
    print(f"Масштаб: {args.scale} м/пиксель")


    G, junctions = process_original_map(
        args.image,
        args.scale,
        None,
        args.output,
        args.show_steps
    )


    path, length = find_optimal_route(G, args.start, args.end)

    if path:
        print(f"Найден оптимальный маршрут длиной {length:.2f} секунд")


        original_image = cv2.imread(args.image)


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