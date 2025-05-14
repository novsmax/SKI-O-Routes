import cv2
from app.graph_builder import process_original_map, find_optimal_route, visualize_route

# Укажите РЕАЛЬНЫЙ путь к вашему изображению
IMAGE_PATH = "YEyQ1MkbKcY.jpg"  # Измените на ваш реальный путь

# Обработка исходного изображения и построение графа
G, junctions = process_original_map(
    IMAGE_PATH,
    scale_factor=1.5
)

print(f"Количество вершин в графе: {G.number_of_nodes()}")
print(f"Количество рёбер в графе: {G.number_of_edges()}")

# Если в графе есть рёбра, попробуем найти маршрут
if G.number_of_edges() > 0:
    # Использовать существующие номера вершин
    nodes = list(G.nodes())
    if len(nodes) >= 2:
        start_node = nodes[50]
        end_node = nodes[200]

        print(f"Поиск маршрута между перекрестками {start_node} и {end_node}")
        path, length = find_optimal_route(G, start_node, end_node)

        if path:
            print(f"Найден маршрут длиной {length:.2f} секунд")

            # Визуализация маршрута
            original_image = cv2.imread(IMAGE_PATH)
            if original_image is not None:
                result = visualize_route(original_image, G, path, junctions)
            else:
                print(f"Ошибка: не удалось загрузить изображение {IMAGE_PATH}")
        else:
            print("Не удалось найти маршрут")
    else:
        print("В графе недостаточно вершин для построения маршрута")
else:
    print("В графе нет рёбер. Исправьте проблему с построением графа перед поиском маршрута.")