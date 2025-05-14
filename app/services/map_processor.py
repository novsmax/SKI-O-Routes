"""
Обновленный MapProcessor с поддержкой поиска ближайших перекрестков
"""

import os
import cv2
import numpy as np
from PIL import Image
import networkx as nx
from typing import Tuple, Optional, List, Dict

# Импортируем функции из модулей анализа

from app.trail_analyzer import process_map_with_dashed_lines
from app.graph_builder import build_graph_from_skeleton, find_optimal_route, visualize_route, remove_disconnected_vertices



class MapProcessor:
    def __init__(self, map_path: str):
        """
        Инициализация процессора карт.

        Args:
            map_path: Путь к файлу карты
        """
        self.map_path = map_path
        self.graph = None
        self.junctions = None
        self.skeleton = None

        # HSV параметры для выделения лыжней по умолчанию
        self.hsv_params = {
            "h_min": 55,
            "h_max": 70,
            "s_min": 50,
            "s_max": 255,
            "v_min": 50,
            "v_max": 215
        }

        # Масштабный коэффициент для преобразования пикселей в метры
        self.scale_factor = 1.0

    def process(self) -> str:
        """
        Обработка карты и создание визуализации оптимальных маршрутов.

        Returns:
            Путь к обработанной карте
        """
        # Создаем директории для результатов
        base_dir = "processed"
        os.makedirs(base_dir, exist_ok=True)

        try:
            # 1. Выделение лыжней и перекрестков
            print(f"Обработка карты: {self.map_path}")
            mask, self.skeleton, self.junctions = process_map_with_dashed_lines(
                self.map_path,
                self.hsv_params,
                base_dir,
                show_steps=False,
                alpha=0.7
            )

            filename = os.path.basename(self.map_path)
            base_name = os.path.splitext(filename)[0]
            save_dir = os.path.join(base_dir, base_name)

            # 2. Построение графовой модели
            self.graph = build_graph_from_skeleton(self.skeleton, self.junctions, self.scale_factor)

            # Удаляем несвязанные вершины для создания более чистого графа
            self.graph = remove_disconnected_vertices(self.graph)

            # 3. Визуализация результатов с примером оптимального маршрута
            output_path = self._get_output_path()

            # Если в графе есть вершины, попробуем найти маршрут между двумя точками
            if self.graph.number_of_nodes() >= 2:
                # Берем две достаточно удаленные друг от друга вершины
                nodes = list(self.graph.nodes())
                # Выбираем первую и точку в середине списка вершин для демонстрации
                start_node = nodes[0]
                end_node = nodes[min(len(nodes) - 1, len(nodes) // 2)]

                # Ищем оптимальный маршрут
                path, length = find_optimal_route(self.graph, start_node, end_node)

                if path:
                    # Визуализируем маршрут
                    original_image = cv2.imread(self.map_path)
                    if original_image is None:
                        raise ValueError(f"Не удалось загрузить изображение: {self.map_path}")

                    result = visualize_route(
                        original_image,
                        self.graph,
                        path,
                        self.junctions,
                        save_path=output_path,
                        show=False
                    )
                    print(f"Визуализация маршрута сохранена: {output_path}")
                else:
                    # Если маршрут не найден, просто выводим обработанную карту с лыжнями
                    self._save_skeleton_visualization(output_path)
                    print(f"Маршрут не найден. Сохранена визуализация скелета: {output_path}")
            else:
                # Если в графе недостаточно вершин, просто выводим обработанную карту с лыжнями
                self._save_skeleton_visualization(output_path)
                print(f"Недостаточно вершин в графе. Сохранена визуализация скелета: {output_path}")

            return output_path

        except Exception as e:
            print(f"Ошибка при обработке карты: {str(e)}")

            # В случае ошибки создаем простую визуализацию, чтобы не ломать интерфейс
            output_path = self._get_output_path()
            self._create_error_image(output_path, str(e))

            return output_path

    def _create_error_image(self, output_path: str, error_message: str) -> None:
        """
        Создает изображение с сообщением об ошибке.

        Args:
            output_path: Путь для сохранения изображения
            error_message: Сообщение об ошибке
        """
        try:
            # Загружаем исходное изображение или создаем пустое
            try:
                img = cv2.imread(self.map_path)
                if img is None:
                    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            except:
                img = np.ones((400, 600, 3), dtype=np.uint8) * 255

            # Добавляем сообщение об ошибке
            cv2.putText(
                img,
                "Ошибка обработки карты:",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            # Ограничиваем длину сообщения и разбиваем на строки
            error_lines = []
            max_length = 60
            while len(error_message) > max_length:
                pos = error_message[:max_length].rfind(' ')
                if pos == -1:
                    pos = max_length
                error_lines.append(error_message[:pos])
                error_message = error_message[pos:].lstrip()
            error_lines.append(error_message)

            # Добавляем строки сообщения об ошибке
            for i, line in enumerate(error_lines):
                cv2.putText(
                    img,
                    line,
                    (50, 100 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1
                )

            # Сохраняем изображение
            cv2.imwrite(output_path, img)
        except Exception as e:
            print(f"Ошибка при создании изображения с ошибкой: {str(e)}")

            # В крайнем случае, создаем минимальное изображение
            img = np.ones((200, 400, 3), dtype=np.uint8) * 255
            cv2.putText(
                img,
                "Ошибка обработки",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.imwrite(output_path, img)

    def _save_skeleton_visualization(self, output_path: str) -> None:
        """
        Сохраняет визуализацию скелета лыжней.

        Args:
            output_path: Путь для сохранения изображения
        """
        try:
            # Создаем цветное изображение из скелета
            if self.skeleton is not None:
                # Загружаем исходное изображение
                original_image = cv2.imread(self.map_path)
                if original_image is None:
                    # Если не удалось загрузить, создаем пустое черное изображение
                    original_image = np.zeros((self.skeleton.shape[0], self.skeleton.shape[1], 3), dtype=np.uint8)

                # Создаем наложение для скелета
                overlay = original_image.copy()
                for y, x in zip(*np.where(self.skeleton > 0)):
                    cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)  # Зеленый цвет для лыжней

                # Добавляем перекрестки
                if self.junctions is not None:
                    for y, x in self.junctions:
                        cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)  # Красный цвет для перекрестков

                # Сохраняем результат
                cv2.imwrite(output_path, overlay)
            else:
                # Если скелет не был создан, создаем изображение с сообщением
                img = np.ones((400, 600, 3), dtype=np.uint8) * 255
                cv2.putText(
                    img,
                    "Лыжни не обнаружены",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                cv2.imwrite(output_path, img)
        except Exception as e:
            print(f"Ошибка при сохранении визуализации скелета: {str(e)}")
            self._create_error_image(output_path, f"Ошибка визуализации: {str(e)}")

    def _get_output_path(self) -> str:
        """
        Создает путь для сохранения обработанной карты.

        Returns:
            Путь к файлу обработанной карты
        """
        filename = os.path.basename(self.map_path)
        output_dir = os.path.join("processed")
        os.makedirs(output_dir, exist_ok=True)

        return os.path.join(output_dir, f"processed_{filename}")

    def find_route(self, start_point: Tuple[int, int], end_point: Tuple[int, int]) -> Optional[str]:
        """
        Находит и визуализирует оптимальный маршрут между двумя точками.

        Args:
            start_point: Координаты начальной точки (x, y)
            end_point: Координаты конечной точки (x, y)

        Returns:
            Путь к визуализации маршрута или None, если маршрут не найден
        """
        if self.graph is None:
            print("Граф не построен. Сначала вызовите метод process().")
            return None

        if self.junctions is None or len(self.junctions) == 0:
            print("Перекрестки не найдены. Невозможно построить маршрут.")
            return None

        # Находим ближайшие перекрестки к указанным точкам
        start_node = self._find_nearest_node(start_point)
        end_node = self._find_nearest_node(end_point)

        if start_node is None or end_node is None:
            print("Не удалось найти ближайшие перекрестки.")
            return None

        # Ищем оптимальный маршрут
        path, length = find_optimal_route(self.graph, start_node, end_node)

        if path:
            # Визуализируем маршрут
            output_path = os.path.join("processed", f"route_{os.path.basename(self.map_path)}")
            original_image = cv2.imread(self.map_path)
            if original_image is None:
                print(f"Не удалось загрузить изображение: {self.map_path}")
                return None

            result = visualize_route(
                original_image,
                self.graph,
                path,
                self.junctions,
                save_path=output_path,
                show=False
            )

            return output_path
        else:
            print(f"Не удалось найти маршрут между перекрестками {start_node} и {end_node}")
            return None

    def find_nearest_junction(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Находит ближайший перекресток (развилку) к указанной точке.

        Args:
            point: Координаты точки (x, y)

        Returns:
            Координаты ближайшего перекрестка (x, y) или None, если перекрестки не найдены
        """
        if self.junctions is None or len(self.junctions) == 0:
            return None

        # Преобразуем координаты из (x, y) в (y, x), так как junctions хранятся в формате (y, x)
        x, y = point

        print(f"Ищем для точек {x} {y}")

        debug_img = cv2.imread(self.map_path)
        # Вычисляем расстояния до всех перекрестков
        min_dist = float('inf')
        nearest_junction = None

        for junction in self.junctions:
            jy, jx = junction
            dist = (jx - x) ** 2 + (jy - y) ** 2

            if dist < min_dist:
                min_dist = dist
                nearest_junction = (jx, jy)  # Возвращаем в формате (x, y)

        for jy, jx in self.junctions:
            jx_int = int(jx)
            jy_int = int(jy)
            # Зеленый кружок
            cv2.circle(debug_img, (jx_int, jy_int), 5, (0, 255, 0), -1)
            # Подпись координат справа от точки
            cv2.putText(
                debug_img,
                f"({jx_int}, {jy_int})",
                (jx_int + 10, jy_int + 5),  # Смещение текста
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  # Уменьшенный размер шрифта
                (0, 255, 0),  # Зеленый цвет
                1
            )

            # Сохраняем изображение
        cv2.imwrite("debug_junctions_with_labels.jpg", debug_img)


        return nearest_junction

    def _find_nearest_node(self, point: Tuple[int, int]) -> Optional[int]:
        """
        Находит ближайшую вершину (перекресток) к указанной точке.

        Args:
            point: Координаты точки (x, y)

        Returns:
            Индекс ближайшей вершины или None, если граф не построен
        """
        if self.graph is None or self.junctions is None:
            return None

        # Преобразуем координаты из (x, y) в (y, x), так как junctions хранятся в формате (y, x)
        x, y = point

        # Вычисляем расстояния до всех перекрестков
        min_dist = float('inf')
        nearest_node = None

        for node in self.graph.nodes():
            if node >= len(self.junctions):
                continue

            jy, jx = self.junctions[node]
            dist = (jx - x) ** 2 + (jy - y) ** 2

            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node