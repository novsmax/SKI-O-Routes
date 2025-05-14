"""
Скрипт для миграции базы данных с добавлением полей для хранения графа
"""
import sqlite3
import os
import sys


def update_database_schema():
    """
    Добавляет новые столбцы в таблицу maps для хранения графа и перекрестков
    """
    print("🔄 Обновление структуры базы данных...")

    # Путь к базе данных
    db_path = os.getenv("DATABASE_URL", "sqlite:///./orienteering.db")

    # Если путь в формате SQLAlchemy, извлекаем файловый путь
    if db_path.startswith("sqlite:///"):
        db_path = db_path[10:]

    # Проверяем существование файла БД
    if not os.path.exists(db_path):
        print(f"❌ База данных не найдена: {db_path}")
        return False

    try:
        # Подключаемся к базе данных
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Проверяем наличие таблицы maps
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='maps';")
        if not cursor.fetchone():
            print("❌ Таблица 'maps' не найдена в базе данных")
            conn.close()
            return False

        # Проверяем существование новых столбцов
        cursor.execute("PRAGMA table_info(maps);")
        columns = [info[1] for info in cursor.fetchall()]

        # Добавляем столбцы, если они не существуют
        if 'graph_data' not in columns:
            print("➕ Добавление столбца 'graph_data'...")
            cursor.execute("ALTER TABLE maps ADD COLUMN graph_data TEXT;")

        if 'junctions_data' not in columns:
            print("➕ Добавление столбца 'junctions_data' для хранения перекрестков...")
            cursor.execute("ALTER TABLE maps ADD COLUMN junctions_data BLOB;")

        # Сохраняем изменения
        conn.commit()

        # Проверяем, что столбцы были добавлены
        cursor.execute("PRAGMA table_info(maps);")
        new_columns = [info[1] for info in cursor.fetchall()]

        if 'graph_data' in new_columns and 'junctions_data' in new_columns:
            print("✅ Схема базы данных успешно обновлена")
            conn.close()
            return True
        else:
            print("❌ Ошибка при добавлении столбцов")
            conn.close()
            return False

    except Exception as e:
        print(f"❌ Ошибка при обновлении базы данных: {str(e)}")
        return False


if __name__ == "__main__":
    update_database_schema()