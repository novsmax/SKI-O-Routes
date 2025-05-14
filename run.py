"""
Скрипт для запуска приложения и создания необходимых директорий
"""

import os
import sys
import uvicorn
import shutil
import sqlite3
from pathlib import Path

def setup_environment():
    """Настройка окружения для приложения"""
    print("Настройка окружения...")

    # Создаем необходимые директории, если они не существуют
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    os.makedirs("app/static/images", exist_ok=True)

    # Создаем простой placeholder, если он не существует
    placeholder_path = "app/static/images/placeholder.png"
    if not os.path.exists(placeholder_path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (300, 200), color=(240, 240, 240))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, 299, 199], outline=(200, 200, 200))
            d.text((100, 100), "Нет изображения", fill=(100, 100, 100))
            img.save(placeholder_path)
            print(f"✅ Создан placeholder: {placeholder_path}")
        except Exception as e:
            print(f"⚠️ Не удалось создать placeholder: {e}")

    # Проверяем существование базы данных
    db_path = Path("orienteering.db")
    if db_path.exists():
        print(f"📊 База данных существует: {db_path}")
        try:
            # Проверяем структуру БД
            conn = sqlite3.connect("orienteering.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"📋 Таблицы в БД: {', '.join([t[0] for t in tables])}")
            conn.close()
        except Exception as e:
            print(f"⚠️ Ошибка при проверке БД: {e}")
    else:
        print("🔄 База данных будет создана при первом запуске приложения")

    # Проверяем зависимости
    try:
        import fastapi
        import sqlalchemy
        import jinja2
        import aiofiles
        print("✅ Все необходимые зависимости установлены")
    except ImportError as e:
        print(f"❌ Не установлены необходимые зависимости: {e}")
        print("🔄 Установка зависимостей...")
        if os.path.exists("requirements.txt"):
            os.system(f"{sys.executable} -m pip install -r requirements.txt")
            print("✅ Зависимости установлены")
        else:
            print("❌ Файл requirements.txt не найден")
            return False

    return True

def reset_database():
    """Сброс базы данных"""
    print("🔄 Сброс базы данных...")
    try:
        # Пытаемся удалить существующую БД
        db_path = "orienteering.db"
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print("✅ База данных удалена")
            except PermissionError:
                # Если файл заблокирован, работаем с новым файлом
                print("⚠️ База данных заблокирована другим процессом")
                print("🔄 Создание новой базы данных с другим именем...")

                # Обновляем переменную окружения для использования новой БД
                import random
                new_db_path = f"orienteering_{random.randint(1000, 9999)}.db"
                os.environ["DATABASE_URL"] = f"sqlite:///./{new_db_path}"
                print(f"✅ Будет использована новая база данных: {new_db_path}")

        # Удаляем загруженные файлы
        for file in os.listdir("uploads"):
            if file != ".gitkeep":
                try:
                    os.remove(os.path.join("uploads", file))
                except PermissionError:
                    print(f"⚠️ Не удалось удалить файл: {file} (файл занят)")

        for file in os.listdir("processed"):
            if file != ".gitkeep":
                try:
                    os.remove(os.path.join("processed", file))
                except PermissionError:
                    print(f"⚠️ Не удалось удалить файл: {file} (файл занят)")

        print("✅ Загруженные файлы удалены")
        return True
    except Exception as e:
        print(f"❌ Ошибка при сбросе базы данных: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        # Если передан флаг --reset, сбрасываем базу данных
        if reset_database():
            print("✅ База данных успешно сброшена")
        else:
            print("❌ Ошибка при сбросе базы данных")
            sys.exit(1)

    # Настройка окружения
    if not setup_environment():
        print("❌ Ошибка при настройке окружения")
        sys.exit(1)

    print("\n🚀 Запуск приложения...")
    print("📱 Открой в браузере: http://127.0.0.1:8000")
    print("🛑 Для остановки нажми Ctrl+C\n")

    try:
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Приложение остановлено")
    except Exception as e:
        print(f"\n❌ Ошибка при запуске приложения: {e}")
        sys.exit(1)