"""
Скрипт для принудительного освобождения блокировки файла базы данных
"""

import os
import psutil
import sys


def find_and_kill_process_locking_file(file_path):
    """
    Находит и завершает процесс, блокирующий указанный файл
    """
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не существует")
        return False

    file_path = os.path.abspath(file_path)
    print(f"Поиск процессов, блокирующих файл: {file_path}")

    found = False
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            proc_info = proc.info
            if proc_info['open_files']:
                for file in proc_info['open_files']:
                    if file.path == file_path:
                        found = True
                        print(f"Найден блокирующий процесс: PID {proc.pid} ({proc_info['name']})")

                        try:
                            proc.terminate()
                            print(f"Процесс {proc.pid} успешно завершен")
                            return True
                        except psutil.AccessDenied:
                            print(f"Ошибка доступа при попытке завершить процесс {proc.pid}")
                        except Exception as e:
                            print(f"Ошибка при завершении процесса {proc.pid}: {e}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not found:
        print("Блокирующие процессы не найдены")

    return found


if __name__ == "__main__":
    db_path = "orienteering.db"

    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    print(f"Пытаемся освободить файл: {db_path}")

    if find_and_kill_process_locking_file(db_path):
        print("Готово!")
    else:
        print("Не удалось найти или завершить блокирующие процессы")