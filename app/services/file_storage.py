import os
from fastapi import UploadFile
import aiofiles
from PIL import Image


async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """
    Сохраняет загруженный файл в указанное место назначения.

    Args:
        upload_file: Загруженный файл
        destination: Путь, куда сохранить файл

    Returns:
        Путь к сохраненному файлу
    """
    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Сохраняем файл
        async with aiofiles.open(destination, "wb") as buffer:
            content = await upload_file.read()
            await buffer.write(content)

        return destination
    except Exception as e:
        # В реальном приложении добавить логирование ошибки
        print(f"Ошибка при сохранении файла: {str(e)}")
        raise