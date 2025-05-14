import os
from fastapi import UploadFile
import aiofiles


async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        async with aiofiles.open(destination, "wb") as buffer:
            content = await upload_file.read()
            await buffer.write(content)

        return destination
    except Exception as e:
        print(f"Ошибка при сохранении файла: {str(e)}")
        raise