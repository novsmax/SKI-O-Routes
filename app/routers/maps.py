"""
Обновленный роутер для использования сохраненного графа
"""
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import os
import uuid
from datetime import datetime
import numpy as np
import shutil
from typing import List, Optional, Dict, Any, Tuple

from ..database import get_db
from ..models import Map
from ..schemas import MapResponse
from ..services.file_storage import save_upload_file
from ..services.map_processor import MapProcessor

router = APIRouter(
    prefix="/maps",
    tags=["maps"],
    responses={404: {"description": "Not found"}},
)

templates = Jinja2Templates(directory="templates")


@router.get("/test")
async def test_api():
    return {"status": "ok", "message": "API работает корректно"}


@router.get("/list")
async def list_maps(db: Session = Depends(get_db)):
    """Получение списка всех карт."""
    try:
        maps = db.query(Map).all()
        result = []

        for map_obj in maps:
            map_dict = {
                "id": map_obj.id,
                "title": map_obj.title,
                "location": map_obj.location,
                "original_filename": map_obj.original_filename,
                "stored_filename": map_obj.stored_filename,
                "processed_filename": map_obj.processed_filename,
                "is_processed": map_obj.is_processed,
                "created_at": map_obj.created_at.isoformat() if map_obj.created_at else None,
                "updated_at": map_obj.updated_at.isoformat() if map_obj.updated_at else None
            }
            result.append(map_dict)

        return result
    except Exception as e:
        print(f"Ошибка при получении списка карт: {e}")
        # В случае ошибки возвращаем пустой список
        return []


@router.post("/upload")
async def upload_map(
        request: Request,
        file: UploadFile = File(...),
        title: str = Form(...),
        location: str = Form(...),
        db: Session = Depends(get_db)
):
    """Загрузка новой карты."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    file_path = os.path.join("uploads", filename)

    await save_upload_file(file, file_path)

    new_map = Map(
        title=title,
        location=location,
        original_filename=file.filename,
        stored_filename=filename,
        is_processed=False
    )

    db.add(new_map)
    db.commit()
    db.refresh(new_map)

    return RedirectResponse(url=f"/maps/{new_map.id}", status_code=303)




@router.get("/{map_id}", response_class=HTMLResponse)
async def get_map(request: Request, map_id: int, db: Session = Depends(get_db)):
    """Страница просмотра карты."""
    map_data = db.query(Map).filter(Map.id == map_id).first()
    if not map_data:
        raise HTTPException(status_code=404, detail="Карта не найдена")

    return templates.TemplateResponse(
        "map_view.html",
        {
            "request": request,
            "map": map_data,
            "upload_path": f"/uploads/{map_data.stored_filename}",
            "processed_path": f"/processed/{map_data.processed_filename}" if map_data.processed_filename else None
        }
    )


@router.post("/{map_id}/analyze")
async def analyze_map(request: Request, map_id: int, db: Session = Depends(get_db)):
    """
    Анализ карты и нахождение оптимальных маршрутов.
    Использует функционал из модулей trail_analyzer и graph_builder.
    """
    try:
        map_data = db.query(Map).filter(Map.id == map_id).first()
        if not map_data:
            raise HTTPException(status_code=404, detail="Карта не найдена")

        original_path = os.path.join("uploads", map_data.stored_filename)

        if not os.path.exists(original_path):
            return {"success": False, "error": "Исходный файл карты не найден"}

        processor = MapProcessor(original_path, db=db, map_id=map_id)
        processed_path = processor.process()

        processed_filename = os.path.basename(processed_path)

        map_data.processed_filename = processed_filename
        map_data.is_processed = True
        db.commit()

        return {"success": True, "processed_path": f"/processed/{processed_filename}"}
    except Exception as e:
        print(f"Ошибка при анализе карты: {str(e)}")
        return {"success": False, "error": str(e)}


@router.post("/{map_id}/find_junctions")
async def find_junctions(
    request: Request,
    map_id: int,
    start_x: int = Form(...),
    start_y: int = Form(...),
    end_x: int = Form(...),
    end_y: int = Form(...),
    db: Session = Depends(get_db)
):
    """
    Находит ближайшие перекрестки (развилки) к выбранным точкам.
    """
    try:
        map_data = db.query(Map).filter(Map.id == map_id).first()
        if not map_data:
            raise HTTPException(status_code=404, detail="Карта не найдена")

        if not map_data.is_processed:
            return {"success": False, "error": "Карта еще не обработана. Сначала нужно выполнить анализ карты."}

        original_path = os.path.join("uploads", map_data.stored_filename)

        processor = MapProcessor(original_path, db=db, map_id=map_id)

        if processor.graph is None:
            processor.process()

        start_junction = processor.find_nearest_junction((start_x, start_y))
        end_junction = processor.find_nearest_junction((end_x, end_y))

        if not start_junction or not end_junction:
            return {"success": False, "error": "Не удалось найти ближайшие перекрестки"}

        return {
            "success": True,
            "start_junction": {
                "x": int(start_junction[0]),
                "y": int(start_junction[1])
            },
            "end_junction": {
                "x": int(end_junction[0]),
                "y": int(end_junction[1])
            }
        }

    except Exception as e:
        print(f"Ошибка при поиске перекрестков: {str(e)}")
        return {"success": False, "error": str(e)}


@router.post("/{map_id}/find_route")
async def find_route(
    request: Request,
    map_id: int,
    start_x: int = Form(...),
    start_y: int = Form(...),
    end_x: int = Form(...),
    end_y: int = Form(...),
    db: Session = Depends(get_db)
):
    """
    Находит оптимальный маршрут между двумя точками на карте.
    """
    try:
        map_data = db.query(Map).filter(Map.id == map_id).first()
        if not map_data:
            raise HTTPException(status_code=404, detail="Карта не найдена")

        if not map_data.is_processed:
            return {"success": False, "error": "Карта еще не обработана. Сначала нужно выполнить анализ карты."}

        original_path = os.path.join("uploads", map_data.stored_filename)

        processor = MapProcessor(original_path, db=db, map_id=map_id)

        result_path = processor.find_route((start_x, start_y), (end_x, end_y))

        if result_path:
            route_filename = os.path.basename(result_path)
            return {"success": True, "route_path": f"/processed/{route_filename}"}
        else:
            return {"success": False, "error": "Не удалось найти маршрут между указанными точками"}
    except Exception as e:
        print(f"Ошибка при поиске маршрута: {str(e)}")
        return {"success": False, "error": str(e)}