from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import os

from ..database import get_db
from ..models import Map

router = APIRouter(
    tags=["admin"],
    responses={404: {"description": "Not found"}},
)

templates = Jinja2Templates(directory="templates")


@router.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    """Административная панель."""
    maps = db.query(Map).all()
    return templates.TemplateResponse(
        "admin/dashboard.html",
        {"request": request, "maps": maps}
    )


@router.post("/maps/{map_id}/update")
async def update_map(
        map_id: int,
        title: str = Form(...),
        location: str = Form(...),
        db: Session = Depends(get_db)
):
    """Обновление информации о карте."""
    map_data = db.query(Map).filter(Map.id == map_id).first()
    if not map_data:
        raise HTTPException(status_code=404, detail="Карта не найдена")

    map_data.title = title
    map_data.location = location
    db.commit()

    return RedirectResponse(url="/admin/dashboard", status_code=303)


@router.post("/maps/{map_id}/delete")
async def delete_map(map_id: int, db: Session = Depends(get_db)):
    """Удаление карты."""
    map_data = db.query(Map).filter(Map.id == map_id).first()
    if not map_data:
        raise HTTPException(status_code=404, detail="Карта не найдена")

    # Удаляем файлы
    if map_data.stored_filename:
        file_path = os.path.join("uploads", map_data.stored_filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    if map_data.processed_filename:
        processed_path = os.path.join("processed", map_data.processed_filename)
        if os.path.exists(processed_path):
            os.remove(processed_path)

    # Удаляем запись из базы данных
    db.delete(map_data)
    db.commit()

    return RedirectResponse(url="/admin/dashboard", status_code=303)