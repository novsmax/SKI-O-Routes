from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from app.database import get_db, init_db
from app.routers import maps, admin
from app.models import Map

os.makedirs("templates/includes", exist_ok=True)
app = FastAPI(title="Цифровой ассистент спортивного ориентирования")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("app/static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/processed", StaticFiles(directory="processed"), name="processed")

templates = Jinja2Templates(directory="templates")

app.include_router(maps.router)
app.include_router(admin.router)


@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        print("База данных инициализирована")
    except Exception as e:
        print(f"Ошибка инициализации базы данных: {e}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница приложения."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/library", response_class=HTMLResponse)
async def library(request: Request, db=Depends(get_db)):
    """Страница библиотеки карт."""
    maps = db.query(Map).all()

    locations = set()
    for map_item in maps:
        if map_item.location:
            locations.add(map_item.location)

    locations = sorted(list(locations))

    return templates.TemplateResponse(
        "library.html",
        {"request": request, "maps": maps, "locations": locations}
    )

if __name__ == "__main__":
    import uvicorn
    print("Запуск приложения...")
    print("Откройте браузер по адресу: http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)