from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Получаем URL базы данных из переменной окружения или используем значение по умолчанию
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./orienteering.db")
print(f"Используется база данных: {DATABASE_URL}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Инициализация базы данных."""
    from .models import Map  # Импортируем модели здесь во избежание циклических импортов
    Base.metadata.create_all(bind=engine)
    print("База данных инициализирована")