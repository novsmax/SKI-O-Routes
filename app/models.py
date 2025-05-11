from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from .database import Base

class Map(Base):
    """Модель для хранения информации о загруженных картах."""
    __tablename__ = "maps"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    location = Column(String)
    original_filename = Column(String)
    stored_filename = Column(String)  # Имя файла в системе
    processed_filename = Column(String, nullable=True)  # Имя обработанного файла
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_processed = Column(Boolean, default=False)  # Флаг, указывающий, обработана ли карта