from sqlalchemy import Column, Integer, String, DateTime, Boolean, LargeBinary, Text
from sqlalchemy.sql import func
from .database import Base


class Map(Base):
    """Модель для хранения информации о загруженных картах."""
    __tablename__ = "maps"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    location = Column(String)
    original_filename = Column(String)
    stored_filename = Column(String)
    processed_filename = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_processed = Column(Boolean, default=False)

    graph_data = Column(Text, nullable=True)  #  граф в формате JSON
    junctions_data = Column(LargeBinary, nullable=True)   # перекрестки в формате pickle