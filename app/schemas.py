from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class MapBase(BaseModel):
    """Базовая схема для карты."""
    title: str
    location: str

class MapCreate(MapBase):
    """Схема для создания карты."""
    pass

class MapResponse(BaseModel):
    """Схема для ответа с данными карты."""
    id: int
    title: str
    location: str
    original_filename: str
    stored_filename: str
    processed_filename: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_processed: bool

    model_config = ConfigDict(from_attributes=True)