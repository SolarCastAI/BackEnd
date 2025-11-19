from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# (DashboardSummary, RegionPowerData, PowerForecast 클래스는 그대로)
class DashboardSummary(BaseModel):
    current_power: float
    today_total: float
    accuracy: float
    today_date: str
    class Config:
        from_attributes = True

class RegionPowerData(BaseModel):
    region: str
    power: float
    revenue: int
    latitude: float
    longitude: float
    class Config:
        from_attributes = True

class PowerForecast(BaseModel):
    time: str
    actual: Optional[float]
    predicted: float
    class Config:
        from_attributes = True
        
# --- PredictionRequest 수정 ---
class PredictionRequest(BaseModel):
    """
    POST /predict 요청 본문
    'features'를 선택 사항(Optional)으로 변경
    """
    region_id: int
    sequence_length: int
    features: Optional[List[Dict[str, Any]]] = None # <-- (Q1) DB에서 읽어올 것이므로 선택 사항

class PredictionData(BaseModel):
    ts: str
    predicted_kwh: float

class PredictionResponse(BaseModel):
    status: str
    data: List[PredictionData]