from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# (DashboardSummary, RegionPowerData, PowerForecast 클래스는 그대로)
# schemas.py

class DashboardSummary(BaseModel):
    """대시보드 요약 정보"""
    current_power: float  # 현재 발전량 (Kw)
    today_total: float    # 오늘의 누적 발전량 (Kwh)
    today_revenue: int    
    
    accuracy: float       # 예측 정확도 (%)
    today_date: str       # 오늘 날짜
    
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