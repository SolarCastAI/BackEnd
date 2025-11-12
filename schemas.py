from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# ==================== Pydantic 데이터 모델 ====================
# (DB 모델(models.py)과 연동하기 위해 Config 클래스 추가)

class DashboardSummary(BaseModel):
    """대시보드 요약 정보"""
    current_power: float  # 현재 발전량 (Kw)
    today_total: float    # 오늘의 누적 발전량 (Kwh)
    accuracy: float       # 예측 정확도 (%)
    today_date: str       # 오늘 날짜
    
    class Config:
        from_attributes = True  # Pydantic v2 (ORM 모델 -> Pydantic 모델)
        # orm_mode = True       # Pydantic v1

class RegionPowerData(BaseModel):
    """지역별 발전량 데이터"""
    region: str           # 지역명
    power: float          # 발전량 (Kwh)
    revenue: int          # 수익 (원)
    latitude: float       # 위도
    longitude: float      # 경도
    
    class Config:
        from_attributes = True

class PowerForecast(BaseModel):
    """시간대별 발전량 예측"""
    time: str             # 시간
    actual: Optional[float]    # 실제 발전량
    predicted: float      # 예측 발전량
    
    class Config:
        from_attributes = True