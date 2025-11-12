from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DashboardSummary(BaseModel):
    """대시보드 요약 정보"""
    current_power: float  # 현재 발전량 (Kw)
    today_total: float    # 오늘의 누적 발전량 (Kwh)
    accuracy: float       # 예측 정확도 (%)
    today_date: str       # 오늘 날짜
    
    class Config:
        from_attributes = True

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
    actual: Optional[float]    # 실제 발전량 (Kwh)
    predicted: float      # 예측 발전량 (Kwh)
    
    class Config:
        from_attributes = True

# ====================================
# ▼▼▼ AI 예측 API용 모델 (추가) ▼▼▼
# ====================================

class PredictionRequest(BaseModel):
    """
    POST /predict 요청 본문
    main.py의 `request.region_id` 등을 참조
    """
    region_id: int
    sequence_length: int
    features: List[Dict[str, Any]] # 예: [ { "temp_c": 15, ... }, ... ]

class PredictionData(BaseModel):
    """
    예측 결과 데이터 (개별)
    main.py의 `predictions_list` 항목
    """
    ts: str
    predicted_kwh: float

class PredictionResponse(BaseModel):
    """
    POST /predict 응답 본문
    main.py의 `response_model`
    """
    status: str
    data: List[PredictionData]