from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import random

app = FastAPI(title="SolarCast API")

# CORS 설정 - React 앱과 연동
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 데이터 모델 ====================

class DashboardSummary(BaseModel):
    """대시보드 요약 정보"""
    current_power: float  # 현재 발전량 (Kw)
    today_total: float    # 오늘의 누적 발전량 (Kwh)
    accuracy: float       # 예측 정확도 (%)
    today_date: str       # 오늘 날짜

class RegionPowerData(BaseModel):
    """지역별 발전량 데이터"""
    region: str           # 지역명
    power: float          # 발전량 (Kwh)
    revenue: int          # 수익 (원)
    latitude: float       # 위도
    longitude: float      # 경도

class PowerForecast(BaseModel):
    """시간대별 발전량 예측"""
    time: str             # 시간
    actual: Optional[float]    # 실제 발전량
    predicted: float      # 예측 발전량

# ==================== 임시 데이터 ====================

# 지역별 데이터
regions_data = [
    {"region": "서울", "power": 4850, "revenue": 843900, "lat": 37.5665, "lng": 126.9780},
    {"region": "부산", "power": 5240, "revenue": 911760, "lat": 35.1796, "lng": 129.0756},
    {"region": "대구", "power": 4650, "revenue": 809100, "lat": 35.8714, "lng": 128.6014},
    {"region": "인천", "power": 4720, "revenue": 821280, "lat": 37.4563, "lng": 126.7052},
    {"region": "대전", "power": 4100, "revenue": 713400, "lat": 36.3504, "lng": 127.3845},
]

# 시간대별 발전량 데이터 생성 함수
def generate_power_data(hours: int = 24):
    """시간대별 발전량 데이터 생성"""
    data = []
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for i in range(hours):
        time = (base_time + timedelta(hours=i)).strftime("%H:%M")
        
        # 낮 시간대에 발전량이 높도록 설정
        hour = i
        if 6 <= hour <= 18:
            base_power = 200 + (hour - 6) * 30
            if hour > 12:
                base_power = 200 + (18 - hour) * 30
        else:
            base_power = 10
        
        # 실제 발전량 (현재 시간까지만)
        actual = None
        if i <= datetime.now().hour:
            actual = base_power + random.uniform(-20, 20)
        
        # 예측 발전량
        predicted = base_power + random.uniform(-15, 15)
        
        data.append({
            "time": time,
            "actual": round(actual, 2) if actual else None,
            "predicted": round(predicted, 2)
        })
    
    return data

# ==================== API 엔드포인트 ====================

@app.get("/")
def read_root():
    """API 루트"""
    return {
        "message": "SolarCast API - 태양광 발전량 예측 시스템",
        "version": "1.0.0",
        "endpoints": {
            "dashboard": "/api/dashboard/summary",
            "regions": "/api/regions",
            "forecast": "/api/forecast/{hours}"
        }
    }

@app.get("/api/dashboard/summary", response_model=DashboardSummary)
def get_dashboard_summary():
    """대시보드 요약 정보 조회"""
    return DashboardSummary(
        current_power=round(random.uniform(300, 350), 1),
        today_total=round(random.uniform(4500, 4600), 0),
        accuracy=round(random.uniform(96, 99), 1),
        today_date=datetime.now().strftime("%m/%d(%a)")
    )

@app.get("/api/regions", response_model=List[RegionPowerData])
def get_regions_data():
    """지역별 발전량 데이터 조회"""
    return [
        RegionPowerData(
            region=region["region"],
            power=region["power"] + random.uniform(-100, 100),
            revenue=region["revenue"] + random.randint(-10000, 10000),
            latitude=region["lat"],
            longitude=region["lng"]
        )
        for region in regions_data
    ]

@app.get("/api/regions/{region_name}", response_model=RegionPowerData)
def get_region_data(region_name: str):
    """특정 지역 발전량 데이터 조회"""
    region = next((r for r in regions_data if r["region"] == region_name), None)
    
    if not region:
        raise HTTPException(status_code=404, detail=f"지역 '{region_name}'을 찾을 수 없습니다.")
    
    return RegionPowerData(
        region=region["region"],
        power=region["power"] + random.uniform(-100, 100),
        revenue=region["revenue"] + random.randint(-10000, 10000),
        latitude=region["lat"],
        longitude=region["lng"]
    )

@app.get("/api/forecast/{hours}", response_model=List[PowerForecast])
def get_power_forecast(hours: int = 24):
    """시간대별 발전량 예측 데이터 조회
    
    Args:
        hours: 예측 시간 (24, 48, 72)
    """
    if hours not in [24, 48, 72]:
        raise HTTPException(status_code=400, detail="hours는 24, 48, 72 중 하나여야 합니다.")
    
    data = generate_power_data(hours)
    return [PowerForecast(**item) for item in data]

@app.get("/api/download/csv/{region}")
def download_csv_data(region: str):
    """CSV 다운로드용 데이터 (실제로는 파일 생성 필요)"""
    region_data = next((r for r in regions_data if r["region"] == region), None)
    
    if not region_data:
        raise HTTPException(status_code=404, detail=f"지역 '{region}'을 찾을 수 없습니다.")
    
    # 실제로는 CSV 파일을 생성하고 FileResponse로 반환
    return {
        "message": f"{region} 지역 CSV 데이터 생성 완료",
        "region": region,
        "data_points": 24,
        "download_url": f"/api/download/csv/{region}/file"
    }

@app.get("/health")
def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)