from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime

# --- 로컬 모듈 임포트 ---
import crud, models, schemas  # crud, models, schemas 모듈
from database import async_session, engine, Base  # DB 세션
from schemas import DashboardSummary, RegionPowerData, PowerForecast # Pydantic 모델

# (SQLAlchemy 등 다른 임포트...)

# --- SQLAlchemy 관련 모듈 ---
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI(title="SolarCast API")

# CORS 설정 - React 앱과 연동
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (선택) 서버 시작 시 DB 테이블 생성 (개발용)
# @app.on_event("startup")
# async def startup_event():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)

# ==================== DB 세션 의존성 함수 ====================

async def get_db() -> AsyncSession:
    """
    API 요청마다 비동기 DB 세션을 생성하고 반환합니다.
    """
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

# ==================== Pydantic 데이터 모델 ====================
#
# (이전 main.py에 있던 Pydantic 모델들은 
#  crud.py와의 순환 참조를 피하기 위해
#  schemas.py 파일로 분리되었습니다.)
#
# ==========================================================


# ==================== 임시 데이터 (모두 제거) ====================
#
# (regions_data, generate_power_data 함수 등
#  Mock 데이터는 crud.py의 DB 연동으로 대체되어 모두 제거되었습니다.)
#
# ==========================================================


# ==================== API 엔드포인트 (DB 연동) ====================

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
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    """대시보드 요약 정보 조회 (DB 연동)"""
    
    # crud.py의 함수가 DB에서 실제 데이터를 계산하여 반환 (Dict 형태)
    summary_data = await crud.get_dashboard_summary(db)
    
    # Pydantic 모델로 응답 반환
    return DashboardSummary(
        current_power=summary_data["current_power"],
        today_total=summary_data["today_total"],
        accuracy=summary_data["accuracy"],
        today_date=datetime.now().strftime("%m/%d(%a)")
    )

@app.get("/api/regions", response_model=List[RegionPowerData])
async def get_regions_data(db: AsyncSession = Depends(get_db)):
    """지역별 발전량 데이터 조회 (DB 연동)"""
    
    # crud.py가 DB에서 조회/계산한 RegionPowerData 모델 리스트를 반환
    regions_from_db = await crud.get_regions_data(db)
    
    # Pydantic 모델 리스트를 반환하면 FastAPI가 JSON 리스트로 직렬화
    return regions_from_db

@app.get("/api/regions/{region_name}", response_model=RegionPowerData)
async def get_region_data(region_name: str, db: AsyncSession = Depends(get_db)):
    """특정 지역 발전량 데이터 조회 (DB 연동)"""
    
    # crud.py에서 특정 지역의 데이터를 조회 (RegionPowerData 모델 반환)
    region = await crud.get_region_data(db, region_name)
    
    if not region:
        raise HTTPException(status_code=404, detail=f"지역 '{region_name}'을 찾을 수 없습니다.")
    
    return region

@app.get("/api/forecast/{hours}", response_model=List[PowerForecast])
async def get_power_forecast(hours: int = 24, db: AsyncSession = Depends(get_db)):
    """시간대별 발전량 예측 데이터 조회 (DB 연동)
    
    Args:
        hours: 예측 시간 (24, 48, 72)
    """
    if hours not in [24, 48, 72]:
        raise HTTPException(status_code=400, detail="hours는 24, 48, 72 중 하나여야 합니다.")
    
    # crud.py에서 forecast_ts(예측)와 generation_ts(실제)를 조회하여 반환
    data = await crud.get_power_forecast(db, hours)
    return data

@app.get("/api/download/csv/{region}")
async def download_csv_data(region: str, db: AsyncSession = Depends(get_db)):
    """CSV 다운로드용 데이터 (DB 연동)"""
    
    # crud.py에서 데이터 존재 여부 확인
    region_exists = await crud.check_region_exists(db, region)
    
    if not region_exists:
        raise HTTPException(status_code=404, detail=f"지역 '{region}'을 찾을 수 없습니다.")
    
    # TODO:
    # 1. crud.py에서 해당 지역의 CSV 데이터 생성 로직 호출
    # 2. FileResponse 또는 StreamingResponse를 사용하여 실제 파일 반환
    
    # (임시로 JSON 응답 반환)
    return {
        "message": f"{region} 지역 CSV 데이터 생성 준비 완료 (구현 필요)",
        "region": region,
        "data_points": 24, # 예시
        "download_url": f"/api/download/csv/{region}/file" # 가상 URL
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
    # `reload=True`는 개발 중 코드 자동 리로딩을 위함
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)