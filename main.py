import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import asyncio

# --- 모든 모듈 임포트 ---
import crud
import models
import schemas
import serving  # 팀원의 AI 모듈
from database import async_session, engine, Base
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI(title="SolarCast API")

# --- CORS 설정 ---
origins = [
    "http://localhost:3000",  # React 개발 서버
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI 모델 로드 ---
# (서버 시작 시 AI 모델을 메모리에 로드)
try:
    models_ai = serving.predict_()
    '''models_ai 반환값 = {
                    'time': target_time.strftime('%H:%M'),
                    'lstm': lstm_pred,
                    'gru': gru_pred,
                    'ensemble': ensemble_pred
                }을 담고 있는 리스트 형태 결국 결론은 ensemble을 사용해야 됨'''
    for line in models_ai:
        print(line)
    print("✅ AI 모델 추론 성공!")

except Exception as e:
    print(f"❌ AI 모델 로딩 실패: {e}")
    models_ai = None

# --- DB 세션 의존성 ---
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

# ====================================
# API 엔드포인트 (DB 연동 완료)
# ====================================

@app.get("/")
def read_root():
    return {"message": "SolarCast API"}

@app.get("/api/dashboard/summary", response_model=schemas.DashboardSummary)
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    """
    대시보드 요약 정보 조회 (실제 DB 연동)
    """
    # [Mock 데이터 제거] -> [crud.py 호출]
    summary_data = await crud.get_dashboard_summary(db)
    
    return schemas.DashboardSummary(
        current_power=summary_data["current_power"],
        today_total=summary_data["today_total"],
        accuracy=summary_data["accuracy"],
        today_date=datetime.now().strftime("%m/%d(%a)")
    )

@app.get("/api/regions", response_model=List[schemas.RegionPowerData])
async def get_regions_data(db: AsyncSession = Depends(get_db)):
    """
    지역별 발전량 데이터 조회 (실제 DB 연동)
    """
    # [Mock 데이터 제거] -> [crud.py 호출]
    return await crud.get_regions_data(db)

@app.get("/api/forecast/hourly", response_model=List[schemas.PowerForecast])
async def get_hourly_forecast(db: AsyncSession = Depends(get_db)):
    """
    시간대별 발전량 예측 데이터 조회 (실제 DB 연동)
    """
    # [Mock 데이터 제거] -> [crud.py 호출]
    # (임시로 24시간 조회, 필요시 hours 파라미터 추가)
    return await crud.get_power_forecast(db, hours=24)

# ====================================
# AI 예측 및 DB 저장 API (핵심)
# ====================================

@app.post("/predict", response_model=schemas.PredictionResponse)
async def predict(
    request: schemas.PredictionRequest, 
    db: AsyncSession = Depends(get_db) # <-- DB 세션 주입
):
    """
    프론트엔드에서 예측 요청을 받아 AI 추론을 수행하고,
    입력값과 출력값을 DB에 저장합니다.
    """
    if not models_ai:
        raise HTTPException(status_code=503, detail="AI 모델이 로드되지 않았습니다.")

    # --- 1. (AI 작업) 팀원의 AI 추론 코드 ---
    try:
        # (serving.py의 함수 형식에 맞게 호출)
        # 예시: request.features가 DataFrame에 필요한 dict라고 가정
        features_df = pd.DataFrame(request.features)
        
        # serving.py의 예측 함수 호출
        ai_result_dict = serving.predict_(
            new_data=features_df,
            models_dict=models_ai,
            sequence_length=request.sequence_length
        )
        # (ai_result_dict의 형식을 serving.py에 맞게 조정 필요)
        
        # (임시) serving.py가 아래 형식으로 반환한다고 가정:
        # ai_result_dict = {
        #     "stacked_predictions": [120.5, 130.2, ...],
        #     "metadata": {"model": "XGBoost", "version": "1.2"}
        # }
        
        # (임시) 프론트엔드/DB 저장용 데이터로 가공
        predictions_list = []
        base_time = datetime.utcnow() # (임시)
        for i, pred_val in enumerate(ai_result_dict.get("stacked_predictions", [])):
            predictions_list.append({
                "ts": (base_time + pd.Timedelta(hours=i+1)).isoformat() + "Z",
                "predicted_kwh": pred_val
            })
            
        model_info = ai_result_dict.get("metadata", {"model": "XGBoost", "version": "1.2"})

    except Exception as e:
        print(f"❌ AI 추론 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"AI 추론 실패: {e}")

    # --- 2. (DB 작업) 출력값 저장 ---
    try:
        await crud.save_forecast_results(
            db=db, 
            region_id=request.region_id,
            model_name=model_info["model"],
            model_ver=model_info["version"],
            predictions=predictions_list
        )
    except Exception as e:
        # (선택) DB 저장이 실패해도 프론트엔드에는 예측 결과를 반환할 수 있음
        print(f"⚠️ DB 저장 실패 (하지만 예측은 성공): {e}")

    # --- 3. (Frontend 반환) AI 결과를 프론트엔드에 반환 ---
    return schemas.PredictionResponse(
        status="success",
        data=predictions_list
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)