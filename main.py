import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from datetime import datetime
import asyncio

# 로컬 모듈
import crud
import schemas
import serving
from database import async_session
from sqlalchemy.ext.asyncio import AsyncSession

# --- 서버 시작 시 모델 로드 ---
try:
    models_ai = serving.load_jeju_pretrained_models() 
    print("✅ AI 모델 로딩 성공!")
except Exception as e:
    print(f"❌ AI 모델 로딩 실패: {e}")
    models_ai = None

app = FastAPI(title="SolarCast API")

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB 세션 의존성 ---
async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

# ====================================
# API 엔드포인트
# ====================================

@app.get("/")
def read_root():
    return {"message": "SolarCast API"}

@app.get("/api/dashboard/summary", response_model=schemas.DashboardSummary)
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    summary_data = await crud.get_dashboard_summary(db)
    return schemas.DashboardSummary(
        current_power=summary_data["current_power"],
        today_total=summary_data["today_total"],
        accuracy=summary_data["accuracy"],
        today_date=datetime.now().strftime("%m/%d(%a)")
    )

@app.get("/api/regions", response_model=List[schemas.RegionPowerData])
async def get_regions_data(db: AsyncSession = Depends(get_db)):
    return await crud.get_regions_data(db)

@app.get("/api/forecast/hourly", response_model=List[schemas.PowerForecast])
async def get_hourly_forecast(db: AsyncSession = Depends(get_db)):
    return await crud.get_power_forecast(db, hours=24)

# ====================================
# (Q1, Q2, Q3) AI 예측 및 DB 저장 API
# ====================================
@app.post("/predict", response_model=schemas.PredictionResponse)
async def predict(
    request: schemas.PredictionRequest, 
    db: AsyncSession = Depends(get_db)
):
    """
    1. DB에서 최근 데이터를 가져와 DataFrame 생성
    2. serving.py의 run_prediction 함수로 예측 수행
    3. 결과를 DB에 저장하고 응답 반환
    """
    if not models_ai:
        raise HTTPException(status_code=503, detail="AI 모델이 로드되지 않았습니다.")

    # --- 1. DB에서 AI 입력 데이터 가져오기 ---
    try:
        # request.sequence_length 만큼의 데이터를 가져오려면, 
        # DB에서는 그보다 조금 넉넉하게 가져와서 serving.py에서 처리하는 것이 안전합니다.
        features_df = await crud.get_training_data(
            db=db, 
            region_id=request.region_id, 
            limit=500 # 최근 500개 데이터를 가져옴 (충분한 시퀀스 확보용)
        )
        
        if features_df.empty:
            raise ValueError(f"DB(region_id: {request.region_id})에 데이터가 없습니다.")
            
    except Exception as e:
        print(f"❌ DB 데이터 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"DB 데이터 조회 실패: {e}")

    # --- 2. AI 추론 실행 ---
    try:
        # serving.py의 run_prediction 호출 (DataFrame 전달)
        ai_results = serving.run_prediction(
            df_input=features_df,
            loaded_models=models_ai
        )
        # ai_results = [{'예측일시': datetime, '앙상블_발전량(MWh)': float}, ...]

    except Exception as e:
        print(f"❌ AI 추론 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"AI 추론 실패: {e}")

    # --- 3. 결과 데이터 변환 (List[Dict] -> DB Schema) ---
    try:
        predictions_list = []
        model_info = {"model": "XGBoost-Stack", "version": "20251112"} # 메타데이터 예시

        for item in ai_results:
            predictions_list.append({
                "ts": item['예측일시'], # 이미 datetime 객체임
                "predicted_kwh": float(item['앙상블_발전량(MWh)']) * 1000 # MWh -> kWh 변환 필요시 확인 (일단 값 그대로 사용 시 1000 곱하기 제거)
            })

        if not predictions_list:
             raise ValueError("AI가 유효한 예측값을 반환하지 않았습니다.")

    except Exception as e:
        print(f"❌ 결과 변환 오류: {e}")
        raise HTTPException(status_code=500, detail=f"결과 변환 실패: {e}")
        
    # --- 4. DB에 저장 ---
    try:
        await crud.save_forecast_results(
            db=db, 
            region_id=request.region_id,
            model_name=model_info["model"],
            model_ver=model_info["version"],
            predictions=predictions_list
        )
    except Exception as e:
        print(f"⚠️ DB 저장 실패 (예측값은 반환됨): {e}")

    # --- 5. 응답 반환 ---
    # 프론트엔드가 이해할 수 있는 ISO 포맷 문자열로 변환하여 반환
    response_data = []
    for p in predictions_list:
        response_data.append({
            "ts": p["ts"].isoformat(),
            "predicted_kwh": p["predicted_kwh"]
        })

    return schemas.PredictionResponse(
        status="success",
        data=response_data
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)