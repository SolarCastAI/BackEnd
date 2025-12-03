import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
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
    lstm_model, gru_model, scaler_X, scaler_y = serving.preload_models()
    print("✅ AI 모델 로딩 성공!")
    models_loaded = True
except Exception as e:
    print(f"❌ AI 모델 로딩 실패: {e}")
    models_loaded = False
    lstm_model = gru_model = scaler_X = scaler_y = None

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

# [수정] region_id 쿼리 파라미터 받기 (기본값 1)
@app.get("/api/dashboard/summary", response_model=schemas.DashboardSummary)
async def get_dashboard_summary(
    region_id: int = 1, 
    db: AsyncSession = Depends(get_db)
):
    summary_data = await crud.get_dashboard_summary(db, region_id)
    return schemas.DashboardSummary(
        current_power=summary_data["current_power"],
        today_total=summary_data["today_total"],
        accuracy=summary_data["accuracy"],
        today_revenue=summary_data["today_revenue"], # 수익 추가
        today_date=datetime.now().strftime("%m/%d(%a)")
    )

@app.get("/api/regions", response_model=List[schemas.RegionPowerData])
async def get_regions_data(db: AsyncSession = Depends(get_db)):
    return await crud.get_regions_data(db)

# [수정] region_id 쿼리 파라미터 받기
@app.get("/api/forecast/hourly", response_model=List[schemas.PowerForecast])
async def get_hourly_forecast(
    hours: int = 24, 
    region_id: int = 1, 
    db: AsyncSession = Depends(get_db)
):
    return await crud.get_power_forecast(db, hours=hours, region_id=region_id)

# ====================================
# (수정됨) AI 예측 및 DB 저장 API
# ====================================

@app.post("/predict", response_model=schemas.PredictionResponse)
async def predict(
    request: schemas.PredictionRequest, 
    db: AsyncSession = Depends(get_db)
):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="AI 모델이 로드되지 않았습니다.")

    # --- 1. DB에서 AI 입력 데이터 가져오기 ---
    try:
        # (crud.py는 그대로 사용 가능)
        features_df = await crud.get_features_for_prediction(
            db=db, 
            region_id=request.region_id, 
            sequence_length=request.sequence_length
        )
        if features_df.empty:
            raise ValueError(f"예측에 필요한 데이터가 DB(region_id: {request.region_id})에 부족합니다.")
            
    except Exception as e:
        print(f"❌ DB 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {e}")

    # --- 2. AI 추론 실행 (수정됨) ---
    try:
        # serving.py의 predict_future_daegu 함수 호출
        # 이 함수는 (lstm, gru, scaler_X, scaler_y, data_sequence, ...)를 인자로 받습니다.
        
        # DataFrame을 numpy array로 변환 (입력 데이터)
        # (주의: serving.py의 feature_columns 순서와 crud.py의 조회 순서가 일치해야 함)
        input_data = features_df.values 
        
        # 예측 실행
        # (predict_future_daegu는 내부적으로 24시간을 예측하도록 되어 있음)
        ai_results = serving.predict_future_daegu(
            lstm_model=lstm_model,
            gru_model=gru_model,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            data_sequence=input_data,
            solar_capacity=446.0, # (임시: 설비용량. DB에서 가져오거나 상수로 지정)
            hours_ahead=24 # 24시간 예측
        )
        # ai_results = [{'time': '10:00', 'lstm': ..., 'gru': ..., 'ensemble': ...}, ...]

    except Exception as e:
        print(f"❌ AI 추론 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"AI 추론 실패: {e}")

    # --- 3. 결과 변환 ---
    try:
        predictions_list = []
        model_info = {"model": "Daegu-Transfer-Ensemble", "version": "20251130"}
        
        # 현재 시간 (기준)
        base_time = datetime.utcnow()
        # 오늘 날짜 (문자열 '10:00'에 날짜를 붙여주기 위함)
        today_str = base_time.strftime("%Y-%m-%d")

        for item in ai_results:
            # item['time']은 '10:00' 같은 문자열임. 날짜를 붙여서 datetime으로 만듦
            time_str = f"{today_str} {item['time']}:00"
            
            # (만약 내일 시간으로 넘어갔다면 날짜 하루 추가 로직이 필요할 수 있음)
            # 여기서는 간단히 처리
            
            predictions_list.append({
                "ts": time_str, # ISO 포맷이 아니어도 fromisoformat이 처리 가능할 수 있음
                # 앙상블 결과를 최종 예측값으로 사용
                "predicted_kwh": float(item['ensemble'])
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
            "ts": p["ts"], # (혹은 ISO 포맷으로 변환)
            "predicted_kwh": p["predicted_kwh"]
        })

    return schemas.PredictionResponse(
        status="success",
        data=response_data
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)