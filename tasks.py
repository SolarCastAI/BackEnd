# solarcastai/backend/.../tasks.py
import sys
import os
from dotenv import load_dotenv
import asyncio
import pandas as pd
from celery_app import celery_app
from database import async_session
import crud
import serving


load_dotenv()

# Windows 이벤트 루프 설정
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 비동기(async) 함수를 Celery(동기)에서 실행하기 위한 래퍼 함수
def run_async(coro):
    return asyncio.run(coro)

@celery_app.task(name='tasks.run_hourly_prediction')
def run_hourly_prediction(region_id: int):
    """
    1시간마다 실행되는 Celery 작업
    DB 데이터 조회 -> AI 예측 -> DB 저장
    """
    print(f"🕒 [Task Started] Region {region_id} 예측 작업 시작")

    async def _process():
        async with async_session() as db:
            try:
                # =======================================
                # 1. (NEW) 가짜 데이터 생성 (센서 역할)
                # =======================================
                print("   1. 가짜 센서 데이터 생성 중...")
                await crud.insert_dummy_sensor_data(db, region_id)
                
                # =======================================
                # 2. DB에서 데이터 가져오기
                # =======================================
                print("   2. DB 데이터 조회 중... (방금 넣은 데이터 포함)")
                # 이제 방금 넣은 최신 데이터가 포함되어 조회됨!
                features_df = await crud.get_training_data(db, region_id, limit=500)
                
                if features_df.empty:
                    print("   ⚠️ 데이터 부족으로 예측 중단")
                    return "No Data"

                # 2. AI 예측 수행 (serving.py)
                # (매번 모델을 새로 로드하므로 메모리 효율적, 속도가 중요하면 전역 변수로 뺼 수 있음)
                print("   2. AI 모델 예측 수행 중...")
                ai_results = serving.run_prediction(features_df)
                
                if not ai_results:
                    print("   ⚠️ AI 예측 결과 없음")
                    return "Prediction Failed"

                # 3. 결과 포맷 변환
                predictions_list = []
                for item in ai_results:
                    predictions_list.append({
                        "ts": item['예측일시'],
                        "predicted_kwh": float(item['앙상블_발전량(MWh)']) * 1000
                    })

                # 4. DB에 저장
                print(f"   3. 결과 DB 저장 중... ({len(predictions_list)}건)")
                await crud.save_forecast_results(
                    db=db,
                    region_id=region_id,
                    model_name="XGBoost-Stack-Auto",
                    model_ver="v1.0-hourly",
                    predictions=predictions_list
                )
                return f"Success: {len(predictions_list)} predictions saved."

            except Exception as e:
                print(f"   ❌ 작업 중 에러 발생: {e}")
                return f"Error: {e}"

    # 비동기 로직 실행
    result = run_async(_process())
    print(f"✅ [Task Finished] {result}")
    return result

@celery_app.task(name='tasks.run_daily_retraining')
def run_daily_retraining(region_id: int):
    """
    하루 1번 실행: 최근 데이터를 모아서 모델을 재학습(Update)함
    """
    print(f"🌙 [Retraining Task] Region {region_id} 모델 재학습 작업 시작")

    async def _process():
        async with async_session() as db:
            try:
                # 1. 어제 모델 성적표 채점하기 📝
                await crud.calculate_daily_accuracy(db, region_id)
                # 2. 학습 데이터 조회 (지난 30일치 정도? limit=2000개면 충분)
                print("   1. 학습용 데이터 조회 중...")
                # limit을 넉넉하게 잡아서 가져옵니다.
                df_train = await crud.get_training_data(db, region_id, limit=3000)
                
                if len(df_train) < 100:
                    print("   ⚠️ 데이터가 너무 적어 재학습을 건너뜁니다.")
                    return "Skipped: Not enough data"

                # 3. serving.py의 재학습 함수 호출 (동기 함수이므로 바로 호출)
                # (GPU가 있다면 여기서 시간이 좀 걸립니다)
                print(f"   2. 모델 재학습 시작 (데이터 {len(df_train)}건)...")
                success = serving.retrain_model(df_train)
                
                if success:
                    return "Success: Model Updated"
                else:
                    return "Failed: Training Error"

            except Exception as e:
                print(f"   ❌ 작업 중 에러: {e}")
                return f"Error: {e}"

    return run_async(_process())