
import asyncio
import pandas as pd
from sqlalchemy import select
from celery_app import celery_app
from database import async_session
import crud
import serving
import models  # Region 모델 조회를 위해 필요

# ---------------------------------------------------------------------------
# [Helper] 비동기 함수 실행기
# ---------------------------------------------------------------------------
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ---------------------------------------------------------------------------
# [Helper] 모든 지역 ID 조회 함수 (마스터 태스크용)
# ---------------------------------------------------------------------------
async def get_all_region_ids():
    async with async_session() as db:
        # DB에 등록된 모든 Region의 ID를 리스트로 가져옴
        result = await db.execute(select(models.Region.region_id))
        return result.scalars().all()

# ===========================================================================
# 1. [Master] 매 시간마다 모든 지역의 예측 작업을 스케줄링 (Beat가 호출)
# ===========================================================================
@celery_app.task(name='tasks.schedule_hourly_prediction_all_regions')
def schedule_hourly_prediction_all_regions():
    """
    [반장 태스크]
    DB에서 모든 지역 ID를 조회한 뒤, 각 지역별로 run_hourly_prediction 태스크를 실행시킵니다.
    """
    print("🌍 [Master] 모든 지역의 시간별 예측 작업을 스케줄링합니다...")
    
    try:
        # 모든 지역 ID 조회
        region_ids = run_async(get_all_region_ids())
        
        # 각 지역별로 작업 배포 (Fan-out)
        for rid in region_ids:
            # delay()를 쓰면 비동기로 워커 큐에 들어갑니다.
            run_hourly_prediction.delay(region_id=rid)
            print(f"   --> Region {rid} 예측 작업 큐 등록 완료")
            
        return f"Scheduled prediction for {len(region_ids)} regions."
    
    except Exception as e:
        print(f"❌ [Master Error] 지역 목록 조회 실패: {e}")
        return f"Error: {e}"

# ===========================================================================
# 2. [Worker] 개별 지역 예측 실행 (실제 일꾼)
# ===========================================================================
@celery_app.task(name='tasks.run_hourly_prediction')
def run_hourly_prediction(region_id: int):
    """
    [일꾼 태스크]
    특정 지역(region_id)의 가짜 데이터를 만들고, 예측을 수행합니다.
    """
    print(f"🕒 [Worker] Region {region_id} 예측 작업 시작")

    async def _process():
        async with async_session() as db:
            try:
                # 1. 가짜 센서 데이터 생성 (센서 시뮬레이션)
                #    각 지역 ID에 맞는 데이터를 생성해서 DB에 넣음
                await crud.insert_dummy_sensor_data(db, region_id)
                
                # 2. DB 데이터 조회 (방금 넣은 데이터 포함)
                features_df = await crud.get_training_data(db, region_id, limit=500)
                
                if features_df.empty:
                    print(f"   ⚠️ [Region {region_id}] 데이터 부족으로 중단")
                    return "No Data"

                # 3. AI 예측 수행
                ai_results = serving.run_prediction(features_df)
                
                if not ai_results:
                    print(f"   ⚠️ [Region {region_id}] AI 예측 결과 없음")
                    return "Prediction Failed"

                # 4. 결과 포맷 변환
                predictions_list = []
                for item in ai_results:
                    # serving.py의 반환 키값과 일치시켜야 함 ('앙상블_현재_발전량(MWh)')
                    val = item.get('앙상블_현재_발전량(MWh)', item.get('앙상블_발전량(MWh)', 0))
                    predictions_list.append({
                        "ts": item['예측_일시'], # serving.py의 키 확인 ('예측_일시')
                        "predicted_kwh": float(val) * 1000
                    })

                # 5. DB에 저장
                await crud.save_forecast_results(
                    db=db,
                    region_id=region_id,
                    model_name="XGBoost-Stack-Auto",
                    model_ver="v1.0-hourly",
                    predictions=predictions_list
                )
                return f"Success: Saved {len(predictions_list)} predictions for Region {region_id}"

            except Exception as e:
                print(f"   ❌ [Region {region_id}] 에러 발생: {e}")
                return f"Error: {e}"

    result = run_async(_process())
    print(f"✅ [Worker Finished] Region {region_id}: {result}")
    return result


# ===========================================================================
# 3. [Master] 매일 자정 모델 재학습 스케줄링 (Beat가 호출)
# ===========================================================================
@celery_app.task(name='tasks.schedule_daily_retraining_all_regions')
def schedule_daily_retraining_all_regions():
    """
    [반장 태스크]
    모든 지역 ID를 조회하여 재학습 작업을 배포합니다.
    """
    print("🌙 [Master] 모든 지역의 모델 재학습을 스케줄링합니다...")
    try:
        region_ids = run_async(get_all_region_ids())
        for rid in region_ids:
            run_daily_retraining.delay(region_id=rid)
            print(f"   --> Region {rid} 재학습 작업 큐 등록 완료")
        return f"Scheduled retraining for {len(region_ids)} regions."
    except Exception as e:
        print(f"❌ [Master Error] {e}")
        return f"Error: {e}"

# ===========================================================================
# 4. [Worker] 개별 지역 모델 재학습 (실제 일꾼)
# ===========================================================================
@celery_app.task(name='tasks.run_daily_retraining')
def run_daily_retraining(region_id: int):
    """
    [일꾼 태스크]
    특정 지역의 데이터를 모아 모델을 업데이트합니다.
    """
    print(f"🌙 [Worker] Region {region_id} 재학습 시작")

    async def _process():
        async with async_session() as db:
            try:
                # 1. 어제 모델 성적 채점
                await crud.calculate_daily_accuracy(db, region_id)
                
                # 2. 학습 데이터 조회
                df_train = await crud.get_training_data(db, region_id, limit=3000)
                
                if len(df_train) < 100:
                    return "Skipped: Not enough data"

                # 3. 재학습 실행 (GPU 부하가 클 수 있음)
                #    Worker가 여러 개면 병렬로 돌지만, GPU 메모리 주의 필요
                success = serving.retrain_model(df_train)
                
                if success:
                    return f"Success: Region {region_id} Model Updated"
                else:
                    return f"Failed: Region {region_id} Training Error"

            except Exception as e:
                print(f"   ❌ [Region {region_id}] 재학습 에러: {e}")
                return f"Error: {e}"

    return run_async(_process())