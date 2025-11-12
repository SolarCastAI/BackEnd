from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_
import models, schemas
from typing import List, Optional, Dict
from datetime import datetime, timedelta, date
import math

# (참고) DB 스키마(models.py)와 프론트엔드 요구(schemas.py) 간의 단위 변환
# - DB: generation_mwh (메가와트시), capacity_mw (메가와트)
# - API/Frontend: Kwh (킬로와트시), Kw (킬로와트)
# - 1 MWh = 1000 KWh
# - 1 MW = 1000 KW

async def get_dashboard_summary(db: AsyncSession) -> Dict:
    """
    대시보드 요약 정보를 DB에서 조회/계산합니다.
    """
    today = date.today()
    
    # 1. 'generation_ts'에서 오늘의 누적 발전량(today_total) 계산
    # (generation_mwh * 1000 = generation_kwh)
    today_total_query = select(func.sum(models.GenerationTs.generation_mwh * 1000))\
        .where(func.date(models.GenerationTs.ts) == today)
    
    today_total_result = await db.execute(today_total_query)
    today_total_kwh = today_total_result.scalar_one_or_none() or 0.0
    
    # 2. 'generation_ts'에서 현재 발전량(current_power) 조회
    # (가장 최근 'generation_mwh' 값을 '현재 시간의 발전량'으로 간주, Kw로 변환)
    # (참고: generation_mwh는 1시간 누적값이므로 '현재' 전력(Kw)과 다를 수 있으나,
    #       스키마 상 가장 근접한 값입니다.)
    current_power_query = select(models.GenerationTs.generation_mwh * 1000)\
        .order_by(models.GenerationTs.ts.desc())\
        .limit(1)
    
    current_power_result = await db.execute(current_power_query)
    current_power_kw = current_power_result.scalar_one_or_none() or 0.0

    # 3. 'eval_daily'에서 정확도(accuracy) 계산
    # (가장 최근 MAPE 값을 가져와 100에서 뺌)
    accuracy_query = select(models.EvalDaily.mape)\
        .order_by(models.EvalDaily.date.desc())\
        .limit(1)
    
    accuracy_result = await db.execute(accuracy_query)
    mape = accuracy_result.scalar_one_or_none() or 0.0
    accuracy_percent = max(0.0, 100.0 - mape)
    
    return {
        "current_power": round(current_power_kw, 1),
        "today_total": round(today_total_kwh, 0),
        "accuracy": round(accuracy_percent, 1)
    }

async def get_regions_data(db: AsyncSession) -> List[schemas.RegionPowerData]:
    """
    모든 지역의 오늘 발전량 합계 및 기타 정보를 조회합니다.
    """
    today = date.today()
    
    # 'regions' 테이블과 'generation_ts' 테이블을 JOIN하여
    # 오늘 날짜 기준, 지역별 발전량(Kwh) 합계 조회
    query = select(
        models.Region.name,
        func.sum(models.GenerationTs.generation_mwh * 1000).label("total_power_kwh")
    )\
    .join(models.GenerationTs, models.Region.region_id == models.GenerationTs.region_id)\
    .group_by(models.Region.name)
    #.where(func.date(models.GenerationTs.ts) == today)\
    
    result = await db.execute(query)
    db_data = result.all() # (region_name, total_power_kwh) 리스트
    
    response_list = []
    
    # TODO: 위도(latitude), 경도(longitude)는 'regions' 테이블에 컬럼을 추가하는 것이 좋습니다.
    # (임시로 Mock 데이터의 위/경도 사용)
    mock_geo_data = {
        "서울": {"lat": 37.5665, "lng": 126.9780},
        "부산": {"lat": 35.1796, "lng": 129.0756},
        "대구": {"lat": 35.8714, "lng": 128.6014},
        "인천": {"lat": 37.4563, "lng": 126.7052},
        "대전": {"lat": 36.3504, "lng": 127.3845},
        "제주": {"lat": 33.4996, "lng": 126.5312} # 스키마 기본값
    }

    for name, total_power_kwh in db_data:
        geo = mock_geo_data.get(name, {"lat": 37.0, "lng": 127.5}) # 기본값
        # --- 수정된 부분 ---
        # total_power_kwh가 None이거나 NaN(숫자 아님)인지 확인
        if total_power_kwh is None or math.isnan(total_power_kwh):
            power_kwh = 0.0
        else:
            power_kwh = round(total_power_kwh, 0)
        # --- 여기까지 ---
        
        # TODO: 수익(revenue) 계산 로직 (예: kWh당 단가)은 비즈니스 로직에 맞게 수정 필요
        revenue = int(power_kwh * 174) # (임시 단가 174원/kWh)
        
        response_list.append(schemas.RegionPowerData(
            region=name,
            power=power_kwh,
            revenue=revenue,
            latitude=geo["lat"],
            longitude=geo["lng"]
        ))
        
    return response_list

async def get_region_data(db: AsyncSession, region_name: str) -> Optional[schemas.RegionPowerData]:
    """
    특정 지역의 오늘 발전량 합계 및 기타 정보를 조회합니다.
    """
    today = date.today()
    
    query = select(
        models.Region.name,
        func.sum(models.GenerationTs.generation_mwh * 1000).label("total_power_kwh")
    )\
    .join(models.GenerationTs, models.Region.region_id == models.GenerationTs.region_id)\
    .where(
        and_(
            func.date(models.GenerationTs.ts) == today,
            models.Region.name == region_name
        )
    )\
    .group_by(models.Region.name)
    
    result = await db.execute(query)
    db_data = result.first() # (region_name, total_power_kwh)
    
    if not db_data:
        return None
    
    # (get_regions_data와 동일한 임시/TODO 로직 사용)
    mock_geo_data = {"제주": {"lat": 33.4996, "lng": 126.5312}} # (간략화)
    geo = mock_geo_data.get(db_data[0], {"lat": 37.0, "lng": 127.5})
    power_kwh = round(db_data[1] or 0.0, 0)
    revenue = int(power_kwh * 174)
    
    return schemas.RegionPowerData(
        region=db_data[0],
        power=power_kwh,
        revenue=revenue,
        latitude=geo["lat"],
        longitude=geo["lng"]
    )

async def get_power_forecast(db: AsyncSession, hours: int) -> List[schemas.PowerForecast]:
    """
    시간대별 실제/예측 발전량 데이터를 조회합니다.
    """
    now = datetime.now()
    # 쿼리 기준 시간 (현재 시간의 정각)
    start_time = now.replace(minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=hours)
    
    # TODO: API가 특정 지역을 지정할 수 있도록 수정해야 합니다.
    # (임시로 '제주' (region_id=1)를 기본값으로 사용)
    DEFAULT_REGION_ID = 1 
    
    # 1. 실제 발전량 데이터(Kwh) 조회
    actual_query = select(
        models.GenerationTs.ts,
        (models.GenerationTs.generation_mwh * 1000).label("actual_kwh")
    ).where(
        models.GenerationTs.region_id == DEFAULT_REGION_ID,
        models.GenerationTs.ts >= start_time,
        models.GenerationTs.ts < end_time
    )
    actual_results = await db.execute(actual_query)
    actual_data = {ts: actual_kwh for ts, actual_kwh in actual_results.all()}

    # 2. 예측 발전량 데이터(Kwh) 조회
    # (gen_pred_kwh가 이미 Kwh 단위라고 가정)
    predicted_query = select(
        models.ForecastTs.ts,
        models.ForecastTs.gen_pred_kwh
    ).where(
        models.ForecastTs.region_id == DEFAULT_REGION_ID,
        models.ForecastTs.ts >= start_time,
        models.ForecastTs.ts < end_time,
        # TODO: (선택) 특정 모델 버전만 가져오도록 필터링
        # models.ForecastTs.model == 'XGBoost', 
    ).order_by(models.ForecastTs.generated_at.desc()) # 최신 예측 순
    
    predicted_results = await db.execute(predicted_query)
    
    # 동일 시간(ts)에 여러 예측이 있다면, 가장 최신(generated_at) 예측만 사용
    predicted_data = {}
    for ts, pred_kwh in predicted_results.all():
        if ts not in predicted_data:
            predicted_data[ts] = pred_kwh
            
    # 3. 데이터 취합
    response_list = []
    for i in range(hours):
        current_ts = start_time + timedelta(hours=i)
        time_str = current_ts.strftime("%H:%M")
        
        actual = actual_data.get(current_ts)
        predicted = predicted_data.get(current_ts)
        
        # 실제 또는 예측 데이터 둘 중 하나라도 존재하면 리스트에 추가
        if actual is not None or predicted is not None:
            response_list.append(schemas.PowerForecast(
                time=time_str,
                actual=round(actual, 2) if actual is not None else None,
                predicted=round(predicted or 0.0, 2) # 예측값이 없으면 0.0으로
            ))
            
    return response_list

async def check_region_exists(db: AsyncSession, region_name: str) -> bool:
    """
    DB에 해당 이름의 지역이 존재하는지 확인합니다. (CSV 다운로드 API용)
    """
    query = select(models.Region).where(models.Region.name == region_name)
    result = await db.execute(query)
    return result.first() is not None