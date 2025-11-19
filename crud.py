import math
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_
from sqlalchemy.dialects.postgresql import insert
from typing import List, Optional, Dict
from datetime import datetime, timedelta, date
import random

import models
import schemas

# ================================================
# (Q1) AI 예측을 위한 DB 조회 함수
# ================================================
async def get_training_data(db: AsyncSession, region_id: int, limit: int = 2000) -> pd.DataFrame:
    """
    AI 모델 학습/추론용 데이터를 DB에서 추출하여 DataFrame으로 반환합니다.
    WeatherTs(기상)와 GenerationTs(발전량)를 시간(ts) 기준으로 조인합니다.
    serving.py 모델이 사용하는 영문 컬럼명으로 반환합니다.
    """
    stmt = select(
        models.WeatherTs.ts.label('datetime'),           # 발전일자 -> datetime
        models.WeatherTs.temp_c.label('temperature'),    # 기온
        models.WeatherTs.precip_mm.label('precipitation'), # 강우량
        models.WeatherTs.humidity.label('humidity'),     # 습도
        models.WeatherTs.snow_cm.label('snow'),          # 적설량
        models.WeatherTs.cloud_10.label('cloud_cover'),  # 전운량
        models.WeatherTs.sunshine_hr.label('sunshine_duration'), # 일조
        models.WeatherTs.solar_irr.label('solar_radiation'),     # 일사량
        models.GenerationTs.capacity_mw.label('solar_capacity'), # 설비용량
        models.GenerationTs.generation_mwh.label('solar_generation') # 발전량
    ).join(
        models.GenerationTs,
        and_(
            models.WeatherTs.ts == models.GenerationTs.ts,
            models.WeatherTs.region_id == models.GenerationTs.region_id
        )
    ).where(
        models.WeatherTs.region_id == region_id
    ).order_by(
        models.WeatherTs.ts.asc() # 과거 -> 현재 순서 정렬
    )
    
    if limit > 0:
        stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return pd.DataFrame() 

    # DataFrame 변환 (컬럼명 명시)
    df = pd.DataFrame(rows, columns=[
        'datetime', 'temperature', 'precipitation', 'humidity', 
        'snow', 'cloud_cover', 'sunshine_duration', 'solar_radiation', 
        'solar_capacity', 'solar_generation'
    ])

    # 데이터 타입 보정
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df


# ================================================
# 기존 Dashboard 및 데이터 조회 함수들
# ================================================
async def get_dashboard_summary(db: AsyncSession) -> Dict:
    today = date.today()
    today_total_query = select(func.sum(models.GenerationTs.generation_mwh * 1000)).where(func.date(models.GenerationTs.ts) == today)
    today_total_result = await db.execute(today_total_query)
    today_total_kwh = today_total_result.scalar_one_or_none() or 0.0
    
    current_power_query = select(models.GenerationTs.generation_mwh * 1000).order_by(models.GenerationTs.ts.desc()).limit(1)
    current_power_result = await db.execute(current_power_query)
    current_power_kw = current_power_result.scalar_one_or_none() or 0.0

    accuracy_query = select(models.EvalDaily.mape).order_by(models.EvalDaily.date.desc()).limit(1)
    accuracy_result = await db.execute(accuracy_query)
    mape = accuracy_result.scalar_one_or_none() or 0.0
    accuracy_percent = max(0.0, 100.0 - mape)
    
    return {
        "current_power": round(current_power_kw, 1),
        "today_total": round(today_total_kwh, 0),
        "accuracy": round(accuracy_percent, 1)
    }

async def get_regions_data(db: AsyncSession) -> List[schemas.RegionPowerData]:
    query = select(
        models.Region.name,
        func.sum(models.GenerationTs.generation_mwh * 1000).label("total_power_kwh")
    )\
    .join(models.GenerationTs, models.Region.region_id == models.GenerationTs.region_id)\
    .group_by(models.Region.name)
    
    result = await db.execute(query)
    db_data = result.all()
    
    response_list = []
    mock_geo_data = {
        "서울": {"lat": 37.5665, "lng": 126.9780}, "부산": {"lat": 35.1796, "lng": 129.0756},
        "대구": {"lat": 35.8714, "lng": 128.6014}, "인천": {"lat": 37.4563, "lng": 126.7052},
        "대전": {"lat": 36.3504, "lng": 127.3845}, "제주": {"lat": 33.4996, "lng": 126.5312}
    }

    for name, total_power_kwh in db_data:
        geo = mock_geo_data.get(name, {"lat": 37.0, "lng": 127.5})
        if total_power_kwh is None or math.isnan(total_power_kwh):
            power_kwh = 0.0
        else:
            power_kwh = round(total_power_kwh, 0)
        revenue = int(power_kwh * 174)
        response_list.append(schemas.RegionPowerData(
            region=name, power=power_kwh, revenue=revenue,
            latitude=geo["lat"], longitude=geo["lng"]
        ))
    return response_list

async def get_power_forecast(db: AsyncSession, hours: int) -> List[schemas.PowerForecast]:
    now = datetime.now()
    start_time = now.replace(minute=0, second=0, microsecond=0)
    DEFAULT_REGION_ID = 1 
    
    actual_query = select(
        models.GenerationTs.ts,
        (models.GenerationTs.generation_mwh * 1000).label("actual_kwh")
    ).where(models.GenerationTs.region_id == DEFAULT_REGION_ID)
    actual_results = await db.execute(actual_query)
    actual_data = {ts: actual_kwh for ts, actual_kwh in actual_results.all()}

    predicted_query = select(
        models.ForecastTs.ts,
        models.ForecastTs.gen_pred_kwh
    ).where(models.ForecastTs.region_id == DEFAULT_REGION_ID)\
     .order_by(models.ForecastTs.generated_at.desc())
    
    predicted_results = await db.execute(predicted_query)
    
    predicted_data = {}
    for ts, pred_kwh in predicted_results.all():
        if ts not in predicted_data:
            predicted_data[ts] = pred_kwh
            
    response_list = []
    all_timestamps = sorted(list(set(actual_data.keys()) | set(predicted_data.keys())))

    for current_ts in all_timestamps:
        if len(response_list) >= 24:
             break
        time_str = current_ts.strftime("%H:%M")
        actual = actual_data.get(current_ts)
        predicted = predicted_data.get(current_ts)
        response_list.append(schemas.PowerForecast(
            time=time_str,
            actual=round(actual, 2) if actual is not None else None,
            predicted=round(predicted or 0.0, 2)
        ))
    return response_list

async def check_region_exists(db: AsyncSession, region_name: str) -> bool:
    query = select(models.Region).where(models.Region.name == region_name)
    result = await db.execute(query)
    return result.first() is not None

async def save_forecast_results(
    db: AsyncSession, 
    region_id: int, 
    model_name: str, 
    model_ver: str, 
    predictions: List[dict]
):
    """
    AI 예측 결과를 DB에 저장 (UPSERT)
    predictions: [{'ts': datetime, 'predicted_kwh': float}, ...]
    """
    if not predictions:
        print("DB에 저장할 예측 결과가 없습니다.")
        return

    objects_to_save = []
    generated_at_time = datetime.utcnow()

    for pred in predictions:
        # 이미 datetime 객체라면 변환 건너뛰기, 문자열이면 변환
        ts_val = pred["ts"]
        if isinstance(ts_val, str):
            ts_datetime = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        else:
            ts_datetime = ts_val

        objects_to_save.append({
            "ts": ts_datetime,
            "region_id": region_id,
            "horizon": 0, # 필요시 horizon 계산 로직 추가
            "gen_pred_kwh": pred["predicted_kwh"],
            "model": model_name,
            "ver": model_ver,
            "generated_at": generated_at_time
        })

    stmt = insert(models.ForecastTs).values(objects_to_save)
    stmt = stmt.on_conflict_do_update(
        index_elements=['ts', 'region_id', 'horizon', 'model', 'ver'],
        set_={
            "gen_pred_kwh": stmt.excluded.gen_pred_kwh,
            "generated_at": stmt.excluded.generated_at
        }
    )
    await db.execute(stmt)
    await db.commit()
    print(f"✅ 예측 결과 {len(objects_to_save)}건 DB 저장 완료")

async def insert_dummy_sensor_data(db: AsyncSession, region_id: int):
    """
    (고지능 가짜 센서) 계절, 시간, 날씨 상태를 반영하여 
    현실적인 더미 데이터를 생성하고 DB에 저장합니다.
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    month = now.month
    hour = now.hour

    # --- 1. 계절별 기본 설정 (기온, 일출/일몰, 최대 일사량) ---
    if month in [12, 1, 2]:  # 겨울
        base_temp = -2.0
        base_humid = 40
        sunrise, sunset = 7, 18
        max_irr_season = 0.5
    elif month in [6, 7, 8]:  # 여름
        base_temp = 26.0
        base_humid = 75
        sunrise, sunset = 5, 20
        max_irr_season = 0.9
    elif month in [9, 10, 11]: # 가을
        base_temp = 18.0
        base_humid = 60
        sunrise, sunset = 6, 19
        max_irr_season = 0.7
    else:  # 봄
        base_temp = 15.0
        base_humid = 55
        sunrise, sunset = 6, 19
        max_irr_season = 0.8

    # --- 2. 시간대별 기온 변동 (Diurnal Cycle) ---
    # 하루 중 14시에 가장 덥고, 새벽 4시에 가장 춥도록 코사인 곡선 적용
    # 시간 차이(hour - 14)를 이용해 변동폭 -5도 ~ +5도 설정
    temp_adjustment = 5 * -math.cos(math.pi * (hour - 4) / 12)
    current_temp = base_temp + temp_adjustment + random.uniform(-1.5, 1.5)

    # --- 3. 날씨 랜덤 이벤트 (맑음 70%, 흐림 20%, 비 10%) ---
    weather_type = random.choices(['sunny', 'cloudy', 'rainy'], weights=[70, 20, 10])[0]

    precip_mm = 0.0
    snow_cm = 0.0
    cloud_10 = 0
    sunshine_hr = 0.0
    solar_irr = 0.0
    
    # 낮 시간인지 확인
    is_daytime = sunrise <= hour < sunset

    if is_daytime:
        # 태양 고도에 따른 일사량 계산 (정오에 피크인 포물선)
        # day_progress: 0(일출) ~ 1(일몰)
        day_progress = (hour - sunrise) / (sunset - sunrise)
        # 포물선 공식 y = 4x(1-x) : x=0.5일 때 1이 됨
        sun_intensity = 4 * day_progress * (1 - day_progress)
        
        solar_irr = max_irr_season * sun_intensity * random.uniform(0.9, 1.1)
        sunshine_hr = 1.0 # 기본 1시간
        
    # 날씨에 따른 값 보정
    if weather_type == 'cloudy':
        cloud_10 = random.randint(5, 8)
        solar_irr *= 0.4      # 흐리면 일사량 40%로 감소
        sunshine_hr = 0.0     # 햇빛 없음
        current_temp -= 1.0   # 기온 약간 하강
    elif weather_type == 'rainy':
        cloud_10 = random.randint(9, 10)
        precip_mm = random.uniform(1.0, 15.0) # 비 옴
        solar_irr = 0.0       # 비 오면 발전량 거의 없음
        sunshine_hr = 0.0
        current_temp -= 2.0   # 기온 하강
        base_humid += 30      # 습도 대폭 상승

    # 습도 최종 계산 (0~100 제한)
    current_humid = min(100, max(0, base_humid + random.uniform(-10, 10)))

    # --- 4. 발전량 계산 (물리 법칙 반영) ---
    capacity = 100.0  # 설비 용량 100MW 가정
    # 효율: 기온이 25도보다 높으면 효율이 떨어지는 태양광 패널 특성 반영
    temp_efficiency_loss = max(0, (current_temp - 25) * 0.005) 
    efficiency = 0.85 - temp_efficiency_loss + random.uniform(-0.02, 0.02)
    
    generation_mwh = solar_irr * capacity * efficiency
    if generation_mwh < 0: generation_mwh = 0

    # --- 5. DB 저장용 딕셔너리 생성 ---
    dummy_weather = {
        "ts": now,
        "region_id": region_id,
        "temp_c": round(current_temp, 1),
        "precip_mm": round(precip_mm, 1),
        "humidity": round(current_humid, 1),
        "snow_cm": round(snow_cm, 1),
        "cloud_10": cloud_10,
        "sunshine_hr": round(sunshine_hr, 1),
        "solar_irr": round(solar_irr, 2),
    }

    dummy_generation = {
        "ts": now,
        "region_id": region_id,
        "capacity_mw": capacity,
        "generation_mwh": round(generation_mwh, 2)
    }

    # --- 6. DB Insert (UPSERT) ---
    stmt_weather = insert(models.WeatherTs).values(dummy_weather)
    stmt_weather = stmt_weather.on_conflict_do_update(
        index_elements=['ts', 'region_id'],
        set_=dummy_weather
    )
    
    stmt_gen = insert(models.GenerationTs).values(dummy_generation)
    stmt_gen = stmt_gen.on_conflict_do_update(
        index_elements=['ts', 'region_id'],
        set_=dummy_generation
    )

    await db.execute(stmt_weather)
    await db.execute(stmt_gen)
    await db.commit()
    
    # 로그 출력
    weather_desc = "☀️" if weather_type == 'sunny' else ("☁️" if weather_type == 'cloudy' else "🌧️")
    if not is_daytime: weather_desc = "🌙"
    
    print(f"✅ [Dummy Sensor] {now.strftime('%H:%M')} {weather_desc} | "
          f"기온: {current_temp:.1f}℃, 일사량: {solar_irr:.2f}, 발전량: {generation_mwh:.2f} MWh")