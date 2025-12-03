import math
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_
from sqlalchemy.dialects.postgresql import insert
from typing import List, Optional, Dict
from datetime import datetime, timedelta, date, timezone
import random

import models
import schemas


async def get_training_data(db: AsyncSession, region_id: int, limit: int = 2000) -> pd.DataFrame:
    """
    AI 모델 재학습(Fine-tuning)을 위해 과거의 날씨(X)와 발전량(y) 데이터를 조회합니다.
    """
    query = select(
        models.WeatherTs.ts.label("datetime"),
        models.WeatherTs.temp_c.label("기온"),
        models.WeatherTs.precip_mm.label("강수량(mm)"),
        models.WeatherTs.humidity.label("습도"),
        models.WeatherTs.snow_cm.label("적설(cm)"),
        models.WeatherTs.cloud_10.label("전운량(10분위)"),
        models.WeatherTs.sunshine_hr.label("일조(hr)"),
        models.WeatherTs.solar_irr.label("일사량"),
        models.GenerationTs.capacity_mw.label("태양광 설비용량(MW)"),
        models.GenerationTs.generation_mwh.label("태양광 발전량(MWh)") # <-- 학습용 정답(Label) 포함
    )\
    .join(models.GenerationTs, 
          and_(
              models.WeatherTs.ts == models.GenerationTs.ts,
              models.WeatherTs.region_id == models.GenerationTs.region_id
          ))\
    .where(models.WeatherTs.region_id == region_id)\
    .order_by(models.WeatherTs.ts.desc())\
    .limit(limit)
    
    result = await db.execute(query)
    rows = result.fetchall()
    
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame([row._asdict() for row in rows])
    # 시간 오름차순 정렬 (과거 -> 현재)
    df = df.iloc[::-1].reset_index(drop=True)
    
    return df
# ================================================
# (Q1) AI 예측을 위한 DB 조회 함수
# ================================================
async def get_features_for_prediction(
    db: AsyncSession, 
    region_id: int, 
    sequence_length: int
) -> pd.DataFrame:
    """
    AI 예측에 필요한 최근 N개(sequence_length)의 데이터를 조회합니다.
    (serving.py의 feature_columns 순서와 정확히 일치해야 합니다)
    """
    
    # 1. 쿼리 작성 (순서 중요!)
    # serving.py: ['기온', '강수량(mm)', '습도', '적설(cm)', '전운량(10분위)', '일조(hr)', '일사량', '태양광 설비용량(MW)']
    query = select(
        models.WeatherTs.ts.label("datetime"), # (참고용: 시간)
        models.WeatherTs.temp_c.label("기온"),
        models.WeatherTs.precip_mm.label("강수량(mm)"),
        models.WeatherTs.humidity.label("습도"),
        models.WeatherTs.snow_cm.label("적설(cm)"),
        models.WeatherTs.cloud_10.label("전운량(10분위)"),
        models.WeatherTs.sunshine_hr.label("일조(hr)"),
        models.WeatherTs.solar_irr.label("일사량"),
        models.GenerationTs.capacity_mw.label("태양광 설비용량(MW)")
    )\
    .join(models.GenerationTs, 
          and_(
              models.WeatherTs.ts == models.GenerationTs.ts,
              models.WeatherTs.region_id == models.GenerationTs.region_id
          ))\
    .where(models.WeatherTs.region_id == region_id)\
    .order_by(models.WeatherTs.ts.desc())\
    .limit(sequence_length)
    
    # 2. 실행
    result = await db.execute(query)
    rows = result.fetchall()
    
    # 3. 데이터가 없을 경우 빈 DF 반환
    if not rows:
        return pd.DataFrame()
        
    # 4. DataFrame 변환
    df = pd.DataFrame([row._asdict() for row in rows])
    
    # 5. 시간 순서 뒤집기 (DB: 최신->과거 / AI: 과거->최신)
    df = df.iloc[::-1].reset_index(drop=True)
    
    # 6. NaN(빈 값) 처리 (안전장치)
    df = df.fillna(0.0)
    
    return df

# -------------------------------------------------------------------
# [1] 대시보드 요약 정보
# -------------------------------------------------------------------
async def get_dashboard_summary(db: AsyncSession, region_id: int) -> Dict:
    today = date.today()
    
    # 1. 오늘의 누적 발전량
    today_total_query = select(func.sum(models.GenerationTs.generation_mwh * 1000))\
        .where(
            func.date(models.GenerationTs.ts) == today,
            models.GenerationTs.region_id == region_id
        )
    today_total_result = await db.execute(today_total_query)
    today_total_kwh = today_total_result.scalar_one_or_none() or 0.0
    
    # 2. 현재 발전량
    current_power_query = select(models.GenerationTs.generation_mwh * 1000)\
        .where(models.GenerationTs.region_id == region_id) \
        .order_by(models.GenerationTs.ts.desc())\
        .limit(1)
    
    current_power_result = await db.execute(current_power_query)
    current_power_kw = current_power_result.scalar_one_or_none() or 0.0

    # 3. 정확도
    accuracy_query = select(models.EvalDaily.mape)\
        .where(models.EvalDaily.region_id == region_id) \
        .order_by(models.EvalDaily.date.desc())\
        .limit(1)
    
    accuracy_result = await db.execute(accuracy_query)
    mape = accuracy_result.scalar_one_or_none() or 0.0
    accuracy_percent = max(0.0, 100.0 - mape)
    
    today_revenue = int(today_total_kwh * 174)

    return {
        "current_power": round(current_power_kw, 1),
        "today_total": round(today_total_kwh, 0),
        "today_revenue": today_revenue,
        "accuracy": round(accuracy_percent, 1)
    }

# -------------------------------------------------------------------
# [2] 시간대별 예측 데이터 (수정됨: 시간 매칭 로직 개선)
# -------------------------------------------------------------------
async def get_power_forecast(db: AsyncSession, hours: int, region_id: int) -> List[schemas.PowerForecast]:
    # 현재 시간 (정각 기준)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_time = now - timedelta(hours=24)
    end_time = now + timedelta(hours=hours)
    
    # 1. 실제 발전량 조회
    actual_query = select(
        models.GenerationTs.ts,
        (models.GenerationTs.generation_mwh * 1000).label("actual_kwh")
    ).where(
        models.GenerationTs.region_id == region_id,
        models.GenerationTs.ts >= start_time,
        models.GenerationTs.ts < end_time
    )
    actual_results = await db.execute(actual_query)
    
    # [수정 1] 실제 데이터도 '분'을 버리고 '정각'으로 맞춤 (안전장치)
    actual_data = {}
    for row in actual_results.all():
        ts_utc = row.ts.replace(tzinfo=timezone.utc) if row.ts.tzinfo is None else row.ts.astimezone(timezone.utc)
        # 분/초 제거 -> 정각 만들기
        ts_rounded = ts_utc.replace(minute=0, second=0, microsecond=0)
        key = ts_rounded.strftime("%Y-%m-%d %H:%M")
        actual_data[key] = row.actual_kwh

    # 2. 예측 발전량 조회
    predicted_query = select(
        models.ForecastTs.ts,
        models.ForecastTs.gen_pred_kwh
    ).where(
        models.ForecastTs.region_id == region_id,
        models.ForecastTs.ts >= start_time,
        models.ForecastTs.ts < end_time
    ).order_by(models.ForecastTs.generated_at.desc())
    
    predicted_results = await db.execute(predicted_query)
    
    # [수정 2] 예측 데이터도 '분'을 버리고 '정각' 키로 저장 ★핵심★
    predicted_data = {}
    for row in predicted_results.all():
        ts_utc = row.ts.replace(tzinfo=timezone.utc) if row.ts.tzinfo is None else row.ts.astimezone(timezone.utc)
        # 예: 06:15:12 -> 06:00:00 으로 변환
        ts_rounded = ts_utc.replace(minute=0, second=0, microsecond=0)
        key = ts_rounded.strftime("%Y-%m-%d %H:%M")
        
        # 같은 시간대에 여러 예측이 있으면 최신 것(쿼리 순서상 먼저 나온 것)만 유지
        if key not in predicted_data:
            predicted_data[key] = row.gen_pred_kwh
            
    # 3. 데이터 병합 (KST 시간으로 변환하여 반환)
    response_list = []
    kst_tz = timezone(timedelta(hours=9))
    total_hours = 24 + hours 

    for i in range(total_hours):
        current_ts_utc = start_time + timedelta(hours=i)
        
        # 검색용 키 (UTC 정각 기준)
        key = current_ts_utc.strftime("%Y-%m-%d %H:%M")
        
        actual = actual_data.get(key)
        predicted = predicted_data.get(key)
        
        # 표시용 시간 (KST 변환)
        current_ts_kst = current_ts_utc.astimezone(kst_tz)
        time_str = current_ts_kst.strftime("%m/%d %H:%M")
        
        response_list.append(schemas.PowerForecast(
            time=time_str,
            actual=round(actual, 2) if actual is not None else None,
            predicted=round(predicted or 0.0, 2)
        ))
            
    return response_list

# -------------------------------------------------------------------
# [3] 기타 함수들 (기존 유지)
# -------------------------------------------------------------------
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
    # 임시 좌표 데이터 (지도용)
    mock_geo_data = {
        "서울": {"lat": 37.5665, "lng": 126.9780}, "부산": {"lat": 35.1796, "lng": 129.0756},
        "대구": {"lat": 35.8714, "lng": 128.6014}, "인천": {"lat": 37.4563, "lng": 126.7052},
        "대전": {"lat": 36.3504, "lng": 127.3845}, "제주": {"lat": 33.4996, "lng": 126.5312},
        "광주": {"lat": 35.1595, "lng": 126.8526}, "울산": {"lat": 35.5384, "lng": 129.3114},
        "세종": {"lat": 36.4800, "lng": 127.2890}, "경기": {"lat": 37.4138, "lng": 127.5183},
        "강원": {"lat": 37.8228, "lng": 128.1555}, "충북": {"lat": 36.6350, "lng": 127.4914},
        "충남": {"lat": 36.6588, "lng": 126.6728}, "전북": {"lat": 35.7175, "lng": 127.1530},
        "전남": {"lat": 34.8161, "lng": 126.4629}, "경북": {"lat": 36.5760, "lng": 128.5056},
        "경남": {"lat": 35.2383, "lng": 128.6924},
    }

    for name, total_power_kwh in db_data:
        geo = mock_geo_data.get(name, {"lat": 36.5, "lng": 127.5}) 
        power = round(total_power_kwh or 0.0, 0)
        revenue = int(power * 174)
        response_list.append(schemas.RegionPowerData(
            region=name, power=power, revenue=revenue,
            latitude=geo["lat"], longitude=geo["lng"]
        ))
    return response_list

async def get_training_data(db: AsyncSession, region_id: int, limit: int = 2000) -> pd.DataFrame:
    # (기존 코드와 동일)
    query = select(
        models.WeatherTs.ts.label("datetime"),
        models.WeatherTs.temp_c.label("기온"),
        models.WeatherTs.precip_mm.label("강수량(mm)"),
        models.WeatherTs.humidity.label("습도"),
        models.WeatherTs.snow_cm.label("적설(cm)"),
        models.WeatherTs.cloud_10.label("전운량(10분위)"),
        models.WeatherTs.sunshine_hr.label("일조(hr)"),
        models.WeatherTs.solar_irr.label("일사량"),
        models.GenerationTs.capacity_mw.label("태양광 설비용량(MW)"),
        models.GenerationTs.generation_mwh.label("태양광 발전량(MWh)")
    )\
    .join(models.GenerationTs, 
          and_(
              models.WeatherTs.ts == models.GenerationTs.ts,
              models.WeatherTs.region_id == models.GenerationTs.region_id
          ))\
    .where(models.WeatherTs.region_id == region_id)\
    .order_by(models.WeatherTs.ts.desc())\
    .limit(limit)
    
    result = await db.execute(query)
    rows = result.fetchall()
    if not rows: return pd.DataFrame()
    df = pd.DataFrame([row._asdict() for row in rows])
    return df.iloc[::-1].reset_index(drop=True)

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

    if not predictions: return
    
    stmt = insert(models.ForecastTs).values([
        {
            "ts": datetime.fromisoformat(p["ts"].replace("Z", "+00:00")) if isinstance(p["ts"], str) else p["ts"],
            "region_id": region_id,
            "horizon": 0,
            "gen_pred_kwh": p["predicted_kwh"],
            "model": model_name,
            "ver": model_ver,
            "generated_at": datetime.now(timezone.utc)
        } for p in predictions
    ])
    stmt = stmt.on_conflict_do_update(
        index_elements=['ts', 'region_id', 'horizon', 'model', 'ver'],
        set_={"gen_pred_kwh": stmt.excluded.gen_pred_kwh}
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
    
# 모델 예측 점수 
async def calculate_daily_accuracy(db: AsyncSession, region_id: int):
    """
    [일일 평가] 어제 날짜의 '실제 vs 예측'을 비교하여 정확도를 계산하고 DB에 저장합니다.
    """
    # 1. 어제 날짜 구하기 (UTC 기준)
    now = datetime.now(timezone.utc)
    yesterday = (now - timedelta(days=1)).date()
    
    print(f"📝 [Evaluation] {yesterday} 일자 모델 성능 평가 시작...")

    # 2. 어제 하루치 '실제 발전량' 총합 (MWh -> kWh 변환)
    actual_query = select(func.sum(models.GenerationTs.generation_mwh * 1000))\
        .where(
            models.GenerationTs.region_id == region_id,
            func.date(models.GenerationTs.ts) == yesterday
        )
    actual_total = (await db.execute(actual_query)).scalar() or 0.0

    # 3. 어제 하루치 '예측 발전량' 총합 (kWh)
    pred_query = select(func.sum(models.ForecastTs.gen_pred_kwh))\
        .where(
            models.ForecastTs.region_id == region_id,
            func.date(models.ForecastTs.ts) == yesterday
        )
    pred_total = (await db.execute(pred_query)).scalar() or 0.0

    # 4. 정확도 계산 (0으로 나누기 방지)
    if actual_total == 0:
        accuracy = 0.0 # 실제 발전량이 없으면 정확도 0 처리
    else:
        # 오차율 = |실제 - 예측| / 실제
        error_rate = abs(actual_total - pred_total) / actual_total
        accuracy = max(0, (1 - error_rate) * 100) # 100점 만점 환산

    # 5. 점수 저장 (eval_daily 테이블)
    eval_data = {
        "date": yesterday,
        "region_id": region_id,
        "model": "XGBoost-Stack", # 사용 중인 모델명
        "ver": "v1.0",
        "mae": abs(actual_total - pred_total), # 오차 절대값
        "rmse": 0.0, # (약식) 필요시 구현
        "mape": 100 - accuracy, # 오차율(%)
        "samples": 24 # 24시간 데이터
    }

    stmt = insert(models.EvalDaily).values(eval_data)
    stmt = stmt.on_conflict_do_update(
        index_elements=['date', 'region_id', 'model', 'ver'],
        set_=eval_data
    )
    
    await db.execute(stmt)
    await db.commit()
    
    print(f"✅ [Evaluation] {yesterday} 평가 완료: 실제 {actual_total:.1f} vs 예측 {pred_total:.1f} -> 정확도 {accuracy:.1f}%")
    return accuracy