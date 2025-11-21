import asyncio
from datetime import datetime, timedelta, timezone
from database import async_session
from sqlalchemy.dialects.postgresql import insert
import models
import random
import math

async def seed_past_24h():
    async with async_session() as db:
        print("🌱 지난 24시간 데이터 채우기 시작...")
        
        # 현재 시간(UTC) 기준으로 지난 24시간을 계산
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_time = now - timedelta(hours=24)
        
        current = start_time
        while current <= now:
            # 1. 시간대별 현실적인 데이터 생성 (낮엔 발전량 있고 밤엔 0)
            # 한국 시간(KST) 기준으로 낮/밤 계산 (UTC+9)
            kst_hour = (current.hour + 9) % 24
            is_daytime = 6 <= kst_hour <= 19
            
            # 태양 고도에 따른 일사량 곡선 흉내 (사인파)
            solar_irr = 0.0
            if is_daytime:
                # 낮 12~1시에 피크
                solar_irr = max(0, math.sin((kst_hour - 6) * math.pi / 13)) 
            
            # 2. 발전량 및 날씨 데이터 생성
            capacity = 100.0
            # 약간의 랜덤성을 줘서 그래프가 자연스럽게 보이도록 함
            generation = solar_irr * capacity * 0.85 * random.uniform(0.9, 1.1) if is_daytime else 0.0
            
            weather = {
                "ts": current, "region_id": 1,
                "temp_c": 15 + (5 * solar_irr),
                "precip_mm": 0, "humidity": 50, "snow_cm": 0,
                "cloud_10": 2, "sunshine_hr": 1 if is_daytime else 0,
                "solar_irr": round(solar_irr, 2)
            }
            
            gen = {
                "ts": current, "region_id": 1,
                "capacity_mw": capacity,
                "generation_mwh": round(generation, 2)
            }
            
            # 3. DB 저장 (덮어쓰기)
            for model, data in [(models.WeatherTs, weather), (models.GenerationTs, gen)]:
                stmt = insert(model).values(data)
                stmt = stmt.on_conflict_do_update(index_elements=['ts', 'region_id'], set_=data)
                await db.execute(stmt)
            
            print(f"   - {current.strftime('%H:%M')} (KST {kst_hour}시): {gen['generation_mwh']:.2f} MWh")
            current += timedelta(hours=1)
            
        await db.commit()
        print("✅ 24시간 데이터 주입 완료!")

if __name__ == "__main__":
    asyncio.run(seed_past_24h())