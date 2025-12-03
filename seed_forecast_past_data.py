import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
import random
from database import async_session
import models

async def seed_forecast_history():
    async with async_session() as db:
        print("📜 [과거 예측 기록] 생성 시작 (최근 5일)...")
        
        regions = (await db.execute(select(models.Region.region_id))).scalars().all()
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(days=5) # 5일 전부터
        
        for rid in regions:
            # 실제 데이터 조회
            query = select(models.GenerationTs).where(
                models.GenerationTs.region_id == rid,
                models.GenerationTs.ts >= start_time
            )
            rows = (await db.execute(query)).scalars().all()
            
            for row in rows:
                # 오차(Noise) 추가하여 예측값 생성
                # 밤(0)이면 예측도 0에 가깝게, 낮이면 오차 적용
                val = row.generation_mwh * 1000 # MWh -> kWh
                if val < 1:
                    pred_val = 0
                else:
                    pred_val = val * random.uniform(0.9, 1.1)
                
                forecast_data = {
                    "ts": row.ts, "region_id": rid, "horizon": 0,
                    "gen_pred_kwh": pred_val, "model": "Historical-Seed", "ver": "v0.0",
                    "generated_at": row.ts - timedelta(hours=1)
                }
                
                stmt = insert(models.ForecastTs).values(forecast_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['ts', 'region_id', 'horizon', 'model', 'ver'],
                    set_=forecast_data
                )
                await db.execute(stmt)
                
        await db.commit()
        print("✅ 과거 예측 데이터 생성 완료!")

if __name__ == "__main__":
    asyncio.run(seed_forecast_history())