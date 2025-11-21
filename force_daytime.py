import asyncio
from datetime import datetime, date
from database import async_session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text
import models

async def force_today_data():
    async with async_session() as db:
        print("☀️ [Test] 오늘 날짜(낮 시간) 데이터 강제 주입 중...")
        
        # 1. 오늘 낮 1시 데이터 생성 (발전량 150 MWh 가정)
        now = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        region_id = 1
        
        # 날씨 & 발전량 넣기
        weather = {
            "ts": now, "region_id": region_id, "temp_c": 20.0, "precip_mm": 0, 
            "humidity": 50, "snow_cm": 0, "cloud_10": 0, "sunshine_hr": 1, "solar_irr": 0.8
        }
        gen = {
            "ts": now, "region_id": region_id, 
            "capacity_mw": 100.0, "generation_mwh": 150.55 # 발전량 150.55
        }
        
        for model, data in [(models.WeatherTs, weather), (models.GenerationTs, gen)]:
            stmt = insert(model).values(data)
            stmt = stmt.on_conflict_do_update(index_elements=['ts', 'region_id'], set_=data)
            await db.execute(stmt)
            
        print(f"   ✅ 발전량 데이터 주입 완료: {gen['generation_mwh']} MWh")

        # 2. 정확도 점수 강제 주입 (95.5점)
        print("📝 [Test] 정확도 점수 강제 주입 중...")
        eval_data = {
            "date": date.today(), # 오늘 날짜 점수
            "region_id": region_id,
            "model": "XGBoost-Stack", "ver": "v1.0",
            "mae": 5.0, "rmse": 7.0, 
            "mape": 4.5, # 오차율 4.5% -> 정확도 95.5%
            "samples": 24
        }
        stmt_eval = insert(models.EvalDaily).values(eval_data)
        stmt_eval = stmt_eval.on_conflict_do_update(
            index_elements=['date', 'region_id', 'model', 'ver'], set_=eval_data
        )
        await db.execute(stmt_eval)
        print("   ✅ 정확도 점수 주입 완료 (95.5%)")
        
        await db.commit()

if __name__ == "__main__":
    asyncio.run(force_today_data())