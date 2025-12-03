import asyncio
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
import models
from database import async_session

# 우리가 사용하는 6개 지역
TARGET_REGIONS = [
    (1, '제주'), (2, '서울'), (3, '부산'), 
    (4, '대구'), (6, '대전'), (7, '광주')
]

async def reset_db_only():
    async with async_session() as db:
        print("🧹 [1단계] 데이터베이스 초기화 (데이터 삭제)...")
        # 모든 데이터 삭제 (CASCADE)
        await db.execute(text("TRUNCATE TABLE regions, weather_ts, generation_ts, forecast_ts, eval_daily CASCADE;"))
        
        print("🔧 [2단계] 지역(Region) 정보 등록...")
        for rid, name in TARGET_REGIONS:
            stmt = insert(models.Region).values(region_id=rid, name=name, tz='Asia/Seoul')
            await db.execute(stmt)
            
        await db.commit()
        print(f"   ✅ 6개 지역 등록 완료: {[name for _, name in TARGET_REGIONS]}")

if __name__ == "__main__":
    asyncio.run(reset_db_only())