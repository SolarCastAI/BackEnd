# 터미널에서 python3 실행 후:
import asyncio
from database import async_session
from crud import calculate_daily_accuracy

async def test_eval():
    async with async_session() as db:
        # 강제로 채점 실행
        score = await calculate_daily_accuracy(db, 1)
        print(f"채점 결과: {score}%")

asyncio.run(test_eval())