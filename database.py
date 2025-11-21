import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
from sqlalchemy.pool import NullPool

# .env 파일에서 환경 변수 로드
load_dotenv()

# DB 접속 URL (.env 파일에서 가져옴)
# 예: "postgresql+asyncpg://user:password@localhost:5432/solarcast_db"
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경 변수를 찾을 수 없습니다. .env 파일을 확인하세요.")

# 비동기 엔진 생성
engine = create_async_engine(
    DATABASE_URL, 
    echo=True, 
    poolclass=NullPool 
)

# 비동기 세션 메이커
# expire_on_commit=False 는 FastAPI에서 세션 객체를 계속 사용하기 위해 필요
async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# ORM 모델의 기본이 될 Base 클래스
Base = declarative_base()