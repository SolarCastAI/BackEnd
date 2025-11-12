import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, Float, Text, TIMESTAMP, Date, ForeignKey
from sqlalchemy.orm import relationship
from database import Base  # 2단계에서 만든 Base 클래스 import

class Region(Base):
    """
    'regions' 테이블: 지역 마스터 정보
    (alembic: f5bf5baeb347...py)
    """
    __tablename__ = 'regions'
    
    region_id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    tz = Column(Text, nullable=False, server_default='Asia/Seoul')

    # 다른 테이블과의 관계 설정 (ORM에서 region.weather_ts 등으로 접근 가능)
    weather_ts = relationship("WeatherTs", back_populates="region")
    generation_ts = relationship("GenerationTs", back_populates="region")
    forecast_ts = relationship("ForecastTs", back_populates="region")
    eval_daily = relationship("EvalDaily", back_populates="region")

class WeatherTs(Base):
    """
    'weather_ts' 하이퍼테이블: 시계열 기상 데이터
    (alembic: f5bf5baeb347...py)
    """
    __tablename__ = 'weather_ts'
    
    ts = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.region_id'), nullable=False, primary_key=True)
    
    temp_c = Column(Float)
    precip_mm = Column(Float)
    humidity = Column(Float)
    snow_cm = Column(Float)
    cloud_10 = Column(Float)
    sunshine_hr = Column(Float)
    solar_irr = Column(Float)
    
    # Alembic 스크립트의 유니크 제약조건을 기반으로 복합 기본 키 설정
    __table_args__ = (
        sa.UniqueConstraint('region_id', 'ts', name='uq_weather_ts_region_ts'),
    )

    # Region 테이블과의 관계
    region = relationship("Region", back_populates="weather_ts")

class GenerationTs(Base):
    """
    'generation_ts' 하이퍼테이블: 시계열 발전량 데이터
    (alembic: f5bf5baeb347...py)
    """
    __tablename__ = 'generation_ts'
    
    ts = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.region_id'), nullable=False, primary_key=True)
    
    capacity_mw = Column(Float)
    generation_mwh = Column(Float)
    
    # Alembic 스크립트의 유니크 제약조건을 기반으로 복합 기본 키 설정
    __table_args__ = (
        sa.UniqueConstraint('region_id', 'ts', name='uq_generation_ts_region_ts'),
    )
    
    # Region 테이블과의 관계
    region = relationship("Region", back_populates="generation_ts")

class ForecastTs(Base):
    """
    'forecast_ts' 하이퍼테이블: 시계열 예측 결과 데이터
    (alembic: f5bf5baeb347...py)
    """
    __tablename__ = 'forecast_ts'
    
    # 예측 결과를 특정하는 복합 기본 키
    ts = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)    # 예측 시각
    region_id = Column(Integer, ForeignKey('regions.region_id'), nullable=False, primary_key=True)
    horizon = Column(Integer, nullable=False, primary_key=True)                # 예측 시점 (+N 시간)
    model = Column(String(32), nullable=False, primary_key=True)
    ver = Column(String(16), nullable=False, primary_key=True)
    
    # 예측 값 및 생성 시각
    gen_pred_kwh = Column(Float, nullable=False)
    generated_at = Column(TIMESTAMP(timezone=True), nullable=False)
    
    # Region 테이블과의 관계
    region = relationship("Region", back_populates="forecast_ts")

class EvalDaily(Base):
    """
    'eval_daily' 테이블: 일일 단위 모델 평가 지표
    (alembic: f5bf5baeb347...py)
    """
    __tablename__ = 'eval_daily'
    
    # Alembic 스크립트에 정의된 복합 기본 키
    date = Column(Date, nullable=False, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.region_id'), nullable=False, primary_key=True)
    model = Column(String(32), nullable=False, primary_key=True)
    ver = Column(String(16), nullable=False, primary_key=True)
    
    # 평가 지표
    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    samples = Column(Integer)
    
    __table_args__ = (
        sa.PrimaryKeyConstraint('date', 'region_id', 'model', 'ver'),
    )
    
    # Region 테이블과의 관계
    region = relationship("Region", back_populates="eval_daily")

class Job(Base):
    """
    'jobs' 테이블: 데이터 수집/예측 작업 로그
    (alembic: f5bf5baeb347...py)
    """
    __tablename__ = 'jobs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task = Column(String(64), nullable=False)
    started_at = Column(TIMESTAMP(timezone=True), nullable=False)
    ended_at = Column(TIMESTAMP(timezone=True)) # 작업이 끝나면 채워짐
    status = Column(String(16), nullable=False) # 'success' or 'failed'
    meta_json = Column(Text) # JSON 문자열로 메타데이터 저장