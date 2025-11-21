# solarcastai/backend/.../celery_app.py
from celery import Celery
from celery.schedules import crontab

# 1. Celery 앱 생성 (Redis 주소는 환경에 맞게 수정하세요. 로컬이면 localhost, 도커면 redis 컨테이너명)
# 예: broker='redis://localhost:6379/0'
celery_app = Celery(
    "solarcast_worker",
    broker="redis://localhost:6379/0",  
    backend="redis://localhost:6379/0",
    include=['tasks']  
)

# 2. 기본 설정
celery_app.conf.update(
    timezone='Asia/Seoul',
    enable_utc=False,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
)

# 3. 주기적 작업(Beat) 설정
celery_app.conf.beat_schedule = {
    'predict-every-hour': {
        'task': 'tasks.run_hourly_prediction',  # tasks.py의 함수 이름
        'schedule': crontab(minute=0),          # 매시 정각(0분)마다 실행
        #'schedule': 60.0,                     # 테스트용: 60초마다 실행하려면 주석 해제 -> 테스트 완
        'args': (1,)                            # region_id=1 (제주) 전달
    },

    'retrain-daily': {
        'task': 'tasks.run_daily_retraining',
        'schedule': crontab(hour=0, minute=0), # 매일 밤 12시 0분
        # 테스트 할 때는 'schedule': 120.0 (2분마다) 등으로 바꿔서 확인 가능
        # 'schedule': 60.0,  -> 테스트완        
        'args': (1,)
    }
}