import os
from celery import Celery
from celery.schedules import crontab

celery_app = Celery(
"solarcast_worker",
broker="redis://localhost:6379/0",
backend="redis://localhost:6379/0",
include=['tasks']
)

# 3. 기본 설정 업데이트
celery_app.conf.update(
    timezone='Asia/Seoul',
    enable_utc=False,          # 한국 시간 기준 사용
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    broker_connection_retry_on_startup=True, # 최신 버전 경고 방지
)

# 4. 주기적 작업(Beat) 스케줄 설정
# ★ 중요: 개별 지역(run_...)이 아니라 '마스터 태스크(schedule_...)'를 호출해야 합니다.
celery_app.conf.beat_schedule = {
    
    # (1) 시간별 예측 마스터 태스크 (매시 정각 실행)
    'schedule-hourly-prediction-all-regions': {
        'task': 'tasks.schedule_hourly_prediction_all_regions',  # tasks.py의 마스터 함수
        'schedule': crontab(minute=0),          # 매시 0분마다
        #'schedule': 60.0,                     # (테스트용) 60초마다
        'args': ()                              # 인자 없음 (DB에서 ID 조회함)
    },

    # (2) 일일 재학습 마스터 태스크 (매일 자정 실행)
    'schedule-daily-retraining-all_regions': {
        'task': 'tasks.schedule_daily_retraining_all_regions',   # tasks.py의 마스터 함수
        'schedule': crontab(hour=0, minute=0),  # 매일 00:00
        # 'schedule': 120.0,                    # (테스트용) 2분마다
        'args': ()                              # 인자 없음
    }
}

if __name__ == '__main__':
    celery_app.start()