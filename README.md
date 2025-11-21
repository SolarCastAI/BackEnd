# BackEnd

프론트엔드와 연결하기위한 간단한 FastApi 프로젝트

## 실행방법(프론트포함)

`pip3 install -r requirements.txt`나 `pip install -r requriements.txt` 이후 `python main.py`나 `python3 main.py`으로 실행후 FrontEnd 레포에서 `npm start`를 하면 실시간 연동이 됩니다.

---

## MLOps 자동화 시스템 실행 방법 (Celery)

이 프로젝트는 **Celery**를 사용하여 1시간 단위 데이터 수집/예측 및 일일 재학습을 자동화합니다.
서버(`main.py`)와 별도로 **Worker**와 **Beat**를 실행해야 합니다.

### 1. 필수 준비 사항
* Redis 서버가 실행 중이어야 합니다. (Docker 컨테이너 등)
* `.env` 파일 혹은 `celery_app.py`에 Redis URL이 올바르게 설정되어 있어야 합니다.

### 2. 실행 명령어

터미널을 2개 열고 각각 아래 명령어를 실행하세요.

 Terminal A: Celery Worker 
실제 AI 예측 및 데이터 처리를 수행합니다.
```bash
# Mac / Linux
celery -A celery_app worker --loglevel=info

# Windows
celery -A celery_app worker --loglevel=info -P solo

Terminal B: Celery Beat (스케줄러)

celery -A celery_app beat --loglevel=info