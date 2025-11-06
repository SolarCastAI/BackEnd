# BackEnd

프론트엔드와 연결하기위한 간단한 FastApi 프로젝트

## 실행방법(프론트포함)

`pip3 install -r requirements.txt`나 `pip install -r requriements.txt` 이후 `python main.py`나 `python3 main.py`으로 실행후 FrontEnd 레포에서 `npm start`를 하면 실시간 연동이 됩니다.


# 📖 코드(main.py) 상세 설명

## 1️⃣ 임포트 및 기본 설정 (1-21줄)
```python
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd  # CSV 처리용
```

- **FastAPI**: 웹 API 프레임워크
- **CORSMiddleware**: React와 통신하기 위한 CORS 설정
- **Pydantic BaseModel**: 데이터 검증 및 타입 안정성
- **pandas**: CSV 파일 읽기/쓰기

---

## 2️⃣ 데이터 모델 (25-44줄)
```python
class DashboardSummary(BaseModel):
    current_power: float  # 현재 발전량
    today_total: float    # 오늘 누적량
    accuracy: float       # 예측 정확도
    today_date: str       # 날짜
```

→ API 응답 데이터의 구조를 정의

---

## 3️⃣ CSV 데이터 디렉토리 (48-50줄)
```python
CSV_DATA_DIR = "csv_data"
os.makedirs(CSV_DATA_DIR, exist_ok=True)
```

→ CSV 파일을 저장할 폴더 자동 생성

---

## 4️⃣ 임시 데이터 (52-93줄)
```python
regions_data = [
    {"region": "서울", "power": 4850, ...},
    # 하드코딩된 테스트 데이터
]
```

→ 실제로는 CSV나 DB에서 가져와야 함

---

## 5️⃣ API 엔드포인트

### 📌 대시보드 요약 (110-118줄)
```python
@app.get("/api/dashboard/summary")
def get_dashboard_summary():
    return DashboardSummary(...)
```

### 📌 지역별 데이터 (120-148줄)
```python
@app.get("/api/regions")
def get_regions_data():
    # 모든 지역 데이터 반환
```

### 📌 예측 데이터 (150-161줄)
```python
@app.get("/api/forecast/{hours}")
def get_power_forecast(hours: int):
    # 24/48/72시간 예측 데이터
```

---

## 6️⃣ CSV 파일 업로드 (189-247줄)
```python
@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    # CSV 파일 업로드 및 저장
```

---

## 7️⃣ CSV 데이터 조회 (249-311줄)
```python
@app.get("/api/chart/data/{filename}")
def get_chart_data(filename: str):
    # 특정 CSV 파일 데이터 조회
```


