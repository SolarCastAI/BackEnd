import os
import glob
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
import models

# [접속 정보] 로컬 DB 충돌 방지용 127.0.0.1 사용
SYNC_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/solar_db"

DATA_DIR = "data"
REGION_MAP = { "제주": 1, "서울": 2, "부산": 3, "대구": 4, "대전": 6, "광주": 7 }
LOAD_DAYS = 30  # 30일치 데이터 로드

def load_real_data_aligned():
    print(f"🔌 DB 접속: {SYNC_DATABASE_URL}")
    engine = create_engine(SYNC_DATABASE_URL)

    # [시간 기준] 데이터의 끝을 '오늘 현재 시간'에 맞춤
    # 이렇게 하면 그래프가 끊기지 않고 현재 시점까지 꽉 참
    kst_tz = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst_tz)
    
    # 분/초는 0으로 맞춤 (깔끔하게)
    end_point_kst = now_kst.replace(minute=0, second=0, microsecond=0)
    start_point_kst = end_point_kst - timedelta(days=LOAD_DAYS)
    
    print(f"🎯 데이터 기간(KST): {start_point_kst} ~ {end_point_kst}")

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("❌ CSV 파일 없음")
        return

    with engine.begin() as conn:
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            target_rid = None
            for key, rid in REGION_MAP.items():
                if key in filename: target_rid = rid; break
            if target_rid is None: continue

            print(f"📦 적재 중: {filename} -> Region {target_rid}")

            try: df = pd.read_csv(filepath, encoding='euc-kr')
            except: df = pd.read_csv(filepath, encoding='utf-8')

            # 컬럼 정리
            col_map = {
                "일시": "ts_str", "발전일자": "ts_str", "기온": "temp_c", "기온(°C)": "temp_c",
                "강수량(mm)": "precip_mm", "강우량(mm)": "precip_mm", "습도": "humidity", "습도(%)": "humidity",
                "적설(cm)": "snow_cm", "적설량(mm)": "snow_mm", "전운량(10분위)": "cloud_10", "적운량(10분위)": "cloud_10",
                "일조(hr)": "sunshine_hr", "일사량": "solar_irr", "일사(MJ/m2)": "solar_irr",
                "태양광 설비용량(MW)": "capacity_mw", "설비용량(MW)": "capacity_mw",
                "태양광 발전량(MWh)": "generation_mwh", "발전량(MWh)": "generation_mwh"
            }
            df = df.rename(columns=col_map)
            df['ts'] = pd.to_datetime(df['ts_str'])
            
            # 00시 데이터 찾기 (시작점)
            df['hour'] = df['ts'].dt.hour
            midnight_indices = df.index[df['hour'] == 0].tolist()
            if not midnight_indices: continue
            
            target_len = LOAD_DAYS * 24
            start_idx = midnight_indices[-1]
            for idx in reversed(midnight_indices):
                if idx + target_len <= len(df): start_idx = idx; break
            
            df_target = df.iloc[start_idx : start_idx + target_len].copy().reset_index(drop=True)
            
            # 결측치 0 처리
            for c in ["temp_c", "precip_mm", "humidity", "snow_cm", "cloud_10", "sunshine_hr", "solar_irr", "capacity_mw", "generation_mwh"]:
                if c not in df_target.columns: df_target[c] = 0
            df_target = df_target.fillna(0)
            if "snow_mm" in df_target.columns and "snow_cm" not in df_target.columns:
                df_target["snow_cm"] = df_target.get("snow_mm", 0) / 10.0

            # Insert Buffers
            weather_buffer, gen_buffer = [], []

            for i, row in df_target.iterrows():
                # [시간 매핑] 시작점부터 1시간씩 증가
                current_kst = start_point_kst + timedelta(hours=i)
                # UTC로 변환 (-9H) 및 timezone 정보 부착
                current_utc = (current_kst - timedelta(hours=9)).replace(tzinfo=timezone.utc)
                
                # 미래 데이터(현재 시간보다 뒤)는 넣지 않음 (그래프 현실성 위해)
                # 하지만 넉넉하게 채우기 위해 그냥 다 넣습니다. (API에서 필터링함)
                
                common = {"ts": current_utc, "region_id": target_rid}
                w_row = {**common, "temp_c": row["temp_c"], "precip_mm": row["precip_mm"], "humidity": row["humidity"], "snow_cm": row["snow_cm"], "cloud_10": row["cloud_10"], "sunshine_hr": row["sunshine_hr"], "solar_irr": row["solar_irr"]}
                g_row = {**common, "capacity_mw": row["capacity_mw"], "generation_mwh": row["generation_mwh"]}
                
                weather_buffer.append(w_row)
                gen_buffer.append(g_row)

            if weather_buffer:
                conn.execute(insert(models.WeatherTs), weather_buffer)
                conn.execute(insert(models.GenerationTs), gen_buffer)

    print("\n✅ [성공] 30일치 실제 데이터 적재 완료!")

if __name__ == "__main__":
    load_real_data_aligned()