import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
import os
import json
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# --- 기본 설정 ---
# GPU/CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
model_dir = "./saved_models"

# ==========================================
# 1. Dataset 클래스 (누락되었던 부분)
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. AI 모델 클래스 정의 (LSTM, GRU)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc(lstm_out)
        return output

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out[:, -1, :])
        output = self.fc(gru_out)
        return output

# ==========================================
# 3. 유틸리티 함수 (모델 로드, 전이학습)
# ==========================================
def load_jeju_pretrained_models(model_dir='./saved_models', timestamp=None):
    """저장된 모델 불러오기"""
    if timestamp is None:
        latest_path = os.path.join(model_dir, 'latest_models.json')
        if not os.path.exists(latest_path):
            # 파일이 없으면 예외 처리 대신 None 반환하거나 빈 껍데기 생성 로직 필요
            # 여기서는 에러를 띄웁니다.
            raise FileNotFoundError(f"최신 모델 정보 파일을 찾을 수 없습니다: {latest_path}")
        with open(latest_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        timestamp = model_info['timestamp']
    
    metadata_path = os.path.join(model_dir, f'metadata_{timestamp}.json')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    lstm_path = os.path.join(model_dir, f'lstm_model_{timestamp}.pth')
    lstm_checkpoint = torch.load(lstm_path, map_location=device, weights_only=False)
    lstm_model = LSTMModel(**lstm_checkpoint['model_config']).to(device)
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    
    gru_path = os.path.join(model_dir, f'gru_model_{timestamp}.pth')
    gru_checkpoint = torch.load(gru_path, map_location=device, weights_only=False)
    gru_model = GRUModel(**gru_checkpoint['model_config']).to(device)
    gru_model.load_state_dict(gru_checkpoint['model_state_dict'])
    
    return lstm_model, gru_model, metadata

def transfer_learning(model, train_loader, val_loader, criterion, 
                     num_epochs=10, patience=3, learning_rate=0.0001, 
                     freeze_layers=False, device='cpu', model_name='Model'):
    """전이학습 (Fine-tuning) 수행"""
    print(f"   >> {model_name} 학습 시작 (LR: {learning_rate})...")
    
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()
    
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_model_state)
    return model, [], []

def predict_future(model, scaler_X, scaler_y, last_sequence, target_datetime, solar_capacity, device='cpu'):
    """단일 시점 미래 예측"""
    model.eval()
    
    # UTC -> KST 변환하여 시간 계산
    from datetime import timezone, timedelta
    kst_tz = timezone(timedelta(hours=9))
    
    # target_datetime이 UTC라면 KST로 변환
    if target_datetime.tzinfo is None:
        target_kst = target_datetime  # timezone 없으면 그대로 사용
    else:
        target_kst = target_datetime.astimezone(kst_tz)
    
    month = target_kst.month
    hour = target_kst.hour  # ← KST 기준 시간!
    
    # 계절별/시간대별 기상 패턴 생성 (간략화)
    if month in [11, 12, 1, 2]: base_temp, base_humid, base_cloud = 5, 60, 5
    elif month in [3, 4, 5]: base_temp, base_humid, base_cloud = 15, 55, 4
    elif month in [6, 7, 8]: base_temp, base_humid, base_cloud = 25, 70, 6
    else: base_temp, base_humid, base_cloud = 15, 65, 5
    
    if 6 <= hour <= 12: temperature = base_temp + (hour - 6) * 1.5
    elif 12 < hour <= 18: temperature = base_temp + 9 - (hour - 12) * 1.0
    else: temperature = base_temp - 3
    
    # KST 기준으로 낮/밤 판단
    if 6 <= hour <= 18:
        sunshine_duration = 0.8 if 9 <= hour <= 15 else 0.3
        solar_radiation = 600 if 9 <= hour <= 15 else 200
    else:
        sunshine_duration = 0
        solar_radiation = 0
    
    # 모델 학습 때 사용한 특성 순서 준수
    new_features = np.array([[
        temperature, 0, base_humid, base_cloud,
        sunshine_duration, solar_radiation, solar_capacity, hour
    ]])
    
    new_features_scaled = scaler_X.transform(new_features)
    new_sequence = np.vstack([last_sequence[1:], new_features_scaled])
    new_sequence_tensor = torch.FloatTensor(new_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_scaled = model(new_sequence_tensor).cpu().numpy()
        prediction = scaler_y.inverse_transform(prediction_scaled)[0, 0]
    
    return max(0, prediction), new_sequence

# ==========================================
# 4. 데이터 로딩 함수 (Inference용 vs Training용)
# ==========================================
def load_data_from_db(df, sequence_length=24):
    """[추론용] DB 데이터를 모델 입력 형태로 변환"""
    df_renamed = df.copy()
    if 'datetime' in df_renamed.columns:
        df_renamed['datetime'] = pd.to_datetime(df_renamed['datetime'])
    elif '발전일자' in df_renamed.columns:
        df_renamed['datetime'] = pd.to_datetime(df_renamed['발전일자'])
    
    cols_to_fill_zero = ['precipitation', 'snow', 'sunshine_duration', 'solar_radiation']
    for col in cols_to_fill_zero:
        if col in df_renamed.columns:
            df_renamed[col] = df_renamed[col].fillna(0)
            
    df_renamed['hour'] = df_renamed['datetime'].dt.hour
    
    feature_cols = [
        'temperature', 'precipitation', 'humidity', 'cloud_cover',
        'sunshine_duration', 'solar_radiation', 'solar_capacity', 'hour'
    ]
    
    target_col = 'solar_generation'
    # 추론 시에는 타겟값이 없어도 되지만, 시퀀스 생성을 위해 dropna를 사용하기도 함
    # 여기서는 generation 데이터가 있는 구간만 유효하다고 가정
    df_valid = df_renamed.dropna(subset=[target_col]).copy()
    
    X = df_valid[feature_cols].values
    y = df_valid[target_col].values.reshape(-1, 1)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_seq = []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
    
    X_seq = np.array(X_seq)
    
    # 5개 값 반환 (추론용)
    return X_seq, scaler_X, scaler_y, feature_cols, df_valid

def load_train_data_from_db(df, sequence_length=24):
    """[재학습용] 데이터 전처리 및 Train/Val/Test 분할 함수"""
    df_renamed = df.copy()
    if 'datetime' in df_renamed.columns:
        df_renamed['datetime'] = pd.to_datetime(df_renamed['datetime'])
    
    cols_to_fill_zero = ['precipitation', 'snow', 'sunshine_duration', 'solar_radiation']
    for col in cols_to_fill_zero:
        if col in df_renamed.columns:
            df_renamed[col] = df_renamed[col].fillna(0)

    df_renamed['hour'] = df_renamed['datetime'].dt.hour
    
    feature_cols = [
        'temperature', 'precipitation', 'humidity', 'cloud_cover',
        'sunshine_duration', 'solar_radiation', 'solar_capacity', 'hour'
    ]
    
    target_col = 'solar_generation'
    df_valid = df_renamed.dropna(subset=[target_col]).copy()
    
    X = df_valid[feature_cols].values
    y = df_valid[target_col].values.reshape(-1, 1)
    dates = df_valid['datetime'].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_seq, y_seq, date_seq = [], [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_scaled[i+sequence_length])
        date_seq.append(dates[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    if len(X_seq) < 10:
        return [], [], [], [], [], [], scaler_X, scaler_y, feature_cols, df_valid, []

    X_temp, X_test, y_temp, y_test, date_temp, date_test = train_test_split(
        X_seq, y_seq, date_seq, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val, date_train, date_val = train_test_split(
        X_temp, y_temp, date_temp, test_size=0.111, random_state=42
    )
    
    # 11개 값 반환 (학습용)
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            scaler_X, scaler_y, feature_cols, df_valid, date_test)

# ==========================================
# 5. Main Entry Points (Celery Task에서 호출)
# ==========================================
def run_prediction(df_input, loaded_models=None):
    """
    [1시간 주기] 예측 실행 함수
    """
    try:
        # 1. 모델 로드
        if loaded_models:
            lstm_model, gru_model, _ = loaded_models
        else:
            lstm_model, gru_model, _ = load_jeju_pretrained_models(model_dir="./saved_models")
        
        SEQUENCE_LENGTH = 24
        
        # 2. 데이터 전처리 (추론용 함수 사용)
        if df_input.empty:
            return []
            
        X_seq, scaler_X, scaler_y, _, df_valid = load_data_from_db(df_input, SEQUENCE_LENGTH)
        
        if len(X_seq) == 0:
            print("시퀀스를 만들 데이터가 부족합니다.")
            return []

        # 3. 미래 예측 (가장 마지막 시점 기준)
        current_time = df_valid['datetime'].iloc[-1]
        solar_capacity = df_valid['solar_capacity'].iloc[0]
        last_sequence = X_seq[-1]
        
        all_predictions = []
        temp_sequence = last_sequence.copy()
        
        # 72시간 예측 수행
        for h in range(1, 73):
            target_time = current_time + timedelta(hours=h)
            
            lstm_pred, temp_sequence = predict_future(
                lstm_model, scaler_X, scaler_y, temp_sequence,
                target_time, solar_capacity, device
            )
            gru_pred, _ = predict_future(
                gru_model, scaler_X, scaler_y, temp_sequence,
                target_time, solar_capacity, device
            )
            
            ensemble_pred = (lstm_pred + gru_pred) / 2
            
            all_predictions.append({
                '예측일시': target_time,
                '앙상블_발전량(MWh)': max(0, ensemble_pred)
            })

        return all_predictions

    except Exception as e:
        print(f"예측 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return []

def retrain_model(df_train):
    """
    [하루 1번] 재학습 실행 함수 (수정됨)
    """
    try:
        print("\n🚀 [Model Retraining] 시작...")
        
        # 1. 모델 로드
        lstm_model, gru_model, metadata = load_jeju_pretrained_models(model_dir=model_dir)
        
        SEQUENCE_LENGTH = 24
        
        # 2. 데이터 전처리
        (X_train, X_val, X_test, y_train, y_val, y_test,
         scaler_X, scaler_y, feature_cols, df_valid, date_test) = load_train_data_from_db(df_train, SEQUENCE_LENGTH)
        
        if len(X_train) == 0:
            print("⚠️ 학습할 데이터가 너무 적습니다.")
            return False

        # 3. DataLoader 생성
        BATCH_SIZE = 32
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # 4. 전이학습 수행
        criterion = nn.MSELoss()
        
        lstm_model, _, _ = transfer_learning(
            lstm_model, train_loader, val_loader, criterion,
            num_epochs=10, patience=3, learning_rate=0.0001,
            freeze_layers=False, device=device, model_name='LSTM'
        )
        
        gru_model, _, _ = transfer_learning(
            gru_model, train_loader, val_loader, criterion,
            num_epochs=10, patience=3, learning_rate=0.0001,
            freeze_layers=False, device=device, model_name='GRU'
        )
        
        # ==========================================
        # 5. 모델 저장 (에러 수정 부분)
        # ==========================================
        new_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 기존 메타데이터에 config가 없으면 기본값 사용 (안전장치)
        default_config = {
            'input_size': 8,     # 특성 8개
            'hidden_size': 128,  # 모델 기본값
            'num_layers': 2,
            'dropout': 0.2
        }
        
        # .get()을 써서 키가 없으면 default_config를 가져오게 함
        lstm_config = metadata.get('lstm_config', default_config)
        gru_config = metadata.get('gru_config', default_config)
        
        # LSTM 저장
        torch.save({
            'model_state_dict': lstm_model.state_dict(),
            'model_config': lstm_config
        }, os.path.join(model_dir, f'lstm_model_{new_timestamp}.pth'))
        
        # GRU 저장
        torch.save({
            'model_state_dict': gru_model.state_dict(),
            'model_config': gru_config
        }, os.path.join(model_dir, f'gru_model_{new_timestamp}.pth'))
        
        # 메타데이터 갱신 (다음번엔 에러 안 나도록 config도 같이 저장)
        metadata['timestamp'] = new_timestamp
        metadata['lstm_config'] = lstm_config
        metadata['gru_config'] = gru_config
        
        with open(os.path.join(model_dir, f'metadata_{new_timestamp}.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        with open(os.path.join(model_dir, 'latest_models.json'), 'w', encoding='utf-8') as f:
            json.dump({'timestamp': new_timestamp}, f, indent=4)
            
        print(f"✅ 재학습 완료! 새로운 모델 버전: {new_timestamp}")
        return True

    except Exception as e:
        print(f"❌ 재학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False