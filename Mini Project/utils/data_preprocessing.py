import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_dummy_data(n_samples=500):
    np.random.seed(42)
    ipk_sem1 = np.random.uniform(2.0, 4.0, n_samples)
    kehadiran = np.random.uniform(60, 100, n_samples)
    jam_belajar = np.random.uniform(1, 8, n_samples)
    
    skor = (ipk_sem1 * 10) + (kehadiran * 0.5) + (jam_belajar * 2)
    status_lulus = np.where(skor > 75, 1, 0)
    
    df = pd.DataFrame({
        'ipk_sem1': np.round(ipk_sem1, 2),
        'kehadiran': np.round(kehadiran, 1),
        'jam_belajar_harian': np.round(jam_belajar, 1),
        'status_lulus': status_lulus
    })
    return df

def preprocess_features(df, is_training=True, scaler=None):
    features = ['ipk_sem1', 'kehadiran', 'jam_belajar_harian']
    X = df[features]
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X)
        return X_scaled
