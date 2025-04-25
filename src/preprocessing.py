import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath):
    """Carrega e pré-processa os dados brutos"""
    df = pd.read_csv(filepath)
    
    # Remoção de colunas
    df = df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 
                 'medical_specialty', 'diag_1', 'diag_2', 'diag_3'], axis=1)
    
    # Tratamento de valores ausentes
    df.replace('?', np.nan, inplace=True)
    
    # Preenchimento de valores ausentes
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Para numéricos: preencher com a mediana
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Para categóricos: preencher com 'Unknown'
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    # Codificação de categorias
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Normalização
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Verificação final de NaN
    if np.isnan(scaled_data).any():
        scaled_data = np.nan_to_num(scaled_data)
    
    return scaled_data, df, scaler