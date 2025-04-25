from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def predict_cluster(new_data, model, scaler, reference_df):
    """Prediz o cluster para novos dados"""
    # Criar DataFrame com a estrutura correta
    processed = pd.DataFrame(np.zeros((1, len(reference_df.columns))), 
                           columns=reference_df.columns)
    
    # Preencher com os novos dados
    for col in new_data.columns:
        if col in processed.columns:
            processed[col] = new_data[col].values[0]
    
    # Codificar variáveis categóricas
    cat_cols = processed.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if processed[col].notna().all():  # Só codificar se não for NaN
            processed[col] = LabelEncoder().fit_transform(processed[col].astype(str))
        else:
            processed[col] = 0  # Preencher com valor padrão se for NaN
    
    # Remover coluna 'cluster' se existir
    if 'cluster' in processed.columns:
        processed = processed.drop(columns=['cluster'])
    
    # Substituir quaisquer NaN restantes por 0
    processed = processed.fillna(0)
    
    # Verificar NaN antes de escalar
    if processed.isna().any().any():
        raise ValueError("Dados contêm NaN após pré-processamento")
    
    # Escalar os dados
    scaled_data = scaler.transform(processed)
    
    # Verificar NaN após escalar
    if np.isnan(scaled_data).any():
        scaled_data = np.nan_to_num(scaled_data)
    
    # Prever o cluster
    return model.predict(scaled_data)[0]