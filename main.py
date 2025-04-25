import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).parent))

from src.preprocessing import load_and_preprocess_data
from src.clustering import find_optimal_clusters, train_kmeans_model
from src.visualization import plot_cluster_analysis
from src.inference import predict_cluster

def check_nan(data):
    """Verifica e reporta valores NaN"""
    nan_count = np.isnan(data).sum()
    total = data.size
    print(f"Valores NaN encontrados: {nan_count}/{total} ({nan_count/total:.2%})")
    if nan_count > 0:
        print("Atenção: Existem valores NaN nos dados!")
    return data

def main():
    try:
        # 1. Pré-processamento
        print("1/4 - Carregando e pré-processando dados...")
        scaled_data, processed_df, scaler = load_and_preprocess_data('data/raw/diabetic_data.csv')
        
        # Verificação de NaN
        print("\n2/4 - Verificando qualidade dos dados...")
        check_nan(scaled_data)
        
        # 2. Clusterização
        print("\n3/4 - Executando clusterização...")
        optimal_k = 4  # Definido com base na análise prévia
        
        # Verificação e tratamento final de NaN
        if np.isnan(scaled_data).any():
            print("Substituindo valores NaN restantes por 0...")
            scaled_data = np.nan_to_num(scaled_data)
        
        # Chamada corrigida da função de clusterização
        kmeans_model, clustered_df = train_kmeans_model(
            data=scaled_data,
            n_clusters=optimal_k,
            original_df=processed_df
        )
        
        print(f"Clusterização concluída com {optimal_k} clusters.")
        
        # 3. Análise
        print("\n4/4 - Gerando visualizações...")
        plot_cluster_analysis(clustered_df)
        
         # 4. Exemplo de inferência
        print("\nTestando módulo de inferência...")
        sample_data = pd.DataFrame([{
            'race': 'Caucasian',
            'gender': 'Female',
            'age': '[50-60)',
            'admission_type_id': 1,
            'discharge_disposition_id': 1,
            'admission_source_id': 7,
            'time_in_hospital': 4,
            'num_lab_procedures': 45,
            'num_procedures': 2,
            'num_medications': 15,
            'number_outpatient': 0,
            'number_emergency': 0,
            'number_inpatient': 0,
            'number_diagnoses': 7,
            'max_glu_serum': 'None',
            'A1Cresult': 'None',
            'metformin': 'No',
            'change': 'No',
            'diabetesMed': 'Yes',
            'readmitted': 'NO'
        }])
        
        # tratamento
        try:
            cluster = predict_cluster(sample_data, kmeans_model, scaler, processed_df)
            print(f"\nResultado da inferência:")
            print(f"Cluster predito: {cluster}")
        except Exception as e:
            print(f"\nFalha na inferência: {str(e)}")
            print("Verifique os dados de entrada e o modelo")
            raise
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    main()