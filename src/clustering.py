from sklearn.cluster import KMeans
import pandas as pd

def find_optimal_clusters(data, max_k=10):
    """Determina o número ótimo de clusters usando método do cotovelo"""
    wcss = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

def train_kmeans_model(data, n_clusters, original_df=None):
    """
    Treina o modelo K-Means e retorna:
    - modelo treinado
    - dados originais com labels de cluster (se fornecido)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(data)
    
    if original_df is not None:
        clustered_df = original_df.copy()
        clustered_df['cluster'] = clusters
        return kmeans, clustered_df
    else:
        return kmeans