# 📊 Sistema de Clusterização de Dados Hospitalares

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.2.3-brightgreen.svg)

## 📝 Descrição

Projeto de análise de dados de pacientes diabéticos utilizando clusterização com K-Means para identificar padrões em internações hospitalares.

## 🚀 Funcionalidades

- **Pré-processamento** automático de dados
- Determinação do número **ótimo de clusters**
- Visualização interativa dos resultados
- Módulo de **inferência** para novos casos

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/sistemas_inteligentes_trabalho.git
cd sistemas_inteligentes_trabalho
Instale as dependências:

bash
pip install -r requirements.txt
🏗️ Estrutura do Projeto
.
├── data/
│   ├── raw/               # Dados brutos (CSV)
│   └── processed/         # Dados processados
├── notebooks/             # Análises exploratórias
├── src/
│   ├── preprocessing.py   # Pipeline de limpeza
│   ├── clustering.py      # Algoritmo K-Means
│   ├── visualization.py   # Gráficos e plots
│   ├── inference.py       # Classificação
│   └── main.py            # Execução principal
└── results/               # Outputs e visualizações
💻 Uso
Execute o fluxo completo:

python src/main.py
Ou utilize partes específicas:

from src.preprocessing import load_and_preprocess_data
from src.clustering import train_kmeans_model

# Carregar e processar dados
data, df, scaler = load_and_preprocess_data('data/raw/dados.csv')

# Treinar modelo
model, clustered_data = train_kmeans_model(data, n_clusters=4)
📊 Exemplo de Saída
plaintext
[SYSTEM] Pipeline de análise iniciado...
✅ Pré-processamento concluído | 0 valores NaN
🔍 Identificados 4 clusters ótimos
📈 Visualizações salvas em /results/
🎯 Novo caso classificado no Cluster 2
