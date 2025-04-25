import matplotlib.pyplot as plt

def plot_elbow_method(wcss):
    """Plota gráfico do método do cotovelo"""
    plt.figure(figsize=(10,6))
    plt.plot(range(2, len(wcss)+2), wcss, marker='o')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('WCSS')
    plt.savefig('results/cluster_analysis/elbow_plot.png')

def plot_cluster_analysis(df):
    """Gera visualizações da distribuição por cluster"""
    for col in ['time_in_hospital', 'num_lab_procedures']:
        df.boxplot(column=col, by='cluster')
        plt.savefig(f'results/cluster_analysis/{col}_distribution.png')