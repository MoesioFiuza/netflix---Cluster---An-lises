import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

caminho_arquivo = r'cole aqui o caminho do arquivo csv'
df = pd.read_csv(caminho_arquivo)

df['time'] = df['time'].str.extract('(\d+)').astype(float)

caracteristicas = df[['year', 'time']].dropna()

escalador = StandardScaler()
caracteristicas_normalizadas = escalador.fit_transform(caracteristicas)

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(caracteristicas_normalizadas)

df['cluster'] = kmeans.labels_
plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='time', hue='cluster', data=df, palette='viridis', alpha=0.6, edgecolor=None)
plt.title('Clustering de Títulos por Ano e Duração')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Duração (minutos)')
plt.legend(title='Cluster')
plt.show()

contagem_clusters = df['cluster'].value_counts()
print(contagem_clusters)

estatisticas_clusters = df.groupby('cluster')[['year', 'time']].describe()
print(estatisticas_clusters)
