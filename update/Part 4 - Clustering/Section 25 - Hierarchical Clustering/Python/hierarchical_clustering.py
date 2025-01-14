# Clustering Jerárquico (Hierarchical Clustering)

#Explicación:
#Dendrograma: Utilizamos el dendrograma para visualizar cómo se agrupan los clientes a medida que vamos fusionando los clusters. El corte en el dendrograma indica el número de clusters óptimos.
#Modelo de Clustering Jerárquico: Usamos el método de Clustering Jerárquico Agglomerativo, que es un tipo de clustering basado en la aglomeración de puntos en clusters según la distancia mínima entre ellos. El parámetro n_clusters se ajusta a 5, basado en el análisis del dendrograma.
#Visualización de clusters: Los diferentes clusters se visualizan en un gráfico de dispersión con colores diferentes para cada cluster. El gráfico permite observar cómo se agrupan los clientes según su ingreso anual y puntuación de gasto.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Cargando el dataset
dataset = pd.read_csv('Mall_Customers.csv')  # Leer el archivo CSV con los datos
X = dataset.iloc[:, [3, 4]].values  # Seleccionar las columnas de Ingreso Anual y Puntuación de Gasto

# Usando el dendrograma para encontrar el número óptimo de clusters
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))  # Usamos el método 'ward' para calcular las distancias
plt.title('Dendrograma')  # Título del gráfico
plt.xlabel('Clientes')  # Etiqueta del eje X
plt.ylabel('Distancias Euclidianas')  # Etiqueta del eje Y
plt.show()  # Mostrar el dendrograma

# Entrenando el modelo de Clustering Jerárquico en el dataset
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')  # Usamos 'ward' para la fusión de clusters
y_hc = hc.fit_predict(X)  # Ajustamos el modelo y predecimos el cluster al que pertenece cada punto

# Visualizando los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')  # Clientes del Cluster 1
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')  # Clientes del Cluster 2
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')  # Clientes del Cluster 3
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')  # Clientes del Cluster 4
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')  # Clientes del Cluster 5

# Visualizamos el gráfico final con los centroides de los clusters
plt.title('Clusters de Clientes')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación de Gasto (1-100)')
plt.legend()  # Añadimos la leyenda para identificar los clusters
plt.show()  # Mostrar el gráfico
