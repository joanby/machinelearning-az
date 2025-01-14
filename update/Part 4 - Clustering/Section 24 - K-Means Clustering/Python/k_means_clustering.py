# Clustering con K-Means

#Explicación:
#Selección de datos: Se seleccionan las columnas correspondientes al ingreso anual y la puntuación de gasto de los clientes del dataset.
#Método del Codo: Se utiliza el método del codo para determinar el número óptimo de clusters. Este método muestra cómo la suma de cuadrados dentro de los clusters (WCSS) disminuye a medida que se aumentan los clusters. El "codo" en el gráfico indica el número de clusters ideal.
#Entrenamiento del modelo K-Means: Se ajusta el modelo K-Means con el número óptimo de clusters (en este caso, 5) y se predicen los clusters para cada cliente.
#Visualización: Se visualizan los diferentes clusters en un gráfico de dispersión, con cada cluster marcado con un color diferente. Además, se marcan los centroides de los clusters con un color amarillo para ver cómo K-Means ha dividido los datos.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Cargando el dataset
dataset = pd.read_csv('Mall_Customers.csv')  # Lee el archivo CSV con los datos
X = dataset.iloc[:, [3, 4]].values  # Seleccionamos las columnas de Ingreso Anual y Puntuación de Gasto

# Usando el método del codo (Elbow Method) para encontrar el número óptimo de clusters
wcss = []  # Lista para almacenar el valor de la suma de cuadrados dentro de los clusters (WCSS)
for i in range(1, 11):  # Probamos con 1 a 10 clusters
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)  # Inicialización con el método 'k-means++' para mejorar la convergencia
    kmeans.fit(X)  # Ajustamos el modelo al conjunto de datos
    wcss.append(kmeans.inertia_)  # Almacenamos la inercia (WCSS) para cada número de clusters

# Graficamos el gráfico del codo
plt.plot(range(1, 11), wcss)
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()  # Muestra el gráfico

# Entrenando el modelo K-Means con 5 clusters (según lo indicado por el gráfico del codo)
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)  # Ajustamos el modelo y predecimos el cluster de cada punto

# Visualización de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')  # Clientes del Cluster 1
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')  # Clientes del Cluster 2
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')  # Clientes del Cluster 3
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')  # Clientes del Cluster 4
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')  # Clientes del Cluster 5

# Visualizamos los centros de los clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroides')  # Los centroides de los clusters
plt.title('Clusters de Clientes')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación de Gasto (1-100)')
plt.legend()  # Añadimos la leyenda para identificar los clusters
plt.show()  # Muestra el gráfico
