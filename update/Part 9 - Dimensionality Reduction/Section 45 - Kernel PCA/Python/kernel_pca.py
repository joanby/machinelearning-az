# Kernel PCA

"""
Explicación de los pasos:
Importación del conjunto de datos:
Cargamos el archivo CSV (Wine.csv), que contiene las características de los vinos y las etiquetas correspondientes.
División de los datos:
Se divide el conjunto de datos en entrenamiento y prueba, con un 80% para entrenamiento y un 20% para prueba.
Escalado de características:
Utilizamos StandardScaler para escalar las características y asegurar que todas tengan la misma escala.
Aplicación de Kernel PCA:
Kernel PCA se utiliza para reducir la dimensionalidad de los datos de forma no lineal. Usamos el núcleo radial (RBF), que es adecuado para datos no lineales.
Se reducen las características a 2 componentes principales.
Entrenamiento del modelo de Regresión Logística:
Se entrena un modelo de Regresión Logística utilizando las características transformadas por Kernel PCA.
Matriz de Confusión y Precisión:
Se genera la matriz de confusión para evaluar las predicciones del modelo y se calcula la precisión para el conjunto de prueba.
Visualización de los resultados:
Se muestran dos gráficos:
Uno para el conjunto de entrenamiento.
Otro para el conjunto de prueba.
Los puntos en los gráficos están coloreados según la clase predicha.
Los límites de decisión del modelo se muestran como contornos en el gráfico.
Visualización de los resultados:
Rojo: Clase 1 (puede ser el tipo de vino 1).
Verde: Clase 2 (puede ser el tipo de vino 2).
Azul: Clase 3 (puede ser el tipo de vino 3).
Los gráficos muestran cómo el modelo de Regresión Logística clasifica los datos en el espacio reducido por Kernel PCA.

Comentarios:
Kernel PCA es útil cuando las relaciones entre las características son no lineales, lo que lo hace más flexible que el PCA tradicional.
Los gráficos y la matriz de confusión permiten evaluar visualmente la eficacia del modelo.
"""

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importación del conjunto de datos
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values  # Características
y = dataset.iloc[:, -1].values   # Etiquetas

# Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de características
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Aplicar Kernel PCA (Análisis de Componentes Principales no lineal)
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')  # Usando el núcleo 'rbf' (Radial Basis Function)
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Entrenamiento del modelo de Regresión Logística sobre el conjunto de entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Crear la matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)  # Mostrar la matriz de confusión
print("Precisión: ", accuracy_score(y_test, y_pred))  # Precisión del modelo

# Visualización de los resultados del conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Regresión Logística (Conjunto de Entrenamiento)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualización de los resultados del conjunto de prueba
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Regresión Logística (Conjunto de Prueba)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
