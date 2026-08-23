# Análisis de Componentes Principales (PCA)


"""
Explicación de los pasos:
Importación del conjunto de datos:
Cargamos los datos desde un archivo CSV (Wine.csv). Este conjunto de datos contiene características de diferentes tipos de vino, y la última columna es la etiqueta o clase (tipo de vino).
División de los datos:
Se dividen los datos en un conjunto de entrenamiento y un conjunto de prueba con un 80% de los datos para entrenamiento y un 20% para la prueba.
Escalado de características:
Se utiliza StandardScaler para escalar las características a un rango común, lo que es importante para los métodos de aprendizaje automático.
Aplicación de PCA (Análisis de Componentes Principales):
PCA se utiliza para reducir la dimensionalidad de los datos. En este caso, se reduce a 2 componentes principales para facilitar la visualización.
Entrenamiento del modelo de Regresión Logística:
Se entrena un modelo de Regresión Logística utilizando los datos transformados con PCA (es decir, los 2 componentes principales).
Matriz de Confusión y Precisión:
Se genera una matriz de confusión para evaluar las predicciones del modelo y se calcula la precisión del modelo en el conjunto de prueba.
Visualización de los resultados:
Se visualizan los resultados del conjunto de entrenamiento y del conjunto de prueba en un gráfico 2D donde se muestran los puntos de datos y los límites de decisión generados por el modelo de regresión logística.
Se usa el gráfico para ver cómo el modelo clasifica los datos según las dos primeras componentes principales (PC1 y PC2).
Visualización de resultados:
En los gráficos generados:
Rojo: Clase 1 (puede ser el tipo de vino 1).
Verde: Clase 2 (puede ser el tipo de vino 2).
Azul: Clase 3 (puede ser el tipo de vino 3).
Los gráficos muestran cómo el modelo ha clasificado los datos del conjunto de entrenamiento y de prueba, y cómo las componentes principales se utilizan para la clasificación.
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

# Aplicar PCA (Análisis de Componentes Principales)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)  # Reducir las características a 2 componentes principales
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Entrenamiento del modelo de Regresión Logística sobre el conjunto de entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Crear la matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)  # Mostrar la matriz de confusión
print("Accuracy: ", accuracy_score(y_test, y_pred))  # Precisión del modelo

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
