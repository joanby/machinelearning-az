# Grid Search

"""
Explicación de cada bloque:
Importación de bibliotecas:
Se importan las bibliotecas necesarias para manejar los datos, crear el modelo y visualizar los resultados.
Carga y preparación de los datos:
Se carga el conjunto de datos Social_Network_Ads.csv, que contiene información sobre usuarios y si hicieron clic en un anuncio o no (en función de su edad y salario estimado).
Se separan las características (X) y las etiquetas (y).
División de los datos:
Se divide el conjunto de datos en entrenamiento (75%) y prueba (25%).
Escalado de características:
Se aplica StandardScaler para escalar las características, asegurando que todas tengan la misma magnitud.
Entrenamiento del modelo:
Se entrena un SVM con kernel RBF (función de base radial) con los datos escalados de entrenamiento.
Matriz de confusión y precisión:
Se evalúa el rendimiento del modelo utilizando la matriz de confusión y se calcula la precisión del modelo con el conjunto de prueba.
Validación cruzada con k-Fold:
Se realiza una validación cruzada con 10 pliegues para estimar la precisión del modelo de manera más robusta.
Búsqueda de hiperparámetros con Grid Search:
Se aplica Grid Search para encontrar los mejores parámetros para el modelo. Se prueban combinaciones de valores para C y gamma con dos núcleos posibles (linear y rbf).
Se imprimen la mejor precisión y los mejores parámetros encontrados.
Visualización de los resultados:
Se visualizan los resultados para ambos conjuntos (entrenamiento y prueba).
En los gráficos, cada punto es un usuario representado por su edad y salario estimado, y los colores representan las predicciones (rojo o verde).
Resultados esperados:
Matriz de confusión que muestra el número de predicciones correctas e incorrectas del modelo.
Precisión del modelo para el conjunto de prueba.
Precisión media y desviación estándar a través de la validación cruzada.
Mejores parámetros encontrados por Grid Search.
Gráficos de decisión para visualizar cómo el modelo clasifica los datos en el espacio de características.
Este enfoque utiliza Grid Search para la optimización de hiperparámetros, lo que mejora la precisión del modelo al encontrar la mejor combinación de parámetros.
"""

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importación del conjunto de datos
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values  # Características
y = dataset.iloc[:, -1].values   # Etiquetas

# División del conjunto de datos en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Escalado de características
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenamiento del modelo Kernel SVM sobre el conjunto de entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Creación de la matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)  # Mostrar la matriz de confusión
print("Precisión:", accuracy_score(y_test, y_pred))  # Precisión del modelo

# Aplicación de k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Precisión media: {:.2f} %".format(accuracies.mean()*100))
print("Desviación estándar: {:.2f} %".format(accuracies.std()*100))

# Aplicación de Grid Search para encontrar el mejor modelo y los mejores parámetros
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Mejor Precisión: {:.2f} %".format(best_accuracy*100))
print("Mejores Parámetros:", best_parameters)

# Visualización de los resultados sobre el conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()

# Visualización de los resultados sobre el conjunto de prueba
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Conjunto de Prueba)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()
