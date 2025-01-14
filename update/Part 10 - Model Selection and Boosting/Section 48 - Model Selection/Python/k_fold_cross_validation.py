# k-Fold Cross Validation

"""
Explicación Detallada:
Importación de bibliotecas:
Se importan las bibliotecas necesarias, como numpy, matplotlib, pandas y las funciones de sklearn para trabajar con el modelo SVM y la validación cruzada.
Carga del conjunto de datos:
Se carga el conjunto de datos Social_Network_Ads.csv y se separan las características (X) y las etiquetas (y).
División de los datos:
Se divide el conjunto de datos en dos partes: un conjunto de entrenamiento (75%) y un conjunto de prueba (25%) utilizando train_test_split de sklearn.
Escalado de características:
Se aplica el escalado estándar para que las características tengan media 0 y desviación estándar 1, lo que es importante para los modelos como SVM.
Entrenamiento del modelo SVM:
Se crea un modelo SVM con kernel RBF y se ajusta (entrena) sobre los datos de entrenamiento (X_train, y_train).
Matriz de confusión y precisión:
Se genera una matriz de confusión que muestra los valores de predicción correctos e incorrectos.
Se calcula la precisión del modelo sobre el conjunto de prueba (X_test, y_test).
Aplicación de k-Fold Cross Validation:
Se aplica validación cruzada con k-Fold usando cross_val_score, con 10 pliegues para obtener una estimación más robusta de la precisión del modelo.
Se imprime la precisión media y la desviación estándar de las precisiones obtenidas a través de los pliegues.
Visualización de los resultados:
Se visualizan los resultados para el conjunto de entrenamiento y el conjunto de prueba utilizando matplotlib.
Los puntos de datos se representan en un gráfico de dispersión, donde los colores indican las predicciones del modelo (rojo o verde).
Se generan gráficos de decisión que muestran las regiones que el modelo clasifica como una clase o la otra.
Resultados Esperados:
Matriz de Confusión: Te ayudará a ver cuántas veces el modelo ha clasificado correctamente las observaciones de las clases.
Precisión: Indicará qué tan bien está funcionando el modelo en el conjunto de prueba.
Precisión media y desviación estándar de la validación cruzada, lo que proporciona una estimación más confiable del rendimiento del modelo.
Gráficos de decisión: Muestran cómo el modelo divide el espacio de características en dos clases (rojo o verde) para el conjunto de entrenamiento y el de prueba.
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
