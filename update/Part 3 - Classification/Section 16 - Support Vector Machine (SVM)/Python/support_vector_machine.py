# Máquinas de Vectores de Soporte (SVM)

#Explicación:
#Entrenamiento y prueba: Este script comienza cargando los datos y dividiéndolos en conjuntos de entrenamiento y prueba. Los estudiantes pueden ver cómo se separan los datos para evaluar el modelo en datos no vistos.
#Escalado de características: Se aplica el StandardScaler para normalizar las características, lo que es esencial para el modelo SVM, ya que es sensible a la escala de los datos.
#Entrenamiento del modelo: Se utiliza el SVC con un kernel lineal para entrenar el modelo y luego se realiza una predicción sobre un nuevo ejemplo (edad 30, salario 87000).
#Evaluación: Se genera la matriz de confusión y se calcula la precisión del modelo, lo que ayuda a evaluar su rendimiento.
#Visualización: Se visualizan los resultados en el conjunto de entrenamiento y prueba, mostrando las áreas de decisión del modelo.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargando el dataset
dataset = pd.read_csv('Social_Network_Ads.csv')  # Lee el archivo CSV con los datos
X = dataset.iloc[:, :-1].values  # Extrae las características (Edad, Salario Estimado)
y = dataset.iloc[:, -1].values  # Extrae la variable objetivo (Compra o No Compra)

# Dividiendo el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print("Conjunto de entrenamiento (X_train):")
print(X_train)
print("Etiquetas de entrenamiento (y_train):")
print(y_train)
print("Conjunto de prueba (X_test):")
print(X_test)
print("Etiquetas de prueba (y_test):")
print(y_test)

# Escalado de características (Feature Scaling)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Ajuste y transformación para el conjunto de entrenamiento
X_test = sc.transform(X_test)  # Transformación para el conjunto de prueba
print("Conjunto de entrenamiento escalado (X_train):")
print(X_train)
print("Conjunto de prueba escalado (X_test):")
print(X_test)

# Entrenando el modelo SVM sobre el conjunto de entrenamiento
classifier = SVC(kernel = 'linear', random_state = 0)  # Usamos el kernel lineal
classifier.fit(X_train, y_train)  # Ajusta el modelo a los datos de entrenamiento

# Predicción de un nuevo resultado (Ejemplo: Edad = 30, Salario = 87000)
resultado = classifier.predict(sc.transform([[30, 87000]]))  # Predice si comprará o no
print(f"Predicción para edad 30 y salario 87000: {resultado}")

# Predicción sobre el conjunto de prueba
y_pred = classifier.predict(X_test)  # Predice las etiquetas para el conjunto de prueba
print("Predicciones sobre el conjunto de prueba:")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))  # Compara las predicciones con las etiquetas reales

# Creando la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)  # Genera la matriz de confusión
print("Matriz de Confusión:")
print(cm)
print("Precisión del modelo:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)  # Muestra la precisión del modelo

# Visualización de los resultados en el conjunto de entrenamiento
X_set, y_set = sc.inverse_transform(X_train), y_train  # Inversa para visualizar en la escala original
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()

# Visualización de los resultados en el conjunto de prueba
X_set, y_set = sc.inverse_transform(X_test), y_test  # Inversa para visualizar en la escala original
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Conjunto de Prueba)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()
