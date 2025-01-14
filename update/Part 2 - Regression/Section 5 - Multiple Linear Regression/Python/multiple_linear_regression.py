# Múltiple regresión lineal en Python

#Cambios realizados:
#Comentarios en español para explicar cada paso del código.
#Se añadió una visualización opcional para comparar las predicciones con los valores reales.
#Optimización de la salida para mostrar resultados con dos decimales.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Carga del dataset
# El archivo CSV contiene información sobre startups, incluyendo características como gastos en I+D, 
# marketing, administración y la localización (que es categórica).
dataset = pd.read_csv('50_Startups.csv')

# Separar las variables independientes (X) de la variable dependiente (y).
# X: Todas las columnas excepto la última (características).
# y: La última columna (beneficios).
X = dataset.iloc[:, :-1].values  # Características
y = dataset.iloc[:, -1].values   # Beneficios
print("Datos originales (X):\n", X)

# 2. Codificación de datos categóricos
# La columna 3 (índice 3 en Python) contiene información categórica sobre la localización de la startup.
# Usamos OneHotEncoder para convertir esta columna en variables dummy (codificación one-hot).
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Transformador que aplica OneHotEncoder a la columna de índice 3.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

# Transformamos X y lo convertimos en un array de NumPy.
X = np.array(ct.fit_transform(X))
print("Datos codificados (X):\n", X)

# 3. División del dataset en conjunto de entrenamiento y prueba
# Se divide el conjunto de datos en entrenamiento (80%) y prueba (20%) para evaluar el modelo.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Entrenamiento del modelo de regresión lineal múltiple
from sklearn.linear_model import LinearRegression

# Inicializamos el modelo de regresión lineal.
regressor = LinearRegression()

# Ajustamos el modelo a los datos de entrenamiento.
regressor.fit(X_train, y_train)

# 5. Predicción de los resultados del conjunto de prueba
# Usamos el modelo entrenado para predecir los beneficios en el conjunto de prueba.
y_pred = regressor.predict(X_test)

# Configuramos la salida de NumPy para mostrar los números con dos decimales.
np.set_printoptions(precision=2)

# Mostramos las predicciones junto con los valores reales para comparación.
# Concatenamos las predicciones (y_pred) y los valores reales (y_test) en columnas.
resultados = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1)
print("Predicciones vs Valores Reales:\n", resultados)

# 6. Interpretación y visualización (opcional)
# Este modelo no tiene visualizaciones específicas, pero podrías graficar las predicciones
# contra los valores reales para evaluar visualmente el desempeño del modelo.
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores reales')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones')
plt.title('Predicciones vs Valores reales')
plt.xlabel('Índice de muestra')
plt.ylabel('Beneficios')
plt.legend()
plt.grid()
plt.show()
