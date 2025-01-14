# Regresión Lineal Simple

#Explicaciones clave:
#Importación de datos: Se asume que el archivo Salary_Data.csv contiene dos columnas: años de experiencia y salario.
#División del dataset: Usamos train_test_split para dividir los datos en un conjunto de entrenamiento (2/3) y prueba (1/3).
#Entrenamiento del modelo: Utilizamos LinearRegression de sklearn para ajustar un modelo lineal a los datos de entrenamiento.
#Visualización de resultados:
#Gráfico para el conjunto de entrenamiento: muestra los puntos reales y la línea de regresión ajustada.
#Gráfico para el conjunto de prueba: muestra los puntos reales del conjunto de prueba y la misma línea ajustada con el conjunto de entrenamiento.

# Importar las bibliotecas necesarias
import numpy as np  # Para operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Para visualización de datos
import pandas as pd  # Para manipulación y análisis de datos

# Importar el dataset
# Asegúrate de que 'Salary_Data.csv' esté en el mismo directorio o proporciona la ruta completa.
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Variable independiente (años de experiencia)
y = dataset.iloc[:, -1].values  # Variable dependiente (salario)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
# Usamos 2/3 para entrenamiento y 1/3 para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Entrenar el modelo de Regresión Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # Crear el modelo de regresión lineal
regressor.fit(X_train, y_train)  # Ajustar el modelo a los datos de entrenamiento

# Predecir los resultados del conjunto de prueba
y_pred = regressor.predict(X_test)  # Generar predicciones para los datos de prueba

# Visualizar los resultados del conjunto de entrenamiento
plt.scatter(X_train, y_train, color='red')  # Puntos reales en el conjunto de entrenamiento
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Línea de regresión ajustada
plt.title('Salario vs Experiencia (Conjunto de Entrenamiento)')  # Título del gráfico
plt.xlabel('Años de Experiencia')  # Etiqueta del eje X
plt.ylabel('Salario')  # Etiqueta del eje Y
plt.show()  # Mostrar el gráfico

# Visualizar los resultados del conjunto de prueba
plt.scatter(X_test, y_test, color='red')  # Puntos reales en el conjunto de prueba
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Línea de regresión ajustada (misma que en entrenamiento)
plt.title('Salario vs Experiencia (Conjunto de Prueba)')  # Título del gráfico
plt.xlabel('Años de Experiencia')  # Etiqueta del eje X
plt.ylabel('Salario')  # Etiqueta del eje Y
plt.show()  # Mostrar el gráfico
