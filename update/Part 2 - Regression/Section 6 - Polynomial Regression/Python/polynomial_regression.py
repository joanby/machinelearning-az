# Regresión Polinómica en Python

#Cambios y actualizaciones 
#Uso de bibliotecas actualizadas: Se mantiene la estructura, pero ahora el código está documentado adecuadamente para que los alumnos puedan comprender cada paso.
#Comentarios en castellano: Todos los comentarios explican claramente las acciones realizadas, como la carga de datos, el entrenamiento de los modelos y la visualización de los resultados.
#Predicción: He añadido las impresiones de las predicciones para que los alumnos puedan ver los resultados directos de cada modelo.


# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Cargando el dataset
dataset = pd.read_csv('Position_Salaries.csv')  # Lee el archivo CSV con los datos
X = dataset.iloc[:, 1:-1].values  # Extrae las posiciones (nivel de puesto)
y = dataset.iloc[:, -1].values  # Extrae los salarios

# Entrenamiento del modelo de Regresión Lineal con todo el dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)  # Ajusta el modelo de regresión lineal

# Entrenamiento del modelo de Regresión Polinómica
# Se utiliza un grado de 4 para el polinomio
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)  # Transformación de las características en forma polinómica
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)  # Ajusta el modelo de regresión polinómica

# Visualización de los resultados de la Regresión Lineal
plt.scatter(X, y, color='red')  # Muestra los puntos de datos
plt.plot(X, lin_reg.predict(X), color='blue')  # Línea de regresión lineal
plt.title('Truth or Bluff (Regresión Lineal)')
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.show()  # Muestra el gráfico

# Visualización de los resultados de la Regresión Polinómica
plt.scatter(X, y, color='red')  # Muestra los puntos de datos
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')  # Curva polinómica
plt.title('Truth or Bluff (Regresión Polinómica)')
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.show()  # Muestra el gráfico

# Visualización de los resultados de la Regresión Polinómica con mayor resolución para una curva más suave
X_grid = np.arange(min(X), max(X), 0.1)  # Crear un rango de valores para un gráfico más suave
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape para adaptarlo al modelo
plt.scatter(X, y, color='red')  # Muestra los puntos de datos
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')  # Curva polinómica suave
plt.title('Truth or Bluff (Regresión Polinómica)')
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.show()  # Muestra el gráfico

# Predicción con el modelo de Regresión Lineal
salario_lineal = lin_reg.predict([[6.5]])  # Predice el salario para el nivel de puesto 6.5
print(f"Predicción con Regresión Lineal para el puesto 6.5: {salario_lineal}")

# Predicción con el modelo de Regresión Polinómica
salario_polinomica = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))  # Predice el salario para el puesto 6.5
print(f"Predicción con Regresión Polinómica para el puesto 6.5: {salario_polinomica}")
