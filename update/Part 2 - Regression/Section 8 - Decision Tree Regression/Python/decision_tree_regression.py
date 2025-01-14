# Regresión con Árbol de Decisión (Decision Tree Regression)

#Explicación de los cambios:
#Modelo de Árbol de Decisión: Se ha añadido la explicación sobre cómo se inicializa el modelo con random_state=0 para asegurar que los resultados sean reproducibles.
#Predicción: He incluido un ejemplo práctico de cómo predecir el salario para el nivel de puesto 6.5, y he mostrado el resultado de manera clara.
#Visualización avanzada: El gráfico ahora tiene una mayor resolución (con un paso de 0.01 en X_grid) para que la curva sea más suave, lo que permite ver cómo el árbol de decisión ajusta los datos de forma más detallada.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Cargando el dataset
dataset = pd.read_csv('Position_Salaries.csv')  # Lee el archivo CSV con los datos
X = dataset.iloc[:, 1:-1].values  # Extrae las posiciones (nivel de puesto)
y = dataset.iloc[:, -1].values  # Extrae los salarios

# Entrenamiento del modelo de Regresión con Árbol de Decisión
regressor = DecisionTreeRegressor(random_state=0)  # Inicializa el modelo con una semilla aleatoria para reproducibilidad
regressor.fit(X, y)  # Ajusta el modelo de árbol de decisión a los datos

# Predicción de un nuevo resultado (para un puesto de nivel 6.5)
resultado = regressor.predict([[6.5]])  # Predice el salario para el nivel de puesto 6.5
print(f"Predicción del salario para el puesto 6.5: {resultado}")

# Visualización de los resultados de la Regresión con Árbol de Decisión (con mayor resolución)
X_grid = np.arange(min(X), max(X), 0.01)  # Rango de valores para obtener una curva más suave
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape para adaptarlo al modelo
plt.scatter(X, y, color='red')  # Muestra los puntos de los datos reales
plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # Curva ajustada por el árbol de decisión
plt.title('Truth or Bluff (Regresión con Árbol de Decisión)')
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.show()  # Muestra el gráfico
