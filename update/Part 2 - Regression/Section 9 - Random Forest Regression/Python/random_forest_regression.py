# Regresión con Bosque Aleatorio (Random Forest Regression)

#Explicación de los cambios:
#Modelo de Bosque Aleatorio: Se ha añadido la explicación sobre la inicialización del modelo RandomForestRegressor con 10 árboles (n_estimators=10) y una semilla aleatoria para asegurar la reproducibilidad de los resultados.
#Predicción: Incluí un ejemplo claro de cómo predecir el salario para el nivel de puesto 6.5.
#Visualización avanzada: El gráfico ahora tiene una mayor resolución (con un paso de 0.01 en X_grid) para obtener una curva más suave, lo que permite ver cómo el modelo de bosque aleatorio ajusta los datos de forma detallada.


# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Cargando el dataset
dataset = pd.read_csv('Position_Salaries.csv')  # Lee el archivo CSV con los datos
X = dataset.iloc[:, 1:-1].values  # Extrae las posiciones (nivel de puesto)
y = dataset.iloc[:, -1].values  # Extrae los salarios

# Entrenamiento del modelo de Regresión con Bosque Aleatorio
regressor = RandomForestRegressor(n_estimators=10, random_state=0)  # Inicializa el modelo con 10 árboles en el bosque
regressor.fit(X, y)  # Ajusta el modelo de bosque aleatorio a los datos

# Predicción de un nuevo resultado (para un puesto de nivel 6.5)
resultado = regressor.predict([[6.5]])  # Predice el salario para el nivel de puesto 6.5
print(f"Predicción del salario para el puesto 6.5: {resultado}")

# Visualización de los resultados de la Regresión con Bosque Aleatorio (con mayor resolución)
X_grid = np.arange(min(X), max(X), 0.01)  # Rango de valores para obtener una curva más suave
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape para adaptarlo al modelo
plt.scatter(X, y, color='red')  # Muestra los puntos de los datos reales
plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # Curva ajustada por el bosque aleatorio
plt.title('Truth or Bluff (Regresión con Bosque Aleatorio)')
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.show()  # Muestra el gráfico
