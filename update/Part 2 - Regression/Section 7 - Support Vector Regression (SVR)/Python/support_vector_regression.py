# Regresión con Máquinas de Vectores de Soporte (SVR)

#Explicación de los cambios:
#Escalado de características: He añadido comentarios detallados sobre el escalado de las características X y la variable dependiente y, ya que SVR suele funcionar mejor con variables estandarizadas.
#Redimensionamiento de y: He aclarado que la variable y es redimensionada a una columna para que funcione correctamente con el escalado y el modelo SVR.
#Visualización avanzada: He añadido un gráfico con mayor resolución para que la curva sea más suave y los resultados sean más claros.
#Predicción e inversión del escalado: He incluido ejemplos detallados sobre cómo invertir el escalado para obtener predicciones en el rango original de los datos.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib.ticker import FormatStrFormatter

# Cargando el dataset
dataset = pd.read_csv('Position_Salaries.csv')  # Lee el archivo CSV con los datos
X = dataset.iloc[:, 1:-1].values  # Extrae las posiciones (nivel de puesto)
y = dataset.iloc[:, -1].values  # Extrae los salarios
print("X (Posiciones):", X)
print("y (Salarios):", y)

# Redimensionando y (salarios) para aplicar la escala
y = y.reshape(len(y), 1)  # Reajusta el array de salarios para ser una columna

# Escalado de las características
sc_X = StandardScaler()  # Inicializa el escalador para X
sc_y = StandardScaler()  # Inicializa el escalador para y
X = sc_X.fit_transform(X)  # Escala las posiciones
y = sc_y.fit_transform(y)  # Escala los salarios
print("X escalado:", X)
print("y escalado:", y)

# Entrenamiento del modelo SVR (Regresión con Máquinas de Vectores de Soporte)
regressor = SVR(kernel='rbf')  # Se utiliza el kernel radial (rbf)
regressor.fit(X, y)  # Ajusta el modelo SVR a los datos

# Predicción de un nuevo resultado (para un puesto de nivel 6.5)
resultado = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
print("Predicción del salario para el puesto 6.5:", resultado)

# Visualización de los resultados del modelo SVR
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Puntos de datos reales
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')  # Curva ajustada por SVR
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.show()  # Muestra el gráfico

# Visualización de los resultados del SVR con mayor resolución para una curva más suave
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)  # Rango de valores para una curva más suave
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape para adaptarlo al modelo
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Puntos de datos reales
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color='blue')  # Curva ajustada con mayor resolución
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.show()  # Muestra el gráfico

# ADICIONAL: Representando el modelo sin que los ejes X e Y estén estandarizados
X_model = np.linspace(X.min(), X.max(), num=100).reshape(-1, 1)  # Rango de valores para predicción del modelo
y_model = regressor.predict(X_model).reshape(-1, 1)  # Predicción del modelo SVR
X_model = sc_X.inverse_transform(X_model)  # Desescalado de X
y_model = sc_y.inverse_transform(y_model)  # Desescalado de y

# Visualización con formato de ejes sin estandarización
plt.figure(figsize=(8, 6))
plt.axes().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # Formato de los valores en el eje Y
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red', label='Datos Reales')  # Datos reales
plt.plot(X_model, y_model, color='green', label='Modelo SVR')  # Modelo ajustado
plt.title('Regresión con SVM (SVR)', fontsize=16)
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.grid()  # Muestra la cuadrícula
plt.legend()  # Muestra la leyenda
plt.show()  # Muestra el gráfico
