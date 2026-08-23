# Plantilla de preprocesamiento de datos

#Cambios realizados:
#Verificación de valores nulos: Se añade un chequeo para detectar si el dataset tiene valores nulos y advertir al usuario.
#Comentarios detallados: Cada sección incluye explicaciones claras sobre qué hace el código.
#Mensajes de depuración: Se añaden prints para verificar las dimensiones de los conjuntos de entrenamiento y prueba.

# Importando las bibliotecas necesarias
import numpy as np  # Para manejo de vectores y matrices
import matplotlib.pyplot as plt  # Para visualización de datos (aunque no se usa aquí)
import pandas as pd  # Para manipulación y análisis de datos

# Importar el dataset
# Asegúrate de que 'Data.csv' esté en el mismo directorio o proporciona la ruta completa.
dataset = pd.read_csv('Data.csv')

# Verificar si hay valores nulos en el dataset
if dataset.isnull().values.any():
    print("El dataset contiene valores nulos. Considera imputarlos o eliminarlos antes de continuar.")

# Variables independientes (X) y variable objetivo (y)
# Seleccionamos todas las columnas excepto la última como variables independientes (X)
X = dataset.iloc[:, :-1].values
# Seleccionamos la última columna como la variable objetivo (y)
y = dataset.iloc[:, -1].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split

# Dividimos los datos: 80% para entrenamiento y 20% para prueba
# random_state asegura que los resultados sean reproducibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Mostrar información básica para verificar las dimensiones de los datos
print("Dimensiones de X_train:", X_train.shape)
print("Dimensiones de X_test:", X_test.shape)
print("Dimensiones de y_train:", y_train.shape)
print("Dimensiones de y_test:", y_test.shape)


