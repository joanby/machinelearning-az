# Apriori

#Explicación:
#Preprocesamiento de datos: Se cargan las transacciones del archivo CSV y se convierten en una lista de transacciones, donde cada transacción es un conjunto de productos.
#Modelo Apriori: Se entrena el modelo Apriori usando las transacciones cargadas. Los parámetros min_support, min_confidence, min_lift, min_length, y max_length controlan las reglas generadas, como la frecuencia mínima de los productos, la confianza mínima en las reglas y el "lift" (que mide la importancia de la regla).
#Visualización de resultados: El resultado del modelo se convierte en un DataFrame, con columnas que muestran los productos del lado izquierdo (Left Hand Side), los productos del lado derecho (Right Hand Side), el soporte (Support), la confianza (Confidence) y el "lift" (Lift) de cada regla.
#Ordenación de resultados: Los resultados se muestran sin ordenar y luego se ordenan por el "lift" para mostrar las reglas más interesantes.


# Instalación del paquete apyori si no está instalado
# Ejecuta este comando en la terminal: pip install apyori

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Preprocesamiento de los datos
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)  # Cargar el dataset sin encabezado
transactions = []  # Lista para almacenar las transacciones
for i in range(0, 7501):  # Iterar sobre cada transacción
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])  # Convertir los datos en transacciones de items

# Entrenamiento del modelo Apriori sobre el dataset
from apyori import apriori  # Importar la librería apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)  # Generar las reglas de asociación

# Visualización de los resultados

## Mostrar los primeros resultados provenientes directamente de la función apriori
results = list(rules)  # Convertir el resultado en lista
results

# Organizar los resultados de manera más legible en un DataFrame de Pandas
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]  # Elemento de la izquierda de la regla
    rhs         = [tuple(result[2][0][1])[0] for result in results]  # Elemento de la derecha de la regla
    supports    = [result[1] for result in results]  # Soporte de la regla
    confidences = [result[2][0][2] for result in results]  # Confianza de la regla
    lifts       = [result[2][0][3] for result in results]  # Lift de la regla
    return list(zip(lhs, rhs, supports, confidences, lifts))  # Combinar todo en una lista

# Convertir los resultados en un DataFrame para facilitar la visualización
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Mostrar los resultados sin ordenar
resultsinDataFrame

# Mostrar los resultados ordenados por el valor de Lift en orden descendente
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')  # Top 10 reglas con mayor Lift
