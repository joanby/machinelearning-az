# Eclat

#Explicación:
#Preprocesamiento de datos: Se cargan las transacciones del archivo CSV y se transforman en una lista, donde cada transacción es un conjunto de productos.
#Modelo Eclat: Aunque el código usa apyori para generar reglas de asociación mediante Apriori, Eclat también puede generar reglas de asociación basadas en el soporte, pero con un enfoque diferente. El modelo utiliza las transacciones cargadas y se aplican parámetros como min_support, min_confidence, min_lift, entre otros, para filtrar las reglas.
#Visualización de resultados: Los resultados del modelo se organizan en un DataFrame de Pandas que muestra los productos del lado izquierdo (Product 1), los productos del lado derecho (Product 2) y su soporte (Support). Los resultados se ordenan para mostrar las reglas más fuertes según el soporte.

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

# Entrenamiento del modelo Eclat sobre el dataset
from apyori import apriori  # Importar la librería apriori (Eclat también puede estar presente en algunos contextos)
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
    return list(zip(lhs, rhs, supports))  # Combinar todo en una lista

# Convertir los resultados en un DataFrame para facilitar la visualización
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

# Mostrar los resultados ordenados por el soporte en orden descendente
resultsinDataFrame.nlargest(n = 10, columns = 'Support')  # Top 10 reglas con mayor soporte
