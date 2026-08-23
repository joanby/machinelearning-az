# Thompson Sampling

#Explicación:
#Importación de datos: El archivo Ads_CTR_Optimisation.csv contiene información sobre la tasa de clics (CTR) de diferentes anuncios. Se carga en un DataFrame de pandas.
#Algoritmo de Thompson Sampling: Este algoritmo utiliza la distribución Beta para modelar la probabilidad de éxito de un anuncio. En cada ronda, se selecciona el anuncio con la mayor muestra de la distribución Beta (basada en los clics previos). La idea es equilibrar la exploración y explotación mediante muestras aleatorias para decidir qué anuncios mostrar.
#Visualización: Se genera un histograma que muestra cuántas veces cada anuncio fue seleccionado, lo que ayuda a visualizar el rendimiento de cada uno durante las 10,000 rondas.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importación del dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')  # Cargar el dataset de optimización de clics en anuncios

# Implementación de Thompson Sampling
import random
N = 10000  # Número total de rondas
d = 10  # Número de anuncios
ads_selected = []  # Lista para almacenar los anuncios seleccionados
numbers_of_rewards_1 = [0] * d  # Contador de recompensas 1 (clics) para cada anuncio
numbers_of_rewards_0 = [0] * d  # Contador de recompensas 0 (sin clics) para cada anuncio
total_reward = 0  # Recompensa total acumulada

# Algoritmo de Thompson Sampling
for n in range(0, N):  # Para cada ronda
    ad = 0  # Inicializar el anuncio seleccionado
    max_random = 0  # Inicializar el valor máximo aleatorio
    for i in range(0, d):  # Evaluar cada anuncio
        # Generar una muestra aleatoria de la distribución Beta
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        # Seleccionar el anuncio con la mayor muestra aleatoria
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)  # Añadir el anuncio seleccionado a la lista
    reward = dataset.values[n, ad]  # Obtener la recompensa (clic) del anuncio seleccionado
    # Actualizar los contadores de recompensas para cada anuncio
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward  # Sumar la recompensa total

# Visualización de los resultados - Histograma
plt.hist(ads_selected)  # Crear el histograma de las selecciones de anuncios
plt.title('Histograma de selecciones de anuncios')  # Título del gráfico
plt.xlabel('Anuncios')  # Etiqueta en el eje X
plt.ylabel('Número de veces que se seleccionó cada anuncio')  # Etiqueta en el eje Y
plt.show()  # Mostrar el gráfico
