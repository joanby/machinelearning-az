# Red Neuronal Artificial (ANN)

"""
Explicación de los pasos:
Preprocesamiento de datos:
Se realiza una codificación de variables categóricas para la columna "Gender" mediante LabelEncoder.
Se realiza una codificación One Hot para la columna "Geography" para convertir las categorías en columnas binarias.
Se divide el conjunto de datos en un conjunto de entrenamiento (80%) y prueba (20%).
Se aplica un escalado de las características con StandardScaler para normalizar los valores.
Construcción de la ANN:
Se crea una red neuronal con Keras. La estructura tiene:
Capa de entrada y primera capa oculta con 6 unidades y función de activación ReLU.
Una segunda capa oculta con 6 unidades y ReLU.
Capa de salida con una unidad y función de activación sigmoid para obtener una salida binaria (0 o 1).
Entrenamiento del modelo:
El modelo es entrenado con el optimizador Adam y la función de pérdida binary_crossentropy, ideal para tareas de clasificación binaria.
Se entrena durante 100 épocas con un batch size de 32.
Predicción y Evaluación:
Se predice si un cliente específico dejará el banco.
Se evalúa el rendimiento del modelo usando la matriz de confusión y la precisión.
"""

# Importación de las bibliotecas necesarias
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__  # Verificar la versión de TensorFlow

# Parte 1 - Preprocesamiento de los Datos

# Importación del dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values  # Tomar las características
y = dataset.iloc[:, -1].values  # Tomar la etiqueta (si el cliente se fue o no)
print(X)
print(y)

# Codificación de los datos categóricos
# Codificación de la columna "Gender" (Género) con LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # Transformar los valores de la columna "Gender"
print(X)

# Codificación One Hot para la columna "Geography" (Geografía)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # Aplicar la codificación one-hot para "Geography"
print(X)

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de las características
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Escalar el conjunto de entrenamiento
X_test = sc.transform(X_test)  # Escalar el conjunto de prueba

# Parte 2 - Construcción de la Red Neuronal Artificial (ANN)

# Inicializando la red neuronal (ANN)
ann = tf.keras.models.Sequential()

# Añadir la capa de entrada y la primera capa oculta
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Añadir la segunda capa oculta
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Añadir la capa de salida
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Parte 3 - Entrenamiento de la Red Neuronal Artificial (ANN)

# Compilando la red neuronal
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenando la red neuronal con el conjunto de entrenamiento
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Parte 4 - Predicción y Evaluación del Modelo

# Predicción del resultado para una observación individual

"""
Tarea:
Usar el modelo ANN para predecir si el cliente con la siguiente información dejará el banco: 
- Geografía: Francia
- Puntuación crediticia: 600
- Género: Masculino
- Edad: 40 años
- Tenencia: 3 años
- Saldo: $ 60000
- Número de productos: 2
- ¿Este cliente tiene tarjeta de crédito? Sí
- ¿Es este cliente miembro activo? Sí
- Salario estimado: $ 50000
¿Debemos despedir a este cliente?

Solución:
"""

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

"""
Por lo tanto, nuestro modelo ANN predice que este cliente se queda en el banco.
Nota importante 1: Observa que los valores de las características se introdujeron en un par de corchetes dobles. Esto se debe a que el método "predict" siempre espera una matriz 2D como formato de entrada.
Nota importante 2: Además, observa que el país "Francia" no se introdujo como una cadena en la última columna, sino como "1, 0, 0" en las primeras tres columnas. Esto se debe a que el método predict espera los valores codificados con one-hot de la geografía.
"""

# Predicción de los resultados para el conjunto de prueba
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)  # Convertir las predicciones a 0 o 1
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))  # Comparar predicciones con los valores reales

# Crear la matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)  # Generar la matriz de confusión
print(cm)  # Mostrar la matriz de confusión
accuracy_score(y_test, y_pred)  # Calcular la precisión del modelo
