# Herramientas de preprocesamiento de datos

#Explicaciones clave:
#Manejo de valores faltantes: Utiliza SimpleImputer para reemplazar valores nulos con la media.
#Codificación de datos categóricos:
#Para variables independientes, se aplica OneHotEncoder.
#Para la variable dependiente, se usa LabelEncoder.
#División del dataset: Separamos los datos en conjuntos de entrenamiento y prueba usando train_test_split.
#Escalado de características: Se aplica estandarización para mejorar el rendimiento de ciertos algoritmos que son sensibles a las escalas de las variables.

# Importar las bibliotecas necesarias
import numpy as np  # Para operaciones con matrices y vectores
import matplotlib.pyplot as plt  # Para visualización de datos (opcional)
import pandas as pd  # Para manipulación y análisis de datos

# Importar el dataset
# Asegúrate de que el archivo 'Data.csv' esté en el mismo directorio o proporciona la ruta completa.
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Seleccionamos todas las columnas excepto la última como variables independientes
y = dataset.iloc[:, -1].values  # Seleccionamos la última columna como la variable dependiente (objetivo)
print("Datos originales de X:")
print(X)
print("Datos originales de y:")
print(y)

# Manejo de datos faltantes
# Utilizamos SimpleImputer para reemplazar valores nulos con la media de la columna
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # Estrategia: reemplazar valores nulos con la media
imputer.fit(X[:, 1:3])  # Aplicamos el ajuste a las columnas 1 y 2 (índices 1:3)
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Reemplazamos los valores nulos en las columnas seleccionadas
print("Datos de X después de manejar valores faltantes:")
print(X)

# Codificación de datos categóricos
# Codificación de la variable independiente (X)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Aplicamos OneHotEncoder a la primera columna (índice 0)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # Transformamos X y lo convertimos a un array de NumPy
print("Datos de X después de la codificación OneHot:")
print(X)

# Codificación de la variable dependiente (y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)  # Transformamos las etiquetas categóricas en valores numéricos
print("Datos de y después de la codificación:")
print(y)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
# Usamos 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Conjunto de entrenamiento (X_train):")
print(X_train)
print("Conjunto de prueba (X_test):")
print(X_test)
print("Etiquetas del conjunto de entrenamiento (y_train):")
print(y_train)
print("Etiquetas del conjunto de prueba (y_test):")
print(y_test)

# Escalado de características
# Estandarizamos los valores para que tengan media 0 y desviación estándar 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])  # Ajustamos y transformamos las características numéricas del conjunto de entrenamiento
X_test[:, 3:] = sc.transform(X_test[:, 3:])  # Transformamos el conjunto de prueba usando el ajuste del conjunto de entrenamiento
print("Conjunto de entrenamiento después del escalado de características:")
print(X_train)
print("Conjunto de prueba después del escalado de características:")
print(X_test)
