# Procesamiento de Lenguaje Natural (NLP)

"""
Explicación del flujo:
Limpieza de textos:
Se eliminan caracteres no alfabéticos y se convierte el texto a minúsculas.
Se elimina las palabras comunes (stopwords) que no aportan valor informativo, salvo la palabra "not" que es importante para el análisis de sentimientos.
Se aplica el stemmer de Porter para reducir las palabras a su raíz (por ejemplo, "running" se convierte en "run").
Modelo Bag of Words:
Se crea una matriz de características usando las 1500 palabras más frecuentes del corpus.
Entrenamiento y prueba:
Se divide el dataset en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%).
Se entrena un modelo de Naive Bayes sobre el conjunto de entrenamiento.
Evaluación del modelo:
Se predicen las etiquetas para el conjunto de prueba y se comparan con las etiquetas reales.
Se genera una matriz de confusión para evaluar el rendimiento del modelo y se calcula la precisión.
"""

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importación del dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Limpieza de los textos
import re
import nltk
nltk.download('stopwords')  # Descargar las stopwords de NLTK
from nltk.corpus import stopwords  # Importar las stopwords
from nltk.stem.porter import PorterStemmer  # Importar el Stemmer de Porter

corpus = []  # Lista para almacenar las reseñas procesadas
for i in range(0, 1000):  # Procesar las primeras 1000 reseñas
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # Eliminar caracteres no alfabéticos
    review = review.lower()  # Convertir el texto a minúsculas
    review = review.split()  # Separar las palabras
    ps = PorterStemmer()  # Inicializar el stemmer
    all_stopwords = stopwords.words('english')  # Obtener las stopwords en inglés
    all_stopwords.remove('not')  # Mantener la palabra 'not' (importante en análisis de sentimientos)
    # Aplicar stemming y eliminar stopwords
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)  # Unir las palabras procesadas en una sola cadena
    corpus.append(review)  # Añadir la reseña procesada al corpus
print(corpus)

# Creación del modelo Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  # Usar las 1500 palabras más frecuentes
X = cv.fit_transform(corpus).toarray()  # Crear la matriz de características
y = dataset.iloc[:, -1].values  # Obtener las etiquetas (sentimiento de las reseñas)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)  # Dividir 80/20

# Entrenamiento del modelo Naive Bayes en el conjunto de entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()  # Inicializar el clasificador Naive Bayes
classifier.fit(X_train, y_train)  # Ajustar el modelo al conjunto de entrenamiento

# Predicción de los resultados en el conjunto de prueba
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))  # Comparar predicciones y valores reales

# Crear la matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)  # Generar la matriz de confusión
print(cm)  # Mostrar la matriz de confusión
accuracy_score(y_test, y_pred)  # Calcular la precisión del modelo
