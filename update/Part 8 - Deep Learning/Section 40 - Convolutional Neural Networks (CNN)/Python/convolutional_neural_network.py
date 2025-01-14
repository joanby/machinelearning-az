# Red Neuronal Convolucional (CNN)

"""
Explicación de los pasos:
Preprocesamiento de los Datos:
Se utiliza ImageDataGenerator para aplicar transformaciones como escalado, rotación y volteo horizontal a las imágenes del conjunto de entrenamiento. Esto mejora la capacidad de generalización del modelo.
Las imágenes de prueba solo se escalan para mantener la consistencia.
Se cargan las imágenes desde directorios específicos (training_set y test_set), y se asigna una etiqueta binaria (1 para perros y 0 para gatos).
Construcción de la CNN:
Se crea un modelo secuencial de Keras con varias capas:
Capa de Convolución: Detecta características locales en las imágenes.
Max Pooling: Reducción del tamaño de las imágenes para evitar sobreajuste y reducir el cómputo.
Flattening: Convierte las características extraídas en un vector unidimensional.
Capa densa (Fully Connected): Conecta todas las neuronas de la capa anterior con las neuronas de esta capa.
Capa de salida: Utiliza una función de activación sigmoid para obtener un valor entre 0 y 1, que se interpreta como la probabilidad de que la imagen sea un perro.
Entrenamiento de la CNN:
El modelo se compila con Adam como optimizador y binary_crossentropy como la función de pérdida, que es adecuada para clasificación binaria.
El modelo se entrena durante 25 épocas y se valida con las imágenes del conjunto de prueba.
Hacer una predicción:
Se carga una imagen de prueba (por ejemplo, una imagen de un gato o perro).
La imagen se preprocesa (cambiando su tamaño y convirtiéndola en un arreglo NumPy).
El modelo predice si la imagen es de un perro (1) o un gato (0).
Se muestra la predicción (ya sea "dog" o "cat").
"""

# Importación de las bibliotecas necesarias
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__  # Verificar la versión de TensorFlow

# Parte 1 - Preprocesamiento de los Datos

# Preprocesamiento del conjunto de entrenamiento
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocesamiento del conjunto de prueba
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Parte 2 - Construcción de la CNN

# Inicializando la Red Neuronal Convolucional (CNN)
cnn = tf.keras.models.Sequential()

# Paso 1 - Capa de Convolución
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Paso 2 - Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Añadir una segunda capa convolucional
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Paso 3 - Aplanamiento (Flatten)
cnn.add(tf.keras.layers.Flatten())

# Paso 4 - Conexión Completa
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Paso 5 - Capa de Salida
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Parte 3 - Entrenamiento de la CNN

# Compilando la CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenando la CNN con el conjunto de entrenamiento y validándola con el conjunto de prueba
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Parte 4 - Hacer una predicción para una imagen individual

import numpy as np
import keras.utils as image

# Cargar una imagen para predecir (por ejemplo, una imagen de un gato o perro)
test_image = image.load_img('dataset/test_image/Hadelin_Dog.jpg', target_size = (64, 64))

# Convertir la imagen a un arreglo NumPy
test_image = image.img_to_array(test_image)

# Ampliar las dimensiones para que se ajuste al formato de entrada de la CNN
test_image = np.expand_dims(test_image, axis = 0)

# Realizar la predicción
result = cnn.predict(test_image)

# Ver los índices de las clases (gato o perro)
training_set.class_indices

# Asignar la predicción: 1 = perro, 0 = gato
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
