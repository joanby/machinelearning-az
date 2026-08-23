# Cambios 2026

Este documento explica qué se ha tocado en el código Python del curso para que
vuelva a ejecutarse en 2026, y por qué. No es un diff: es la explicación de
qué decidió cada librería entre 2019 (cuando se grabó el curso) y ahora, y qué
se ha hecho al respecto.

Todo lo que hay en esta rama son las mismas clases del vídeo. Nada de lo que
sigue cambia lo que aprendes ni el resultado que deberías obtener — salvo un
caso, marcado bien claro más abajo, donde sí cambia y se dice por qué.

## Versiones con las que se ha verificado

Verificado en **agosto de 2026**, en **macOS con Apple Silicon**:

| Paquete | Versión |
|---|---|
| Python | 3.13.15 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| nltk | 3.10.3 |
| xgboost | 3.4.1 |
| apyori | 1.1.2 |
| tensorflow | 2.21.0 |
| keras | 3.15.1 |

Si usas otro sistema operativo o instalas versiones distintas de estas
librerías, es posible que algo de lo que se cuenta aquí ya no aplique tal
cual, para bien o para mal.

## Los cuatro cambios

Con 60 ficheros de por medio (30 `.py` + 30 `.ipynb`) uno esperaría una lista
larga. En realidad todo lo que se ha tocado se reduce a cuatro causas.

### 1. numpy 2 dejó de convertir arrays a números por las buenas

**Afecta a:** `polynomial_regression`, `support_vector_regression`,
`decision_tree_regression` y `random_forest_regression` (`.py` y `.ipynb`,
8 ficheros en total, Parte 2 — Regresión).

**Qué se rompía.** La línea que dibuja la "curva suave" de la regresión —
`np.arange(min(X), max(X), 0.1)` — moría con
`TypeError: only 0-dimensional arrays can be converted to Python scalars`.

**Por qué lo cambió la librería.** `X` no es una lista de números: es una
matriz columna de una sola columna (shape `(n, 1)`). El `min()`/`max()`
**built-in** de Python, aplicado a algo así, no te da el número más pequeño —
itera fila a fila y te devuelve la fila mínima, que sigue siendo un array,
solo que de un elemento. Durante años eso no importó: NumPy convertía ese
array de un solo elemento a un número real por debajo, sin preguntar (con un
aviso silencioso que nadie leía). NumPy 2 decidió que esa conversión
automática era ambigua y la eliminó: si le das un array con `ndim > 0`,
aunque tenga un único elemento, ya no lo acepta donde espera un número. Y
`np.arange` espera un número. Es un cambio de NumPy, no de scikit-learn — a
scikit-learn no le toca nada aquí.

**Qué se hace ahora.** Se cambia `min(X)` por `X.min()` (y lo mismo con
`max`). No es una función distinta que "hace otra cosa": es el método propio
del array de NumPy, que sí sabe reducir cualquier array a un único número
real, sea cual sea su forma. El valor que devuelve es exactamente el mismo
que antes — solo cambia cómo se pide, no lo que se obtiene. La curva que ves
en el gráfico es idéntica a la del vídeo.

### 2. scikit-learn le cambió el nombre a un argumento

**Afecta a:** `hierarchical_clustering` (`.py` y `.ipynb`, Parte 4 —
Clustering).

**Qué se rompía.**
`AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')`
moría con
`TypeError: AgglomerativeClustering.__init__() got an unexpected keyword argument 'affinity'`.

**Por qué lo cambió la librería.** scikit-learn tenía dos nombres distintos
para "qué distancia usar" según el algoritmo: `affinity` en unos sitios,
`metric` en otros. Decidieron unificarlo bajo `metric` en toda la librería,
avisaron de la deprecación en la versión 1.2 y quitaron `affinity` del todo
en la 1.4. Con scikit-learn 1.9.0 el argumento ya no existe: es un error de
inicialización, no de cálculo.

**Qué se hace ahora.** Se renombra `affinity` a `metric`. El valor no
cambia: `'euclidean'` sigue siendo `'euclidean'` (y de hecho es el valor por
defecto de `metric` hoy en día). El curso pedía distancia euclídea y sigue
pidiendo distancia euclídea; solo cambia la palabra con la que se lo dices a
la función.

### 3. Keras 3 se llevó por delante `ImageDataGenerator` — y con él, una forma entera de hacer las cosas

**Afecta a:** `convolutional_neural_network` (`.py` y `.ipynb`, sección 40,
Parte 8 — Deep Learning). Es el único cambio de las cuatro donde el código
deja de parecerse línea a línea a lo que ves en el vídeo, así que merece la
explicación larga.

**Qué se rompía.**
`from keras.preprocessing.image import ImageDataGenerator` moría directamente
con `ImportError: cannot import name 'ImageDataGenerator' from 'keras.preprocessing.image'`.

**Por qué lo cambió la librería, y qué significa de verdad.**
`ImageDataGenerator` era una única clase que hacía cuatro trabajos distintos
a la vez sobre las fotos de una carpeta: las cargaba, las redimensionaba,
las reescalaba a valores entre 0 y 1 (`rescale = 1./255`), y les aplicaba
aumento de datos aleatorio en cada pasada — pequeños recortes, zooms y giros
(`shear_range`, `zoom_range`, `horizontal_flip`) para que la red no se
aprendiera de memoria las 8.000 fotos de entrenamiento y generalizara mejor.
Keras 3 (a partir de TensorFlow 2.16) eliminó esa clase entera, junto con
todo el pipeline que representaba.

Lo que la sustituye, `image_dataset_from_directory`, solo resuelve **dos**
de esos cuatro trabajos: carga las imágenes y las redimensiona. Ni reescala
ni aumenta datos por sí sola. Y esto no es una carencia que haya que
rodear con más configuración — es un cambio de filosofía deliberado: el
aumento de datos deja de ser un parámetro de "el objeto que carga fotos" y
pasa a ser **capas normales de la red neuronal** (`Rescaling`, `RandomShear`,
`RandomZoom`, `RandomFlip`), que se ejecutan sobre cada lote de imágenes
igual que se ejecuta una capa convolucional. Antes preparabas los datos
*antes* de dárselos a la red; ahora esa preparación es parte del propio
grafo de cómputo de la red.

**Qué se hace ahora, y qué es distinto de lo que ves en el vídeo.** El
fichero se ha reescrito con el visto bueno explícito del autor del curso —
es la única sección donde el mínimo cambio no era una línea, sino esto:

- Antes de cargar las imágenes, se declaran las transformaciones como capas:
  `Rescaling(1./255)` para el reescalado, y un `Sequential` con
  `RandomShear`, `RandomZoom` y `RandomFlip` para el aumento de datos.
- `train_datagen.flow_from_directory(...)` pasa a ser
  `tf.keras.utils.image_dataset_from_directory(...)` (mismo directorio,
  mismo `batch_size = 32`; `target_size` ahora se llama `image_size` y
  `class_mode` ahora se llama `label_mode`), seguido de un `.map(...)` que
  aplica el reescalado y el aumento de datos a cada lote. El conjunto de
  test se reescala igual, pero sin aumento de datos — igual que en el
  vídeo.
- `class_indices` (un diccionario) pasa a ser `class_names` (una lista): el
  nuevo pipeline no expone el diccionario de siempre.
- El resto del fichero —la CNN en sí, su compilación, el entrenamiento con
  `epochs = 25`, y la predicción sobre una imagen suelta— es idéntico al
  vídeo. Ahí no hay nada que Keras 3 haya tocado.

Dos valores del aumento de datos no tienen un equivalente exacto entre la
clase vieja y las capas nuevas — se ha mantenido el número que pide el
vídeo, pero el efecto visual no es idéntico, y te lo decimos para que no te
sorprenda:

- `shear_range = 0.2` era un ángulo en grados. `RandomShear(0.2)` interpreta
  ese `0.2` como una fracción del ancho de la imagen — la cizalladura que
  verás es notablemente más fuerte que la del original.
- `zoom_range = 0.2` hacía zoom aleatorio hacia dentro y hacia fuera a la
  vez. `RandomZoom(0.2)` con un solo número solo hace zoom hacia fuera.

`horizontal_flip` y el reescalado sí son exactamente equivalentes.

**Un fallo aparte que apareció al verificar esto, sin relación con Keras.**
El `.ipynb` de esta sección buscaba una imagen que no existe en el dataset
del curso (`dataset/single_prediction/cat_or_dog_1.jpg`) — un resto de la
reorganización del material a español, que sí actualizó esa ruta en el
`.py` pero se olvidó del `.ipynb`. No lo rompió ninguna librería: era un
enlace roto que llevaba ahí desde antes, invisible porque el entrenamiento
completo nunca se ejecutaba en las verificaciones de rutina. Se ha
corregido para que apunte a la misma imagen que usa el `.py`.

### 4. La malla de los gráficos de clasificación se disparó a 136 millones de puntos — y esto no lo rompió ninguna librería

**Afecta a:** las 7 secciones de la Parte 3 — Clasificación
(`logistic_regression`, `k_nearest_neighbors`, `support_vector_machine`,
`kernel_svm`, `naive_bayes`, `decision_tree_classification`,
`random_forest_classification`; `.py` y `.ipynb`, 14 ficheros).

Aquí toca ser honestos: a diferencia de los tres casos anteriores, esto no
lo causó ninguna versión nueva de numpy ni de scikit-learn. Fue un descuido
nuestro, en la revisión de enero de 2025.

**Qué se rompía.** Al dibujar la región de decisión de cada clasificador,
`support_vector_machine` y `kernel_svm` tardaban más de 300 segundos y
llegaban a consumir unos 9,5 GB de RAM antes de que se les cortara la
ejecución. Los otros cinco no llegaban a reventar, pero tardaban entre 13 y
57 segundos en pintar un gráfico que debería tardar menos de 5.

**Por qué pasaba.** El código original de 2019 dibujaba esa región sobre
los ejes **escalados** (valores pequeños, de -3 a +3 aproximadamente) con un
paso de malla de `0.01` — una malla de en torno a 360.000 puntos, razonable.
La revisión de enero de 2025 cambió el gráfico para mostrar los ejes **sin
escalar** (edad real y salario real, que se lee mucho mejor — una mejora
legítima) pero se dejó el mismo paso de malla, pensado para un rango de
apenas 6 unidades, aplicado ahora a un salario que va de 15.000 a 150.000.
El resultado: una malla de 136 millones de puntos en vez de 360.000. Nadie
cambió una librería; cambiamos los ejes y se nos olvidó ajustar el paso.

**Qué se hace ahora.** Se mantienen los ejes sin escalar — esa mejora se
queda — y se corrige el paso del eje del salario para que guarde la misma
proporción con su rango que ya tenía el paso del eje de la edad. En la
práctica: el paso del salario pasa de `0.25` a `500` (de `1` a `2000` en
K-NN, que ya usaba un paso distinto de partida). La malla baja de 136
millones de puntos a unos 68.000, y el gráfico que ves es visualmente
equivalente al que la revisión de 2025 ya pretendía dibujar — mismos ejes,
mismo rango, misma resolución aparente — solo que se calcula en segundos en
vez de minutos, o de nunca. Se ha comprobado, además, que este cambio no
toca ni una predicción ni una matriz de confusión: la malla solo alimenta el
dibujo, nunca un número que se imprime por pantalla.

## Lo que puede verse distinto al vídeo

- **Sección 40 — CNN.** Como se explica arriba, el código de carga y
  aumento de datos es nuevo. La arquitectura de la red, su entrenamiento y
  la predicción final son los mismos del vídeo.
- **Sección 24 — K-Means, el codo.** `KMeans` cambió el valor por defecto
  de `n_init` (cuántas veces prueba distintos puntos de partida antes de
  quedarse con el mejor) de `10` a `'auto'`, que equivale a solo 1 intento.
  El fichero del curso nunca fijaba `n_init` explícitamente, así que en
  2026 el codo de la curva salía un poco distinto del que ves en el vídeo a
  partir de k=6. Se ha fijado `n_init = 10` a mano, así que el codo vuelve
  a ser exactamente el de siempre.
- **`thompson_sampling.py`.** Este fichero no fija ninguna semilla aleatoria
  — no lo hacía tampoco en la revisión de 2025, así que no es un cambio de
  esta actualización. Pero sí significa que **cada vez que lo ejecutes
  obtendrás un resultado ligeramente distinto**, y distinto también del que
  ves en el vídeo. Es el comportamiento esperado del fichero tal como está
  escrito, no un error.

## Lo que NO se ha ejecutado

**El entrenamiento completo de la CNN de la sección 40: 25 épocas sobre
8.000 imágenes.** Verificar eso entero se cuenta en horas, no en segundos,
y no es viable como parte de una revisión rutinaria. Se ha verificado el
fichero con **1 sola época**, solo para confirmar que el código corre de
principio a fin sin errores — la CNN se entrena, sí, pero muy poco. El
accuracy que da esa prueba (en torno al 58-64% según el fichero) **no es el
accuracy del vídeo y no se presenta como tal**: con una sola época no hay
manera honesta de que lo sea. El fichero que tienes delante conserva sus 25
épocas — cuando lo ejecutes tú, con tiempo, deberías acercarte al resultado
del vídeo.

## R sigue sin actualizar

Este curso se llama *"Machine Learning de A a la Z: R y Python para Data
Science"*, y esta actualización, dicho sin rodeos, **solo cubre la mitad
Python**. Los 47 scripts `.R` del curso no se han tocado — ni se han
arreglado, ni se han borrado, siguen tal cual estaban.

De esos 47, **19 dependen de `library(ElemStatLearn)`** (con 38 llamadas en
total, dos por fichero: una para el gráfico de entrenamiento y otra para el
de test) — un paquete que **CRAN archivó en 2020** y que ya no se instala
con un simple `install.packages()`. Son, en concreto, los scripts de
Clasificación (Parte 3), Reducción de la Dimensión (Parte 9) y Selección de
Modelos (Parte 10).

No lo escondemos: si sigues el curso en R, te vas a encontrar con este
muro. Ponerlo al día es una segunda fase, todavía pendiente, que no entra
en esta actualización.

---

¿Dudas sobre algún cambio? Este documento intenta explicar el porqué, no
solo el qué — si algo no cuadra, dínoslo.
