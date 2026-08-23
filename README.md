# 🚀 Repositorio del Curso: **[Machine Learning de A a la Z: R y Python para Data Science](https://cursos.frogamesformacion.com/courses/machine-learning-az)**  
**Creado por** [Kirill Eremenko](https://www.udemy.com/user/kirilleremenko/) y [Hadelin de Ponteves](https://www.udemy.com/user/hadelin-de-ponteves/)  
**Traducido al español por** [Juan Gabriel Gomila Salas](https://www.udemy.com/user/juangabriel2)  

**Última actualización**: Agosto de 2026

---

## 🆕 Esta es la rama `update-2026`

Estás en la versión del código **puesta al día y verificada en agosto de
2026**, con Python 3.13 y las librerías tal como están hoy (numpy 2,
scikit-learn 1.9, TensorFlow/Keras 3...). El curso se grabó en 2019 y
algunas librerías han cambiado de comportamiento desde entonces; aquí el
código Python vuelve a ejecutarse de principio a fin.

**Qué encontrarás — y qué no — en esta rama:**

- Aquí solo está el código **Python** (`.py` y `.ipynb`), puesto al día y en
  la raíz del repositorio, organizado por partes del curso igual que
  siempre (`Part 1 - Data Preprocessing`, `Part 2 - Regression`, etc.).
- **No hay carpetas `original/` ni `update/`.** Esta rama es plana: lo que
  ves en cada `Part X` es directamente el código que necesitas.
- **`original/` no ha desaparecido: vive en la rama [`master`](https://github.com/joanby/machinelearning-az/tree/master)**, junto con la revisión de enero de 2025. Si quieres ver el código exactamente como se grabó el vídeo, es ahí donde está.
- **Los 47 scripts en R sí están en esta rama**, en una carpeta `R/` junto
  al Python de cada sección. No se han tocado — ni arreglados, ni
  borrados, siguen tal cual estaban. 19 de ellos dependen de
  `library(ElemStatLearn)`, archivado en CRAN desde 2020, así que fallarán
  al ejecutarlos — ver el aviso sobre R más abajo.
- `additional materials/` (las transparencias de teoría) se mantiene igual
  que en `master`.

**Qué ha cambiado exactamente y por qué:** todo está explicado, cambio a
cambio y con el porqué de cada librería, en
[`CAMBIOS-2026.md`](CAMBIOS-2026.md). Incluye también qué partes pueden
verse distintas al vídeo, qué no se ha llegado a ejecutar (el entrenamiento
completo de la CNN) y el aviso claro sobre el estado de R. Léelo antes de
reportar como error algo que ya está documentado ahí.

---

## 🐍 Cómo montar el entorno

El curso usa un entorno conda. Con [Miniforge](https://github.com/conda-forge/miniforge) o Anaconda instalado:

```bash
conda env create -f machinelearningaz.yml
conda activate machinelearningaz
```

Esto te deja Python 3.13 con numpy, pandas, matplotlib, scikit-learn, scipy,
nltk, xgboost, Jupyter, Spyder y (por pip) apyori y TensorFlow. Las
versiones exactas con las que se ha verificado esta rama están en la tabla
de [`CAMBIOS-2026.md`](CAMBIOS-2026.md#versiones-con-las-que-se-ha-verificado).

---

## 📚 **Temario del Curso**

Este curso está diseñado para llevarte de la teoría a la práctica, y cubre una amplia gama de temas esenciales de **Machine Learning**. Aquí tienes un resumen de los módulos:

### 1️⃣ **Preprocesamiento de Datos**
### 2️⃣ **Regresión**  
- Regresión Lineal Simple, Múltiple, Polinomial  
- SVR, Árboles de Decisión, Bosques Aleatorios

### 3️⃣ **Clasificación**  
- Regresión Logística, K-NN, SVM  
- Naive Bayes, Árboles de Decisión, Bosques Aleatorios

### 4️⃣ **Clustering**  
- K-Means, Clustering Jerárquico

### 5️⃣ **Aprendizaje por Reglas de Asociación**  
- Apriori, Eclat

### 6️⃣ **Reinforcement Learning**  
- Límite de Confianza Superior, Muestreo Thompson

### 7️⃣ **Procesamiento Natural del Lenguaje (NLP)**  
- Modelo de Bag-of-Words, algoritmos de NLP

### 8️⃣ **Deep Learning**  
- Redes Neuronales Artificiales y Convolucionales

### 9️⃣ **Reducción de la Dimensión**  
- ACP, LDA, Kernel ACP

### 🔟 **Selección de Modelos & Boosting**  
- k-fold Cross Validation, Ajuste de Parámetros, Grid Search, XGBoost

---

### 🧑‍💻 **Lo que aprenderás en este curso**  
- **Dominar Machine Learning** con Python y R.
- **Crear modelos predictivos robustos** y generar análisis precisos.
- **Aplicar técnicas avanzadas** como **Reinforcement Learning**, **NLP** y **Deep Learning**.
- **Crear valor en tu negocio** mediante el uso de **Modelos de Machine Learning**.
- **Desarrollar una librería de modelos** para resolver problemas de Data Science.

---

## ⚡ **¿Para quién es este curso?**

Este curso está dirigido a todos aquellos interesados en aprender **Machine Learning**. Ideal para:  
- **Principiantes**: Si tienes conocimientos básicos de matemáticas (nivel secundaria y bachillerato).
- **Estudiantes de Data Science** que quieran profundizar en el mundo del Machine Learning.
- **Analistas de Datos** que busquen mejorar sus habilidades en programación y ML.
- **Profesionales** que deseen mejorar sus habilidades analíticas y crear valor en sus empresas.

---

## 🎯 **¿Qué requisitos necesitas?**  
No es necesario ser un experto en programación, solo es recomendable tener conocimientos básicos de matemáticas a nivel de **secundaria o bachillerato**.

---

## 🎁 **¡Bonus!**  
Este curso incluye **todo el código en Python y R** que puedes descargar y aplicar a tus propios proyectos, así como ejemplos prácticos basados en datos reales. El código en R está aquí mismo, en esta rama, sin tocar (19 de los 47 scripts dependen de `ElemStatLearn`, archivado en CRAN desde 2020, y no llegan a ejecutarse).

---

## 🎥 **¿Listo para aprender Machine Learning de manera fácil y divertida?**  
Apúntate al curso con **descuento exclusivo** [desde aquí](https://cursos.frogamesformacion.com/courses/machine-learning-az) y empieza a aprender hoy mismo.

---

¡Esperamos verte pronto en el curso! Si tienes dudas o preguntas, no dudes en contactar con nosotros. ¡Estamos aquí para ayudarte! 💡
