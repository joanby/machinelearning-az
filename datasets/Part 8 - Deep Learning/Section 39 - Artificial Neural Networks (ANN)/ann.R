# Redes Neuronales Artificiales

# Importar el dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[, 4:14]

# Codificar los factores para la RNA
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c("France", "Spain", "Germany"),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                  levels = c("Female", "Male"),
                                  labels = c(1,2)))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
training_set[,-11] = scale(training_set[,-11])
testing_set[,-11] = scale(testing_set[,-11])

# Crear la red Neuronal
#install.packages("h2o")
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = "Exited",
                              training_frame = as.h2o(training_set),
                              activation = "Rectifier",
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

# Predicción de los resultados con el conjunto de testing
prob_pred = h2o.predict(classifier, 
                        newdata = as.h2o(testing_set[,-11]))
y_pred = (prob_pred>0.5)
y_pred = as.vector(y_pred)

# Crear la matriz de confusión
cm = table(testing_set[, 11], y_pred)

# Cerrar la sesión de H2O
h2o.shutdown()
