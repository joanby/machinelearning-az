# Random Forest Regression

# Importar el dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)


# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar Modelo de Random Forest con el Conjunto de Datos
# install.packages("randomForest")
library(randomForest)
set.seed(1234)
regression = randomForest(x = dataset[1],
                          y = dataset$Salary,
                          ntree = 500)

# Predicción de nuevos resultados con Regresión 
y_pred = predict(regression, newdata = data.frame(Level = 6.5))


# Visualización del modelo de Random Forest
# install.packages("ggplot2")
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, 
                                        newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Predicción (Random Forest)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")
