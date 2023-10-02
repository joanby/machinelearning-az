#PLANTILLA DE PRE PROCESADO
#IMPORTAR EL DATASET
dataset = read.csv('Data.csv')

#LIMPIEZA DE DATOS O TRATAMIENTO DE LOS NA(NULL)
dataset$Age = ifelse (is.na(dataset$Age),
                      ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                      dataset$Age)

dataset$Salary = ifelse (is.na(dataset$Salary),
                      ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                      dataset$Salary)