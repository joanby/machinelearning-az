"""
Created on Tue May 21 23:49:01 2019
"""
#PLANTILLA DE PRE PROCESADO

#IMPORTAR LAS LIBRERIAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#IMPORTAR EL DATASET
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[ : , :-1 ].values #MATRIZ DE DATOS INDEPENDIENTES

y = dataset.iloc[ : , 3 ].values #VECTOR DE DATOS DEPENDIENTES

#LIMPIEZA DE DATOS O TRATAMIENTO DE LOS NA(NULL)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

x