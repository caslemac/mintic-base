import numpy as np
import random
from .random_forest import random_forest

def bootstrap_sample (X, y, random_state= None):
    semilla= np.random.RandomState (random_state)
    filas= X.shape [0]
    indices= semilla.randint (0, filas, size= filas)

    Xi= X[indices]
    yi= y[indices]

    return Xi, yi