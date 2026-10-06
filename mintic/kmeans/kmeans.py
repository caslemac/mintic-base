import numpy as np
import pandas as pd
def euclidean_distance (point1, point2):
    p1= np.asarray(point1, dtype= float)
    p2= np.asarray (point2, dtype= float)
    distancia= float (np.sqrt (np.sum((p1-p2)**2)))
    return distancia

def create_kmeans (data, n_clusters=8, max_iter=1000, tol=0.0001,randomstate=None):
    puntos = np.asarray (data, dtype= float)
    filas = puntos.shape[0]
    semilla= np.random.default_rng (randomstate)
    indices= semilla.choice (filas, size= n_clusters, replace = False)
    centroides = puntos [indices]
    diccionario = {
        'centroids' : centroides,
        'n_clusters' : n_clusters,
        'max_iter' : max_iter,
        'tol' : tol,
        'randomState' : randomstate,
        'labels' : None
    }
    return diccionario

    