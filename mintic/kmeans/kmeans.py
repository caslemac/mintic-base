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


def kmeans_fit (kmeans,verbose=False):
    puntos = kmeans ['data']
    centroides = kmeans ['centroids'].copy()
    max_iter = kmeans ['max_iter']
    tol = kmeans ['tol']
    k = kmeans ['n_clusters']

    etiqueta= None

    for itera in range (1, max_iter + 1):
        distancia= np.zeros ((puntos.shape[0], k)) 
        
        for i in range (puntos.shape[0]):
            for j in range (k):         
 
               distancia [i,j] =euclidean_distance (puntos[i], centroides[j])
                
        etiquetas = np.argmin(distancia, axis=1)    
        centroides_actualizados = centroides.copy()        

        for j in range(k):                          
            grupo = puntos[etiquetas == j]        
            if len(grupo) > 0:                   
                centroides_actualizados[j] = grupo.mean(axis=0) 
     
        pasos = [euclidean_distance(centroides_actualizados[j], centroides[j]) for j in range(k)]  
        paso = np.max(pasos)          

        centroides = centroides_actualizados              

        if verbose:                                 
            print(f"Número de iteración {itera}: movimiento máximo = {paso:.6f}")

        if paso < tol:                        
            break                                  

    kmeans['centroids'] = centroides                 
    kmeans['labels'] = etiquetas                
    return kmeans  
    