import numpy as np
import pandas as pd
def euclidean_distance (point1, point2):
    p1= np.asarray(point1, dtype= float)
    p2= np.asarray (point2, dtype= float)
    distancia= float (np.sqrt (np.sum((p1-p2)**2)))
    return distancia