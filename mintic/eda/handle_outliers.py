import numpy as np
import pandas as pd
import math
from .handle_outliers import handle_outliers
def handle_outliers (data, method='iqr', action='trim', threshold=1.5):
    if action == 'trim':
        outliers= detect_outliers (data, method=method, threshold=1.5)
        without_outliers= ~outliers.any(axis=1)
        data= data [without_outliers]
    elif action == 'cap':
        for c in data.columns:
            if method == 'iqr':
                q1=np.quantile (data[c],0.25)
                q3=np.quantile (data[c],0.75)
                IQR= q3-q1
                lower_bound= ( q1- threshold * (IQR))
                upper_bound = ( q3 + threshold * (IQR)) 
                
            if method == 'zscore':
                n=data[c].count()
                mean= data[c].sum() / data[c].count()
                var= ((data[c]- mean)**2).sum() /n
                desv= math.sqrt (var)
                lower_bound= mean- threshold *desv
                upper_bound= mean + threshold*desv
            data[c] = data[c].clip(lower= lower_bound, upper= upper_bound)
    return data