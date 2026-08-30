import numpy as np
import pandas as pd
import math
def detect_outliers (data, method='iqr', threshold=1.5):
    data= data.copy()
    outliers = pd.DataFrame(False, index=data.index, columns=data.columns)
    for c in data.columns:
        is_numeric= pd.api.types.is_numeric_dtype (data[c])
        if is_numeric:
            if method == 'iqr':
                q1=np.quantile (data[c],0.25)
                q2=np.quantile (data[c],0.50)
                q3=np.quantile (data[c],0.75)
                IQR= q3-q1
                lower_bound= ( q1- threshold * (IQR))
                upper_bound = ( q3 + threshold * (IQR)) 
                outliers[c] = (data[c] < lower_bound) | (data[c] > upper_bound)
        
            if method == 'zscore':
                n=data[c].count()
                mean= data[c].sum() / data[c].count()
                var= ((data[c]- mean)**2).sum() /n
                desv= math.sqrt (var)
                z= (data[c]-mean)/ desv
                outliers [c] = (z > threshold) | (z < -threshold)
    return outliers   