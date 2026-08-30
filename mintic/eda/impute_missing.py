import numpy as np
import pandas as pd
def impute_missing (data, strategy= 'mean', columns= None):
    data= data.copy()

    if columns == None:
        columns= data.columns.tolist()

    for c in columns:
        is_numeric= pd.api.types.is_numeric_dtype (data[c])

        if strategy == 'mean':
            if is_numeric:
                mean= data[c].sum() / data[c].count()
                data[c] = data[c].fillna(mean)
                
        elif strategy == 'median':
            if is_numeric:
                values= data[c].dropna()
                sort_v= values.sort_values().reset_index(drop=True)
                n = len(sort_v)
                half= n//2
                if n%2 == 1:
                    median= sort_v [half]
                    data[c] = data[c].fillna(median)
                else:
                    median= ((sort_v[half -1 ] + sort_v[half]) /2)
                    data[c] = data[c].fillna(median)

        elif strategy == 'mode':
            if not is_numeric:
                values= data[c].dropna()
                contador ={}
                for v in values:
                    if v in contador:
                        contador[v] = contador[v]+1
                    else:
                        contador[v] =1
                mode= max (contador, key=contador.get)
                data[c] = data[c].fillna(mode)
                        
    return data      