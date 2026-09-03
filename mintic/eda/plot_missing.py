import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .plot_missing import plot_missing

def plot_missing (data):
    missing = data.isnull().sum()
    missing= missing [missing  > 0]
    plt.figure (figsize= (10,8))
    plt.bar (missing.index, missing.values, color = "thistle")
    plt.xlabel ('Columns')
    plt.ylabel ('Missing values')
    plt.show()

    return None