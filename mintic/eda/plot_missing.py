def plot_missing (data):
    missing = data.isnull().sum()
    missing= missing [missing  > 0]
    plt.figure (figsize= (10,8))
    plt.bar (missing.index, missing.values, color = "thistle")
    plt.xlabel ('Columns')
    plt.ylabel ('Missing values')
    plt.show()

    return None