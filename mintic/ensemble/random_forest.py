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


def calcular_entropia (y):
    valores, clases_conteo = np.unique (y, return_counts= True)
    proporciones= clases_conteo / len (y)
    return -np.sum(proporciones * np.log2(proporciones))

def ganancias (X_col, y):
    entropia= calcular_entropia(y)
    valores, conteos_col = np.unique(X_col, return_counts=True)

    entropia_ponderada=0
    for valor, conteo in zip(valores, conteos_col):
        subconjunto= y[X_col == valor]
        entropia_ponderada += (conteo / len (y))* calcular_entropia (subconjunto)
    return entropia - entropia_ponderada

def clase_mayoritaria(y):
    valores, clase_mas_frecuente = np.unique(y, return_counts=True)
    return valores[np.argmax(clase_mas_frecuente)]

def build_id3_tree(X, y, feature_indices=None, max_features=None, rng=None):
    if feature_indices is None:
        feature_indices = list(range(X.shape[1]))
    if rng is None:
        rng = np.random.RandomState()

    if len(np.unique(y)) == 1:
        return {'leaf': True, 'class': y[0]}
    elif len(feature_indices) == 0:
        return {'leaf': True, 'class': clase_mayoritaria(y)}
    elif len(y) == 0:
        return {'leaf': True, 'class': None}
    if max_features is None:
        candidatas = feature_indices
    else:
        k = min(max_features, len(feature_indices))
        candidatas = list(rng.choice(feature_indices, size=k, replace=False))

    ganancia = [ganancias(X[:, col], y) for col in candidatas]
    mejor_columna = candidatas[np.argmax(ganancia)]

    clase_max = clase_mayoritaria(y)
    restantes = [f for f in feature_indices if f != mejor_columna]
    ramas = {}
    for valor in np.unique(X[:, mejor_columna]):
        mascara = X[:, mejor_columna] == valor
        ramas[valor] = build_id3_tree(X[mascara], y[mascara],
                                      restantes, max_features, rng)

    return {'leaf': False, 'feature': mejor_columna,
            'branches': ramas, 'default': clase_max}


def build_random_forest(X, y, n_trees=10, max_features=None, random_state=None):
    if max_features is None:
        max_features = max(1, int(np.sqrt(X.shape[1])))

    arboles_entrenados = []
    for n in range(n_trees):
        semilla = random_state + n if random_state is not None else None
        rng = np.random.RandomState(semilla)

        muestra_X, muestra_y = bootstrap_sample(X, y, random_state=semilla)
        arbol = build_id3_tree(muestra_X, muestra_y,
                               max_features=max_features, rng=rng)
        arboles_entrenados.append(arbol)

    return arboles_entrenados