import numpy as np
from numpy.linalg import inv


# Distância de Minkowski
def distancia_minkowski(vetor_1, vetor_2, p=2):
    """
    Calcula a distância de Minkowski entre dois vetores.

    Parâmetros:

    vetor_1 : array-like, shape (d,)
        Primeiro vetor.
    vetor_2 : array-like, shape (d,)
        Segundo vetor.
    p : float >= 1
        Parâmetro da distância de Minkowski (p=1 -> Manhattan, p=2 -> Euclidiana).

    Retorno:

    float
        Distância de Minkowski entre vetor_1 e vetor_2.
    """
    vetor_1 = np.asarray(vetor_1, dtype=float)
    vetor_2 = np.asarray(vetor_2, dtype=float)

    diferencas = np.abs(vetor_1 - vetor_2) ** p
    soma_potencias = np.sum(diferencas)
    distancia = soma_potencias ** (1.0 / p)

    return distancia


def matriz_distancias_minkowski(dados, p=2):
    """
    Calcula a matriz de distâncias de Minkowski entre todos os pares de pontos.

    Parâmetros:

    dados : array, shape (n_instancias, n_atributos)
        Matriz de dados, onde cada linha é um exemplo/ponto.
    p : float >= 1
        Parâmetro da distância de Minkowski (p=1 -> Manhattan, p=2 -> Euclidiana).

    Retorno:

    numpy.ndarray, shape (n_instancias, n_instancias)
        Matriz de distâncias, em que a posição (i, j) contém a
        distância de Minkowski entre dados[i] e dados[j].
    """
    dados = np.asarray(dados, dtype=float)

    diferencas = np.abs(dados[:, None, :] - dados[None, :, :]) ** p
    soma_potencias = np.sum(diferencas, axis=2)
    matriz_distancias = soma_potencias ** (1.0 / p)

    return matriz_distancias


# Distância de Mahalanobis
def distancia_mahalanobis(vetor_1, vetor_2, matriz_cov_inv):
    """
    Calcula a distância de Mahalanobis entre dois vetores,
    dada a matriz de covariância inversa.

    Parâmetros:

    vetor_1 : array, shape (d,)
        Primeiro vetor.
    vetor_2 : array, shape (d,)
        Segundo vetor.
    matriz_cov_inv : numpy.ndarray, shape (d, d)
        Matriz inversa da covariância dos dados.

    Retorno:

    float
        Distância de Mahalanobis entre vetor_1 e vetor_2.
    """
    vetor_1 = np.asarray(vetor_1, dtype=float)
    vetor_2 = np.asarray(vetor_2, dtype=float)
    matriz_cov_inv = np.asarray(matriz_cov_inv, dtype=float)

    diferenca = vetor_1 - vetor_2
    
    distancia_quadrado = diferenca.T @ matriz_cov_inv @ diferenca
    distancia = np.sqrt(distancia_quadrado)

    return distancia


def matriz_distancias_mahalanobis(dados, matriz_cov=None):
    """
    Calcula a matriz de distâncias de Mahalanobis entre todos os pares de pontos.

    Parâmetros:

    dados : array, shape (n_instancias, n_atributos)
        Matriz de dados, onde cada linha é um exemplo/ponto.
    matriz_cov : numpy.ndarray, opcional (default=None)
        Matriz de covariância dos dados. Caso None, será estimada via np.cov
        a partir de 'dados'.

    Retorno:

    numpy.ndarray, shape (n_instancias, n_instancias)
        Matriz de distâncias de Mahalanobis entre todos os pares de pontos.
    """
    dados = np.asarray(dados, dtype=float)

    if matriz_cov is None:
        matriz_cov = np.cov(dados, rowvar=False)

    matriz_cov_inv = inv(matriz_cov)

    diferencas = dados[:, None, :] - dados[None, :, :]

    termo_esquerdo = diferencas @ matriz_cov_inv

    distancias_quadrado = np.sum(termo_esquerdo * diferencas, axis=2)
    matriz_distancias = np.sqrt(distancias_quadrado)

    return matriz_distancias


def dist_minkowski(a, b, p=2):
    return np.sum(np.abs(a - b) ** p) ** (1/p)


def distancias_para_um_ponto(X, ponto, p=2):
    return np.sum(np.abs(X - ponto) ** p, axis=1) ** (1/p)