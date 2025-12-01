import numpy as np
from distancias import distancias_para_um_ponto


# Funções auxiliares
def calcular_raio(matriz_distancias, indices_centros):
    """
    Calcula o raio (máxima distância de cada ponto até seu centro mais próximo)
    dado um conjunto de centros.

    Parâmetros:

    matriz_distancias : numpy.ndarray, shape (n_instancias, n_instancias)
        Matriz de distâncias pré-computada.
    indices_centros : lista de int
        Lista com os índices das instâncias escolhidas como centros.

    Retorno:

    float
        Raio da solução (máxima distância ponto -> centro mais próximo).
    """
    dados = np.asarray(matriz_distancias, dtype=float)
    n_instancias = dados.shape[0]
    distancias_minimas = np.full(n_instancias, np.inf)

    for indice_centro in indices_centros:
        dists = distancias_para_um_ponto(dados, dados[indice_centro])
        distancias_minimas = np.minimum(distancias_minimas, dists)

    raio = float(np.max(distancias_minimas))
    return raio


def atribuir_pontos_a_centros(matriz_distancias, indices_centros):
    """
    Atribui cada ponto ao centro mais próximo.

    Parâmetros:

    matriz_distancias : numpy.ndarray, shape (n_instancias, n_instancias)
        Matriz de distâncias pré-computada.
    indices_centros : lista de int
        Lista com os índices das instâncias escolhidas como centros.

    Retorno:

    numpy.ndarray, shape (n_instancias,)
        Vetor em que a posição i contém o índice do centro (em indices_centros)
        ao qual o ponto i foi atribuído.
    """
    dados = np.asarray(matriz_distancias, dtype=float)
    indices_centros = list(indices_centros)
    n_instancias = dados.shape[0]
    n_centros = len(indices_centros)

    distancias_para_centros = np.empty((n_instancias, n_centros), dtype=float)
    for j, idx_centro in enumerate(indices_centros):
        distancias_para_centros[:, j] = distancias_para_um_ponto(dados, dados[idx_centro])

    indices_centros_locais = np.argmin(distancias_para_centros, axis=1)

    return indices_centros_locais


# Algoritmo guloso de k-centers (2-aproximado)
def k_centros_guloso(matriz_distancias, k, indice_inicial=None):
    """
    Implementa o algoritmo guloso 2-aproximado para o problema dos k-centros.

    Parâmetros:

    matriz_distancias : numpy.ndarray, shape (n_instancias, n_instancias)
        Matriz de distâncias pré-computada.
    k : int
        Número de centros (clusters) desejados.
    indice_inicial : int, opcional (default=None)
        Índice do ponto a ser usado como primeiro centro.
        Se None, será escolhido aleatoriamente.

    Retorno:

    indices_centros : list of int
        Lista com os índices dos centros escolhidos.
    raio : float
        Raio da solução obtida.
    atribuicoes : numpy.ndarray, shape (n_instancias,)
        Vetor com a atribuição de cada ponto ao centro mais próximo.
        O valor em i é o índice (na lista indices_centros) do centro associado.
    """
    dados = np.asarray(matriz_distancias, dtype=float)
    n_instancias = dados.shape[0]

    if indice_inicial is None:
        indice_inicial = np.random.randint(0, n_instancias)

    indices_centros = [indice_inicial]

    distancias_minimas = distancias_para_um_ponto(dados, dados[indice_inicial])

    for _ in range(1, k):
        novo_centro = int(np.argmax(distancias_minimas))
        indices_centros.append(novo_centro)
        dists_novo = distancias_para_um_ponto(dados, dados[novo_centro])
        distancias_minimas = np.minimum(distancias_minimas, dists_novo)

    raio = float(np.max(distancias_minimas))
    atribuicoes = atribuir_pontos_a_centros(dados, indices_centros)

    return indices_centros, raio, atribuicoes


# Teste de viabilidade para um dado raio
def verificar_raio_viavel(matriz_distancias, k, raio):
    """
    Verifica se é possível cobrir todos os pontos com k centros,
    dado um raio máximo, e também devolve os centros utilizados
    na tentativa de cobertura.

    Parâmetros:

    matriz_distancias : numpy.ndarray, shape (n_instancias, n_instancias)
        Matriz de distâncias pré-computada.
    k : int
        Número de centros disponíveis.
    raio : float
        Raio máximo a ser testado.

    Retorno:

    viavel : bool
        True se o raio permite cobrir todos os pontos com k centros, False caso contrário.
    indices_centros : list of int
        Lista de índices escolhidos como centros durante o teste.
        Se viavel == False, corresponde à cobertura parcial obtida.
    """
    dados = np.asarray(matriz_distancias, dtype=float)
    n_instancias = dados.shape[0]
    nao_cobertos = set(range(n_instancias))
    indices_centros = []

    while nao_cobertos and len(indices_centros) < k:
        indice_centro = next(iter(nao_cobertos))
        indices_centros.append(indice_centro)

        dists = distancias_para_um_ponto(dados, dados[indice_centro])
        pontos_a_remover = {i for i in nao_cobertos if dists[i] <= raio}
        nao_cobertos -= pontos_a_remover

    viavel = (len(nao_cobertos) == 0)
    return viavel, indices_centros


# Algoritmo 2-aproximado por refinamento de intervalo no raio
def k_centros_busca_intervalo(matriz_distancias, k, proporcao_largura=0.05, max_iteracoes=100):
    """
    Algoritmo 2-aproximado baseado em refinamento de intervalo para o raio ótimo.

    Parâmetros:

    matriz_distancias : numpy.ndarray
        Matriz de distâncias pré-computada.
    k : int
        Número de centros.
    proporcao_largura : float
        Proporção da largura inicial que define a largura final mínima do intervalo.
    max_iteracoes : int
        Máximo de iterações permitidas no refinamento do intervalo.

    Retorno:

    raio_aproximado : float ou None
        Raio aproximado encontrado. None caso nenhum teste viável tenha ocorrido.
    melhores_centros : list[int] ou None
        Centros encontrados no último teste viável. None se nenhum encontro viável ocorrer.
    """
    dados = np.asarray(matriz_distancias, dtype=float)
    n_instancias = dados.shape[0]

    if n_instancias <= 50:
        indices_amostra = np.arange(n_instancias)
    else:
        rng = np.random.default_rng(0)
        indices_amostra = rng.choice(n_instancias, size=50, replace=False)

    limite_inferior = 0.0
    limite_superior = 0.0

    for idx in indices_amostra:
        dists = distancias_para_um_ponto(dados, dados[idx])
        max_local = float(np.max(dists))
        if max_local > limite_superior:
            limite_superior = max_local

    largura_inicial = limite_superior - limite_inferior
    largura_alvo = largura_inicial * proporcao_largura

    melhores_centros = None
    iteracao = 0

    while (limite_superior - limite_inferior) > largura_alvo and iteracao < max_iteracoes:
        iteracao += 1

        raio_teste = (limite_inferior + limite_superior) / 2.0
        viavel, centros_teste = verificar_raio_viavel(dados, k, raio_teste)

        if viavel:
            limite_superior = raio_teste
            melhores_centros = centros_teste
        else:
            limite_inferior = raio_teste

    if melhores_centros is None:
        return None, None

    raio_aproximado = limite_superior
    return raio_aproximado, melhores_centros