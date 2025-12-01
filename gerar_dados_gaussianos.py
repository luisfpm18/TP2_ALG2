"""
Geração de 10 conjuntos de dados sintéticos 2D com distribuição
normal multivariada, conforme o enunciado do trabalho:

- Clusters com diferentes médias
- Diferentes desvios padrão -> variando sobreposição (baixa, média, alta)
- Clusters circulares e elípticos (covariância controlada)
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np


def gerar_dataset_gaussiano(
    lista_medias,
    lista_covariancias,
    n_por_cluster: int,
    gerador: np.random.RandomState,
):
    """
    Gera um único conjunto de dados gaussianos multivariados 2D.

    Parâmetros:

    lista_medias : list[np.ndarray]
        Lista de vetores de médias (cada um com shape (2,)).
    lista_covariancias : list[np.ndarray]
        Lista de matrizes de covariância 2x2, uma para cada média.
    n_por_cluster : int
        Número de exemplos gerados para cada cluster.
    gerador : np.random.RandomState
        Gerador de números aleatórios (para reprodutibilidade).

    Retorno:

    X : np.ndarray, shape (n_total, 2)
        Matriz de dados (pontos 2D) empilhados.
    y : np.ndarray, shape (n_total,)
        Vetor de rótulos inteiros (0, 1, 2, ...), indicando o cluster de origem.
    """
    X_lista = []
    y_lista = []

    for indice_cluster, (media, cov) in enumerate(zip(lista_medias, lista_covariancias)):
        X_cluster = gerador.multivariate_normal(
            mean=np.asarray(media, dtype=float),
            cov=np.asarray(cov, dtype=float),
            size=n_por_cluster,
        )
        y_cluster = np.full(n_por_cluster, indice_cluster, dtype=int)

        X_lista.append(X_cluster)
        y_lista.append(y_cluster)

    X = np.vstack(X_lista)
    y = np.concatenate(y_lista)

    return X, y


def gerar_10_datasets_gaussianos(
    n_por_cluster: int = 400,
    semente_base: int = 42,
) -> List[Dict[str, Any]]:
    """
    Gera 10 conjuntos de dados gaussianos multivariados 2D.

    Cada conjunto de dados é definido por:
        - número de clusters (2 ou 3)
        - médias dos clusters
        - matrizes de covariância (circulares ou elípticas)
        - grau de sobreposição (baixa, média, alta)

    Parâmetros:

    n_por_cluster : int
        Número de exemplos por cluster (mínimo recomendado: +-350
        para garantir >= 700 pontos mesmo com 2 clusters).
    semente_base : int
        Semente base para o gerador de números aleatórios.

    Retorno:

    lista_conjuntos : list[dict]
        Lista com 10 dicionários, cada um contendo:
            - "nome": str
            - "X": np.ndarray, shape (n_total, 2)
            - "y": np.ndarray, shape (n_total,)
            - "descricao": str (texto explicando o cenário)
    """
    gerador = np.random.RandomState(semente_base)
    lista_conjuntos: List[Dict[str, Any]] = []

    # 1) 2 clusters circulares, bem separados
    medias = [np.array([-5.0, 0.0]), np.array([5.0, 0.0])]
    sigma = 0.5
    covs = [np.eye(2) * sigma**2, np.eye(2) * sigma**2]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_circular_separado",
            "X": X,
            "y": y,
            "descricao": "2 clusters circulares bem separados (baixa sobreposição).",
        }
    )

    # 2) 2 clusters circulares, sobreposição leve
    medias = [np.array([-3.0, 0.0]), np.array([3.0, 0.0])]
    sigma = 1.0
    covs = [np.eye(2) * sigma**2, np.eye(2) * sigma**2]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_circular_sobrepos_leve",
            "X": X,
            "y": y,
            "descricao": "2 clusters circulares com sobreposição leve.",
        }
    )

    # 3) 2 clusters circulares, sobreposição alta
    medias = [np.array([-2.0, 0.0]), np.array([2.0, 0.0])]
    sigma = 1.8
    covs = [np.eye(2) * sigma**2, np.eye(2) * sigma**2]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_circular_sobrepos_alta",
            "X": X,
            "y": y,
            "descricao": "2 clusters circulares com alta sobreposição.",
        }
    )

    # 4) 3 clusters circulares, separação moderada
    medias = [
        np.array([0.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([0.0, 5.0]),
    ]
    sigma = 1.0
    covs = [np.eye(2) * sigma**2 for _ in medias]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_circular_3_clusters",
            "X": X,
            "y": y,
            "descricao": "3 clusters circulares moderadamente separados.",
        }
    )

    # 5) 3 clusters circulares, variâncias bem diferentes
    medias = [
        np.array([0.0, 0.0]),
        np.array([6.0, 0.0]),
        np.array([0.0, 6.0]),
    ]
    sigmas = [0.4, 1.0, 2.0]
    covs = [np.eye(2) * s**2 for s in sigmas]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_circular_var_densidade",
            "X": X,
            "y": y,
            "descricao": "3 clusters circulares com densidades/sigmas bem diferentes.",
        }
    )

    # 6) 2 clusters elípticos, bem separados
    medias = [np.array([-6.0, -2.0]), np.array([6.0, 2.0])]
    covs = [
        np.array([[3.0, 0.0],
                  [0.0, 0.5]]),
        np.array([[3.0, 0.0],
                  [0.0, 0.5]]),
    ]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_eliptico_separado",
            "X": X,
            "y": y,
            "descricao": "2 clusters elípticos bem separados (eixos alongados).",
        }
    )

    # 7) 2 clusters elípticos, sobreposição leve
    medias = [np.array([-4.0, -2.0]), np.array([4.0, 2.0])]
    covs = [
        np.array([[3.0, 1.0],
                  [1.0, 1.0]]),
        np.array([[3.0, 1.0],
                  [1.0, 1.0]]),
    ]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_eliptico_sobrepos_leve",
            "X": X,
            "y": y,
            "descricao": "2 clusters elípticos com sobreposição leve.",
        }
    )

    # 8) 2 clusters elípticos, alta sobreposição
    medias = [np.array([-2.0, -1.0]), np.array([2.0, 1.0])]
    covs = [
        np.array([[4.0, 1.5],
                  [1.5, 2.0]]),
        np.array([[4.0, 1.5],
                  [1.5, 2.0]]),
    ]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_eliptico_sobrepos_alta",
            "X": X,
            "y": y,
            "descricao": "2 clusters elípticos com alta sobreposição.",
        }
    )

    # 9) 3 clusters elípticos, orientações diferentes
    medias = [
        np.array([0.0, 0.0]),
        np.array([5.0, 5.0]),
        np.array([-5.0, 5.0]),
    ]
    covs = [
        np.array([[2.0, 1.5],
                  [1.5, 2.0]]),
        np.array([[1.0, -0.8],
                  [-0.8, 2.0]]),
        np.array([[3.0, 0.5],
                  [0.5, 1.0]]),
    ]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_eliptico_3_orientacoes",
            "X": X,
            "y": y,
            "descricao": "3 clusters elípticos com diferentes orientações.",
        }
    )

    # 10) 3 clusters mistos (um circular, dois elípticos)
    medias = [
        np.array([0.0, 0.0]),    # circular
        np.array([6.0, 0.0]),    # elíptico 1
        np.array([0.0, 6.0]),    # elíptico 2
    ]
    covs = [
        np.eye(2) * 0.8**2,      # circular
        np.array([[3.0, 1.0],
                  [1.0, 0.7]]),  # elíptico
        np.array([[2.0, -1.0],
                  [-1.0, 1.5]]),  # elíptico
    ]
    X, y = gerar_dataset_gaussiano(medias, covs, n_por_cluster, gerador)
    lista_conjuntos.append(
        {
            "nome": "gauss_misto_circ_eliptico",
            "X": X,
            "y": y,
            "descricao": "3 clusters: 1 circular e 2 elípticos com sobreposição moderada.",
        }
    )

    return lista_conjuntos


def salvar_datasets_gaussianos_npz(
    lista_conjuntos: List[Dict[str, Any]],
    pasta_saida: str = "dados_gaussianos_multivariados",
) -> None:
    """
    Salva cada conjunto gaussiano em um arquivo .npz (X, y) na pasta indicada.

    O nome dos arquivos segue o padrão:
        <nome>.npz
    Exemplo:
        gauss_circular_separado.npz

    Parâmetros:

    lista_conjuntos : list[dict]
        Lista retornada por gerar_10_datasets_gaussianos.
    pasta_saida : str
        Diretório onde os arquivos .npz serão salvos.
    """
    os.makedirs(pasta_saida, exist_ok=True)

    for conjunto in lista_conjuntos:
        nome = conjunto["nome"]
        X = conjunto["X"]
        y = conjunto["y"]

        caminho = os.path.join(pasta_saida, f"{nome}.npz")
        np.savez(caminho, X=X, y=y)

        print(f"Salvo: {caminho}")


if __name__ == "__main__":
    conjuntos_gauss = gerar_10_datasets_gaussianos(n_por_cluster=400)
    salvar_datasets_gaussianos_npz(conjuntos_gauss)