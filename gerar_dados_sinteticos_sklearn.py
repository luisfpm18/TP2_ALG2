"""
Geração dos 30 conjuntos de dados sintéticos tipo scikit-learn, conforme o exemplo:

https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

Para cada um dos 6 tipos de dados (noisy_circles, noisy_moons, blobs,
varied, aniso, no_structure), geraram-se 5 variações diferentes
(aleatoriedade + parâmetros), totalizando 30 datasets.
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn import datasets


def gerar_30_datasets_sinteticos(
    n_amostras: int = 1000,
    sementes_base: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Gera 30 conjuntos de dados sintéticos 2D inspirados no exemplo
    de comparação de algoritmos de clustering do scikit-learn.

    Para cada um dos 6 tipos de dados:
        - noisy_circles
        - noisy_moons
        - blobs (isotrópicos)
        - varied (blobs com variâncias diferentes)
        - aniso (blobs anisotrópicos)
        - no_structure (ruído puro)
    são geradas 5 variações diferentes.

    Parâmetros:

    n_amostras : int
        Número de exemplos em cada conjunto de dados.
        Deve ser >= 700.
    sementes_base : list[int], opcional
        Lista de 5 sementes para controlar a aleatoriedade.
        Se None, será usado [0, 1, 2, 3, 4].

    Retorno:

    lista_conjuntos : list[dict]
        Lista de dicionários, cada um com as chaves:
            - "tipo": str   (nome do tipo, ex.: "noisy_circles")
            - "id": int     (0..4, índice da variação)
            - "X": np.ndarray, shape (n_amostras, 2)
            - "y": np.ndarray ou None (rótulos verdadeiros, se existirem)
    """
    if sementes_base is None:
        sementes_base = [0, 1, 2, 3, 4]

    lista_conjuntos: List[Dict[str, Any]] = []

    lista_cluster_std_varied = [
        [1.0, 2.5, 0.5],
        [0.5, 2.0, 0.5],
        [1.5, 3.0, 0.3],
        [0.3, 1.0, 2.5],
        [2.0, 0.7, 0.7],
    ]

    lista_transformacoes_aniso = [
        np.array([[0.6, -0.6], [-0.4, 0.8]]),
        np.array([[1.0, 0.3], [-0.2, 0.5]]),
        np.array([[0.3, -0.9], [0.8, 0.4]]),
        np.array([[0.9, 0.1], [0.0, 0.7]]),
        np.array([[0.5, -0.2], [0.2, 1.0]]),
    ]

    for idx, semente in enumerate(sementes_base):
        # 1) noisy_circles
        # Variou-se o nível de ruído para criar conjuntos mais/menos difíceis
        ruido_circles = 0.03 + 0.01 * idx
        X, y = datasets.make_circles(
            n_samples=n_amostras,
            factor=0.5,
            noise=ruido_circles,
            random_state=semente,
        )
        lista_conjuntos.append(
            {"tipo": "noisy_circles", "id": idx, "X": X, "y": y}
        )

        # 2) noisy_moons
        ruido_moons = 0.03 + 0.015 * idx
        X, y = datasets.make_moons(
            n_samples=n_amostras,
            noise=ruido_moons,
            random_state=semente + 100,
        )
        lista_conjuntos.append(
            {"tipo": "noisy_moons", "id": idx, "X": X, "y": y}
        )

        # 3) blobs (isotrópicos)
        # Variou-se o desvio padrão para mudar o grau de sobreposição
        std_blobs = 0.7 + 0.2 * idx
        X, y = datasets.make_blobs(
            n_samples=n_amostras,
            centers=3,
            cluster_std=std_blobs,
            random_state=semente + 200,
        )
        lista_conjuntos.append(
            {"tipo": "blobs_isotropicos", "id": idx, "X": X, "y": y}
        )

        # 4) varied (blobs com variâncias diferentes)
        X, y = datasets.make_blobs(
            n_samples=n_amostras,
            centers=3,
            cluster_std=lista_cluster_std_varied[idx],
            random_state=semente + 300,
        )
        lista_conjuntos.append(
            {"tipo": "blobs_varied", "id": idx, "X": X, "y": y}
        )

        # 5) aniso (blobs anisotrópicos)
        X_base, y = datasets.make_blobs(
            n_samples=n_amostras,
            centers=3,
            cluster_std=1.0,
            random_state=semente + 400,
        )
        transformacao = lista_transformacoes_aniso[idx]
        X_aniso = X_base @ transformacao.T
        lista_conjuntos.append(
            {"tipo": "blobs_anisotropicos", "id": idx, "X": X_aniso, "y": y}
        )

        # 6) no_structure (ruído puro)
        rng = np.random.RandomState(semente + 500)
        X = rng.rand(n_amostras, 2)
        y = None  # sem rótulo verdadeiro (cenário de "dados sem estrutura")
        lista_conjuntos.append(
            {"tipo": "no_structure", "id": idx, "X": X, "y": y}
        )

    return lista_conjuntos


def salvar_conjuntos_npz(
    lista_conjuntos: List[Dict[str, Any]],
    pasta_saida: str = "dados_sinteticos_sklearn",
) -> None:
    """
    Salva cada conjunto em um arquivo .npz (X, y) na pasta indicada.

    O nome dos arquivos segue o padrão:
        <tipo>_id<id>.npz
    Exemplo:
        noisy_circles_id0.npz
        blobs_anisotropicos_id3.npz

    Parâmetros:

    lista_conjuntos : list[dict]
        Lista retornada por gerar_30_datasets_sinteticos.
    pasta_saida : str
        Diretório onde os arquivos .npz serão salvos.
    """
    os.makedirs(pasta_saida, exist_ok=True)

    for conjunto in lista_conjuntos:
        tipo = conjunto["tipo"]
        idx = conjunto["id"]
        X = conjunto["X"]
        y = conjunto["y"]

        nome_arquivo = f"{tipo}_id{idx}.npz"
        caminho = os.path.join(pasta_saida, nome_arquivo)
        np.savez(caminho, X=X, y=y)

        print(f"Salvo: {caminho}")


if __name__ == "__main__":
    conjuntos = gerar_30_datasets_sinteticos(n_amostras=1000)
    salvar_conjuntos_npz(conjuntos)