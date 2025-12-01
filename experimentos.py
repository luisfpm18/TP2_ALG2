import os
import time
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

from kcentros import (
    k_centros_guloso,
    k_centros_busca_intervalo,
    atribuir_pontos_a_centros,
    calcular_raio,
)


# Parâmetros globais
N_EXECUCOES = 15

LARGURAS_INTERVALO = [0.01, 0.05, 0.10, 0.15, 0.25]

PASTAS_DADOS = [
    "dados_reais_preprocessados",
    "dados_sinteticos_sklearn",
    "dados_gaussianos_multivariados",
]

PASTA_SAIDA = "resultados"
os.makedirs(PASTA_SAIDA, exist_ok=True)


# Funções auxiliares
def carregar_npz(caminho):
    data = np.load(caminho, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    if isinstance(y, np.ndarray) and y.ndim == 0:
        if y.shape == () and y.dtype == object and y.item() is None:
            y = None
        else:
            y = np.array([y.item()])
    if "k" in data:
        return X, y, int(data["k"])
    return X, y, None


def executar_kcenters_guloso(X, y, k):
    """Executa o algoritmo guloso 2-aproximado."""
    inicio = time.time()
    centros, raio, atribuicoes = k_centros_guloso(X, k)
    tempo = time.time() - inicio

    rotulos = atribuicoes
    labels_unicos = np.unique(rotulos)

    if labels_unicos.size < 2 or labels_unicos.size >= X.shape[0]:
        sil = np.nan
    else:
        if X.shape[0] > 5000:
            amostras = np.random.choice(X.shape[0], size=5000, replace=False)
            rot_sub = rotulos[amostras]
            if np.unique(rot_sub).size < 2:
                sil = np.nan
            else:
                sil = silhouette_score(X[amostras], rot_sub)
        else:
            sil = silhouette_score(X, rotulos)

    if y is None:
        ari = np.nan
    else:
        ari = adjusted_rand_score(y, rotulos)

    return {
        "raio": raio,
        "silhouette": sil,
        "ari": ari,
        "tempo": tempo,
        "rotulos": rotulos,
    }


def executar_kcenters_intervalo(X, y, k, proporcao):
    """Executa algoritmo por refinamento de intervalo."""
    inicio = time.time()
    raio, centros = k_centros_busca_intervalo(
        X, k, proporcao_largura=proporcao
    )
    tempo = time.time() - inicio

    if centros is None:
        return {
            "raio": None,
            "silhouette": None,
            "ari": None,
            "tempo": tempo,
            "rotulos": None,
        }

    atribuicoes = atribuir_pontos_a_centros(X, centros)
    rotulos = atribuicoes

    labels_unicos = np.unique(rotulos)

    if labels_unicos.size < 2 or labels_unicos.size >= X.shape[0]:
        sil = np.nan
    else:
        if X.shape[0] > 5000:
            amostras = np.random.choice(X.shape[0], size=5000, replace=False)
            rot_sub = rotulos[amostras]
            if np.unique(rot_sub).size < 2:
                sil = np.nan
            else:
                sil = silhouette_score(X[amostras], rot_sub)
        else:
            sil = silhouette_score(X, rotulos)

    if y is None:
        ari = np.nan
    else:
        ari = adjusted_rand_score(y, rotulos)

    return {
        "raio": raio,
        "silhouette": sil,
        "ari": ari,
        "tempo": tempo,
        "rotulos": rotulos,
    }


def executar_kmeans(X, y, k, seed):
    """Executa K-Means e coleta métricas."""
    modelo = KMeans(n_clusters=k, random_state=seed, n_init=10)

    inicio = time.time()
    rotulos = modelo.fit_predict(X)
    tempo = time.time() - inicio

    labels_unicos = np.unique(rotulos)

    if labels_unicos.size < 2 or labels_unicos.size >= X.shape[0]:
        sil = np.nan
    else:
        if X.shape[0] > 5000:
            amostras = np.random.choice(X.shape[0], size=5000, replace=False)
            rot_sub = rotulos[amostras]
            if np.unique(rot_sub).size < 2:
                sil = np.nan
            else:
                sil = silhouette_score(X[amostras], rot_sub)
        else:
            sil = silhouette_score(X, rotulos)

    if y is None:
        ari = np.nan
    else:
        ari = adjusted_rand_score(y, rotulos)

    return {
        "raio": None,
        "silhouette": sil,
        "ari": ari,
        "tempo": tempo,
        "rotulos": rotulos,
    }


# Loop principal de experimentos
def main():
    resultados = []

    for pasta in PASTAS_DADOS:
        for arquivo in os.listdir(pasta):
            if not arquivo.endswith(".npz"):
                continue

            caminho = os.path.join(pasta, arquivo)
            X, y, k = carregar_npz(caminho)

            if k is None:
                if y is None:
                    continue
                else:
                    k = len(np.unique(y))

            print(f"\n>>> Dataset: {arquivo} | k={k}")

            n = len(X)
            seeds = np.random.randint(0, 10_000, size=N_EXECUCOES)

            for i_exec in range(N_EXECUCOES):
                seed = seeds[i_exec]
                rng = np.random.default_rng(seed)
                idx = rng.permutation(n)

                X_exec = X[idx]
                if y is None:
                    y_exec = None
                else:
                    y_exec = y[idx]

                res_guloso = executar_kcenters_guloso(X_exec, y_exec, k)
                res_guloso["algoritmo"] = "kcenters_guloso"
                res_guloso["dataset"] = arquivo
                res_guloso["execucao"] = i_exec
                resultados.append(res_guloso)

                for prop in LARGURAS_INTERVALO:
                    res_int = executar_kcenters_intervalo(X_exec, y_exec, k, prop)
                    res_int["algoritmo"] = f"kcenters_intervalo_prop{prop}"
                    res_int["dataset"] = arquivo
                    res_int["execucao"] = i_exec
                    resultados.append(res_int)

                res_kmeans = executar_kmeans(X_exec, y_exec, k, seed)
                res_kmeans["algoritmo"] = "kmeans"
                res_kmeans["dataset"] = arquivo
                res_kmeans["execucao"] = i_exec
                resultados.append(res_kmeans)

    import csv

    caminho_csv = os.path.join(PASTA_SAIDA, "resultados.csv")

    colunas = [
        "dataset",
        "algoritmo",
        "execucao",
        "raio",
        "silhouette",
        "ari",
        "tempo",
    ]

    with open(caminho_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colunas)
        w.writeheader()

        for r in resultados:
            w.writerow({c: r.get(c) for c in colunas})

    print("\n>>> Resultados salvos em:", caminho_csv)


if __name__ == "__main__":
    main()