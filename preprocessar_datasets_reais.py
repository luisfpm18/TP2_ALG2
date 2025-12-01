"""
Pré-processamento dos 10 datasets reais.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from io import StringIO


DATASETS_CONFIG: List[Dict[str, Any]] = [
    # 1) Predict students dropout
    {
        "nome_curto": "students_dropout",
        "arquivo": "data.csv",
        "coluna_classe": "Target",
        "tipo": "csv_semicolon_header",
    },

    # 2) Default of Credit Card Clients
    {
        "nome_curto": "default_credit_card",
        "arquivo": "default_of_credit_card_clients.xls",
        "coluna_classe": "default payment next month",
        "tipo": "excel",
        "forcar_numerico": True,
        "header": 1,
    },

    # 3) German Credit
    {
        "nome_curto": "german_credit",
        "arquivo": "german.csv",
        "coluna_classe": "class",
        "tipo": "german_space",
    },

    # 4) Obesity
    {
        "nome_curto": "obesity",
        "arquivo": "ObesityDataSet_raw_and_data_sinthetic.csv",
        "coluna_classe": "NObeyesdad",
        "tipo": "csv_header",
    },

    # 5) Spambase (csv sem header, última coluna é classe)
    {
        "nome_curto": "spambase",
        "arquivo": "spambase.csv",
        "coluna_classe": -1,
        "tipo": "csv_sem_header",
    },

    # 6) Rice Cammeo Osmancik
    {
        "nome_curto": "rice",
        "arquivo": "Rice_Cammeo_Osmancik.arff",
        "coluna_classe": -1,
        "tipo": "arff_like",
    },

    # 7) Banknote Authentication (csv sem header, última coluna é classe)
    {
        "nome_curto": "banknote",
        "arquivo": "data_banknote_authentication.csv",
        "coluna_classe": -1,
        "tipo": "csv_sem_header",
    },

    # 8) Letter Recognition (csv sem header, primeira coluna é classe)
    {
        "nome_curto": "letter",
        "arquivo": "letter-recognition.csv",
        "coluna_classe": 0,
        "tipo": "csv_sem_header",
    },

    # 9) Maternal Health Risk (csv com header)
    {
        "nome_curto": "maternal_health",
        "arquivo": "Maternal Health Risk Data Set.csv",
        "coluna_classe": "RiskLevel",
        "tipo": "csv_header",
    },

    # 10) Raisin Dataset
    {
        "nome_curto": "raisin",
        "arquivo": "Raisin_Dataset.xlsx",
        "coluna_classe": "Class",
        "tipo": "excel",
    },
]


# Função de carregamento
def carregar_dataset(config: Dict[str, Any], pasta_base: str) -> pd.DataFrame:
    caminho = os.path.join(pasta_base, config["arquivo"])
    tipo = config["tipo"]

    print(f"\nCarregando {config['nome_curto']} de {caminho}  (tipo={tipo})")

    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    if tipo == "csv_header":
        df = pd.read_csv(caminho)

    elif tipo == "csv_semicolon_header":
        df = pd.read_csv(caminho, sep=";")

    elif tipo == "csv_sem_header":
        df = pd.read_csv(caminho, header=None)

    elif tipo == "excel":
        # Usa header customizado se existir
        header_row = config.get("header", 0)
        df = pd.read_excel(caminho, header=header_row)


    elif tipo == "german_space":
        raw = pd.read_csv(caminho, header=None, sep=r"\s+")
        n = raw.shape[1]
        raw.columns = [f"feat_{i}" for i in range(n - 1)] + ["class"]
        df = raw

    elif tipo == "arff_like":
        linhas = []
        in_data = False
        with open(caminho, encoding="latin-1") as f:
            for linha in f:
                strip = linha.strip()
                if not in_data:
                    if strip.upper() == "@DATA":
                        in_data = True
                    continue
                if strip == "" or strip.startswith("%"):
                    continue
                linhas.append(strip)
        texto = "\n".join(linhas)
        df = pd.read_csv(StringIO(texto), header=None)

    else:
        raise ValueError(f"Tipo de dataset desconhecido: {tipo}")

    print(f"  -> shape bruto: {df.shape}")
    return df


# Pré-processamento genérico
def preprocessar_df(df: pd.DataFrame, coluna_classe, forcar_numerico: bool = False):
    # Selecionar classe por nome ou por índice
    if isinstance(coluna_classe, int):
        y = df.iloc[:, coluna_classe].values
        X_df = df.drop(df.columns[coluna_classe], axis=1)
    else:
        y = df[coluna_classe].values
        X_df = df.drop(columns=[coluna_classe])

    if forcar_numerico:
        # Tira coluna ID se existir (não é feature)
        if "ID" in X_df.columns:
            X_df = X_df.drop(columns=["ID"])

        # Converte tudo para numérico, forçando lixo para NaN
        X_num = X_df.apply(pd.to_numeric, errors="coerce").astype(float)

        # Se sobrar NaN, preenche com a média da coluna
        if X_num.isna().any().any():
            X_num = X_num.fillna(X_num.mean())

        X = X_num.values

    else:
        col_cat = X_df.select_dtypes(include=["object"]).columns
        col_num = X_df.select_dtypes(exclude=["object"]).columns

        X_num = X_df[col_num].astype(float) if len(col_num) > 0 else None

        if len(col_cat) > 0:
            enc = OrdinalEncoder()
            X_cat = enc.fit_transform(X_df[col_cat]).astype(float)
        else:
            X_cat = None

        if X_num is not None and X_cat is not None:
            X = np.hstack([X_num.values, X_cat])
        elif X_num is not None:
            X = X_num.values
        else:
            X = X_cat

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    k = len(np.unique(y))
    print(f"  -> X shape: {X.shape}, y shape: {y.shape}, k = {k}")
    return X, y, k


# Loop principal: percorre todos e salva .npz
def preprocessar_todos(pasta_base: str = ".", pasta_saida: str = "dados_reais_preprocessados"):
    os.makedirs(pasta_saida, exist_ok=True)

    for config in DATASETS_CONFIG:
        try:
            df = carregar_dataset(config, pasta_base)
            X, y, k = preprocessar_df(
                df,
                config["coluna_classe"],
                config.get("forcar_numerico", False)
            )
            caminho_npz = os.path.join(pasta_saida, f"{config['nome_curto']}.npz")
            np.savez(caminho_npz, X=X, y=y, k=k)
            print(f"  -> Salvo em: {caminho_npz}")
        except Exception as e:
            print(f"\n[ERRO] Ao processar {config['nome_curto']}: {e}")


if __name__ == "__main__":
    preprocessar_todos(pasta_base="datasets_reais")