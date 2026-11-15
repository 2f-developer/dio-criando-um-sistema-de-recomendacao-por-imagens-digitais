"""Fotos de produtos do Caltech-101 (TensorFlow Datasets), só das categorias que lembram um catálogo de loja."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TAMANHO = 160
CATEGORIAS = ("watch", "laptop", "headphone", "camera", "chair", "cup", "lamp", "umbrella", "scissors", "cellphone")
NOMES_PT = {
    "watch": "relógio",
    "laptop": "notebook",
    "headphone": "fone de ouvido",
    "camera": "câmera",
    "chair": "cadeira",
    "cup": "caneca",
    "lamp": "luminária",
    "umbrella": "guarda-chuva",
    "scissors": "tesoura",
    "cellphone": "celular",
}


def carregar(pasta_cache: str | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Devolve {"train": (imagens, rotulos), "test": (...)} com rótulos de 0 a 9 na ordem de CATEGORIAS."""
    divisoes, info = tfds.load("caltech101", split=["train", "test"], with_info=True, as_supervised=True, data_dir=pasta_cache)
    nomes = info.features["label"].names
    mapa = {nomes.index(c): i for i, c in enumerate(CATEGORIAS)}
    saida = {}
    for nome, ds in zip(("train", "test"), divisoes, strict=True):
        imagens, rotulos = [], []
        for imagem, rotulo in tfds.as_numpy(ds):
            if int(rotulo) in mapa:
                imagens.append(tf.image.resize_with_pad(imagem, TAMANHO, TAMANHO).numpy().astype(np.uint8))
                rotulos.append(mapa[int(rotulo)])
        saida[nome] = (np.stack(imagens), np.array(rotulos))
    return saida
