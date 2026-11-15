"""Classificador de produtos cuja penúltima camada serve de "assinatura visual" (embedding)."""

import keras
from keras import layers

from .dados import CATEGORIAS, TAMANHO


def criar(pesos: str | None = "imagenet") -> tuple[keras.Model, keras.Model, keras.Model]:
    """Devolve (classificador, extrator de embeddings, base). O extrator compartilha as camadas do classificador."""
    base = keras.applications.MobileNetV2(input_shape=(TAMANHO, TAMANHO, 3), include_top=False, weights=pesos)
    base.trainable = False
    entrada = keras.Input((TAMANHO, TAMANHO, 3), name="foto_0_255")
    x = layers.RandomFlip("horizontal")(entrada)
    x = layers.RandomRotation(0.08)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Dense(256, activation="relu", name="embedding")(x)
    saida = layers.Dense(len(CATEGORIAS), activation="softmax", name="categoria")(layers.Dropout(0.3)(embedding))
    classificador = keras.Model(entrada, saida, name="classificador_produtos")
    extrator = keras.Model(entrada, embedding, name="extrator_embeddings")
    return classificador, extrator, base
