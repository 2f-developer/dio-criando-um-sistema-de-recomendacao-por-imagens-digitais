"""Treina o classificador de produtos, monta o catálogo de embeddings e mede a qualidade das recomendações.

    python -m recomendacao.treinar

Baixa o Caltech-101 (~130 MB) na primeira execução. Gera:
    modelo/extrator.keras         rede que transforma uma foto em embedding
    catalogo/catalogo.npz         embeddings e miniaturas do catálogo (fica fora do git)
    resultados/metricas.json      precisão@5 antes e depois do treino
    resultados/exemplos.png       consultas e as 5 recomendações de cada uma
"""

import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dados import CATEGORIAS, NOMES_PT, carregar
from .indice import Indice
from .modelo import criar


def _embeddings_imagenet(base: keras.Model, imagens: np.ndarray) -> np.ndarray:
    """Linha de base: vetores da MobileNetV2 só com o treino da ImageNet, sem ajuste nenhum."""
    x = keras.applications.mobilenet_v2.preprocess_input(imagens.astype("float32"))
    return base.predict(x, verbose=0).mean(axis=(1, 2))


def executar(epocas=(6, 4), k: int = 5) -> dict:
    keras.utils.set_random_seed(42)
    dados = carregar()
    (x_tr, y_tr), (x_te, y_te) = dados["train"], dados["test"]
    classificador, extrator, base = criar()

    # Antes de treinar: quão boas são as recomendações só com a ImageNet?
    base_tr, base_te = _embeddings_imagenet(base, x_tr), _embeddings_imagenet(base, x_te)
    precisao_antes = Indice(base_tr, y_tr).precisao_em_k(base_te, y_te, k)

    classificador.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classificador.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=epocas[0], batch_size=32, verbose=2)
    base.trainable = True
    for camada in base.layers[:-30]:
        camada.trainable = False
    classificador.compile(optimizer=keras.optimizers.Adam(2e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classificador.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=sum(epocas), initial_epoch=epocas[0], batch_size=32, verbose=2)

    emb_tr, emb_te = extrator.predict(x_tr, verbose=0), extrator.predict(x_te, verbose=0)
    precisao_depois = Indice(emb_tr, y_tr).precisao_em_k(emb_te, y_te, k)
    acuracia = float((classificador.predict(x_te, verbose=0).argmax(1) == y_te).mean())

    Path("modelo").mkdir(exist_ok=True)
    Path("catalogo").mkdir(exist_ok=True)
    Path("resultados").mkdir(exist_ok=True)
    extrator.save("modelo/extrator.keras")
    catalogo_imgs, catalogo_rot = np.concatenate([x_tr, x_te]), np.concatenate([y_tr, y_te])
    np.savez_compressed(
        "catalogo/catalogo.npz",
        embeddings=np.concatenate([emb_tr, emb_te]).astype(np.float16),
        rotulos=catalogo_rot,
        miniaturas=np.stack([keras.ops.convert_to_numpy(keras.ops.image.resize(i, (96, 96))).astype(np.uint8) for i in catalogo_imgs]),
    )

    metricas = {
        f"precisao_em_{k}_so_imagenet": round(precisao_antes, 4),
        f"precisao_em_{k}_depois_do_treino": round(precisao_depois, 4),
        "acuracia_classificacao_teste": round(acuracia, 4),
        "categorias": [NOMES_PT[c] for c in CATEGORIAS],
        "imagens": {"treino": len(y_tr), "teste": len(y_te)},
    }
    Path("resultados/metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    _exemplos(x_te, y_te, emb_te, Indice(emb_tr, y_tr), x_tr, k)
    return metricas


def _exemplos(x_te, y_te, emb_te, indice, x_tr, k, n=4):
    rng = np.random.default_rng(7)
    consultas = [rng.choice(np.flatnonzero(y_te == c)) for c in rng.choice(len(CATEGORIAS), n, replace=False)]
    fig, eixos = plt.subplots(n, k + 1, figsize=(2.1 * (k + 1), 2.3 * n))
    for linha, q in enumerate(consultas):
        eixos[linha, 0].imshow(x_te[q])
        eixos[linha, 0].set_title(f"busca: {NOMES_PT[CATEGORIAS[y_te[q]]]}", fontsize=9, fontweight="bold")
        ordem, sims = indice.buscar(emb_te[q], k)
        for coluna, (i, s) in enumerate(zip(ordem, sims, strict=True), start=1):
            certo = indice.rotulos[i] == y_te[q]
            eixos[linha, coluna].imshow(x_tr[i])
            eixos[linha, coluna].set_title(f"{s:.2f}", fontsize=9, color="green" if certo else "red")
    for eixo in eixos.ravel():
        eixo.axis("off")
    fig.suptitle("Cada linha: a foto buscada e as 5 recomendações (similaridade de cosseno)", fontsize=10)
    fig.tight_layout()
    fig.savefig("resultados/exemplos.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(json.dumps(executar(), indent=2, ensure_ascii=False))
