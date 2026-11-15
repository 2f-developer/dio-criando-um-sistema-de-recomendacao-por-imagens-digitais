"""Recomenda produtos parecidos com uma foto qualquer.

    python -m recomendacao.recomendar FOTO [--k 5] [--saida recomendacoes.png]

Precisa do modelo e do catálogo gerados por `python -m recomendacao.treinar`.
"""

import argparse
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dados import CATEGORIAS, NOMES_PT, TAMANHO
from .indice import Indice


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m recomendacao.recomendar")
    parser.add_argument("foto", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--saida", type=Path, default=Path("recomendacoes.png"))
    args = parser.parse_args(argv)

    extrator = keras.models.load_model("modelo/extrator.keras")
    catalogo = np.load("catalogo/catalogo.npz")
    indice = Indice(catalogo["embeddings"], catalogo["rotulos"])

    imagem = keras.utils.img_to_array(keras.utils.load_img(args.foto))
    imagem = keras.ops.convert_to_numpy(keras.ops.image.resize(imagem, (TAMANHO, TAMANHO), pad_to_aspect_ratio=True))
    consulta = extrator.predict(imagem[None], verbose=0)[0]
    ordem, sims = indice.buscar(consulta, args.k)

    fig, eixos = plt.subplots(1, args.k + 1, figsize=(2.2 * (args.k + 1), 2.6))
    eixos[0].imshow(imagem.astype(np.uint8))
    eixos[0].set_title("sua foto", fontweight="bold")
    for eixo, i, s in zip(eixos[1:], ordem, sims, strict=True):
        categoria = NOMES_PT[CATEGORIAS[catalogo["rotulos"][i]]]
        eixo.imshow(catalogo["miniaturas"][i])
        eixo.set_title(f"{categoria}\n{s:.2f}", fontsize=9)
        print(f"{categoria:15s} similaridade {s:.3f}")
    for eixo in eixos:
        eixo.axis("off")
    fig.tight_layout()
    fig.savefig(args.saida, dpi=100)
    print(f"imagem salva em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
