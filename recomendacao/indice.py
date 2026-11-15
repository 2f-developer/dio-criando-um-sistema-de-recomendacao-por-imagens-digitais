"""Busca por semelhança: cada imagem vira um vetor (embedding) e a recomendação é a dos vetores mais próximos."""

import numpy as np


def normalizar(vetores: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(vetores, axis=1, keepdims=True)
    return vetores / np.maximum(normas, 1e-12)


class Indice:
    def __init__(self, embeddings: np.ndarray, rotulos: np.ndarray) -> None:
        self.embeddings = normalizar(embeddings.astype(np.float32))
        self.rotulos = rotulos

    def buscar(self, consulta: np.ndarray, k: int = 5, excluir: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Índices e similaridades (cosseno) dos k itens mais parecidos.

        `excluir` tira o próprio item da resposta quando a consulta já faz parte do catálogo:
        recomendar ao cliente o produto que ele está olhando não ajuda.
        """
        similaridades = self.embeddings @ normalizar(consulta.reshape(1, -1)).ravel()
        if excluir is not None:
            similaridades[excluir] = -np.inf
        melhores = np.argpartition(-similaridades, min(k, len(similaridades) - 1))[:k]
        ordem = melhores[np.argsort(-similaridades[melhores])]
        return ordem, similaridades[ordem]

    def precisao_em_k(self, consultas: np.ndarray, rotulos_consulta: np.ndarray, k: int = 5) -> float:
        """Fração das k recomendações que são da mesma categoria da consulta (média sobre as consultas)."""
        acertos = [(self.rotulos[self.buscar(c, k)[0]] == r).mean() for c, r in zip(consultas, rotulos_consulta, strict=True)]
        return float(np.mean(acertos))
