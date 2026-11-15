import numpy as np

from recomendacao.indice import Indice, normalizar


def test_normalizar_deixa_norma_um_e_aguenta_vetor_zero():
    v = normalizar(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(v[0]), 1)
    assert np.all(np.isfinite(v))


def test_busca_ordena_por_similaridade_de_cosseno():
    indice = Indice(np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=float), np.array([0, 0, 1, 2]))
    ordem, sims = indice.buscar(np.array([1.0, 0.0]), k=3)
    assert ordem.tolist() == [0, 1, 2]
    assert sims[0] >= sims[1] >= sims[2]


def test_escala_nao_importa_so_a_direcao():
    indice = Indice(np.array([[10, 0], [0, 0.1]], dtype=float), np.array([0, 1]))
    assert indice.buscar(np.array([0.0, 5.0]), k=1)[0].tolist() == [1]


def test_excluir_tira_o_proprio_item():
    indice = Indice(np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=float), np.array([0, 0, 1]))
    assert 0 not in indice.buscar(np.array([1.0, 0.0]), k=2, excluir=0)[0]


def test_precisao_em_k():
    embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=float)
    indice = Indice(embeddings, np.array([0, 0, 1, 1]))
    assert indice.precisao_em_k(np.array([[1.0, 0.05], [0.05, 1.0]]), np.array([0, 1]), k=2) == 1.0
    assert indice.precisao_em_k(np.array([[1.0, 0.05]]), np.array([1]), k=2) == 0.0


def test_extrator_compartilha_camadas_com_o_classificador():
    from recomendacao.modelo import criar

    classificador, extrator, base = criar(pesos=None)
    assert extrator.get_layer("embedding") is classificador.get_layer("embedding")
    assert extrator.output_shape == (None, 256)
    assert not base.trainable
