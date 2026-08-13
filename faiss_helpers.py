from __future__ import annotations
import faiss, numpy as np

def create_index(dimension: int, index_type: str, nlist: int, pq_m: int, pq_bits: int):
    quantizer = faiss.IndexFlatIP(dimension)
    if index_type == 'ivfpq':
        return faiss.IndexIVFPQ(quantizer, dimension, nlist, pq_m, pq_bits, faiss.METRIC_INNER_PRODUCT)
    if index_type == 'ivfsq8':
        return faiss.IndexIVFScalarQuantizer(quantizer, dimension, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    raise ValueError(index_type)

def train_index(index, vectors: np.ndarray) -> None:
    if not index.is_trained:
        index.train(np.ascontiguousarray(vectors, dtype=np.float32))

def clone_trained_empty(index):
    cloned = faiss.deserialize_index(faiss.serialize_index(index))
    cloned.reset()
    return cloned
