
# C:\dev\GovernEdge_CLI\utils\utils_custom_normalizer.py

import logging
import numpy as np
from numpy.linalg import norm

# ---------------- logging ----------------
logger = logging.getLogger(__name__)

class NormalizedEmbeddingWrapper:
    """
    Wrap an embedding model to guarantee unit-normalized output vectors.
    
    Ensures consistent cosine similarity behavior regardless of the base model,
    since not all HuggingFace / external embeddings are normalized by default.
    """

    def __init__(self, embedding_model):
        """
        Args:
            embedding_model: any model exposing embed_documents() and embed_query().
        """
        self.model = embedding_model
        logger.debug("Initialized NormalizedEmbeddingWrapper for %s", type(embedding_model).__name__)

    def embed_documents(self, texts):
        """
        Embed a list of documents and L2-normalize each vector.
        """
        logger.debug("Embedding %d documents", len(texts))
        vectors = self.model.embed_documents(texts)
        normalized = [v / norm(v) if norm(v) > 0 else v for v in vectors]
        logger.debug("Embedded and normalized %d documents", len(normalized))
        return normalized

    def embed_query(self, text):
        """
        Embed a single query string and L2-normalize the resulting vector.
        """
        logger.debug("Embedding query of length %d chars", len(text))
        v = self.model.embed_query(text)
        n = norm(v)
        if n == 0:
            logger.warning("Zero vector encountered for query embedding.")
            return v
        return v / n

    def __call__(self, input):
        """
        Make the wrapper callable, routing automatically:
          - list[str] → embed_documents
          - str       → embed_query
        """
        if isinstance(input, list):
            logger.debug("Called with list input (%d items)", len(input))
            return self.embed_documents(input)
        else:
            logger.debug("Called with single input (type=%s)", type(input).__name__)
            return self.embed_query(input)





# llm_core_tst/utils_tst/utils_custom_normalizer_tst.py
#import numpy as np

#class NormalizedEmbeddingWrapper:
    #def __init__(self, embedding_model):
        #self.model = embedding_model

    #def _normalize(self, vec):
        #arr = np.asarray(vec, dtype=np.float32)
        #n = float(np.linalg.norm(arr))
        #if not np.isfinite(n) or n == 0.0:
           # return arr  # leave as-is if we somehow got a zero/NaN vector
       # return arr / n

   # def embed_documents(self, texts):
       # vectors = self.model.embed_documents(texts)  # List[List[float]]
       # return [self._normalize(v).tolist() for v in vectors]

    #def embed_query(self, text):
       # v = self.model.embed_query(text)  # List[float]
        #return self._normalize(v).tolist()

   # def __call__(self, input):
        # Some callers treat the embedding function as callable
        #if isinstance(input, list):
           # return self.embed_documents(input)
       # else:
           # return self.embed_query(input)
