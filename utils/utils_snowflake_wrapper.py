
# C:\dev\GovernEdge_CLI\utils\utils_snowflake_wrapper.py

import logging
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------- logging ----------------
logger = logging.getLogger(__name__)


class SnowflakeArcticEmbedder:
    """
    Wrapper around Snowflake's Arctic embedding model.

    Provides normalized embeddings for queries and documents.
    - Queries are prefixed with "query: " to match training format.
    - Documents are embedded without prefix.
    """

    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-m-v2.0"):
        """
        Initialize tokenizer + model and place on GPU if available.
        """
        logger.info("Loading Snowflake Arctic model: %s", model_name)

        # Load tokenizer + model with no pooling (CLS token used)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            trust_remote_code=True,  # required for Arctic
        )

        self.model.eval()  # set to inference mode

        # Resolve device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info("Model loaded on device: %s", self.device)

    def _embed(self, texts, prefix: str = ""):
        """
        Internal helper: embed one or more texts with optional prefix.

        Args:
            texts: list[str] of inputs
            prefix: e.g. "query: " for queries, "" for documents

        Returns:
            numpy.ndarray of normalized embeddings (shape: [len(texts), dim])
        """
        if not texts:
            logger.warning("Called _embed with empty texts list.")
            return []

        # Apply prefix consistently
        inputs = [prefix + text for text in texts]

        # Tokenize and move to device
        tokens = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=8192,
        ).to(self.device)

        logger.debug("Tokenized %d inputs (max length=%d).", len(texts), tokens["input_ids"].shape[1])

        # Forward pass with no gradient tracking
        with torch.no_grad():
            output = self.model(**tokens)
            cls_embeddings = output.last_hidden_state[:, 0, :]  # CLS token
            normalized = F.normalize(cls_embeddings, p=2, dim=1)  # L2-normalize

        logger.debug("Generated embeddings (shape=%s)", normalized.shape)
        return normalized.cpu().numpy()

    def embed_query(self, text: str):
        """
        Embed a single query with "query: " prefix.
        Returns a 1D numpy vector.
        """
        logger.info("Embedding query (length=%d chars)", len(text))
        return self._embed([text], prefix="query: ")[0]

    def embed_documents(self, texts):
        """
        Embed multiple documents (no prefix).
        Returns list of numpy vectors.
        """
        logger.info("Embedding %d documents", len(texts))
        return self._embed(texts, prefix="")
