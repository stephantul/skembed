from __future__ import annotations
from typing import Union, List, Optional, Callable
from functools import partial

import numpy as np
from reach import Reach

from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.base import BaseEstimator, TransformerMixin


ArrayLike = Union[List[str], np.ndarray]
Label = Union[str, int]

POOLING = {"mean": partial(np.mean, axis=0), "max": partial(np.max, axis=0), "sum": partial(np.sum, axis=0)}


class EmbeddingVectorizer(BaseEstimator, TransformerMixin, _VectorizerMixin):

    reach_cls = Reach

    def __init__(
        self,
        path_to_embeddings: str,
        *,
        pooling_function: Union[Callable, str] = "mean",
        input: str ="content",
        encoding: str ="utf-8",
        decode_error: str ="strict",
        strip_accents: Optional[Union[str, Callable]] = None,
        lowercase: bool =True,
        preprocessor: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        stop_words: Optional[Union[str, List[str]]] = None,
        token_pattern: Optional[str] = r"(?u)\b\w\w+\b",
        vocabulary: Optional[List[str]] = None,
    ):
        super().__init__()
        self.path_to_embeddings = path_to_embeddings
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.vocabulary = vocabulary
        self.pooling_function = pooling_function
        
        self.analyzer = "word"
        self.ngram_range = (1, 1)

    def _check_pooling_function(self, function: Callable) -> bool:
        pooled = function(np.random.randn(10, 300))
        if pooled.shape != (300,):
            raise ValueError(f"Your pooling function returned the wrong shape. Expected (300,), got {pooled.shape}")

    def fit(self, X: ArrayLike, y: Optional[List[Label]] = None) -> EmbeddingVectorizer:
        try:
            if callable(self.pooling_function):
                self._check_pooling_function(self.pooling_function)
                self._pooler = self.pooling_function
            else:
                self._pooler = POOLING[self.pooling_function]
        except KeyError:
            raise ValueError(f"Your pooling function was not in the set of pooling functions. Got '{self.pooling_function}', expected one of {set(POOLING.keys())} ")

        # Currently no options passed to reach
        # Could be part of the __init__, but maybe no need
        self._embeddings = self.reach_cls.load(self.path_to_embeddings)
        
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        analyzer = self.build_analyzer()

        out = []

        for doc in X:
            tokens = analyzer(doc)
            vector = self._pooler(self._embeddings.vectorize(tokens, remove_oov=True))
            out.append(vector)

        return np.stack(out)
