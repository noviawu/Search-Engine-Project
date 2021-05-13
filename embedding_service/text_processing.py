"""
basic text processing from previous assignments
it's used to get valid tokens from the text before generating fasttext embeddings
"""
import re
import math
from typing import Any, List
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
from nltk.corpus import stopwords  # type: ignore


class TextProcessing:
    def __init__(self, stemmer, stop_words, *args):
        self.stemmer = stemmer
        self.STOP_WORDS = stop_words

    @classmethod
    def from_nltk(
        cls, stemmer: Any = PorterStemmer().stem, stop_words=None
    ) -> "TextProcessing":
        if stop_words is None:
            stop_words = set(stopwords.words("english"))
        return cls(stemmer, stop_words)

    def is_stop_words(self, token: str) -> bool:
        return token in self.STOP_WORDS

    def is_valid(self, token: str) -> bool:
        return len(token) > 1 and (not self.is_stop_words(token))

    def normalize(self, token: str, use_stemmer: bool) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9\-]", "", token.lower())
        if self.is_valid(normalized):
            if use_stemmer:
                return self.stemmer(normalized)
            else:
                return normalized
        else:
            return ""

    def get_valid_tokens(
        self, title: str, content: str, *, use_stemmer: bool = True
    ) -> List[str]:
        tokens = word_tokenize(content.lower()) + title.lower().split()
        normalized = []
        for tok in tokens:
            normalized_tok = self.normalize(tok, use_stemmer)
            if normalized_tok:
                normalized.append(normalized_tok)

        return normalized

    # adopted from PA4
    @staticmethod
    def idf(N: int, df: int) -> float:
        """
        document frequency (df) is the number of documents the term occurred in
        measures how "informative" a term is
        compute the logarithmic (base 2) idf score (normalized inverse df, so higher idf, more informative the term)
        :param N: document count N
        :param df: document frequency
        :return: the idf given the parameters
        """
        return math.log2(N / df)

    @staticmethod
    def tf(freq: int) -> float:
        """
        term frequency (tf) is the number of times term t occurred in document d
        measures how "relevant" is a document
        compute the logarithmic tf (base 2) score (normalized tf, higher tf, more relevant the document)
        :param freq: raw term frequency
        :return: the weighted tf based on the raw term frequency
        """
        if freq > 0:
            return 1 + math.log2(freq)
        return 0


if __name__ == "__main__":
    pass
