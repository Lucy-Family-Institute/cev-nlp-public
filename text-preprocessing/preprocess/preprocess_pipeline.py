from typeguard import typechecked
from .preprocess import (
    remove_punctuation as _remove_punctuation,
    remove_stopwords as _remove_stopwords,
    lemmatise as _lemmatise,
    tokenise as _tokenise,
    proper_encoding as _proper_encoding,
)
import spacy
from typing import Union, List, Iterable
from nltk.tokenize import MWETokenizer
import os
from tqdm.contrib.concurrent import thread_map
from nltk.corpus import stopwords as nltk_stopwords


class PreprocessPipeline:
    """
    Creates a pipeline to preprocess raw text.
    """

    @typechecked
    def __init__(
        self,
        normalise: bool = True,
        lower: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        lemmatise: bool = True,
        tokenise: bool = True,
        string_punctuation: bool = True,
        symbols: List[str] = None,
        language: str = "spanish",
        additional_stopwords: Union[str, List[str]] = None,
        mwes: Union[str, List[str]] = None,
        mwe_tokeniser: MWETokenizer = None,
        nlp: spacy.Language = None,
    ):
        """
        Initialises a preprocessing pipeline.

        Parameters
        ----------
        normalise: bool
            Whether to normalise text with a proper encoding.
        lower: bool
            Whether to lowercase text.
        remove_punctuation: bool
            Whether to remove punctuation.
        remove_stopwords: bool
            Whether to remove stopwords.
        lemmatise: bool
            Whether to perform lemmatisation.
        tokenise: bool
            Whether to tokenise text or not.
        Other args are defined in the functions from `preprocess.preprocess`
        """
        self._funs = []
        if normalise:
            self._funs.append(_proper_encoding)
        if lower:
            self._funs.append(lambda x: x.lower())
        if lemmatise:
            if nlp is None:
                raise TypeError("You have selected the option to lemmatise. You must provide a spaCy nlp model to `nlp`")
            self._funs.append(lambda x: _lemmatise(x, nlp))
        if remove_punctuation:
            self._funs.append(lambda x: _remove_punctuation(x, string_punctuation, symbols))
        if remove_stopwords:
            stopwords = nltk_stopwords.words(language)
            self._funs.append(lambda x: _remove_stopwords(x, stopwords, language, additional_stopwords, mwes=mwes, mwe_tokeniser=mwe_tokeniser, nlp=nlp))
        if tokenise:
            self._funs.append(lambda x: _tokenise(x, mwes, mwe_tokeniser, nlp))

    def worker(self, x):
        for fun in self._funs:
            x = fun(x)
        return x

    def __call__(self, text: Union[str, Iterable[str]], num_cpus: int = 1) -> List[str]:
        """
        Performs preprocessing on text, or on a batch of text

        Parameters
        ----------
        text: Union[str, Iterable[str]]
            Either a text, or an iterable of texts.
        num_cpus: int
            If text is a batch, multiprocessing will be performed using
            `num_cpus` as the number of cores to distribute the computation.
            Set to -1 to use all available cores.

        Returns
        -------
        List[str]
            List with preprocessed text on each component of the list.
        """
        if num_cpus == -1:
            num_cpus = os.cpu_count()
        if isinstance(text, str):
            return self.worker(text)
        else:
            if num_cpus == 1:
                return [self.worker(x) for x in text]
            else:
                return thread_map(self.worker, text, max_workers=num_cpus)
