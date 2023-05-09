import unicodedata
from typeguard import typechecked
from nltk.corpus import stopwords as nltk_stopwords
from typing import List, Union
from nltk.tokenize import word_tokenize, MWETokenizer, sent_tokenize
import string
import spacy


@typechecked
def lemmatise(text: str, nlp: spacy.Language=None, doc: spacy.tokens.doc.Doc=None) -> str:
    """
    Obtains the lemmas of each word in the text.

    Parameters
    ----------
    text: str
        Any text.
    nlp: spacy.Language
        A spacy Language model.
    doc: spacy.tokens.doc.Doc
        A spacy document. If this one is given, nlp is not needed.

    Returns
    -------
    str
        Lemmatised `text`.
    """
    assert nlp is not None or doc is not None, "Either `nlp` or `doc` must be passed"
    try:
        if doc is None:
            doc = nlp(text)
    except IndexError as e:
        print("FOR DEBUGGING PURPOSES")
        print(text)
        with open("bad_text.txt", "w") as f:
            f.write(text)
        raise e
    return " ".join([w.lemma_ for w in doc])


@typechecked
def tokenise(
    text: str, mwes: Union[str, List[str]] = None, mwe_tokeniser: MWETokenizer = None, nlp: spacy.Language = None,
) -> List[str]:
    """
    Tokenises with the word tokeniser from NLTK, and re-tokenises
    with the multi-word expression (MWE) tokeniser from NLTK.

    Parameters
    ----------
    text: str
        Any text.
    mwes: Union[str, List[str]], optional
        Multi-word Expressions.
        It can be a string or a list of strings. If it is a string,
        it is the path to a file that contains on each line a MWE.
        If it is a list of strings, each string is a MWE.
    mwe_tokeniser: MWETokenizer, optional
        Multi-word expression tokeniser from NLTK.
        Sometimes, many texts are tokenised with the same multi-word
        expressions. Therefore, defining a MWETokenizer for each text
        is a waste of time. Using this option allows fast tokenisation.
    nlp: spacy.Language, optional
        If given, word tokenisation is done with a spacy Language model.

    Returns
    -------
    List[str]
        Tokenised `text`.
    """
    if isinstance(mwes, str):
        with open(mwes, "r") as f:
            mwes_file = mwes
            mwes = f.readlines()
            print(f"{len(mwes)} multi-word expressions found in {mwes_file}")
    if nlp is not None:
        try:
            inner_tokenisation = lambda text: [token.text for token in nlp(text)]
        except IndexError:
            print("FOR DEBUGGING PURPOSES")
            print(text)
            with open("bad_text.txt", "w") as f:
                f.write(text)
            raise IndexError
    else:
        inner_tokenisation = lambda text: word_tokenize(text, language='spanish')
    tokens = inner_tokenisation(text)
    if mwe_tokeniser is None and mwes is not None:
        mwe_tokeniser = MWETokenizer([mwe.split() for mwe in mwes])
    if mwe_tokeniser is not None:
        tokens = mwe_tokeniser.tokenize(tokens)
    return tokens


@typechecked
def proper_encoding(text: str) -> str:
    """
    This function normalises special characters and encodes text
    into a canonical form.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    str
        Normalised version of `text`.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.encode("utf-8", "ignore")
    text = text.decode("utf-8")
    return text


@typechecked
def remove_stopwords(
    text: str,
    stopwords: List[str] = None,
    language: str = "spanish",
    additional_stopwords: Union[str, List[str]] = None,
    **kwargs,
) -> str:
    """
    Removes NLTK's stopwords from a text.

    Parameters
    ----------
    text: str
        Any text.
    stopwords: List[str]
        Preloaded stopwords.
    language: str
        Language of NLTK's stopwords.
    additional_stopwords: Union[str, List[str]], optional
        Can be a string or a list of strings. If it is a string,
        it has to be the path to a file that contains on each line
        a stopword. If it is a list of strings, each string is a
        new stopword.
    **kwargs
        Arguments for `tokenise`

    Returns
    -------
    str
        `text` without the stopwords.
    """
    if stopwords is None:
        stopwords = nltk_stopwords.words(language)
    if isinstance(additional_stopwords, str):
        with open(additional_stopwords, "r") as f:
            additional_stopwords_file = additional_stopwords
            additional_stopwords = f.readlines()
            print(
                f"{len(additional_stopwords)} stopwords found in {additional_stopwords_file}"
            )
    if additional_stopwords is not None:
        stopwords.extend(additional_stopwords)
    return " ".join([w for w in tokenise(text, **kwargs) if w not in stopwords])


@typechecked
def remove_punctuation(
    text: str, string_punctuation: bool = True, symbols: List[str] = None
) -> str:
    """
    Removes punctuation from a text.

    Parameters
    ----------
    text: str
        Any text.
    string_punctuation: bool
        Whether to remove !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ from
        the `text`.
    symbols: List[str]
        List of punctuation symbols to remove from `text`.

    Returns
    -------
    str
        `text` without punctuation.

    Examples
    --------
    Note that if we want to only delete '!?', we can use this function as

    >>> remove_punctuation('hola! cómo estás?', string_punctuation=False, symbols=['!', '?'])
    hola cómo estás
    """
    punkts = ""
    if string_punctuation:
        punkts += string.punctuation
    if symbols is not None:
        punkts += "".join(symbols)
    return text.translate(str.maketrans("", "", punkts))

@typechecked
def tokenise_sentences(text: str, language: str="spanish") -> List[str]:
    """
    Tokenises a text into sentences using NLTK's sentence tokeniser.
    This is essentially a wrapper around NLTK's `sent_tokenize`.

    Parameters
    ----------
    text : str
        Any text.
    language : str
        Language of NLTK's sentence tokeniser to load

    Returns
    -------
    List[str]
        A list of sentences extracted from `text`.
    """
    return sent_tokenize(text, language)
