import sys
from spacy.tokens import Doc, Span, Token
from nltk.stem.snowball import SnowballStemmer



class Steaming(object):
    name = 'stemmer'

    def __init__(self, lang):
        try:
            dict_lang = {'es': 'spanish', 'en': 'english'}
            self.stemmer = SnowballStemmer(dict_lang[lang])
            Token.set_extension('stem', default='', force=True)
        except Exception as e:
            print(e)

    def __call__(self, doc):
        try:
            for token in doc:
                if not token.is_punct and not token.is_stop and not token.is_digit:
                    token._.set('stem', self.stemmer.stem(token.text))
            return doc
        except Exception as e:
            print(e)