import unicodedata
import re


class Common:

    @classmethod
    def clean_text_all(cls, text):
        text = text.lstrip().rstrip().lower()
        text = cls.proper_encoding(text)
        text = cls.delete_special_characters(text)
        text = cls.remove_punctuation(text)
        return text

    @staticmethod
    def proper_encoding(text):
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return text

    @staticmethod
    def delete_special_characters(text):
        text = re.sub('\/|\\|\\.|\,|\;|\:|\n|\?|\)|\(|\!|\¡|\¿|\'|\t', ' ', text)
        text = re.sub("\s+\w\s+", " ", text)
        text = re.sub("\.", "", text)
        text = re.sub("|", "", text)
        text = re.sub("@", "", text)
        text = re.sub("  ", " ", text)
        return text.lower()

    @staticmethod
    def remove_punctuation(text):
        try:
            punctuation = {'/', '"', '(', ')', '.', ',', '%', ';', '?', '¿', '!', '¡',
                           ':', '#', '$', '&', '>', '<', '-', '_', '°', '|', '¬', '\\', '*', '+',
                           '[', ']', '{', '}', '=', "'", '@'}
            for sign in punctuation:
                text = text.replace(sign, '')
            return text
        except:
            # Logging.write_standard_error(sys.exc_info())
            return None