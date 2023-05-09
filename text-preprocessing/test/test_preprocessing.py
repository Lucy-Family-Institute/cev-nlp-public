from preprocess import PreprocessPipeline
import spacy

nlp = spacy.load("es_core_news_sm", disable=['parser', 'ner'])

def test_multiprocessing():
    pp = PreprocessPipeline(nlp=nlp)
    print(pp("hola amiguitos! qué tal todo?"))
    print(pp([f"hola amiguitos! qué tal todo?_{i}" for i in range(1000)], num_cpus=3))

if __name__ == "__main__":
    test_multiprocessing()
